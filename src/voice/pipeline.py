import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any
from contextlib import asynccontextmanager

from ..config import settings
from ..llm import llm_service, ConversationContext, Message, MessageRole
from .models import (
    CallSession, VoiceCallState, AudioChunk, TTSProvider, TTSConfig,
    STTConfig, AudioConfig, VoiceMetrics, TierSwitchEvent, TTSRequest
)
from .tts_manager import TTSManager
from .stt_service import STTService
from .twilio_integration import TwilioIntegration
from .pipecat_integration import pipecat_pipeline

logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(self, use_pipecat: bool = True):
        self.use_pipecat = use_pipecat

        # Initialize all components for compatibility
        self.tts_manager = TTSManager()
        self.stt_service = STTService()
        self.twilio_integration = TwilioIntegration()

        if use_pipecat:
            # Use Pipecat for real-time processing
            self.pipecat_pipeline = pipecat_pipeline

        self.active_calls: Dict[str, CallSession] = {}
        self.is_initialized = False

        # Callbacks
        self._on_call_started: Optional[Callable] = None
        self._on_call_ended: Optional[Callable] = None
        self._on_tier_switched: Optional[Callable] = None
        self._on_transcript: Optional[Callable] = None

    async def initialize(self):
        """Initialize all voice processing components"""
        try:
            logger.info(f"Initializing Voice Pipeline (Pipecat: {self.use_pipecat})...")

            if self.use_pipecat:
                # Initialize Pipecat pipeline and Twilio
                init_tasks = [
                    self.pipecat_pipeline.initialize(),
                    self.twilio_integration.initialize()
                ]
                components = ["Pipecat Pipeline", "Twilio Integration"]
            else:
                # Initialize legacy components
                init_tasks = [
                    self.tts_manager.initialize(),
                    self.stt_service.initialize(),
                    self.twilio_integration.initialize()
                ]
                components = ["TTS Manager", "STT Service", "Twilio Integration"]

            results = await asyncio.gather(*init_tasks, return_exceptions=True)

            # Check results
            for i, result in enumerate(results):
                component = components[i]
                if isinstance(result, Exception):
                    logger.error(f"{component} initialization failed: {result}")
                    return False
                elif not result:
                    logger.error(f"{component} initialization returned False")
                    return False

            self.is_initialized = True
            logger.info("Voice Pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Voice Pipeline initialization failed: {e}")
            return False

    async def start_call(
        self,
        call_id: str,
        phone_number: str,
        lead_data: Optional[Dict[str, Any]] = None,
        initial_tier: TTSProvider = TTSProvider.LOCAL_PIPER,
        twilio_stream_sid: Optional[str] = None
    ) -> bool:
        """Start a new voice call"""
        try:
            if not self.is_initialized:
                logger.error("Voice Pipeline not initialized")
                return False

            if call_id in self.active_calls:
                logger.error(f"Call {call_id} already active")
                return False

            if self.use_pipecat:
                # Use Pipecat pipeline for real-time processing
                if not twilio_stream_sid:
                    logger.error("Twilio stream SID required for Pipecat pipeline")
                    return False

                success = await self.pipecat_pipeline.start_call(
                    call_id=call_id,
                    phone_number=phone_number,
                    twilio_stream_sid=twilio_stream_sid,
                    lead_data=lead_data,
                    initial_tier=initial_tier
                )

                if success:
                    # Get session from Pipecat pipeline
                    session = self.pipecat_pipeline.get_call_session(call_id)
                    if session:
                        self.active_calls[call_id] = session

                        # Set up callbacks
                        self.pipecat_pipeline.on_call_started(self._on_call_started)
                        self.pipecat_pipeline.on_call_ended(self._on_call_ended)
                        self.pipecat_pipeline.on_tier_switched(self._on_tier_switched)
                        self.pipecat_pipeline.on_transcript(self._on_transcript)

                        logger.info(f"Pipecat call {call_id} started successfully")
                        return True

                logger.error(f"Failed to start Pipecat call {call_id}")
                return False

            else:
                # Use legacy pipeline
                return await self._start_call_legacy(call_id, phone_number, lead_data, initial_tier)

        except Exception as e:
            logger.error(f"Error starting call {call_id}: {e}")
            if call_id in self.active_calls:
                del self.active_calls[call_id]
            return False

    async def _start_call_legacy(
        self,
        call_id: str,
        phone_number: str,
        lead_data: Optional[Dict[str, Any]] = None,
        initial_tier: TTSProvider = TTSProvider.LOCAL_PIPER
    ) -> bool:
        """Start call using legacy components"""
        # Create call session
        tts_config = TTSConfig(provider=initial_tier)
        metrics = VoiceMetrics(call_id=call_id)

        session = CallSession(
            call_id=call_id,
            phone_number=phone_number,
            current_tier=initial_tier,
            tts_config=tts_config,
            metrics=metrics,
            lead_data=lead_data or {}
        )

        self.active_calls[call_id] = session

        # Initiate call through Twilio
        call_started = await self.twilio_integration.start_call(
            call_id=call_id,
            phone_number=phone_number
        )

        if call_started:
            session.state = VoiceCallState.RINGING
            session.started_at = time.time()

            # Set up audio processing pipeline
            await self._setup_audio_pipeline(session)

            if self._on_call_started:
                self._on_call_started(session)

            logger.info(f"Legacy call {call_id} started successfully")
            return True
        else:
            # Remove failed call
            del self.active_calls[call_id]
            logger.error(f"Failed to start legacy call {call_id}")
            return False

    async def _setup_audio_pipeline(self, session: CallSession):
        """Set up bidirectional audio processing for a call"""
        try:
            # Start audio stream handlers
            asyncio.create_task(
                self._handle_incoming_audio(session)
            )
            asyncio.create_task(
                self._handle_outgoing_audio(session)
            )

            # Create conversation context for LLM
            conversation_context = ConversationContext(
                call_id=session.call_id,
                lead_info=session.lead_data
            )
            session.conversation_id = session.call_id

            logger.info(f"Audio pipeline set up for call {session.call_id}")

        except Exception as e:
            logger.error(f"Error setting up audio pipeline for {session.call_id}: {e}")
            session.state = VoiceCallState.ERROR
            session.error_message = str(e)

    async def _handle_incoming_audio(self, session: CallSession):
        """Process incoming audio from the call"""
        try:
            session.state = VoiceCallState.LISTENING

            async for audio_chunk in self.twilio_integration.get_audio_stream(session.call_id):
                if session.state == VoiceCallState.ENDED:
                    break

                session.state = VoiceCallState.PROCESSING

                # Transcribe audio
                stt_result = await self.stt_service.transcribe_audio_chunk(
                    audio_chunk,
                    session.call_id
                )

                if stt_result and stt_result.text.strip():
                    logger.info(f"Call {session.call_id} - Transcription: {stt_result.text}")

                    # Update metrics
                    session.metrics.stt_latency_ms = stt_result.processing_time_ms

                    # Trigger transcript callback
                    if self._on_transcript:
                        self._on_transcript(session.call_id, stt_result)

                    # Generate LLM response
                    await self._process_user_input(session, stt_result.text)

                session.state = VoiceCallState.LISTENING

        except Exception as e:
            logger.error(f"Error in incoming audio handler for {session.call_id}: {e}")
            session.state = VoiceCallState.ERROR
            session.error_message = str(e)

    async def _process_user_input(self, session: CallSession, user_text: str):
        """Process user input and generate response"""
        try:
            # Get conversation context
            conversation_context = ConversationContext(
                call_id=session.conversation_id or session.call_id,
                messages=[],  # Would be loaded from database in real implementation
                lead_info=session.lead_data
            )

            # Generate LLM response
            llm_start_time = time.time()
            llm_response = await llm_service.generate_response(
                context=conversation_context,
                user_input=user_text
            )
            llm_processing_time = (time.time() - llm_start_time) * 1000

            if llm_response and llm_response.content:
                # Create TTS request
                tts_request = TTSRequest(
                    text=llm_response.content,
                    call_id=session.call_id,
                    provider=session.current_tier,
                    config=session.tts_config
                )

                # Queue for speech synthesis
                await self._synthesize_and_play(session, tts_request)

                # Check if qualification threshold is met for tier escalation
                await self._check_tier_escalation(session, conversation_context)

        except Exception as e:
            logger.error(f"Error processing user input for {session.call_id}: {e}")

    async def _synthesize_and_play(self, session: CallSession, tts_request: TTSRequest):
        """Synthesize speech and play through call"""
        try:
            session.state = VoiceCallState.SPEAKING

            # Synthesize speech
            tts_response = await self.tts_manager.synthesize(tts_request)

            if tts_response:
                # Update metrics
                session.metrics.tts_latency_ms = tts_response.generation_time_ms
                session.metrics.cost += tts_response.cost

                # Play audio through Twilio
                await self.twilio_integration.play_audio(
                    session.call_id,
                    tts_response.audio_data
                )

                logger.info(f"Call {session.call_id} - Played response: {tts_request.text[:50]}...")
            else:
                logger.error(f"TTS synthesis failed for call {session.call_id}")

        except Exception as e:
            logger.error(f"Error synthesizing speech for {session.call_id}: {e}")

    async def _check_tier_escalation(self, session: CallSession, context: ConversationContext):
        """Check if call should be escalated to premium tier"""
        try:
            if session.current_tier == TTSProvider.ELEVENLABS:
                return  # Already on premium tier

            # Analyze conversation for qualification
            qualification_data = await llm_service.analyze_qualification(context)
            qualification_score = qualification_data.get("qualification_score", 0.0)

            session.metrics.qualification_score = qualification_score

            # Check escalation threshold
            if qualification_score >= settings.tier_escalation_threshold:
                await self.switch_tier(
                    session.call_id,
                    TTSProvider.ELEVENLABS,
                    trigger="qualification"
                )

        except Exception as e:
            logger.error(f"Error checking tier escalation for {session.call_id}: {e}")

    async def switch_tier(
        self,
        call_id: str,
        new_tier: TTSProvider,
        trigger: str = "manual"
    ) -> bool:
        """Switch TTS tier for a call"""
        try:
            if call_id not in self.active_calls:
                logger.error(f"Call {call_id} not found for tier switch")
                return False

            if self.use_pipecat:
                # Use Pipecat tier switching
                return await self.pipecat_pipeline.switch_tier(call_id, new_tier, trigger)
            else:
                # Use legacy tier switching
                return await self._switch_tier_legacy(call_id, new_tier, trigger)

        except Exception as e:
            logger.error(f"Error switching tier for call {call_id}: {e}")
            return False

    async def _switch_tier_legacy(
        self,
        call_id: str,
        new_tier: TTSProvider,
        trigger: str = "manual"
    ) -> bool:
        """Legacy tier switching implementation"""
        session = self.active_calls[call_id]
        old_tier = session.current_tier

        if old_tier == new_tier:
            logger.info(f"Call {call_id} already on tier {new_tier}")
            return True

        session.state = VoiceCallState.TIER_SWITCHING

        # Update TTS configuration
        new_config = TTSConfig(provider=new_tier)
        if new_tier == TTSProvider.ELEVENLABS:
            new_config.voice_id = settings.elevenlabs_voice

        success = await self.tts_manager.switch_tier(call_id, old_tier, new_tier)

        if success:
            session.current_tier = new_tier
            session.tts_config = new_config
            session.metrics.tier_switches += 1

            # Create tier switch event
            switch_event = TierSwitchEvent(
                call_id=call_id,
                from_tier=old_tier,
                to_tier=new_tier,
                trigger=trigger,
                qualification_score=session.metrics.qualification_score
            )

            if self._on_tier_switched:
                self._on_tier_switched(switch_event)

            logger.info(f"Call {call_id}: Tier switched from {old_tier} to {new_tier}")
            session.state = VoiceCallState.CONNECTED
            return True
        else:
            session.state = VoiceCallState.CONNECTED
            return False

    async def _handle_outgoing_audio(self, session: CallSession):
        """Handle outgoing audio stream (placeholder for Pipecat integration)"""
        # This would integrate with Pipecat for real-time audio streaming
        # For now, we rely on the play_audio method in twilio_integration
        pass

    async def end_call(self, call_id: str, reason: str = "completed") -> bool:
        """End a call"""
        try:
            if call_id not in self.active_calls:
                logger.warning(f"Call {call_id} not found for ending")
                return False

            if self.use_pipecat:
                # Use Pipecat end call
                success = await self.pipecat_pipeline.end_call(call_id, reason)
                if success and call_id in self.active_calls:
                    del self.active_calls[call_id]
                return success
            else:
                # Use legacy end call
                return await self._end_call_legacy(call_id, reason)

        except Exception as e:
            logger.error(f"Error ending call {call_id}: {e}")
            return False

    async def _end_call_legacy(self, call_id: str, reason: str = "completed") -> bool:
        """Legacy end call implementation"""
        session = self.active_calls[call_id]
        session.state = VoiceCallState.ENDED
        session.ended_at = time.time()

        # Calculate final metrics
        if session.started_at:
            session.metrics.total_audio_duration_ms = (session.ended_at - session.started_at) * 1000

        # End call through Twilio
        await self.twilio_integration.end_call(call_id)

        # Trigger callback
        if self._on_call_ended:
            self._on_call_ended(session, reason)

        # Remove from active calls
        del self.active_calls[call_id]

        logger.info(f"Call {call_id} ended: {reason}")
        return True

    def get_call_session(self, call_id: str) -> Optional[CallSession]:
        """Get call session by ID"""
        return self.active_calls.get(call_id)

    def get_active_calls(self) -> Dict[str, CallSession]:
        """Get all active calls"""
        return self.active_calls.copy()

    # Callback setters
    def on_call_started(self, callback: Callable[[CallSession], None]):
        self._on_call_started = callback

    def on_call_ended(self, callback: Callable[[CallSession, str], None]):
        self._on_call_ended = callback

    def on_tier_switched(self, callback: Callable[[TierSwitchEvent], None]):
        self._on_tier_switched = callback

    def on_transcript(self, callback: Callable[[str, Any], None]):
        self._on_transcript = callback

    async def health_check(self) -> dict:
        """Check voice pipeline health"""
        try:
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "error": "Voice Pipeline not initialized"
                }

            # Check components
            component_health = await asyncio.gather(
                self.tts_manager.health_check(),
                self.stt_service.health_check(),
                self.twilio_integration.health_check(),
                return_exceptions=True
            )

            tts_health, stt_health, twilio_health = component_health

            return {
                "status": "healthy",
                "components": {
                    "tts": tts_health if not isinstance(tts_health, Exception) else {"error": str(tts_health)},
                    "stt": stt_health if not isinstance(stt_health, Exception) else {"error": str(stt_health)},
                    "twilio": twilio_health if not isinstance(twilio_health, Exception) else {"error": str(twilio_health)}
                },
                "active_calls": len(self.active_calls),
                "initialized": self.is_initialized
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self):
        """Clean up voice pipeline resources"""
        try:
            # End all active calls
            for call_id in list(self.active_calls.keys()):
                await self.end_call(call_id, "shutdown")

            if self.use_pipecat:
                # Clean up Pipecat components
                cleanup_tasks = [
                    self.pipecat_pipeline.cleanup(),
                    self.twilio_integration.cleanup()
                ]
            else:
                # Clean up legacy components
                cleanup_tasks = [
                    self.tts_manager.cleanup(),
                    self.stt_service.cleanup(),
                    self.twilio_integration.cleanup()
                ]

            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            self.is_initialized = False
            logger.info("Voice Pipeline cleanup complete")

        except Exception as e:
            logger.error(f"Error during voice pipeline cleanup: {e}")
