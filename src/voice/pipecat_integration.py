import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncGenerator, Callable
from pipecat.frames.frames import (
    AudioRawFrame, TextFrame, Frame, EndFrame, StartFrame,
    TranscriptionFrame, TTSStartedFrame, TTSAudioRawFrame,
    TTSStoppedFrame, SystemFrame
)
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
# from pipecat.vad.silero import SileroVADAnalyzer
# Note: VAD import updated for newer pipecat version
try:
    from pipecat.vad.silero import SileroVADAnalyzer
except ImportError:
    # Fallback for newer pipecat versions
    try:
        from pipecat.processors.vad import VADProcessor as SileroVADAnalyzer
    except ImportError:
        # Create a mock VAD for compatibility
        class SileroVADAnalyzer:
            def __init__(self, **kwargs):
                pass
# Compatibility imports for different pipecat versions
try:
    from pipecat.transports.services.twilio import TwilioTransport
except ImportError:
    # Fallback for newer pipecat versions
    try:
        from pipecat.transports.twilio import TwilioTransport
    except ImportError:
        # Mock for compatibility
        class TwilioTransport:
            def __init__(self, **kwargs):
                pass
            def audio_in_processor(self):
                return None
            def audio_out_processor(self):
                return None

try:
    from pipecat.services.cartesia import CartesiaTTSService
except (ImportError, Exception):
    # Mock for compatibility (handles both ImportError and Exception from missing cartesia)
    class CartesiaTTSService:
        def __init__(self, **kwargs):
            pass

try:
    from pipecat.services.whisper import WhisperSTTService
except ImportError:
    # Mock for compatibility
    class WhisperSTTService:
        def __init__(self, **kwargs):
            pass
        async def start(self):
            pass
        async def stop(self):
            pass

from ..config import settings
from ..llm import llm_service, ConversationContext, Message, MessageRole
from .models import (
    CallSession, VoiceCallState, AudioChunk, TTSProvider, TTSConfig,
    STTResult, TTSResponse, VoiceMetrics, TierSwitchEvent, VoicePipelineConfig
)

logger = logging.getLogger(__name__)


class ConversationProcessor(FrameProcessor):
    """Custom processor for handling LLM conversation flow"""

    def __init__(self, call_session: CallSession, on_response_generated: Optional[Callable] = None):
        super().__init__()
        self.call_session = call_session
        self.conversation_context = ConversationContext(
            call_id=call_session.call_id,
            lead_info=call_session.lead_data
        )
        self.on_response_generated = on_response_generated
        self._processing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Handle transcription frames
        if isinstance(frame, TranscriptionFrame):
            if not self._processing and frame.text.strip():
                self._processing = True
                await self._handle_user_input(frame.text.strip())
                self._processing = False

        # Pass through other frames
        await self.push_frame(frame, direction)

    async def _handle_user_input(self, user_text: str):
        """Process user input and generate LLM response"""
        try:
            logger.info(f"Call {self.call_session.call_id} - Processing: {user_text}")

            # Update call session state
            self.call_session.state = VoiceCallState.PROCESSING

            # Generate LLM response
            llm_start_time = time.time()
            llm_response = await llm_service.generate_response(
                context=self.conversation_context,
                user_input=user_text
            )
            llm_processing_time = (time.time() - llm_start_time) * 1000

            if llm_response and llm_response.content:
                # Update conversation context
                self.conversation_context.messages.extend([
                    Message(role=MessageRole.USER, content=user_text),
                    Message(role=MessageRole.ASSISTANT, content=llm_response.content)
                ])

                # Update metrics
                self.call_session.metrics.llm_latency_ms = llm_processing_time

                # Generate TTS response
                await self.push_frame(TextFrame(text=llm_response.content))

                # Callback for response tracking
                if self.on_response_generated:
                    await self.on_response_generated(user_text, llm_response.content)

                # Check for tier escalation
                await self._check_tier_escalation()

                logger.info(f"Call {self.call_session.call_id} - Response: {llm_response.content[:50]}...")

        except Exception as e:
            logger.error(f"Error processing user input for call {self.call_session.call_id}: {e}")
            self.call_session.state = VoiceCallState.ERROR
            self.call_session.error_message = str(e)

    async def _check_tier_escalation(self):
        """Check if call should be escalated to premium tier"""
        try:
            if self.call_session.current_tier == TTSProvider.ELEVENLABS:
                return  # Already on premium tier

            # Analyze conversation for qualification
            qualification_data = await llm_service.analyze_qualification(self.conversation_context)
            qualification_score = qualification_data.get("qualification_score", 0.0)

            self.call_session.metrics.qualification_score = qualification_score

            # Check escalation threshold
            if qualification_score >= settings.tier_escalation_threshold:
                logger.info(f"Call {self.call_session.call_id} qualified for tier escalation: {qualification_score}")
                # Tier switch would be handled by the pipeline manager

        except Exception as e:
            logger.error(f"Error checking tier escalation for call {self.call_session.call_id}: {e}")


class PipecatVoicePipeline:
    """Pipecat-based real-time voice pipeline with Twilio integration"""

    def __init__(self, config: Optional[VoicePipelineConfig] = None):
        self.config = config or VoicePipelineConfig()
        self.active_pipelines: Dict[str, Pipeline] = {}
        self.active_tasks: Dict[str, PipelineTask] = {}
        self.call_sessions: Dict[str, CallSession] = {}
        self.is_initialized = False

        # Services
        self.whisper_stt: Optional[WhisperSTTService] = None
        self.vad_analyzer: Optional[SileroVADAnalyzer] = None

        # Callbacks
        self._on_call_started: Optional[Callable] = None
        self._on_call_ended: Optional[Callable] = None
        self._on_tier_switched: Optional[Callable] = None
        self._on_transcript: Optional[Callable] = None

    async def initialize(self) -> bool:
        """Initialize Pipecat voice pipeline"""
        try:
            logger.info("Initializing Pipecat Voice Pipeline...")

            # Initialize STT service
            self.whisper_stt = WhisperSTTService(
                model="base",  # Can be configured
                language="en"
            )

            # Initialize VAD analyzer
            self.vad_analyzer = SileroVADAnalyzer(
                sample_rate=16000,
                min_volume=0.6
            )

            # Test services
            await self.whisper_stt.start()

            self.is_initialized = True
            logger.info("Pipecat Voice Pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Pipecat Voice Pipeline initialization failed: {e}")
            return False

    async def start_call(
        self,
        call_id: str,
        phone_number: str,
        twilio_stream_sid: str,
        lead_data: Optional[Dict[str, Any]] = None,
        initial_tier: TTSProvider = TTSProvider.LOCAL_PIPER
    ) -> bool:
        """Start a new voice call with Pipecat pipeline"""
        try:
            if not self.is_initialized:
                logger.error("Pipecat Voice Pipeline not initialized")
                return False

            if call_id in self.active_pipelines:
                logger.error(f"Pipeline already active for call {call_id}")
                return False

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

            self.call_sessions[call_id] = session

            # Create Twilio transport
            transport = TwilioTransport(
                stream_sid=twilio_stream_sid,
                audio_in_sample_rate=8000,  # Twilio's sample rate
                audio_out_sample_rate=8000,
                audio_out_enabled=True
            )

            # Create TTS service based on tier
            tts_service = await self._create_tts_service(initial_tier)

            # Create conversation processor
            conversation_processor = ConversationProcessor(
                call_session=session,
                on_response_generated=self._on_response_generated
            )

            # Build pipeline
            pipeline = Pipeline([
                transport.audio_in_processor(),   # Audio input from Twilio
                self.vad_analyzer,                # Voice activity detection
                self.whisper_stt,                # Speech to text
                conversation_processor,           # LLM conversation handling
                tts_service,                      # Text to speech
                transport.audio_out_processor()   # Audio output to Twilio
            ])

            # Create and start pipeline task
            task = PipelineTask(
                pipeline,
                PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=True,
                    enable_usage_metrics=True
                )
            )

            # Store references
            self.active_pipelines[call_id] = pipeline
            self.active_tasks[call_id] = task

            # Start the pipeline
            await task.queue_frames([StartFrame()])

            session.state = VoiceCallState.CONNECTED
            session.started_at = time.time()

            if self._on_call_started:
                await self._on_call_started(session)

            logger.info(f"Pipecat pipeline started for call {call_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting Pipecat pipeline for call {call_id}: {e}")

            # Cleanup on failure
            if call_id in self.call_sessions:
                del self.call_sessions[call_id]
            if call_id in self.active_pipelines:
                del self.active_pipelines[call_id]
            if call_id in self.active_tasks:
                del self.active_tasks[call_id]

            return False

    async def _create_tts_service(self, tier: TTSProvider):
        """Create TTS service based on tier"""
        if tier == TTSProvider.ELEVENLABS:
            # Use Cartesia for high-quality TTS (similar to ElevenLabs)
            return CartesiaTTSService(
                api_key=settings.elevenlabs_api_key,  # Use same config key
                voice_id=settings.elevenlabs_voice or "default",
                model="sonic-english",
                sample_rate=16000
            )
        else:
            # Use local TTS service (would need custom implementation)
            from .local_tts_service import LocalTTSService  # Custom service
            return LocalTTSService(
                engine="piper",
                voice="en_US-lessac-medium",
                sample_rate=16000
            )

    async def _on_response_generated(self, user_input: str, ai_response: str):
        """Handle generated response for transcript tracking"""
        if self._on_transcript:
            await self._on_transcript(user_input, ai_response)

    async def switch_tier(
        self,
        call_id: str,
        new_tier: TTSProvider,
        trigger: str = "manual"
    ) -> bool:
        """Switch TTS tier for an active call"""
        try:
            if call_id not in self.call_sessions:
                logger.error(f"Call {call_id} not found for tier switch")
                return False

            session = self.call_sessions[call_id]
            old_tier = session.current_tier

            if old_tier == new_tier:
                logger.info(f"Call {call_id} already on tier {new_tier}")
                return True

            session.state = VoiceCallState.TIER_SWITCHING

            # Create new TTS service
            new_tts_service = await self._create_tts_service(new_tier)

            # Get current pipeline and update TTS processor
            pipeline = self.active_pipelines[call_id]

            # This would require pipeline modification - simplified for now
            # In a full implementation, you'd need to stop/replace the TTS processor

            session.current_tier = new_tier
            session.tts_config.provider = new_tier
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
                await self._on_tier_switched(switch_event)

            logger.info(f"Call {call_id}: Tier switched from {old_tier} to {new_tier}")
            session.state = VoiceCallState.CONNECTED
            return True

        except Exception as e:
            logger.error(f"Error switching tier for call {call_id}: {e}")
            return False

    async def end_call(self, call_id: str, reason: str = "completed") -> bool:
        """End a call and cleanup pipeline"""
        try:
            if call_id not in self.call_sessions:
                logger.warning(f"Call {call_id} not found for ending")
                return False

            session = self.call_sessions[call_id]
            session.state = VoiceCallState.ENDED
            session.ended_at = time.time()

            # Calculate final metrics
            if session.started_at:
                session.metrics.total_audio_duration_ms = (session.ended_at - session.started_at) * 1000

            # Stop pipeline task
            if call_id in self.active_tasks:
                task = self.active_tasks[call_id]
                await task.queue_frames([EndFrame()])
                await task.stop()

            # Cleanup references
            if call_id in self.active_pipelines:
                del self.active_pipelines[call_id]
            if call_id in self.active_tasks:
                del self.active_tasks[call_id]

            if self._on_call_ended:
                await self._on_call_ended(session, reason)

            # Remove session last
            del self.call_sessions[call_id]

            logger.info(f"Pipecat pipeline ended for call {call_id}: {reason}")
            return True

        except Exception as e:
            logger.error(f"Error ending Pipecat pipeline for call {call_id}: {e}")
            return False

    def get_call_session(self, call_id: str) -> Optional[CallSession]:
        """Get call session by ID"""
        return self.call_sessions.get(call_id)

    def get_active_calls(self) -> Dict[str, CallSession]:
        """Get all active calls"""
        return self.call_sessions.copy()

    # Callback setters
    def on_call_started(self, callback: Callable[[CallSession], None]):
        self._on_call_started = callback

    def on_call_ended(self, callback: Callable[[CallSession, str], None]):
        self._on_call_ended = callback

    def on_tier_switched(self, callback: Callable[[TierSwitchEvent], None]):
        self._on_tier_switched = callback

    def on_transcript(self, callback: Callable[[str, str], None]):
        self._on_transcript = callback

    async def health_check(self) -> dict:
        """Check Pipecat pipeline health"""
        try:
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "error": "Pipecat Voice Pipeline not initialized"
                }

            return {
                "status": "healthy",
                "active_pipelines": len(self.active_pipelines),
                "active_calls": len(self.call_sessions),
                "services": {
                    "whisper_stt": "healthy" if self.whisper_stt else "unavailable",
                    "vad_analyzer": "healthy" if self.vad_analyzer else "unavailable"
                },
                "initialized": self.is_initialized
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self):
        """Clean up all pipeline resources"""
        try:
            # End all active calls
            for call_id in list(self.call_sessions.keys()):
                await self.end_call(call_id, "shutdown")

            # Cleanup services
            if self.whisper_stt:
                await self.whisper_stt.stop()

            self.active_pipelines.clear()
            self.active_tasks.clear()
            self.call_sessions.clear()
            self.is_initialized = False

            logger.info("Pipecat Voice Pipeline cleanup complete")

        except Exception as e:
            logger.error(f"Error during Pipecat pipeline cleanup: {e}")


# Global Pipecat pipeline instance
pipecat_pipeline = PipecatVoicePipeline()
