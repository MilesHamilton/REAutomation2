import asyncio
import aiohttp
import logging
import time
import subprocess
import tempfile
import os
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from ..config import settings
from .models import TTSRequest, TTSResponse, TTSProvider, TTSConfig
from .piper_integration import PiperEngine, PIPER_AVAILABLE

logger = logging.getLogger(__name__)


class TTSManager:
    def __init__(self):
        self._providers: Dict[TTSProvider, Any] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize all TTS providers"""
        try:
            # Initialize HTTP session for API calls
            self._session = aiohttp.ClientSession()

            # Initialize local TTS providers
            await self._initialize_piper()
            await self._initialize_coqui()

            # Validate 11Labs if configured
            if settings.elevenlabs_api_key:
                await self._validate_elevenlabs()

            self.is_initialized = True
            logger.info("TTS Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"TTS Manager initialization failed: {e}")
            return False

    async def _initialize_piper(self):
        """Initialize Piper TTS"""
        try:
            if not PIPER_AVAILABLE:
                logger.warning("Piper TTS library not available")
                self._providers[TTSProvider.LOCAL_PIPER] = {"available": False}
                return

            # Initialize PiperEngine
            piper_engine = PiperEngine(
                models_path=settings.piper_models_path,
                default_voice=settings.piper_default_voice,
                use_gpu=settings.piper_use_gpu,
                prewarm=settings.piper_prewarm_model,
                speaker_id=settings.piper_speaker_id
            )

            # Initialize the engine
            success = await piper_engine.initialize()

            if success:
                available_voices = piper_engine.get_available_voices()
                self._providers[TTSProvider.LOCAL_PIPER] = {
                    "available": True,
                    "engine": piper_engine,
                    "voices": available_voices,
                    "default_voice": settings.piper_default_voice
                }
                logger.info(f"Piper TTS initialized with {len(available_voices)} voices")
            else:
                logger.warning("Piper engine initialization failed")
                self._providers[TTSProvider.LOCAL_PIPER] = {"available": False}

        except Exception as e:
            logger.warning(f"Piper TTS initialization failed: {e}")
            self._providers[TTSProvider.LOCAL_PIPER] = {"available": False}

    async def _initialize_coqui(self):
        """Initialize Coqui TTS"""
        try:
            import TTS
            from TTS.api import TTS as CoquiTTS

            # Initialize with a fast model for real-time use
            tts = CoquiTTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False
            )

            self._providers[TTSProvider.LOCAL_COQUI] = {
                "available": True,
                "engine": tts,
                "model": "tacotron2-DDC"
            }
            logger.info("Coqui TTS initialized")

        except ImportError:
            logger.warning("Coqui TTS not installed")
            self._providers[TTSProvider.LOCAL_COQUI] = {"available": False}
        except Exception as e:
            logger.warning(f"Coqui TTS initialization failed: {e}")
            self._providers[TTSProvider.LOCAL_COQUI] = {"available": False}

    async def _validate_elevenlabs(self):
        """Validate 11Labs API access"""
        try:
            url = "https://api.elevenlabs.io/v1/voices"
            headers = {"xi-api-key": settings.elevenlabs_api_key}

            async with self._session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    voices = data.get("voices", [])

                    self._providers[TTSProvider.ELEVENLABS] = {
                        "available": True,
                        "voices": {voice["voice_id"]: voice["name"] for voice in voices},
                        "default_voice": settings.elevenlabs_voice or voices[0]["voice_id"] if voices else None
                    }
                    logger.info(f"11Labs API validated with {len(voices)} voices")
                else:
                    logger.error(f"11Labs API validation failed: {response.status}")
                    self._providers[TTSProvider.ELEVENLABS] = {"available": False}

        except Exception as e:
            logger.error(f"11Labs validation error: {e}")
            self._providers[TTSProvider.ELEVENLABS] = {"available": False}

    async def synthesize(self, request: TTSRequest) -> Optional[TTSResponse]:
        """Synthesize speech using the specified provider"""
        if not self.is_initialized:
            logger.error("TTS Manager not initialized")
            return None

        provider_info = self._providers.get(request.provider)
        if not provider_info or not provider_info.get("available"):
            logger.error(f"TTS provider {request.provider} not available")
            return None

        start_time = time.time()

        try:
            if request.provider == TTSProvider.LOCAL_PIPER:
                audio_data = await self._synthesize_piper(request)
            elif request.provider == TTSProvider.LOCAL_COQUI:
                audio_data = await self._synthesize_coqui(request)
            elif request.provider == TTSProvider.ELEVENLABS:
                audio_data = await self._synthesize_elevenlabs(request)
            else:
                logger.error(f"Unknown TTS provider: {request.provider}")
                return None

            if audio_data:
                generation_time = (time.time() - start_time) * 1000

                # Calculate approximate audio duration (assuming 16kHz, 16-bit mono)
                audio_duration = len(audio_data) / (16000 * 2) * 1000

                # Calculate cost
                cost = self._calculate_cost(request.provider, audio_duration)

                return TTSResponse(
                    audio_data=audio_data,
                    call_id=request.call_id,
                    chunk_id=request.chunk_id,
                    generation_time_ms=generation_time,
                    audio_duration_ms=audio_duration,
                    provider=request.provider,
                    cost=cost
                )

            return None

        except Exception as e:
            logger.error(f"TTS synthesis error for {request.provider}: {e}")
            return None

    async def _synthesize_piper(self, request: TTSRequest) -> Optional[bytes]:
        """Synthesize using Piper TTS"""
        try:
            provider_info = self._providers[TTSProvider.LOCAL_PIPER]
            piper_engine: PiperEngine = provider_info["engine"]

            # Get voice name
            voice_name = request.config.voice_id or provider_info["default_voice"]

            # Synthesize using PiperEngine
            audio_array = await piper_engine.synthesize(
                text=request.text,
                voice_name=voice_name
            )

            if audio_array is None:
                logger.error("Piper synthesis returned None")
                return None

            # Convert numpy array to PCM bytes
            audio_bytes = self._numpy_to_pcm_bytes(audio_array)

            # Wrap in WAV format for compatibility
            audio_wav = self._wrap_pcm_in_wav(
                audio_bytes,
                sample_rate=piper_engine.get_sample_rate(voice_name),
                channels=1,
                bit_depth=16
            )

            return audio_wav

        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            return None

    def _numpy_to_pcm_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy float32 array to 16-bit PCM bytes"""
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Clip to prevent overflow
        audio_array = np.clip(audio_array, -1.0, 1.0)

        # Convert to 16-bit signed integers
        audio_int16 = (audio_array * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _wrap_pcm_in_wav(self, pcm_data: bytes, sample_rate: int, channels: int, bit_depth: int) -> bytes:
        """Wrap raw PCM data in WAV container"""
        import struct

        # Calculate sizes
        data_size = len(pcm_data)
        file_size = 36 + data_size

        # Create WAV header
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',           # ChunkID
            file_size,         # ChunkSize
            b'WAVE',           # Format
            b'fmt ',           # Subchunk1ID
            16,                # Subchunk1Size (PCM)
            1,                 # AudioFormat (PCM)
            channels,          # NumChannels
            sample_rate,       # SampleRate
            sample_rate * channels * bit_depth // 8,  # ByteRate
            channels * bit_depth // 8,  # BlockAlign
            bit_depth,         # BitsPerSample
            b'data',           # Subchunk2ID
            data_size          # Subchunk2Size
        )

        return wav_header + pcm_data

    async def _synthesize_coqui(self, request: TTSRequest) -> Optional[bytes]:
        """Synthesize using Coqui TTS"""
        try:
            provider_info = self._providers[TTSProvider.LOCAL_COQUI]
            tts_engine = provider_info["engine"]

            # Run synthesis in executor to avoid blocking
            loop = asyncio.get_event_loop()

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            await loop.run_in_executor(
                None,
                tts_engine.tts_to_file,
                request.text,
                temp_path
            )

            with open(temp_path, 'rb') as f:
                audio_data = f.read()

            os.unlink(temp_path)
            return audio_data

        except Exception as e:
            logger.error(f"Coqui synthesis error: {e}")
            return None

    async def _synthesize_elevenlabs(self, request: TTSRequest) -> Optional[bytes]:
        """Synthesize using 11Labs API"""
        try:
            provider_info = self._providers[TTSProvider.ELEVENLABS]
            voice_id = request.config.voice_id or provider_info["default_voice"]

            if not voice_id:
                logger.error("No voice ID configured for 11Labs")
                return None

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

            headers = {
                "Accept": "application/json",
                "xi-api-key": settings.elevenlabs_api_key,
                "Content-Type": "application/json"
            }

            payload = {
                "text": request.text,
                "model_id": request.config.model_id or settings.elevenlabs_model,
                "voice_settings": {
                    "stability": request.config.stability,
                    "similarity_boost": request.config.similarity_boost,
                    "use_speaker_boost": True
                },
                "optimize_streaming_latency": request.config.optimize_streaming_latency
            }

            async with self._session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    error_text = await response.text()
                    logger.error(f"11Labs API error {response.status}: {error_text}")
                    return None

        except Exception as e:
            logger.error(f"11Labs synthesis error: {e}")
            return None

    def _calculate_cost(self, provider: TTSProvider, audio_duration_ms: float) -> float:
        """Calculate cost based on provider and audio duration"""
        duration_seconds = audio_duration_ms / 1000

        if provider == TTSProvider.ELEVENLABS:
            # 11Labs pricing (approximate)
            return duration_seconds * 0.003  # $0.003 per second
        else:
            # Local TTS - minimal compute cost
            return duration_seconds * 0.0001  # $0.0001 per second

    async def switch_tier(
        self,
        call_id: str,
        from_provider: TTSProvider,
        to_provider: TTSProvider
    ) -> bool:
        """Switch TTS tier for a call"""
        try:
            to_provider_info = self._providers.get(to_provider)
            if not to_provider_info or not to_provider_info.get("available"):
                logger.error(f"Target provider {to_provider} not available for tier switch")
                return False

            logger.info(f"Call {call_id}: TTS tier switched from {from_provider} to {to_provider}")
            return True

        except Exception as e:
            logger.error(f"Tier switch error for call {call_id}: {e}")
            return False

    def get_available_providers(self) -> list[TTSProvider]:
        """Get list of available TTS providers"""
        return [
            provider for provider, info in self._providers.items()
            if info.get("available", False)
        ]

    async def health_check(self) -> dict:
        """Check TTS manager health"""
        try:
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "error": "TTS Manager not initialized"
                }

            provider_status = {}
            for provider, info in self._providers.items():
                provider_status[provider.value] = {
                    "available": info.get("available", False),
                    "details": {k: v for k, v in info.items() if k != "engine"}
                }

            return {
                "status": "healthy",
                "providers": provider_status,
                "initialized": self.is_initialized
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self):
        """Clean up resources"""
        if self._session:
            await self._session.close()
            self._session = None

        # Cleanup PiperEngine if present
        piper_info = self._providers.get(TTSProvider.LOCAL_PIPER)
        if piper_info and piper_info.get("engine"):
            piper_engine = piper_info["engine"]
            if isinstance(piper_engine, PiperEngine):
                await piper_engine.cleanup()

        self.is_initialized = False
        self._providers.clear()
        logger.info("TTS Manager cleanup complete")