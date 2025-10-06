import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
import numpy as np
import io
import wave
from pipecat.frames.frames import TextFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import TTSService

from .piper_integration import PiperEngine, PIPER_AVAILABLE
from ..config import settings

logger = logging.getLogger(__name__)


class TTSEngine:
    """TTS Engine abstraction for different TTS backends"""

    def __init__(self):
        self._engine_type: Optional[str] = None
        self._model = None
        self._is_initialized = False

    async def initialize(self, engine_type: str = "mock", **kwargs):
        """Initialize the TTS engine"""
        self._engine_type = engine_type

        if engine_type == "piper":
            await self._initialize_piper(**kwargs)
        elif engine_type == "coqui":
            await self._initialize_coqui(**kwargs)
        elif engine_type == "mock":
            await self._initialize_mock(**kwargs)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")

        self._is_initialized = True
        logger.info(f"TTS engine initialized: {engine_type}")

    async def _initialize_piper(self, **kwargs):
        """Initialize Piper TTS engine"""
        try:
            if not PIPER_AVAILABLE:
                logger.warning("Piper not available, using mock")
                self._model = MockPiperModel()
                return

            # Initialize real PiperEngine
            self._model = PiperEngine(
                models_path=kwargs.get('models_path', settings.piper_models_path),
                default_voice=kwargs.get('default_voice', settings.piper_default_voice),
                use_gpu=kwargs.get('use_gpu', settings.piper_use_gpu),
                prewarm=kwargs.get('prewarm', settings.piper_prewarm_model),
                speaker_id=kwargs.get('speaker_id', settings.piper_speaker_id)
            )

            # Initialize the engine
            success = await self._model.initialize()
            if not success:
                logger.warning("PiperEngine initialization failed, using mock")
                self._model = MockPiperModel()
            else:
                logger.info(f"PiperEngine initialized with {len(self._model.get_available_voices())} voices")

        except Exception as e:
            logger.warning(f"Piper initialization error: {e}, using mock")
            self._model = MockPiperModel()

    async def _initialize_coqui(self, **kwargs):
        """Initialize Coqui TTS engine"""
        try:
            from TTS.api import TTS
            # Mock Coqui initialization for testing
            self._model = MockCoquiModel()
        except ImportError:
            logger.warning("Coqui not available, using mock")
            self._model = MockCoquiModel()

    async def _initialize_mock(self, **kwargs):
        """Initialize mock TTS engine"""
        self._model = MockTTSModel()

    async def synthesize(self, text: str, voice_id: str) -> Optional[bytes]:
        """Synthesize text to audio"""
        if not self._is_initialized:
            raise RuntimeError("TTS engine not initialized")

        # Handle PiperEngine vs mock differently
        if isinstance(self._model, PiperEngine):
            # PiperEngine returns numpy array
            audio_array = await self._model.synthesize(text, voice_id)
            if audio_array is not None:
                # Convert float32 numpy array to 16-bit PCM bytes
                return self._numpy_to_pcm_bytes(audio_array)
            return None
        else:
            # Mock model returns bytes directly
            return await self._model.synthesize(text, voice_id)

    def _numpy_to_pcm_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy float32 array to 16-bit PCM bytes"""
        # Ensure array is float32 and normalized to -1.0 to 1.0
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Convert to 16-bit signed integers
        audio_int16 = (audio_array * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def get_available_voices(self) -> List[str]:
        """Get available voices for the engine"""
        # Get from actual engine if available
        if isinstance(self._model, PiperEngine):
            return self._model.get_available_voices()

        # Fallback to defaults
        if self._engine_type == "piper":
            return ["en_US-lessac-medium", "en_US-amy-medium", "en_US-ryan-medium"]
        elif self._engine_type == "coqui":
            return ["ljspeech", "vctk", "mailabs"]
        elif self._engine_type == "mock":
            return ["mock_voice_1", "mock_voice_2", "mock_voice_3"]
        return []

    def validate_voice_id(self, voice_id: str) -> bool:
        """Validate if voice ID is available"""
        return voice_id in self.get_available_voices()

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self._is_initialized and self._model is not None

    async def cleanup(self):
        """Cleanup engine resources"""
        self._is_initialized = False
        self._model = None
        logger.info("TTS engine cleaned up")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        return {
            "engine_type": self._engine_type,
            "is_initialized": self._is_initialized,
            "available_voices": self.get_available_voices()
        }


class MockTTSModel:
    """Mock TTS model for testing"""

    async def synthesize(self, text: str, voice_id: str) -> bytes:
        """Mock synthesis"""
        # Generate simple sine wave audio
        duration = len(text) * 0.05  # 50ms per character
        sample_rate = 16000
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples)
        frequency = 440  # A4 note
        audio = (np.sin(2 * np.pi * frequency * t) * 0.1 * 32767).astype(np.int16)
        
        return audio.tobytes()


class MockPiperModel:
    """Mock Piper model for testing"""

    async def synthesize(self, text: str, voice_id: str) -> bytes:
        """Mock Piper synthesis"""
        return await MockTTSModel().synthesize(text, voice_id)


class MockCoquiModel:
    """Mock Coqui model for testing"""

    async def synthesize(self, text: str, voice_id: str) -> bytes:
        """Mock Coqui synthesis"""
        return await MockTTSModel().synthesize(text, voice_id)


class LocalTTSService(TTSService):
    """Local TTS service using Piper or Coqui TTS"""

    def __init__(
        self,
        engine: str = "piper",
        voice: str = "en_US-lessac-medium",
        sample_rate: int = 16000,
        speed: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.engine = engine
        self.voice = voice
        self.sample_rate = sample_rate
        self.speed = speed
        self._tts_engine = None
        self._is_initialized = False

    async def start(self, frame_processor_params=None):
        """Initialize the TTS engine"""
        try:
            await super().start(frame_processor_params)

            if self.engine == "piper":
                await self._initialize_piper()
            elif self.engine == "coqui":
                await self._initialize_coqui()
            else:
                raise ValueError(f"Unsupported TTS engine: {self.engine}")

            self._is_initialized = True
            logger.info(f"Local TTS service initialized with {self.engine}")

        except Exception as e:
            logger.error(f"Failed to initialize local TTS service: {e}")
            raise

    async def _initialize_piper(self):
        """Initialize Piper TTS engine"""
        try:
            if not PIPER_AVAILABLE:
                logger.warning("Piper not available, using mock")
                self._tts_engine = MockTTSEngine()
                return

            # Initialize PiperEngine
            self._tts_engine = PiperEngine(
                models_path=settings.piper_models_path,
                default_voice=self.voice,
                use_gpu=settings.piper_use_gpu,
                prewarm=settings.piper_prewarm_model,
                speaker_id=settings.piper_speaker_id
            )

            # Initialize the engine
            success = await self._tts_engine.initialize()
            if not success:
                logger.warning("PiperEngine initialization failed, using mock")
                self._tts_engine = MockTTSEngine()
            else:
                logger.info(f"Piper TTS initialized with voice: {self.voice}")

        except Exception as e:
            logger.error(f"Failed to initialize Piper: {e}")
            self._tts_engine = MockTTSEngine()
            logger.warning("Using mock TTS engine for development")

    async def _initialize_coqui(self):
        """Initialize Coqui TTS engine"""
        try:
            from TTS.api import TTS

            # Initialize Coqui TTS
            self._tts_engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            logger.info("Coqui TTS model loaded")

        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install coqui-tts")
            raise
        except Exception as e:
            logger.error(f"Failed to load Coqui model: {e}")
            # Fallback to mock implementation
            self._tts_engine = MockTTSEngine()
            logger.warning("Using mock TTS engine for development")

    async def process_frame(self, frame, direction):
        """Process frames for TTS"""
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if not self._is_initialized:
                logger.error("TTS service not initialized")
                return

            await self._synthesize_text(frame.text)

    async def _synthesize_text(self, text: str):
        """Synthesize text to speech"""
        try:
            if not text or not text.strip():
                return

            # Emit started frame
            await self.push_frame(TTSStartedFrame())

            start_time = time.time()

            # Synthesize audio based on engine type
            if self.engine == "piper":
                audio_data = await self._synthesize_with_piper(text)
            elif self.engine == "coqui":
                audio_data = await self._synthesize_with_coqui(text)
            else:
                audio_data = await self._synthesize_mock(text)

            generation_time = (time.time() - start_time) * 1000

            if audio_data is not None:
                # Create audio frame
                audio_frame = TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.sample_rate,
                    num_channels=1
                )

                await self.push_frame(audio_frame)
                logger.debug(f"TTS generated {len(audio_data)} samples in {generation_time:.1f}ms")

            # Emit stopped frame
            await self.push_frame(TTSStoppedFrame())

        except Exception as e:
            logger.error(f"Error synthesizing text '{text}': {e}")
            await self.push_frame(TTSStoppedFrame())

    async def _synthesize_with_piper(self, text: str) -> Optional[bytes]:
        """Synthesize with Piper TTS"""
        try:
            if isinstance(self._tts_engine, MockTTSEngine):
                return await self._synthesize_mock(text)

            # Use PiperEngine for synthesis
            if isinstance(self._tts_engine, PiperEngine):
                # Check if streaming is enabled
                if settings.piper_enable_streaming:
                    # Use streaming synthesis for lower latency
                    chunks = await self._tts_engine.synthesize_streaming(
                        text,
                        voice_name=self.voice,
                        chunk_size_ms=settings.piper_chunk_size_ms
                    )

                    if chunks:
                        # Combine all chunks into single audio
                        combined_audio = np.concatenate(chunks)
                        # Convert to PCM bytes
                        return self._numpy_to_pcm_bytes(combined_audio)
                else:
                    # Standard synthesis
                    audio_array = await self._tts_engine.synthesize(text, voice_name=self.voice)
                    if audio_array is not None:
                        return self._numpy_to_pcm_bytes(audio_array)

            return None

        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            return None

    def _numpy_to_pcm_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy float32 array to 16-bit PCM bytes"""
        # Ensure array is float32 and normalized to -1.0 to 1.0
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Clip to prevent overflow
        audio_array = np.clip(audio_array, -1.0, 1.0)

        # Convert to 16-bit signed integers
        audio_int16 = (audio_array * 32767).astype(np.int16)
        return audio_int16.tobytes()

    async def _synthesize_with_coqui(self, text: str) -> Optional[bytes]:
        """Synthesize with Coqui TTS"""
        try:
            if isinstance(self._tts_engine, MockTTSEngine):
                return await self._synthesize_mock(text)

            # Run Coqui synthesis in executor
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None,
                self._coqui_synthesize_sync,
                text
            )

            return audio_data

        except Exception as e:
            logger.error(f"Coqui synthesis error: {e}")
            return None

    def _coqui_synthesize_sync(self, text: str) -> bytes:
        """Synchronous Coqui synthesis (runs in executor)"""
        try:
            # Synthesize with Coqui
            wav_data = self._tts_engine.tts(text)

            # Convert numpy array to bytes
            if isinstance(wav_data, np.ndarray):
                # Normalize and convert to 16-bit PCM
                wav_data = (wav_data * 32767).astype(np.int16)
                return wav_data.tobytes()

            return b""

        except Exception as e:
            logger.error(f"Coqui sync synthesis error: {e}")
            return b""

    async def _synthesize_mock(self, text: str) -> bytes:
        """Mock synthesis for development/testing"""
        try:
            # Generate simple tone for testing
            duration = len(text) * 0.05  # 50ms per character
            samples = int(self.sample_rate * duration)

            # Generate a simple sine wave
            t = np.linspace(0, duration, samples)
            frequency = 440  # A4 note
            audio = (np.sin(2 * np.pi * frequency * t) * 0.1 * 32767).astype(np.int16)

            logger.debug(f"Mock TTS generated {len(audio)} samples for: {text[:30]}...")
            return audio.tobytes()

        except Exception as e:
            logger.error(f"Mock synthesis error: {e}")
            return b""

    async def stop(self):
        """Stop the TTS service"""
        try:
            self._is_initialized = False

            # Cleanup PiperEngine if present
            if isinstance(self._tts_engine, PiperEngine):
                await self._tts_engine.cleanup()

            self._tts_engine = None
            await super().stop()
            logger.info("Local TTS service stopped")

        except Exception as e:
            logger.error(f"Error stopping local TTS service: {e}")

    def get_cost_estimate(self, text: str) -> float:
        """Get cost estimate for local TTS (essentially free)"""
        # Local TTS has minimal cost (just compute)
        return 0.001  # $0.001 per synthesis for compute cost

    async def health_check(self) -> dict:
        """Check TTS service health"""
        try:
            return {
                "status": "healthy" if self._is_initialized else "not_initialized",
                "engine": self.engine,
                "voice": self.voice,
                "sample_rate": self.sample_rate,
                "initialized": self._is_initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


class MockTTSEngine:
    """Mock TTS engine for development"""

    def synthesize(self, text: str, output_stream):
        """Mock synthesis that creates a simple WAV file"""
        # Create a simple WAV file with silence
        duration = len(text) * 0.05
        sample_rate = 16000
        samples = int(sample_rate * duration)

        # Write WAV header and silent audio
        with wave.open(output_stream, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            # Write silent audio
            silence = np.zeros(samples, dtype=np.int16)
            wav_file.writeframes(silence.tobytes())

    def tts(self, text: str) -> np.ndarray:
        """Mock Coqui-style synthesis"""
        duration = len(text) * 0.05
        sample_rate = 16000
        samples = int(sample_rate * duration)

        # Return simple sine wave
        t = np.linspace(0, duration, samples)
        return np.sin(2 * np.pi * 440 * t) * 0.1
