"""
Native Piper TTS Integration (Phase 1)

This module provides native integration with Piper TTS using ONNX models
for low-latency, high-quality text-to-speech synthesis.
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

# Try to import piper-tts
try:
    from piper import PiperVoice
    from piper.voice import PiperConfig
    PIPER_AVAILABLE = True
    logger.info("piper-tts library is available")
except ImportError:
    PIPER_AVAILABLE = False
    PiperVoice = None
    PiperConfig = None
    logger.warning("piper-tts not available, will use fallback")


class PiperVoiceModel:
    """Wrapper for Piper voice model with caching and optimization"""

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        use_gpu: bool = False,
        speaker_id: Optional[int] = None
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.use_gpu = use_gpu
        self.speaker_id = speaker_id
        self.voice: Optional[PiperVoice] = None
        self.config: Optional[Dict[str, Any]] = None
        self.is_loaded = False

    def load(self):
        """Load the Piper voice model"""
        try:
            if not PIPER_AVAILABLE:
                raise ImportError("piper-tts not installed")

            # Load voice configuration
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            # Load voice model
            self.voice = PiperVoice.load(
                str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=self.use_gpu
            )

            self.is_loaded = True
            logger.info(f"Loaded Piper model: {self.model_path.name}")

            return True

        except Exception as e:
            logger.error(f"Failed to load Piper model {self.model_path}: {e}")
            return False

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text to audio

        Args:
            text: Text to synthesize

        Returns:
            Audio samples as numpy array (float32, normalized -1 to 1)
        """
        if not self.is_loaded or not self.voice:
            logger.error("Voice model not loaded")
            return None

        try:
            # Synthesize audio
            audio = self.voice.synthesize(
                text,
                speaker_id=self.speaker_id
            )

            # Convert to numpy array if needed
            if isinstance(audio, bytes):
                audio = np.frombuffer(audio, dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0

            return audio

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return None

    def synthesize_streaming(self, text: str, chunk_size_ms: int = 100) -> List[np.ndarray]:
        """
        Synthesize text with streaming output (chunked)

        Args:
            text: Text to synthesize
            chunk_size_ms: Size of each audio chunk in milliseconds

        Returns:
            List of audio chunks as numpy arrays
        """
        if not self.is_loaded or not self.voice:
            logger.error("Voice model not loaded")
            return []

        try:
            # Synthesize full audio
            audio = self.synthesize(text)
            if audio is None:
                return []

            # Get sample rate from config
            sample_rate = self.config.get('audio', {}).get('sample_rate', 22050)

            # Calculate chunk size in samples
            chunk_samples = int((chunk_size_ms / 1000.0) * sample_rate)

            # Split audio into chunks
            chunks = []
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i+chunk_samples]
                if len(chunk) > 0:
                    chunks.append(chunk)

            logger.debug(f"Split audio into {len(chunks)} chunks of ~{chunk_size_ms}ms")
            return chunks

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            return []

    def get_sample_rate(self) -> int:
        """Get the audio sample rate for this voice"""
        if self.config:
            return self.config.get('audio', {}).get('sample_rate', 22050)
        return 22050

    def get_voice_info(self) -> Dict[str, Any]:
        """Get voice metadata"""
        if not self.config:
            return {}

        return {
            'name': self.model_path.stem,
            'language': self.config.get('language', {}).get('code', 'en'),
            'sample_rate': self.get_sample_rate(),
            'num_speakers': self.config.get('num_speakers', 1),
            'quality': self.config.get('audio', {}).get('quality', 'medium')
        }

    def cleanup(self):
        """Clean up resources"""
        self.voice = None
        self.is_loaded = False


class PiperEngine:
    """
    Piper TTS Engine with model caching and optimization
    """

    def __init__(
        self,
        models_path: str = "./models/piper",
        default_voice: str = "en_US-lessac-medium",
        use_gpu: bool = False,
        prewarm: bool = True,
        speaker_id: Optional[int] = None
    ):
        self.models_path = Path(models_path)
        self.default_voice = default_voice
        self.use_gpu = use_gpu
        self.prewarm = prewarm
        self.speaker_id = speaker_id

        self._voices: Dict[str, PiperVoiceModel] = {}
        self._default_voice_model: Optional[PiperVoiceModel] = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize the Piper engine and discover voices"""
        try:
            if not PIPER_AVAILABLE:
                logger.warning("Piper not available, using mock implementation")
                self.is_initialized = True
                return True

            # Create models directory if it doesn't exist
            self.models_path.mkdir(parents=True, exist_ok=True)

            # Discover available voices
            await self._discover_voices()

            # Pre-warm default voice if enabled
            if self.prewarm and self.default_voice:
                await self.load_voice(self.default_voice)

            self.is_initialized = True
            logger.info(f"Piper engine initialized with {len(self._voices)} voices")
            return True

        except Exception as e:
            logger.error(f"Piper engine initialization failed: {e}")
            return False

    async def _discover_voices(self):
        """Discover available Piper voice models"""
        try:
            # Find all .onnx files in models directory
            onnx_files = list(self.models_path.glob("*.onnx"))

            for onnx_file in onnx_files:
                # Check for corresponding .json config
                config_file = onnx_file.with_suffix('.onnx.json')

                if config_file.exists():
                    voice_name = onnx_file.stem
                    self._voices[voice_name] = PiperVoiceModel(
                        model_path=onnx_file,
                        config_path=config_file,
                        use_gpu=self.use_gpu,
                        speaker_id=self.speaker_id
                    )
                    logger.debug(f"Discovered voice: {voice_name}")

            if not self._voices:
                logger.warning(f"No Piper voices found in {self.models_path}")

        except Exception as e:
            logger.error(f"Voice discovery error: {e}")

    async def load_voice(self, voice_name: str) -> bool:
        """
        Load a specific voice model

        Args:
            voice_name: Name of the voice to load

        Returns:
            True if successful
        """
        try:
            voice_model = self._voices.get(voice_name)
            if not voice_model:
                logger.error(f"Voice not found: {voice_name}")
                return False

            if voice_model.is_loaded:
                logger.debug(f"Voice already loaded: {voice_name}")
                return True

            # Load in executor to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, voice_model.load)

            if success and voice_name == self.default_voice:
                self._default_voice_model = voice_model

            return success

        except Exception as e:
            logger.error(f"Error loading voice {voice_name}: {e}")
            return False

    async def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Synthesize text to audio

        Args:
            text: Text to synthesize
            voice_name: Voice to use (defaults to default_voice)

        Returns:
            Audio as numpy array (float32, normalized)
        """
        try:
            # Use default voice if not specified
            voice_name = voice_name or self.default_voice

            # Load voice if not already loaded
            voice_model = self._voices.get(voice_name)
            if not voice_model:
                logger.error(f"Voice not found: {voice_name}")
                return None

            if not voice_model.is_loaded:
                await self.load_voice(voice_name)

            # Run synthesis in executor to avoid blocking
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                None,
                voice_model.synthesize,
                text
            )

            return audio

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return None

    async def synthesize_streaming(
        self,
        text: str,
        voice_name: Optional[str] = None,
        chunk_size_ms: int = 100
    ) -> List[np.ndarray]:
        """
        Synthesize with streaming (chunked) output

        Args:
            text: Text to synthesize
            voice_name: Voice to use
            chunk_size_ms: Chunk size in milliseconds

        Returns:
            List of audio chunks
        """
        try:
            voice_name = voice_name or self.default_voice

            voice_model = self._voices.get(voice_name)
            if not voice_model or not voice_model.is_loaded:
                await self.load_voice(voice_name)
                voice_model = self._voices.get(voice_name)

            if not voice_model:
                return []

            # Run streaming synthesis in executor
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                voice_model.synthesize_streaming,
                text,
                chunk_size_ms
            )

            return chunks

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            return []

    def get_available_voices(self) -> List[str]:
        """Get list of available voice names"""
        return list(self._voices.keys())

    def get_voice_info(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific voice"""
        voice_model = self._voices.get(voice_name)
        if voice_model:
            return voice_model.get_voice_info()
        return None

    def get_sample_rate(self, voice_name: Optional[str] = None) -> int:
        """Get sample rate for a voice"""
        voice_name = voice_name or self.default_voice
        voice_model = self._voices.get(voice_name)
        if voice_model:
            return voice_model.get_sample_rate()
        return 22050  # Default

    async def cleanup(self):
        """Clean up all resources"""
        for voice_model in self._voices.values():
            voice_model.cleanup()
        self._voices.clear()
        self._default_voice_model = None
        self.is_initialized = False
        logger.info("Piper engine cleaned up")
