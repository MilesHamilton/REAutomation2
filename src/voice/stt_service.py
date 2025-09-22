import asyncio
import whisper
import torch
import numpy as np
import logging
import time
from typing import Optional, AsyncGenerator, Callable
from threading import Lock
import wave
import io

from ..config import settings
from .models import AudioChunk, STTResult, STTConfig, AudioConfig

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self, config: STTConfig = None):
        self.config = config or STTConfig()
        self.model: Optional[whisper.Whisper] = None
        self.model_lock = Lock()
        self.is_initialized = False
        self._processing_queue = asyncio.Queue()
        self._results_callbacks: dict[str, Callable] = {}

    async def initialize(self):
        """Initialize the Whisper model"""
        try:
            # Load Whisper model on a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_whisper_model
            )

            self.is_initialized = True
            logger.info(f"STT service initialized with model: {self.config.model}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            return False

    def _load_whisper_model(self) -> whisper.Whisper:
        """Load Whisper model (runs in executor)"""
        # Map our model names to Whisper model names
        model_map = {
            "whisper-small": "small",
            "whisper-medium": "medium",
            "whisper-base": "base",
            "whisper-large": "large"
        }

        whisper_model_name = model_map.get(self.config.model, "small")

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = whisper.load_model(whisper_model_name, device=device)
        logger.info(f"Loaded Whisper model '{whisper_model_name}' on device: {device}")

        return model

    async def transcribe_audio_chunk(
        self,
        audio_chunk: AudioChunk,
        call_id: str
    ) -> Optional[STTResult]:
        """Transcribe a single audio chunk"""
        if not self.is_initialized or not self.model:
            logger.error("STT service not initialized")
            return None

        start_time = time.time()

        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_numpy(
                audio_chunk.data,
                audio_chunk.sample_rate,
                audio_chunk.channels
            )

            # Skip if audio is too short or likely silence
            if len(audio_array) < 1600:  # ~0.1 seconds at 16kHz
                return None

            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_with_whisper,
                audio_array
            )

            processing_time = (time.time() - start_time) * 1000

            # Filter out empty or very short results
            if not result or len(result.strip()) < 2:
                return None

            return STTResult(
                text=result.strip(),
                call_id=call_id,
                confidence=0.8,  # Whisper doesn't provide confidence scores
                processing_time_ms=processing_time,
                is_final=True,
                language=self.config.language,
                chunk_id=audio_chunk.chunk_id
            )

        except Exception as e:
            logger.error(f"STT transcription error for call {call_id}: {e}")
            return None

    def _transcribe_with_whisper(self, audio_array: np.ndarray) -> str:
        """Run Whisper transcription (blocking, runs in executor)"""
        try:
            with self.model_lock:
                result = self.model.transcribe(
                    audio_array,
                    language=self.config.language,
                    task="transcribe",
                    fp16=torch.cuda.is_available(),
                    verbose=False
                )
                return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return ""

    def _bytes_to_numpy(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        channels: int
    ) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32) / 32768.0

            # Handle multi-channel audio by converting to mono
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)
                audio_array = np.mean(audio_array, axis=1)

            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                import scipy.signal
                audio_array = scipy.signal.resample(
                    audio_array,
                    int(len(audio_array) * 16000 / sample_rate)
                )

            return audio_array

        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return np.array([])

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[AudioChunk, None],
        call_id: str,
        callback: Callable[[STTResult], None]
    ):
        """Transcribe streaming audio with real-time callbacks"""
        buffer = []
        buffer_duration = 0.0
        max_buffer_duration = 3.0  # Process every 3 seconds

        try:
            async for chunk in audio_stream:
                buffer.append(chunk)
                buffer_duration += len(chunk.data) / (chunk.sample_rate * chunk.channels * 2)

                # Process buffer when it reaches target duration or on speech boundaries
                if (buffer_duration >= max_buffer_duration or
                    (chunk.silence_duration > self.config.max_silence_duration)):

                    if buffer:
                        # Combine buffer chunks
                        combined_chunk = self._combine_chunks(buffer)

                        result = await self.transcribe_audio_chunk(combined_chunk, call_id)
                        if result and callback:
                            callback(result)

                        # Clear buffer
                        buffer = []
                        buffer_duration = 0.0

        except Exception as e:
            logger.error(f"Stream transcription error for call {call_id}: {e}")

    def _combine_chunks(self, chunks: list[AudioChunk]) -> AudioChunk:
        """Combine multiple audio chunks into one"""
        if not chunks:
            return AudioChunk(data=b'', timestamp=time.time(), chunk_id=0,
                            sample_rate=16000, channels=1)

        # Combine audio data
        combined_data = b''.join(chunk.data for chunk in chunks)

        return AudioChunk(
            data=combined_data,
            timestamp=chunks[0].timestamp,
            chunk_id=chunks[0].chunk_id,
            sample_rate=chunks[0].sample_rate,
            channels=chunks[0].channels,
            is_speech=any(chunk.is_speech for chunk in chunks)
        )

    async def health_check(self) -> dict:
        """Check STT service health"""
        try:
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "error": "Service not initialized"
                }

            # Test with a small audio sample
            test_audio = np.random.randn(8000).astype(np.float32) * 0.1
            start_time = time.time()

            with self.model_lock:
                self.model.transcribe(test_audio, verbose=False)

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "model": self.config.model,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "response_time_ms": response_time,
                "initialized": self.is_initialized
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self):
        """Clean up resources"""
        self.is_initialized = False
        if self.model:
            del self.model
        self.model = None
        logger.info("STT service cleanup complete")