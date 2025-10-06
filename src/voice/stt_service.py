import asyncio
import whisper
import torch
import numpy as np
import logging
import time
from typing import Optional, AsyncGenerator, Callable, Union
from threading import Lock
import wave
import io

from ..config import settings
from .models import AudioChunk, STTResult, STTConfig, AudioConfig

logger = logging.getLogger(__name__)

# Try to import faster-whisper (optional dependency)
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger.info("faster-whisper is available and will be used for optimization")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    FasterWhisperModel = None
    logger.info("faster-whisper not available, using standard whisper")


class STTService:
    def __init__(self, config: STTConfig = None):
        self.config = config or STTConfig()
        self.model: Optional[Union[whisper.Whisper, FasterWhisperModel]] = None
        self.model_lock = Lock()
        self.is_initialized = False
        self._processing_queue = asyncio.Queue()
        self._results_callbacks: dict[str, Callable] = {}
        self.use_faster_whisper = False  # Track which backend is in use

    async def initialize(self):
        """Initialize the Whisper model (standard or faster-whisper)"""
        try:
            # Determine which backend to use
            use_faster = self.config.use_faster_whisper and FASTER_WHISPER_AVAILABLE

            # Load Whisper model on a separate thread to avoid blocking
            loop = asyncio.get_event_loop()

            if use_faster:
                logger.info("Initializing faster-whisper backend...")
                self.model = await loop.run_in_executor(
                    None,
                    self._load_faster_whisper_model
                )
                self.use_faster_whisper = True
            else:
                logger.info("Initializing standard whisper backend...")
                self.model = await loop.run_in_executor(
                    None,
                    self._load_whisper_model
                )
                self.use_faster_whisper = False

            self.is_initialized = True
            backend_name = "faster-whisper" if self.use_faster_whisper else "whisper"
            logger.info(f"STT service initialized with {backend_name} model: {self.config.model}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            return False

    def _load_whisper_model(self) -> whisper.Whisper:
        """Load standard Whisper model (runs in executor)"""
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
        logger.info(f"Loaded standard Whisper model '{whisper_model_name}' on device: {device}")

        return model

    def _load_faster_whisper_model(self) -> FasterWhisperModel:
        """Load faster-whisper model with CTranslate2 optimization (runs in executor)"""
        # Map our model names to Whisper model names
        model_map = {
            "whisper-small": "small",
            "whisper-medium": "medium",
            "whisper-base": "base",
            "whisper-large": "large-v2"
        }

        whisper_model_name = model_map.get(self.config.model, "small")

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Adjust compute type based on device
        compute_type = self.config.compute_type
        if device == "cpu":
            # CPU doesn't support float16
            if compute_type == "float16":
                compute_type = "int8"
                logger.info("Adjusting compute_type to 'int8' for CPU")

        model = FasterWhisperModel(
            whisper_model_name,
            device=device,
            compute_type=compute_type,
            num_workers=self.config.num_workers
        )

        logger.info(f"Loaded faster-whisper model '{whisper_model_name}' on {device} "
                   f"with compute_type={compute_type}, num_workers={self.config.num_workers}")

        return model

    async def transcribe_audio_chunk(
        self,
        audio_chunk: AudioChunk,
        call_id: str
    ) -> Optional[STTResult]:
        """Transcribe a single audio chunk with quality assessment"""
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

            # Assess audio quality (Phase 2)
            quality_metrics = self._assess_audio_quality(audio_array)

            # Skip low-quality audio that's unlikely to transcribe well
            if quality_metrics['quality_score'] < 0.1:
                logger.debug(f"Skipping low-quality audio for call {call_id}: "
                           f"score={quality_metrics['quality_score']:.2f}")
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
            if not result.get('text') or len(result['text'].strip()) < 2:
                return None

            # Calculate confidence from segments
            confidence = self._calculate_confidence(result.get('segments', []))

            return STTResult(
                text=result['text'].strip(),
                call_id=call_id,
                confidence=confidence,
                processing_time_ms=processing_time,
                is_final=True,
                language=self.config.language,
                detected_language=result.get('language'),
                language_probability=result.get('language_probability'),
                chunk_id=audio_chunk.chunk_id,
                # Quality metrics (Phase 2)
                audio_quality_score=quality_metrics['quality_score'],
                snr_db=quality_metrics['snr'],
                clipping_detected=quality_metrics['clipping_rate'] > 0.01,
                quality_assessment=quality_metrics['assessment']
            )

        except Exception as e:
            logger.error(f"STT transcription error for call {call_id}: {e}")
            return None

    def _transcribe_with_whisper(self, audio_array: np.ndarray) -> dict:
        """
        Run Whisper transcription (blocking, runs in executor)
        Supports both standard whisper and faster-whisper backends

        Returns:
            dict with keys: text, segments, language, language_probability
        """
        try:
            with self.model_lock:
                if self.use_faster_whisper:
                    return self._transcribe_with_faster_whisper(audio_array)
                else:
                    return self._transcribe_with_standard_whisper(audio_array)
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                'text': '',
                'segments': [],
                'language': self.config.language,
                'language_probability': 0.0
            }

    def _transcribe_with_standard_whisper(self, audio_array: np.ndarray) -> dict:
        """Transcribe using standard whisper"""
        # Auto-detect language if enabled, otherwise use configured language
        language = None if self.config.auto_detect_language else self.config.language

        result = self.model.transcribe(
            audio_array,
            language=language,
            task="transcribe",
            fp16=torch.cuda.is_available(),
            verbose=False,
            word_timestamps=True  # Enable for better confidence calculation
        )

        return {
            'text': result.get("text", "").strip(),
            'segments': result.get('segments', []),
            'language': result.get('language', self.config.language),
            'language_probability': result.get('language_probability', 1.0)
        }

    def _transcribe_with_faster_whisper(self, audio_array: np.ndarray) -> dict:
        """Transcribe using faster-whisper (CTranslate2 optimization)"""
        # Auto-detect language if enabled, otherwise use configured language
        language = None if self.config.auto_detect_language else self.config.language

        # faster-whisper returns segments as a generator
        segments_gen, info = self.model.transcribe(
            audio_array,
            language=language,
            task="transcribe",
            beam_size=self.config.beam_size,
            word_timestamps=True,
            vad_filter=self.config.vad_enabled,
            vad_parameters=dict(
                threshold=self.config.silence_threshold,
                min_silence_duration_ms=int(self.config.max_silence_duration * 1000)
            ) if self.config.vad_enabled else None
        )

        # Convert generator to list and extract text
        segments = []
        full_text = []

        for segment in segments_gen:
            segment_dict = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'avg_logprob': segment.avg_logprob,
                'no_speech_prob': segment.no_speech_prob
            }
            segments.append(segment_dict)
            full_text.append(segment.text)

        return {
            'text': ' '.join(full_text).strip(),
            'segments': segments,
            'language': info.language if hasattr(info, 'language') else self.config.language,
            'language_probability': info.language_probability if hasattr(info, 'language_probability') else 1.0
        }

    def _calculate_confidence(self, segments: list) -> float:
        """
        Calculate confidence score from Whisper segments

        Whisper provides average log probability for each segment.
        We convert these to a 0-1 confidence score.

        Args:
            segments: List of segment dictionaries from Whisper output

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not segments:
            return 0.0

        try:
            # Extract average log probabilities from segments
            log_probs = []
            for segment in segments:
                if 'avg_logprob' in segment:
                    log_probs.append(segment['avg_logprob'])
                elif 'logprob' in segment:
                    log_probs.append(segment['logprob'])

            if not log_probs:
                # No probability information available
                return 0.7  # Default moderate confidence

            # Calculate mean log probability
            avg_logprob = sum(log_probs) / len(log_probs)

            # Convert log probability to confidence score
            # Whisper log probs typically range from -1.0 (high confidence) to -3.0+ (low confidence)
            # We map this to 0-1 scale
            if avg_logprob >= -0.5:
                confidence = 1.0
            elif avg_logprob <= -3.0:
                confidence = 0.0
            else:
                # Linear mapping: -0.5 -> 1.0, -3.0 -> 0.0
                confidence = 1.0 - (abs(avg_logprob) - 0.5) / 2.5

            # Also consider no_speech_prob if available
            for segment in segments:
                if 'no_speech_prob' in segment:
                    no_speech_prob = segment['no_speech_prob']
                    # Reduce confidence if high probability of no speech
                    if no_speech_prob > 0.5:
                        confidence *= (1.0 - no_speech_prob)

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Return moderate confidence on error

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

    def _assess_audio_quality(self, audio_array: np.ndarray) -> dict:
        """
        Assess audio quality metrics for transcription suitability

        Args:
            audio_array: Audio data as numpy array (float32, normalized)

        Returns:
            dict with quality metrics: snr, clipping_rate, silence_ratio, quality_score
        """
        try:
            if len(audio_array) == 0:
                return {
                    'snr': 0.0,
                    'clipping_rate': 0.0,
                    'silence_ratio': 1.0,
                    'quality_score': 0.0,
                    'assessment': 'empty'
                }

            # Calculate Signal-to-Noise Ratio (SNR)
            snr = self._calculate_snr(audio_array)

            # Calculate clipping rate (samples near ±1.0)
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio_array) > clipping_threshold)
            clipping_rate = clipped_samples / len(audio_array)

            # Calculate silence ratio (samples near 0.0)
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(audio_array) < silence_threshold)
            silence_ratio = silent_samples / len(audio_array)

            # Calculate overall quality score (0-1 scale)
            quality_score = self._calculate_quality_score(snr, clipping_rate, silence_ratio)

            # Determine assessment
            if quality_score >= 0.8:
                assessment = 'excellent'
            elif quality_score >= 0.6:
                assessment = 'good'
            elif quality_score >= 0.4:
                assessment = 'fair'
            elif quality_score >= 0.2:
                assessment = 'poor'
            else:
                assessment = 'very_poor'

            return {
                'snr': float(snr),
                'clipping_rate': float(clipping_rate),
                'silence_ratio': float(silence_ratio),
                'quality_score': float(quality_score),
                'assessment': assessment
            }

        except Exception as e:
            logger.error(f"Error assessing audio quality: {e}")
            return {
                'snr': 0.0,
                'clipping_rate': 0.0,
                'silence_ratio': 0.0,
                'quality_score': 0.5,
                'assessment': 'unknown'
            }

    def _calculate_snr(self, audio_array: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB

        Args:
            audio_array: Audio data as numpy array

        Returns:
            SNR in decibels
        """
        try:
            # Split audio into frames for analysis
            frame_length = 1024
            frames = [audio_array[i:i+frame_length]
                     for i in range(0, len(audio_array), frame_length)]

            if not frames:
                return 0.0

            # Calculate energy for each frame
            frame_energies = [np.sum(frame ** 2) / len(frame) for frame in frames if len(frame) > 0]

            if not frame_energies:
                return 0.0

            # Sort frames by energy
            sorted_energies = sorted(frame_energies)

            # Estimate signal (top 50% of frames) and noise (bottom 25%)
            signal_threshold_idx = len(sorted_energies) // 2
            noise_threshold_idx = len(sorted_energies) // 4

            signal_energy = np.mean(sorted_energies[signal_threshold_idx:])
            noise_energy = np.mean(sorted_energies[:noise_threshold_idx])

            # Calculate SNR in dB
            if noise_energy > 0:
                snr = 10 * np.log10(signal_energy / noise_energy)
            else:
                snr = 60.0  # Maximum SNR if no noise detected

            # Bound SNR to reasonable range
            return float(np.clip(snr, 0.0, 60.0))

        except Exception as e:
            logger.error(f"Error calculating SNR: {e}")
            return 0.0

    def _calculate_quality_score(self, snr: float, clipping_rate: float,
                                 silence_ratio: float) -> float:
        """
        Calculate overall audio quality score from metrics

        Args:
            snr: Signal-to-Noise Ratio in dB (0-60)
            clipping_rate: Proportion of clipped samples (0-1)
            silence_ratio: Proportion of silent samples (0-1)

        Returns:
            Quality score (0-1)
        """
        try:
            # SNR score (0-1): normalize 0-60 dB to 0-1
            # Good speech typically has SNR > 20 dB
            snr_score = np.clip(snr / 40.0, 0.0, 1.0)

            # Clipping penalty: reduce score if clipping detected
            clipping_penalty = 1.0 - np.clip(clipping_rate * 10, 0.0, 1.0)

            # Silence penalty: reduce score if too much silence (>80%) or too little (<5%)
            if silence_ratio > 0.8:
                silence_penalty = 0.5  # Too much silence
            elif silence_ratio < 0.05:
                silence_penalty = 0.8  # No pauses (possibly noise)
            else:
                silence_penalty = 1.0  # Normal silence ratio

            # Weighted combination
            quality_score = (
                snr_score * 0.5 +          # SNR is most important (50%)
                clipping_penalty * 0.3 +    # Clipping is serious (30%)
                silence_penalty * 0.2       # Silence ratio (20%)
            )

            return float(np.clip(quality_score, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5

    def _detect_speech_vad(self, audio_array: np.ndarray) -> dict:
        """
        Enhanced Voice Activity Detection (VAD) using energy-based analysis

        Args:
            audio_array: Audio data as numpy array (float32, normalized)

        Returns:
            dict with speech_detected, speech_ratio, energy_level
        """
        try:
            if len(audio_array) == 0:
                return {
                    'speech_detected': False,
                    'speech_ratio': 0.0,
                    'energy_level': 0.0
                }

            # Calculate frame energy for VAD
            frame_length = 512  # ~32ms at 16kHz
            hop_length = 256    # 50% overlap

            frames = []
            for i in range(0, len(audio_array) - frame_length, hop_length):
                frames.append(audio_array[i:i+frame_length])

            if not frames:
                return {
                    'speech_detected': False,
                    'speech_ratio': 0.0,
                    'energy_level': 0.0
                }

            # Calculate energy for each frame
            frame_energies = np.array([np.sum(frame ** 2) for frame in frames])

            # Calculate overall energy level
            avg_energy = np.mean(frame_energies)

            # Adaptive threshold based on energy distribution
            if len(frame_energies) > 10:
                # Use median + factor * std as threshold
                median_energy = np.median(frame_energies)
                std_energy = np.std(frame_energies)
                energy_threshold = median_energy + (1.5 * std_energy)
            else:
                # Simple threshold for short audio
                energy_threshold = np.mean(frame_energies) * 1.5

            # Count frames above threshold as speech
            speech_frames = np.sum(frame_energies > energy_threshold)
            speech_ratio = speech_frames / len(frame_energies)

            # Determine if speech is detected
            # Require at least 10% of frames to be speech with minimum energy
            speech_detected = speech_ratio >= 0.1 and avg_energy > 0.001

            return {
                'speech_detected': bool(speech_detected),
                'speech_ratio': float(speech_ratio),
                'energy_level': float(avg_energy)
            }

        except Exception as e:
            logger.error(f"Error in VAD: {e}")
            return {
                'speech_detected': True,  # Default to processing on error
                'speech_ratio': 0.5,
                'energy_level': 0.01
            }

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[AudioChunk, None],
        call_id: str,
        callback: Callable[[STTResult], None]
    ):
        """Transcribe streaming audio with adaptive buffer management (Phase 2)"""
        buffer = []
        buffer_duration = 0.0

        # Adaptive buffer parameters
        min_buffer_duration = 0.5  # Minimum 0.5 seconds
        max_buffer_duration = 3.0  # Maximum 3 seconds
        adaptive_threshold = 2.0   # Start at 2 seconds

        # Quality tracking for adaptation
        recent_quality_scores = []

        try:
            async for chunk in audio_stream:
                buffer.append(chunk)
                buffer_duration += len(chunk.data) / (chunk.sample_rate * chunk.channels * 2)

                # Check if we should process the buffer
                should_process = False

                # 1. Maximum duration reached
                if buffer_duration >= max_buffer_duration:
                    should_process = True

                # 2. Speech boundary detected (silence after speech)
                elif buffer_duration >= min_buffer_duration and chunk.silence_duration > self.config.max_silence_duration:
                    should_process = True

                # 3. Adaptive threshold reached with quality check
                elif buffer_duration >= adaptive_threshold:
                    # Quick VAD check on current chunk to avoid processing pure silence
                    chunk_array = self._bytes_to_numpy(chunk.data, chunk.sample_rate, chunk.channels)
                    vad_result = self._detect_speech_vad(chunk_array)

                    if vad_result['speech_detected']:
                        should_process = True

                if should_process and buffer:
                    # Combine buffer chunks
                    combined_chunk = self._combine_chunks(buffer)

                    # Transcribe
                    result = await self.transcribe_audio_chunk(combined_chunk, call_id)

                    if result:
                        # Adapt buffer duration based on quality
                        if result.audio_quality_score:
                            recent_quality_scores.append(result.audio_quality_score)

                            # Keep only last 5 scores
                            if len(recent_quality_scores) > 5:
                                recent_quality_scores.pop(0)

                            # Adjust adaptive threshold based on quality
                            avg_quality = sum(recent_quality_scores) / len(recent_quality_scores)

                            if avg_quality >= 0.8:
                                # High quality: can use shorter buffers
                                adaptive_threshold = min_buffer_duration + 0.5
                            elif avg_quality >= 0.5:
                                # Medium quality: use moderate buffers
                                adaptive_threshold = 2.0
                            else:
                                # Low quality: use longer buffers for better context
                                adaptive_threshold = 2.5

                        # Send result via callback
                        if callback:
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

    async def transcribe_batch(
        self,
        audio_chunks: list[AudioChunk],
        call_id: str
    ) -> list[Optional[STTResult]]:
        """
        Transcribe multiple audio chunks in parallel (Phase 3 optimization)

        Args:
            audio_chunks: List of audio chunks to transcribe
            call_id: Unique identifier for the call

        Returns:
            List of STTResult objects (in same order as input)
        """
        if not self.is_initialized or not self.model:
            logger.error("STT service not initialized")
            return [None] * len(audio_chunks)

        try:
            # Process chunks in parallel using asyncio.gather
            tasks = [
                self.transcribe_audio_chunk(chunk, call_id)
                for chunk in audio_chunks
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error transcribing chunk {i}: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"Batch transcription error for call {call_id}: {e}")
            return [None] * len(audio_chunks)

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