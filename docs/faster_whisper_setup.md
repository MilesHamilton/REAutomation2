# faster-whisper Setup Guide

## Overview

faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, which provides significant performance improvements through optimized inference.

### Performance Benefits

- **2-4x faster** inference compared to standard Whisper
- **Lower memory usage** through quantization (int8, float16)
- **Multi-threading support** for concurrent processing
- **Built-in VAD** (Voice Activity Detection)

## Installation

### Requirements

- Python 3.8+
- CUDA 11.2+ (for GPU acceleration, optional)
- CTranslate2 library

### Install faster-whisper

```bash
pip install faster-whisper>=1.0.0
```

For GPU support with CUDA 11.x:
```bash
pip install faster-whisper[cuda11]
```

For GPU support with CUDA 12.x:
```bash
pip install faster-whisper[cuda12]
```

### Verify Installation

```python
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")
print("faster-whisper installed successfully!")
```

## Configuration

### STTConfig Options (Phase 3)

```python
from src.voice.models import STTConfig

# Enable faster-whisper (default)
config = STTConfig(
    model="whisper-small",
    use_faster_whisper=True,      # Use faster-whisper backend
    compute_type="float16",         # float16, int8, int8_float16
    num_workers=1,                  # Number of worker threads
    beam_size=5,                    # Beam search size (1=greedy, higher=more accurate)
    auto_detect_language=True,      # Auto-detect language
    vad_enabled=True                # Enable Voice Activity Detection
)
```

### Backend Selection

The STT service automatically selects the best available backend:

1. **faster-whisper** (preferred if available and configured)
2. **standard whisper** (fallback)

```python
from src.voice.stt_service import STTService

service = STTService(config)
await service.initialize()

# Check which backend is in use
if service.use_faster_whisper:
    print("Using faster-whisper backend")
else:
    print("Using standard whisper backend")
```

## Compute Types

### float16 (Default for GPU)
- Best quality
- Requires GPU with CUDA
- ~2GB VRAM for small model
- **Recommended for production with GPU**

### int8
- Good quality with smaller size
- Works on CPU and GPU
- ~1GB memory for small model
- **Recommended for CPU or memory-constrained systems**

### int8_float16
- Hybrid approach
- Encoder in int8, decoder in float16
- Balanced performance/quality

## Model Selection

Available models (in order of size/accuracy):

- `whisper-base` - Fastest, lowest accuracy
- `whisper-small` - Good balance (default)
- `whisper-medium` - Higher accuracy, slower
- `whisper-large` - Best accuracy, slowest

### Model Size Comparison

| Model | Params | English-only | Multilingual | Speed (GPU) | VRAM (float16) |
|-------|--------|--------------|--------------|-------------|----------------|
| base  | 74M    | Yes          | Yes          | ~5x         | ~1GB           |
| small | 244M   | Yes          | Yes          | ~4x         | ~2GB           |
| medium| 769M   | Yes          | Yes          | ~2x         | ~5GB           |
| large | 1550M  | No           | Yes          | 1x          | ~10GB          |

## Performance Tuning

### Beam Size

Controls the beam search width:

```python
config = STTConfig(
    beam_size=1,   # Greedy search - fastest, lower quality
    beam_size=5,   # Balanced (default)
    beam_size=10   # More accurate, slower
)
```

### Multi-threading

For CPU inference, increase worker threads:

```python
config = STTConfig(
    device="cpu",
    num_workers=4,  # Use 4 threads
    compute_type="int8"
)
```

### VAD Integration

faster-whisper includes built-in VAD:

```python
config = STTConfig(
    vad_enabled=True,
    silence_threshold=0.5,
    max_silence_duration=2.0
)
```

## Usage Examples

### Basic Transcription

```python
from src.voice.stt_service import STTService
from src.voice.models import STTConfig, AudioChunk
import numpy as np

# Configure service
config = STTConfig(
    model="whisper-small",
    use_faster_whisper=True,
    compute_type="float16"
)

service = STTService(config)
await service.initialize()

# Create audio chunk
audio_data = np.random.randn(16000).astype(np.float32)
audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()

chunk = AudioChunk(
    data=audio_bytes,
    timestamp=time.time(),
    chunk_id=1,
    sample_rate=16000,
    channels=1
)

# Transcribe
result = await service.transcribe_audio_chunk(chunk, "call_123")
print(f"Transcription: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Quality: {result.audio_quality_score}")
```

### Batch Processing

```python
# Transcribe multiple chunks in parallel
chunks = [chunk1, chunk2, chunk3]
results = await service.transcribe_batch(chunks, "call_123")

for i, result in enumerate(results):
    if result:
        print(f"Chunk {i}: {result.text}")
```

### Streaming with Adaptive Buffers

```python
async def audio_stream():
    while True:
        chunk = await get_next_audio_chunk()
        yield chunk

def on_result(result):
    print(f"Transcribed: {result.text}")

await service.transcribe_stream(audio_stream(), "call_123", on_result)
```

## Troubleshooting

### "faster-whisper not available" Warning

If you see this warning, faster-whisper is not installed. Install it:

```bash
pip install faster-whisper
```

The service will automatically fall back to standard whisper.

### CUDA Out of Memory

Reduce model size or use quantization:

```python
config = STTConfig(
    model="whisper-small",  # Use smaller model
    compute_type="int8"     # Use int8 quantization
)
```

### Slow CPU Performance

Increase worker threads and use int8:

```python
config = STTConfig(
    num_workers=4,
    compute_type="int8"
)
```

### Poor Quality Results

Increase beam size:

```python
config = STTConfig(
    beam_size=10,           # More thorough search
    compute_type="float16"  # Higher precision
)
```

## Benchmarking

Run performance benchmarks:

```bash
pytest tests/unit/voice/test_stt_faster_whisper.py -v -m benchmark
```

Expected speedups (GPU):
- faster-whisper vs standard: **2-4x faster**
- Batch vs sequential: **3-5x faster** (for 5 chunks)

## Migration from Standard Whisper

To migrate existing code:

1. **Update configuration:**
   ```python
   config = STTConfig(use_faster_whisper=True)
   ```

2. **No code changes required** - the API is identical

3. **Test thoroughly** - results may differ slightly due to optimizations

## Best Practices

1. **GPU with float16** for production with GPU
2. **CPU with int8 and multi-threading** for CPU deployments
3. **Enable VAD** to reduce processing of silence
4. **Use batch processing** for multiple chunks
5. **Monitor quality metrics** to ensure acceptable accuracy
6. **Start with small model** and scale up if needed

## Advanced Configuration

### Custom Model Path

```python
# Use custom model directory
model = FasterWhisperModel(
    "/path/to/model",
    device="cuda",
    compute_type="float16"
)
```

### Language-Specific Optimization

```python
config = STTConfig(
    language="en",              # Fixed language
    auto_detect_language=False  # Faster without detection
)
```

### Memory-Constrained Systems

```python
config = STTConfig(
    model="whisper-base",       # Smallest model
    compute_type="int8",        # Minimal memory
    num_workers=1               # Single thread
)
```

## References

- [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)
- [Whisper Model Card](https://github.com/openai/whisper)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test suite: `tests/unit/voice/test_stt_faster_whisper.py`
3. Open an issue with performance metrics and configuration details
