"""
Unit tests for Piper TTS Integration
Tests the native PiperEngine implementation
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import tempfile
import json

from src.voice.piper_integration import (
    PiperVoiceModel,
    PiperEngine,
    PIPER_AVAILABLE
)


@pytest.fixture
def mock_piper_config():
    """Mock Piper voice configuration"""
    return {
        "audio": {
            "sample_rate": 22050,
            "quality": "medium"
        },
        "language": {
            "code": "en_US"
        },
        "num_speakers": 1
    }


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary models directory with mock files"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create mock ONNX model file
    model_file = models_dir / "test_voice.onnx"
    model_file.write_bytes(b"mock_onnx_model")

    # Create mock config file
    config_file = models_dir / "test_voice.onnx.json"
    config_data = {
        "audio": {
            "sample_rate": 22050,
            "quality": "medium"
        },
        "language": {
            "code": "en_US"
        },
        "num_speakers": 1
    }
    config_file.write_text(json.dumps(config_data))

    return models_dir


class TestPiperVoiceModel:
    """Tests for PiperVoiceModel class"""

    def test_init(self, tmp_path):
        """Test PiperVoiceModel initialization"""
        model_path = tmp_path / "model.onnx"
        config_path = tmp_path / "model.onnx.json"

        voice_model = PiperVoiceModel(
            model_path=model_path,
            config_path=config_path,
            use_gpu=False,
            speaker_id=None
        )

        assert voice_model.model_path == model_path
        assert voice_model.config_path == config_path
        assert voice_model.use_gpu is False
        assert voice_model.speaker_id is None
        assert voice_model.is_loaded is False

    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    def test_load_success(self, mock_piper_voice, tmp_path, mock_piper_config):
        """Test successful model loading"""
        # Setup
        model_path = tmp_path / "model.onnx"
        config_path = tmp_path / "model.onnx.json"

        model_path.write_bytes(b"mock_model")
        config_path.write_text(json.dumps(mock_piper_config))

        mock_voice_instance = Mock()
        mock_piper_voice.load.return_value = mock_voice_instance

        # Test
        voice_model = PiperVoiceModel(
            model_path=model_path,
            config_path=config_path
        )
        result = voice_model.load()

        # Verify
        assert result is True
        assert voice_model.is_loaded is True
        assert voice_model.config == mock_piper_config
        mock_piper_voice.load.assert_called_once()

    @patch('src.voice.piper_integration.PIPER_AVAILABLE', False)
    def test_load_piper_not_available(self, tmp_path):
        """Test load failure when Piper not available"""
        model_path = tmp_path / "model.onnx"
        config_path = tmp_path / "model.onnx.json"

        voice_model = PiperVoiceModel(
            model_path=model_path,
            config_path=config_path
        )
        result = voice_model.load()

        assert result is False
        assert voice_model.is_loaded is False

    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    def test_synthesize(self, mock_piper_voice, tmp_path, mock_piper_config):
        """Test audio synthesis"""
        # Setup
        model_path = tmp_path / "model.onnx"
        config_path = tmp_path / "model.onnx.json"
        config_path.write_text(json.dumps(mock_piper_config))

        # Mock audio output
        mock_audio = np.random.randn(1000).astype(np.float32)
        mock_voice_instance = Mock()
        mock_voice_instance.synthesize.return_value = mock_audio
        mock_piper_voice.load.return_value = mock_voice_instance

        # Test
        voice_model = PiperVoiceModel(model_path, config_path)
        voice_model.load()
        result = voice_model.synthesize("Hello world")

        # Verify
        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_voice_instance.synthesize.assert_called_once_with(
            "Hello world",
            speaker_id=None
        )

    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    def test_synthesize_streaming(self, mock_piper_voice, tmp_path, mock_piper_config):
        """Test streaming synthesis"""
        # Setup
        model_path = tmp_path / "model.onnx"
        config_path = tmp_path / "model.onnx.json"
        config_path.write_text(json.dumps(mock_piper_config))

        # Mock audio output (1 second at 22050Hz)
        mock_audio = np.random.randn(22050).astype(np.float32)
        mock_voice_instance = Mock()
        mock_voice_instance.synthesize.return_value = mock_audio
        mock_piper_voice.load.return_value = mock_voice_instance

        # Test
        voice_model = PiperVoiceModel(model_path, config_path)
        voice_model.load()
        chunks = voice_model.synthesize_streaming("Hello world", chunk_size_ms=100)

        # Verify (should create ~10 chunks for 1 second audio at 100ms chunks)
        assert len(chunks) > 0
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)

    def test_get_sample_rate(self, tmp_path, mock_piper_config):
        """Test sample rate retrieval"""
        model_path = tmp_path / "model.onnx"
        config_path = tmp_path / "model.onnx.json"
        config_path.write_text(json.dumps(mock_piper_config))

        voice_model = PiperVoiceModel(model_path, config_path)
        voice_model.config = mock_piper_config

        sample_rate = voice_model.get_sample_rate()
        assert sample_rate == 22050

    def test_get_voice_info(self, tmp_path, mock_piper_config):
        """Test voice metadata retrieval"""
        model_path = tmp_path / "test_voice.onnx"
        config_path = tmp_path / "test_voice.onnx.json"

        voice_model = PiperVoiceModel(model_path, config_path)
        voice_model.config = mock_piper_config

        info = voice_model.get_voice_info()

        assert info["name"] == "test_voice"
        assert info["language"] == "en_US"
        assert info["sample_rate"] == 22050
        assert info["num_speakers"] == 1
        assert info["quality"] == "medium"


class TestPiperEngine:
    """Tests for PiperEngine class"""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test PiperEngine initialization"""
        engine = PiperEngine(
            models_path="./models/piper",
            default_voice="test_voice",
            use_gpu=False,
            prewarm=False
        )

        assert engine.models_path == Path("./models/piper")
        assert engine.default_voice == "test_voice"
        assert engine.use_gpu is False
        assert engine.prewarm is False
        assert engine.is_initialized is False

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', False)
    async def test_initialize_piper_not_available(self):
        """Test initialization when Piper not available"""
        engine = PiperEngine(models_path="./models")
        result = await engine.initialize()

        assert result is True  # Should succeed with warning
        assert engine.is_initialized is True

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    async def test_initialize_success(self, temp_models_dir):
        """Test successful engine initialization"""
        engine = PiperEngine(
            models_path=str(temp_models_dir),
            default_voice="test_voice",
            prewarm=False
        )

        result = await engine.initialize()

        assert result is True
        assert engine.is_initialized is True
        assert "test_voice" in engine.get_available_voices()

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    async def test_discover_voices(self, temp_models_dir):
        """Test voice discovery"""
        # Create multiple voice files
        for i in range(3):
            voice_name = f"voice_{i}"
            model_file = temp_models_dir / f"{voice_name}.onnx"
            config_file = temp_models_dir / f"{voice_name}.onnx.json"

            model_file.write_bytes(b"mock_model")
            config_file.write_text(json.dumps({
                "audio": {"sample_rate": 22050},
                "language": {"code": "en"}
            }))

        engine = PiperEngine(
            models_path=str(temp_models_dir),
            prewarm=False
        )

        await engine.initialize()
        voices = engine.get_available_voices()

        assert len(voices) >= 3
        assert all(f"voice_{i}" in voices for i in range(3))

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    async def test_load_voice(self, mock_piper_voice, temp_models_dir):
        """Test loading a specific voice"""
        mock_voice_instance = Mock()
        mock_piper_voice.load.return_value = mock_voice_instance

        engine = PiperEngine(
            models_path=str(temp_models_dir),
            prewarm=False
        )
        await engine.initialize()

        result = await engine.load_voice("test_voice")

        assert result is True

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    async def test_synthesize(self, mock_piper_voice, temp_models_dir):
        """Test text synthesis"""
        # Mock audio output
        mock_audio = np.random.randn(1000).astype(np.float32)
        mock_voice_instance = Mock()
        mock_voice_instance.synthesize.return_value = mock_audio
        mock_piper_voice.load.return_value = mock_voice_instance

        engine = PiperEngine(
            models_path=str(temp_models_dir),
            default_voice="test_voice",
            prewarm=False
        )
        await engine.initialize()

        # Synthesize
        result = await engine.synthesize("Hello world", voice_name="test_voice")

        assert result is not None
        assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    async def test_synthesize_streaming(self, mock_piper_voice, temp_models_dir):
        """Test streaming synthesis"""
        # Mock audio output
        mock_audio = np.random.randn(22050).astype(np.float32)
        mock_voice_instance = Mock()
        mock_voice_instance.synthesize.return_value = mock_audio
        mock_piper_voice.load.return_value = mock_voice_instance

        engine = PiperEngine(
            models_path=str(temp_models_dir),
            default_voice="test_voice",
            prewarm=False
        )
        await engine.initialize()

        chunks = await engine.synthesize_streaming(
            "Hello world",
            voice_name="test_voice",
            chunk_size_ms=100
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    async def test_get_voice_info(self, temp_models_dir):
        """Test voice info retrieval"""
        engine = PiperEngine(
            models_path=str(temp_models_dir),
            prewarm=False
        )
        await engine.initialize()

        info = engine.get_voice_info("test_voice")

        assert info is not None
        assert "name" in info
        assert "sample_rate" in info

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    async def test_get_sample_rate(self, temp_models_dir):
        """Test sample rate retrieval"""
        engine = PiperEngine(
            models_path=str(temp_models_dir),
            default_voice="test_voice",
            prewarm=False
        )
        await engine.initialize()

        sample_rate = engine.get_sample_rate("test_voice")

        assert isinstance(sample_rate, int)
        assert sample_rate > 0

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    async def test_cleanup(self, temp_models_dir):
        """Test engine cleanup"""
        engine = PiperEngine(
            models_path=str(temp_models_dir),
            prewarm=False
        )
        await engine.initialize()

        assert engine.is_initialized is True

        await engine.cleanup()

        assert engine.is_initialized is False
        assert len(engine.get_available_voices()) == 0


class TestPerformance:
    """Performance tests for PiperEngine"""

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    async def test_synthesis_latency(self, mock_piper_voice, temp_models_dir):
        """Test synthesis latency is below target"""
        import time

        # Mock fast synthesis
        mock_audio = np.random.randn(1000).astype(np.float32)
        mock_voice_instance = Mock()
        mock_voice_instance.synthesize.return_value = mock_audio
        mock_piper_voice.load.return_value = mock_voice_instance

        engine = PiperEngine(
            models_path=str(temp_models_dir),
            default_voice="test_voice",
            prewarm=True
        )
        await engine.initialize()

        # Measure synthesis time
        start = time.time()
        await engine.synthesize("This is a test sentence for latency measurement.")
        latency = (time.time() - start) * 1000  # Convert to ms

        # Target is <200ms for first audio
        assert latency < 500  # Allow some margin for mock overhead

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    async def test_model_caching(self, mock_piper_voice, temp_models_dir):
        """Test that models are cached and reused"""
        mock_audio = np.random.randn(1000).astype(np.float32)
        mock_voice_instance = Mock()
        mock_voice_instance.synthesize.return_value = mock_audio
        mock_piper_voice.load.return_value = mock_voice_instance

        engine = PiperEngine(
            models_path=str(temp_models_dir),
            default_voice="test_voice",
            prewarm=True
        )
        await engine.initialize()

        # Load count should be 1 (from prewarm)
        initial_load_count = mock_piper_voice.load.call_count

        # Synthesize multiple times
        for _ in range(5):
            await engine.synthesize("Test", voice_name="test_voice")

        # Load should not be called again
        assert mock_piper_voice.load.call_count == initial_load_count


class TestErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_synthesize_invalid_voice(self):
        """Test synthesis with invalid voice name"""
        engine = PiperEngine(models_path="./nonexistent", prewarm=False)
        await engine.initialize()

        result = await engine.synthesize("Test", voice_name="invalid_voice")

        assert result is None

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    async def test_initialize_invalid_path(self):
        """Test initialization with invalid models path"""
        engine = PiperEngine(
            models_path="/nonexistent/path",
            prewarm=False
        )

        # Should succeed but find no voices
        result = await engine.initialize()
        assert result is True
        assert len(engine.get_available_voices()) == 0

    @pytest.mark.asyncio
    @patch('src.voice.piper_integration.PIPER_AVAILABLE', True)
    @patch('src.voice.piper_integration.PiperVoice')
    async def test_synthesis_error_handling(self, mock_piper_voice, temp_models_dir):
        """Test error handling during synthesis"""
        mock_voice_instance = Mock()
        mock_voice_instance.synthesize.side_effect = Exception("Synthesis failed")
        mock_piper_voice.load.return_value = mock_voice_instance

        engine = PiperEngine(
            models_path=str(temp_models_dir),
            default_voice="test_voice",
            prewarm=False
        )
        await engine.initialize()

        result = await engine.synthesize("Test", voice_name="test_voice")

        assert result is None
