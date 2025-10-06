"""Unit tests for Twilio audio format conversion"""

import pytest
import numpy as np
from src.voice.twilio_integration import TwilioIntegration


class TestAudioConversion:
    """Test audio format conversion methods"""

    @pytest.fixture
    def twilio_integration(self):
        """Create TwilioIntegration instance for testing"""
        return TwilioIntegration()

    def test_mulaw_to_pcm_conversion(self, twilio_integration):
        """Test µ-law to PCM conversion"""
        # Create sample µ-law data (simulate Twilio audio)
        # µ-law encoding of silence and simple tone
        mulaw_data = bytes([0x7F, 0x7F, 0x7F, 0x7F] * 100)  # 400 bytes of µ-law

        # Convert to PCM
        pcm_data = twilio_integration._convert_from_mulaw(mulaw_data)

        # Verify conversion produced data
        assert pcm_data is not None
        assert len(pcm_data) > 0
        # PCM should be 2x size (16-bit vs 8-bit)
        assert len(pcm_data) == len(mulaw_data) * 2

    def test_pcm_to_mulaw_conversion(self, twilio_integration):
        """Test PCM to µ-law conversion"""
        # Create sample PCM data (16-bit signed integers)
        pcm_array = np.array([0, 100, -100, 500, -500] * 100, dtype=np.int16)
        pcm_data = pcm_array.tobytes()

        # Convert to µ-law
        mulaw_data = twilio_integration._convert_to_mulaw(pcm_data)

        # Verify conversion produced data
        assert mulaw_data is not None
        assert len(mulaw_data) > 0
        # µ-law should be half size (8-bit vs 16-bit)
        assert len(mulaw_data) == len(pcm_data) // 2

    def test_bidirectional_mulaw_pcm_conversion(self, twilio_integration):
        """Test round-trip µ-law <-> PCM conversion"""
        # Create original PCM data
        original_pcm = np.array([0, 1000, -1000, 5000, -5000] * 50, dtype=np.int16)
        original_bytes = original_pcm.tobytes()

        # Convert PCM -> µ-law -> PCM
        mulaw_data = twilio_integration._convert_to_mulaw(original_bytes)
        recovered_pcm = twilio_integration._convert_from_mulaw(mulaw_data)

        # Verify sizes match
        assert len(original_bytes) == len(recovered_pcm)

        # Note: µ-law is lossy compression, so exact match not expected
        # But data should be similar in magnitude
        recovered_array = np.frombuffer(recovered_pcm, dtype=np.int16)

        # Check that recovered data is in similar range
        assert recovered_array.max() > 0
        assert recovered_array.min() < 0

    def test_resample_8k_to_16k(self, twilio_integration):
        """Test resampling from 8kHz to 16kHz"""
        # Create 8kHz audio (1 second = 8000 samples)
        audio_8k = np.array([np.sin(2 * np.pi * 440 * t / 8000) * 10000
                             for t in range(8000)], dtype=np.int16)
        audio_8k_bytes = audio_8k.tobytes()

        # Resample to 16kHz
        audio_16k_bytes = twilio_integration._resample_audio(
            audio_data=audio_8k_bytes,
            from_rate=8000,
            to_rate=16000,
            channels=1
        )

        # Verify output
        assert audio_16k_bytes is not None
        audio_16k = np.frombuffer(audio_16k_bytes, dtype=np.int16)

        # Should have approximately 2x samples (8000 -> 16000)
        expected_length = 16000  # 1 second at 16kHz
        # Allow 1% tolerance
        assert abs(len(audio_16k) - expected_length) < expected_length * 0.01

    def test_resample_16k_to_8k(self, twilio_integration):
        """Test resampling from 16kHz to 8kHz"""
        # Create 16kHz audio (1 second = 16000 samples)
        audio_16k = np.array([np.sin(2 * np.pi * 440 * t / 16000) * 10000
                              for t in range(16000)], dtype=np.int16)
        audio_16k_bytes = audio_16k.tobytes()

        # Resample to 8kHz
        audio_8k_bytes = twilio_integration._resample_audio(
            audio_data=audio_8k_bytes,
            from_rate=16000,
            to_rate=8000,
            channels=1
        )

        # Verify output
        assert audio_8k_bytes is not None
        audio_8k = np.frombuffer(audio_8k_bytes, dtype=np.int16)

        # Should have approximately 0.5x samples (16000 -> 8000)
        expected_length = 8000  # 1 second at 8kHz
        # Allow 1% tolerance
        assert abs(len(audio_8k) - expected_length) < expected_length * 0.01

    def test_resample_preserves_audio_characteristics(self, twilio_integration):
        """Test that resampling preserves general audio characteristics"""
        # Create audio with known characteristics
        duration_s = 1.0
        sample_rate_in = 16000
        frequency_hz = 440  # A4 note

        # Generate sine wave
        t = np.arange(0, duration_s, 1/sample_rate_in)
        audio_in = np.sin(2 * np.pi * frequency_hz * t) * 10000
        audio_in = audio_in.astype(np.int16)
        audio_in_bytes = audio_in.tobytes()

        # Resample to 8kHz
        audio_out_bytes = twilio_integration._resample_audio(
            audio_data=audio_in_bytes,
            from_rate=16000,
            to_rate=8000,
            channels=1
        )

        audio_out = np.frombuffer(audio_out_bytes, dtype=np.int16)

        # Check that output has similar amplitude range
        assert audio_out.max() > 8000  # Should be close to 10000
        assert audio_out.min() < -8000

    def test_full_inbound_conversion_pipeline(self, twilio_integration):
        """Test complete inbound audio conversion: µ-law 8kHz -> PCM 16kHz"""
        # Simulate Twilio audio: µ-law 8kHz
        # Create PCM 8kHz first
        audio_8k_pcm = np.array([np.sin(2 * np.pi * 440 * t / 8000) * 10000
                                 for t in range(800)], dtype=np.int16)  # 100ms
        audio_8k_bytes = audio_8k_pcm.tobytes()

        # Convert to µ-law (simulate Twilio format)
        mulaw_data = twilio_integration._convert_to_mulaw(audio_8k_bytes)

        # Now apply full inbound conversion pipeline
        # 1. µ-law -> PCM
        pcm_8k = twilio_integration._convert_from_mulaw(mulaw_data)

        # 2. Resample 8kHz -> 16kHz
        pcm_16k = twilio_integration._resample_audio(
            audio_data=pcm_8k,
            from_rate=8000,
            to_rate=16000,
            channels=1
        )

        # Verify output
        assert pcm_16k is not None
        audio_16k = np.frombuffer(pcm_16k, dtype=np.int16)

        # Should have approximately 2x samples
        expected_length = 1600  # 100ms at 16kHz
        assert abs(len(audio_16k) - expected_length) < expected_length * 0.05

    def test_full_outbound_conversion_pipeline(self, twilio_integration):
        """Test complete outbound audio conversion: PCM 16kHz -> µ-law 8kHz"""
        # Start with PCM 16kHz (typical TTS output)
        audio_16k = np.array([np.sin(2 * np.pi * 440 * t / 16000) * 10000
                              for t in range(1600)], dtype=np.int16)  # 100ms
        audio_16k_bytes = audio_16k.tobytes()

        # Apply full outbound conversion pipeline
        # 1. Resample 16kHz -> 8kHz
        pcm_8k = twilio_integration._resample_audio(
            audio_data=audio_16k_bytes,
            from_rate=16000,
            to_rate=8000,
            channels=1
        )

        # 2. PCM -> µ-law
        mulaw_data = twilio_integration._convert_to_mulaw(pcm_8k)

        # Verify output
        assert mulaw_data is not None

        # µ-law should be ~800 bytes (100ms at 8kHz, 8-bit)
        expected_length = 800
        assert abs(len(mulaw_data) - expected_length) < expected_length * 0.05

    def test_empty_audio_handling(self, twilio_integration):
        """Test that empty audio is handled gracefully"""
        empty_data = bytes()

        # Test µ-law conversion
        result = twilio_integration._convert_from_mulaw(empty_data)
        assert result == empty_data

        result = twilio_integration._convert_to_mulaw(empty_data)
        assert result == empty_data

        # Test resampling
        result = twilio_integration._resample_audio(empty_data, 8000, 16000)
        assert result == empty_data

    def test_invalid_audio_handling(self, twilio_integration):
        """Test that invalid audio data is handled gracefully"""
        # Test with odd number of bytes (invalid for 16-bit PCM)
        invalid_pcm = bytes([0x00, 0x01, 0x02])  # 3 bytes - odd length

        # Should not crash
        result = twilio_integration._convert_to_mulaw(invalid_pcm)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
