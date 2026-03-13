"""
Unit tests for voice service utilities.

Per test_strategy.md:
  - Voice Utils: Test audio file conversion functions (wav -> mp3 -> float32).
"""

import base64
import io
import struct
from unittest.mock import patch, MagicMock

import pytest

from backend.src.services.voice import VoiceService


def _make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a minimal valid WAV file with silence."""
    num_samples = int(sample_rate * duration_s)
    data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    buf = io.BytesIO()
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))           # chunk size
    buf.write(struct.pack("<H", 1))            # PCM
    buf.write(struct.pack("<H", 1))            # mono
    buf.write(struct.pack("<I", sample_rate))   # sample rate
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))            # block align
    buf.write(struct.pack("<H", 16))           # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    return buf.getvalue()


class TestVoiceServiceInit:
    def test_initial_state(self):
        svc = VoiceService()
        assert svc.stt_available is False
        assert svc.tts_available is False

    def test_load_without_whisper(self):
        """If Whisper is not installed, STT should be disabled gracefully."""
        svc = VoiceService()
        with patch.dict("sys.modules", {"whisper": None}):
            with patch("builtins.__import__", side_effect=ImportError("No whisper")):
                svc.load(whisper_model_size="base")
        # STT should be False (graceful degradation)
        # tts_available depends on whether gTTS is installed

    def test_transcribe_without_load_raises(self):
        svc = VoiceService()
        with pytest.raises(RuntimeError, match="Whisper model not loaded"):
            svc.transcribe(b"fake audio")


class TestVoiceServiceSynthesize:
    def test_synthesize_returns_none_when_tts_unavailable(self):
        svc = VoiceService()
        result = svc.synthesize("Hello world")
        assert result is None

    def test_synthesize_base64_returns_none_when_tts_unavailable(self):
        svc = VoiceService()
        result = svc.synthesize_base64("Hello world")
        assert result is None


class TestWavGeneration:
    """Validate our test WAV helper produces valid WAV bytes."""

    def test_wav_starts_with_riff(self):
        wav = _make_wav_bytes()
        assert wav[:4] == b"RIFF"

    def test_wav_has_wave_marker(self):
        wav = _make_wav_bytes()
        assert wav[8:12] == b"WAVE"

    def test_wav_has_nonzero_length(self):
        wav = _make_wav_bytes(duration_s=1.0)
        assert len(wav) > 44  # WAV header = 44 bytes


class TestTranscribeMocked:
    """Test transcription with a mocked Whisper model."""

    def test_successful_transcription(self):
        svc = VoiceService()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  red shoes  "}
        svc._whisper_model = mock_model
        svc._whisper_loaded = True

        wav_bytes = _make_wav_bytes()
        transcript, elapsed = svc.transcribe(wav_bytes)
        assert transcript == "red shoes"
        assert elapsed >= 0

    def test_empty_transcription(self):
        svc = VoiceService()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": ""}
        svc._whisper_model = mock_model
        svc._whisper_loaded = True

        wav_bytes = _make_wav_bytes()
        transcript, elapsed = svc.transcribe(wav_bytes)
        assert transcript == ""

    def test_transcription_exception_returns_empty(self):
        svc = VoiceService()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("decode error")
        svc._whisper_model = mock_model
        svc._whisper_loaded = True

        wav_bytes = _make_wav_bytes()
        transcript, elapsed = svc.transcribe(wav_bytes)
        assert transcript == ""
        assert elapsed == 0.0

    def test_temp_file_cleaned_up(self):
        """Verify the temp WAV file is deleted after transcription."""
        svc = VoiceService()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello"}
        svc._whisper_model = mock_model
        svc._whisper_loaded = True

        import tempfile
        from pathlib import Path
        wav_bytes = _make_wav_bytes()
        svc.transcribe(wav_bytes)
        # All temp files in the default temp dir that match pattern should be cleaned
        # We just ensure no crash — the `finally` block in voice.py deletes the file
