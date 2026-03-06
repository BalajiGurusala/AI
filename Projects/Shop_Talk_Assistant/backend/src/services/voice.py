"""
ShopTalk Backend — Voice Service (STT + TTS).

Per requirements.md:
  - STT: OpenAI Whisper (open-source, runs locally — no API key)
  - TTS: gTTS (Google Text-to-Speech, free) with optional ElevenLabs

Models load once at startup (constitution.md).
"""

import io
import base64
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VoiceService:
    """Speech-to-Text (Whisper) and Text-to-Speech (gTTS) singleton."""

    def __init__(self):
        self._whisper_model = None
        self._whisper_loaded = False
        self._tts_available = False

    @property
    def stt_available(self) -> bool:
        return self._whisper_loaded

    @property
    def tts_available(self) -> bool:
        return self._tts_available

    def load(self, whisper_model_size: str = "base"):
        """Load Whisper model. Call once at startup.

        Model sizes: tiny (39M), base (74M), small (244M), medium (769M)
        'base' is a good tradeoff: fast + accurate enough for product queries.
        """
        # --- Load Whisper ---
        try:
            import whisper
            logger.info(f"Loading Whisper model: {whisper_model_size}")
            t0 = time.time()
            self._whisper_model = whisper.load_model(whisper_model_size)
            self._whisper_loaded = True
            logger.info(f"  Whisper loaded in {time.time()-t0:.1f}s")
        except ImportError:
            logger.warning("openai-whisper not installed — STT disabled. "
                           "Install with: pip install openai-whisper")
        except Exception as e:
            logger.warning(f"Whisper load failed: {e} — STT disabled")

        # --- Check gTTS ---
        try:
            import gtts  # noqa: F401
            self._tts_available = True
            logger.info("gTTS available for text-to-speech")
        except ImportError:
            logger.warning("gTTS not installed — TTS disabled. "
                           "Install with: pip install gTTS")

    def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        """Transcribe audio bytes to text using Whisper.

        Args:
            audio_bytes: Raw audio bytes (WAV, MP3, etc.)

        Returns:
            (transcript, confidence) tuple. Empty string on failure.
        """
        if not self._whisper_loaded:
            raise RuntimeError("Whisper model not loaded")

        # Write to temp file (Whisper expects a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            t0 = time.time()
            result = self._whisper_model.transcribe(
                temp_path,
                language="en",
                fp16=False,  # CPU-safe
            )
            elapsed = time.time() - t0
            transcript = result.get("text", "").strip()
            logger.info(f"Transcribed in {elapsed:.2f}s: '{transcript[:80]}'")
            return transcript, elapsed
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", 0.0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech using gTTS.

        Args:
            text: Text to speak.

        Returns:
            MP3 audio bytes, or None on failure.
        """
        if not self._tts_available:
            return None

        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="en", slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return None

    def synthesize_base64(self, text: str) -> Optional[str]:
        """Synthesize and return base64-encoded MP3 (for API responses)."""
        audio = self.synthesize(text)
        if audio:
            return base64.b64encode(audio).decode("utf-8")
        return None


# Singleton
voice_service = VoiceService()
