import os

import structlog

from app.tts.base import BaseTTSEngine

logger = structlog.get_logger(__name__)


def get_tts_engine() -> BaseTTSEngine:
    """Return the TTS engine selected by the TTS_ENGINE env var.

    Supported values:
        piper   (default) — local Piper TTS
        elevenlabs        — ElevenLabs API (not yet implemented)
    """
    engine_name = os.getenv("TTS_ENGINE", "piper").lower()

    if engine_name == "piper":
        from app.tts.piper_engine import PiperEngine

        return PiperEngine()

    raise ValueError(f"Unknown TTS engine: '{engine_name}'. " f"Supported: 'piper'")
