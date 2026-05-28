import os
import time
from pathlib import Path

import structlog
from faster_whisper import WhisperModel

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")
_DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")


class Transcriber:
    def __init__(
        self,
        model_size: str = _DEFAULT_MODEL,
        device: str = _DEFAULT_DEVICE,
        language: str = "es",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.language = language

        load_start = time.perf_counter()
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8",
        )
        load_ms = (time.perf_counter() - load_start) * 1000

        logger.info(
            "stt.model_loaded",
            model=model_size,
            device=device,
            load_ms=round(load_ms, 1),
        )

    def transcribe(self, audio_path: str | Path) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t0 = time.perf_counter()

        segments, info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "stt.transcribed",
            audio_file=audio_path.name,
            language=info.language,
            language_probability=round(info.language_probability, 3),
            latency_ms=round(latency_ms, 1),
            text_preview=text[:80],
        )

        return text
