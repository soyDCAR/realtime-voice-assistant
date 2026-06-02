import asyncio
import subprocess
import sys
import time
from pathlib import Path

import structlog

from app.tts.base import BaseTTSEngine

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PIPER_BIN_NAME = "piper.exe" if sys.platform == "win32" else "piper"
_DEFAULT_PIPER_BIN = str(_PROJECT_ROOT / "piper" / "piper" / _PIPER_BIN_NAME)
_DEFAULT_MODEL = str(_PROJECT_ROOT / "piper" / "es_ES-sharvard-medium.onnx")


class PiperEngine(BaseTTSEngine):
    def __init__(
        self,
        piper_bin: str = _DEFAULT_PIPER_BIN,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.piper_bin = Path(piper_bin)
        self.model = Path(model)

        if not self.piper_bin.exists():
            raise FileNotFoundError(f"Piper binary not found: {self.piper_bin}")
        if not self.model.exists():
            raise FileNotFoundError(f"Piper model not found: {self.model}")

        logger.info(
            "tts.engine_ready",
            engine="piper",
            model=self.model.name,
        )

    def synthesize(self, text: str) -> bytes:
        """Blocking synthesis — call via asyncio.to_thread in async contexts."""
        t0 = time.perf_counter()

        result = subprocess.run(
            [
                str(self.piper_bin),
                "--model",
                str(self.model),
                "--output_raw",
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Piper failed (exit {result.returncode}): "
                f"{result.stderr.decode('utf-8', errors='replace')}"
            )

        raw_pcm = result.stdout
        wav_bytes = _pcm_to_wav(raw_pcm, sample_rate=22050)

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "tts.synthesized",
            engine="piper",
            text_len=len(text),
            audio_bytes=len(wav_bytes),
            latency_ms=round(latency_ms, 1),
        )

        return wav_bytes

    async def synthesize_async(self, text: str) -> bytes:
        """Non-blocking synthesis for use in async handlers."""
        return await asyncio.to_thread(self.synthesize, text)


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 22050) -> bytes:
    """Wrap raw PCM bytes in a minimal WAV header.

    Piper with --output_raw produces 16-bit signed little-endian PCM.
    The browser's WebAudio API needs a WAV container to decode it.
    """
    import struct

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # subchunk1 size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data
