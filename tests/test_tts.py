from unittest.mock import MagicMock, patch

import pytest

from app.tts.base import BaseTTSEngine
from app.tts.piper_engine import PiperEngine, _pcm_to_wav

# Fake PCM data: 100 samples of silence (16-bit zeros)
FAKE_PCM = b"\x00\x00" * 100


def _make_mock_engine() -> PiperEngine:
    """Instantiate PiperEngine bypassing filesystem checks."""
    with patch("app.tts.piper_engine.Path.exists", return_value=True):
        return PiperEngine(
            piper_bin="/fake/piper.exe",
            model="/fake/model.onnx",
        )


def _make_subprocess_result(returncode: int = 0, stdout: bytes = FAKE_PCM):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = b""
    return result


# ── BaseTTSEngine tests ────────────────────────────────────────────────────────


def test_base_engine_is_abstract():
    with pytest.raises(TypeError):
        BaseTTSEngine()


# ── _pcm_to_wav helper ────────────────────────────────────────────────────────


def test_pcm_to_wav_header():
    wav = _pcm_to_wav(FAKE_PCM, sample_rate=22050)

    # Must start with RIFF
    assert wav[:4] == b"RIFF"
    # WAVE marker at byte 8
    assert wav[8:12] == b"WAVE"
    # fmt chunk marker at byte 12
    assert wav[12:16] == b"fmt "
    # data chunk marker at byte 36
    assert wav[36:40] == b"data"
    # Total length: 44-byte header + PCM
    assert len(wav) == 44 + len(FAKE_PCM)


# ── PiperEngine tests ─────────────────────────────────────────────────────────


def test_synthesize_returns_wav_bytes(mocker):
    engine = _make_mock_engine()
    mocker.patch(
        "subprocess.run",
        return_value=_make_subprocess_result(stdout=FAKE_PCM),
    )

    result = engine.synthesize("Hola mundo")

    assert isinstance(result, bytes)
    assert result[:4] == b"RIFF"
    assert len(result) > 44  # header + at least some audio


def test_synthesize_logs_latency(mocker):
    engine = _make_mock_engine()
    mocker.patch(
        "subprocess.run",
        return_value=_make_subprocess_result(stdout=FAKE_PCM),
    )
    mock_log = mocker.patch("app.tts.piper_engine.logger.info")

    engine.synthesize("Test")

    log_events = [call.args[0] for call in mock_log.call_args_list]
    assert "tts.synthesized" in log_events

    synthesized_call = next(
        c for c in mock_log.call_args_list if c.args[0] == "tts.synthesized"
    )
    assert "latency_ms" in synthesized_call.kwargs


def test_synthesize_raises_on_piper_failure(mocker):
    engine = _make_mock_engine()
    mocker.patch(
        "subprocess.run",
        return_value=_make_subprocess_result(returncode=1, stdout=b""),
    )

    with pytest.raises(RuntimeError, match="Piper failed"):
        engine.synthesize("Texto que falla")


async def test_synthesize_async_returns_wav_bytes(mocker):
    engine = _make_mock_engine()
    mocker.patch(
        "subprocess.run",
        return_value=_make_subprocess_result(stdout=FAKE_PCM),
    )

    result = await engine.synthesize_async("Hola")

    assert isinstance(result, bytes)
    assert result[:4] == b"RIFF"
