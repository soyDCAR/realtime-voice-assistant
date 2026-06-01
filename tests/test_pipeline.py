import struct
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.main as main_module

# ── Fake audio fixtures ───────────────────────────────────────────────────────

_FAKE_PCM = b"\x00\x00" * 100
_data_size = len(_FAKE_PCM)
_FAKE_WAV = struct.pack(
    "<4sI4s4sIHHIIHH4sI",
    b"RIFF", 36 + _data_size, b"WAVE",
    b"fmt ", 16, 1, 1, 22050, 44100, 2, 16,
    b"data", _data_size,
) + _FAKE_PCM


# ── Mock engines ──────────────────────────────────────────────────────────────

def _make_mocks(transcript: str = "Hola, ¿cómo estás?"):
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = transcript

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="Estoy bien, gracias.")

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = _FAKE_WAV

    return mock_transcriber, mock_llm, mock_tts


# ── Test client factory ───────────────────────────────────────────────────────

def _make_client(transcript: str = "Hola, ¿cómo estás?"):
    """
    Build a TestClient with ML globals pre-injected.

    We override the app's lifespan with a no-op so the real Whisper/Piper/LLM
    never load. The module-level globals (_transcriber, _llm_client, _tts_engine)
    are set directly before the client starts.
    """
    mock_transcriber, mock_llm, mock_tts = _make_mocks(transcript)

    @asynccontextmanager
    async def _noop_lifespan(app):
        main_module._transcriber = mock_transcriber
        main_module._llm_client = mock_llm
        main_module._tts_engine = mock_tts
        yield

    main_module.app.router.lifespan_context = _noop_lifespan

    client = TestClient(main_module.app, raise_server_exceptions=True)
    return client, mock_transcriber, mock_llm, mock_tts


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ws_full_pipeline():
    client, mock_transcriber, mock_llm, mock_tts = _make_client()

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_FAKE_WAV)
            response = ws.receive_bytes()

    assert isinstance(response, bytes)
    assert response[:4] == b"RIFF"
    mock_transcriber.transcribe.assert_called_once()
    mock_llm.complete.assert_called_once()
    mock_tts.synthesize.assert_called_once()


def test_ws_empty_transcript_returns_empty_string():
    client, mock_transcriber, mock_llm, mock_tts = _make_client(transcript="")

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_FAKE_WAV)
            response = ws.receive_text()

    assert response == ""
    mock_llm.complete.assert_not_called()
    mock_tts.synthesize.assert_not_called()


def test_ws_pipeline_latency_logged(mocker):
    client, _, _, _ = _make_client()
    mock_log = mocker.patch.object(main_module.logger, "info")

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_FAKE_WAV)
            ws.receive_bytes()

    log_events = [c.args[0] for c in mock_log.call_args_list]
    assert "ws.pipeline_complete" in log_events

    pipeline_call = next(
        c for c in mock_log.call_args_list
        if c.args[0] == "ws.pipeline_complete"
    )
    assert "stt_ms" in pipeline_call.kwargs
    assert "llm_ms" in pipeline_call.kwargs
    assert "tts_ms" in pipeline_call.kwargs
    assert "total_ms" in pipeline_call.kwargs


def test_ws_disconnect_handled():
    client, _, _, _ = _make_client()

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.close()
    # No exception = disconnect was handled cleanly
