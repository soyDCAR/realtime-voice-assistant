---
title: Realtime Voice Assistant
emoji: 🎙
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🎙 realtime-voice-assistant

Conversational voice assistant with real-time STT → LLM → TTS pipeline over WebSocket.  
Speak into the mic → Whisper transcribes → Claude responds → Piper speaks back.  
End-to-end latency target: **< 2.5 seconds**.

[![CI](https://github.com/soyDCAR/realtime-voice-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/soyDCAR/realtime-voice-assistant/actions/workflows/ci.yml)

---

## Architecture

```
Browser (mic + WebSocket client)
    ↕  binary WebSocket frames
FastAPI /ws handler
    → asyncio.to_thread → faster-whisper (STT, base model, int8)
    → await             → Claude API (LLM, Haiku, ≤2 sentences)
    → asyncio.to_thread → Piper TTS (local, es_ES-sharvard-medium)
    ↓  WAV bytes
Browser (WebAudio API playback)
```

**Key design decisions:**
- `asyncio.to_thread` for CPU-bound ops (Whisper, Piper) — never blocks the event loop
- `await` directly for async I/O (Claude API)
- Push-to-talk v1 — no barge-in complexity
- Pluggable TTS interface (`BaseTTSEngine`) — swap Piper → ElevenLabs via env var
- Structured JSON logging with `structlog` — every stage logs `latency_ms`

## Latency budget

| Stage | Target | Measured (local CPU) |
|---|---|---|
| STT (Whisper base, int8) | 600 ms | ~600 ms |
| LLM (Claude Haiku, 2 sentences) | 900 ms | ~700 ms |
| TTS (Piper medium) | 400 ms | ~400 ms |
| **Total** | **< 2500 ms** | **~1700 ms** |

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, uvicorn |
| STT | faster-whisper (CTranslate2, int8 quantized) |
| LLM | Anthropic Claude 3.5 Haiku |
| TTS | Piper TTS (local, ONNX, es_ES-sharvard-medium) |
| Transport | WebSocket (binary frames) |
| Frontend | HTML + JS vanilla, MediaRecorder, WebAudio API |
| Logging | structlog (JSON structured) |
| Testing | pytest, pytest-mock (20 tests, no real API calls) |
| CI | GitHub Actions |
| Deploy | Hugging Face Spaces (Docker) |

## Run locally

```bash
# 1. Clone and set up environment
git clone https://github.com/soyDCAR/realtime-voice-assistant
cd realtime-voice-assistant
uv venv --python 3.11 && uv pip install -e ".[dev]"

# 2. Download Piper binary + model (see .env.example)

# 3. Configure secrets
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```

## Run tests

```bash
# Fast tests only (no model download, CI-equivalent)
pytest tests/ -m "not slow"

# Full suite including real Whisper transcription
pytest tests/
```

## Project structure

```
app/
  stt/engine.py          # Transcriber — faster-whisper wrapper
  llm/client.py          # LLMClient — async Anthropic SDK wrapper
  llm/conversation.py    # ConversationManager — rolling message history
  tts/base.py            # BaseTTSEngine — pluggable interface
  tts/piper_engine.py    # PiperEngine — subprocess + WAV header
  tts/factory.py         # get_tts_engine() — env-based selection
  main.py                # FastAPI app + WebSocket handler + lifespan
static/
  index.html             # Push-to-talk UI (vanilla JS, no frameworks)
tests/
  fixtures/sample.wav    # Real Spanish audio for STT tests
  test_stt.py            # 3 tests (marked slow — use real Whisper)
  test_llm.py            # 7 tests (mocked API)
  test_tts.py            # 6 tests (mocked subprocess)
  test_pipeline.py       # 4 tests (mocked engines, lifespan override)
```

---

Built by [@soyDCAR](https://github.com/soyDCAR) · [LinkedIn](https://www.linkedin.com/in/dilan-acosta-9b408521a/)
