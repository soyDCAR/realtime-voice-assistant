import asyncio
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.llm.client import LLMClient
from app.llm.conversation import ConversationManager
from app.stt.engine import Transcriber
from app.tts.factory import get_tts_engine

logger = structlog.get_logger(__name__)

_STATIC_DIR = Path(__file__).parent.parent / "static"

# ── Shared resources (loaded once at startup) ─────────────────────────────────

_transcriber: Transcriber | None = None
_llm_client: LLMClient | None = None
_tts_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models once at server startup."""
    global _transcriber, _llm_client, _tts_engine

    logger.info("startup.begin")

    _transcriber = Transcriber()
    _llm_client = LLMClient()
    _tts_engine = get_tts_engine()

    logger.info("startup.complete")
    yield
    logger.info("shutdown.complete")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="realtime-voice-assistant", lifespan=lifespan)

# Static assets (JS, CSS, etc.) at /static — explicit path avoids
# a catch-all mount at "/" which would intercept WebSocket upgrades.
if _STATIC_DIR.exists() and any(_STATIC_DIR.iterdir()):
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ── WebSocket handler ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    conversation = ConversationManager(client=_llm_client)
    logger.info("ws.connected", client=websocket.client)

    try:
        while True:
            # 1. Receive audio bytes from browser
            audio_bytes = await websocket.receive_bytes()
            t_start = time.perf_counter()

            logger.info("ws.audio_received", bytes=len(audio_bytes))

            # 2. STT — CPU-bound → asyncio.to_thread
            transcript = await _run_stt(audio_bytes)
            t_stt = time.perf_counter()

            if not transcript:
                logger.info("ws.empty_transcript")
                await websocket.send_text("")
                continue

            logger.info("ws.transcript", text=transcript)

            # 3. LLM — async I/O → await directly
            response_text = await conversation.chat(transcript)
            t_llm = time.perf_counter()

            logger.info("ws.llm_response", text=response_text)

            # 4. TTS — CPU-bound → asyncio.to_thread
            audio_out = await asyncio.to_thread(_tts_engine.synthesize, response_text)
            t_tts = time.perf_counter()

            # 5. Log full pipeline latency breakdown
            logger.info(
                "ws.pipeline_complete",
                stt_ms=round((t_stt - t_start) * 1000, 1),
                llm_ms=round((t_llm - t_stt) * 1000, 1),
                tts_ms=round((t_tts - t_llm) * 1000, 1),
                total_ms=round((t_tts - t_start) * 1000, 1),
            )

            # 6. Send WAV bytes back to browser
            await websocket.send_bytes(audio_out)

    except WebSocketDisconnect:
        logger.info("ws.disconnected", client=websocket.client)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_stt(audio_bytes: bytes) -> str:
    """Write audio to a temp file and transcribe. Runs in a thread pool."""

    def _transcribe_from_bytes(data: bytes) -> str:
        # delete=False + manual cleanup: avoids Windows file-locking issue
        # where NamedTemporaryFile holds an exclusive lock that blocks Whisper.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            return _transcriber.transcribe(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    return await asyncio.to_thread(_transcribe_from_bytes, audio_bytes)
