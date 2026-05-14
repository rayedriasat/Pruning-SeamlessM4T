from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .boundary_vad import BoundaryVAD
from .model import SeamlessTranslator
from .session import Session, SessionConfig

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("duplex.app")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models2" / "phase7_final_merged"))
ADAPTER_PATH = Path(os.environ.get("ADAPTER_PATH", ROOT / "models2" / "boundary_adapter.pt"))
FRONTEND = ROOT / "frontend"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading translator from %s", MODEL_PATH)
    translator = SeamlessTranslator(MODEL_PATH)
    log.info("loading boundary adapter from %s", ADAPTER_PATH)
    vad = BoundaryVAD(translator.model, translator.processor, ADAPTER_PATH, device=translator.device)
    translator.warmup()
    state["translator"] = translator
    state["vad"] = vad
    log.info("ready")
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND / "index.html"))


@app.get("/app.js")
async def app_js():
    return FileResponse(str(FRONTEND / "app.js"), media_type="application/javascript")


@app.get("/worklet.js")
async def worklet_js():
    return FileResponse(str(FRONTEND / "worklet.js"), media_type="application/javascript")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": "translator" in state,
        "vad_loaded": "vad" in state,
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, lang: str = "ben"):
    await ws.accept()
    translator = state.get("translator")
    vad = state.get("vad")
    if translator is None or vad is None:
        await ws.close(code=1011)
        return
    session = Session(ws, translator, vad, SessionConfig(), tgt_lang=lang)
    try:
        await session.run()
    finally:
        await session.cleanup()
