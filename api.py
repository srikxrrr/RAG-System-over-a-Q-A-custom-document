"""FastAPI app: chat endpoint, document list, upload + reindex, and the static web UI."""
from __future__ import annotations

import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import load_config
from .generate import llm_available
from .loaders import discover
from .pipeline import RAGEngine, ingest

load_dotenv()
cfg = load_config()
app = FastAPI(title="CivilCode RAG", version="0.1.0")
_lock = threading.Lock()
_engine: RAGEngine | None = None
WEB = Path(__file__).resolve().parents[2] / "web"
MAX_UPLOAD_MB = 50


def engine() -> RAGEngine:
    global _engine
    if _engine is None:
        try:
            _engine = RAGEngine(cfg)
        except FileNotFoundError:
            raise HTTPException(503, "No index yet. Upload documents or run `python -m civilrag ingest`.")
    return _engine


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[Turn] = []


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm_available(), "indexed": (cfg.path(cfg.store_dir) / "chunks.jsonl").exists()}


@app.post("/api/ask")
def ask(req: AskRequest):
    return engine().ask(req.question.strip(), [t.model_dump() for t in req.history])


@app.get("/api/docs")
def docs():
    folder = cfg.path(cfg.data.dir)
    files = [p.name for p in discover(folder, cfg.data.extensions)] if folder.exists() else []
    counts: dict[str, int] = {}
    if _engine or (cfg.path(cfg.store_dir) / "chunks.jsonl").exists():
        for c in engine().index.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
    return [{"file": f, "chunks": counts.get(f, 0)} for f in files]


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    folder = cfg.path(cfg.data.dir)
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(f.filename or "").name)
        if Path(name).suffix.lower() not in cfg.data.extensions:
            raise HTTPException(400, f"Unsupported file type: {name}")
        data = await f.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"{name} is larger than {MAX_UPLOAD_MB} MB")
        (folder / name).write_bytes(data)
        saved.append(name)
    global _engine
    with _lock:
        ingest(cfg)
        _engine = RAGEngine(cfg)
    return {"saved": saved, "chunks": len(_engine.index.chunks)}
