"""FastAPI app: chat + drawing upload ingest + knowledge-graph health."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from app.envfile import load_local_env

load_local_env()

from app.generate import VALID_GENERATE, configured_providers, resolve_mode
from app.ingest import (
    get_job,
    list_jobs,
    resolve_ocr_python,
    resolve_vlm_mode,
    save_upload,
    start_ingest_async,
)
from app.logutil import setup_logging
from app.rag import DEFAULT_BACKEND, VALID_BACKENDS, RagService, SessionState
from app import store as db

log = setup_logging("cad_agent.api")

MAX_SESSIONS = 200
MAX_HISTORY_MESSAGES = 40
MAX_SESSION_ID_LEN = 64
STATIC_DIR = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "datasets" / "images"
GRAPH_PATH = ROOT / "datasets" / "index" / "knowledge_graph.json"

BACKEND = os.environ.get("CAD_RAG_BACKEND", DEFAULT_BACKEND)
if BACKEND not in VALID_BACKENDS:
    raise SystemExit(f"CAD_RAG_BACKEND must be one of {sorted(VALID_BACKENDS)}")

GENERATE = os.environ.get("CAD_GENERATE", "auto")
GENERATE_RESOLVED = resolve_mode(GENERATE)

app = FastAPI(title="CAD Agent (Engineering Knowledge Base)", version="0.3.0")
rag = RagService(backend=BACKEND, generate=GENERATE)

_STEM_RE = re.compile(
    r"^[\w .\-()+\u4e00-\u9fff]+(?:__ps_[\w .\-]+)?$",
    re.UNICODE,
)
_SESSION_ID_RE = re.compile(r"^[\w.\-]+$", re.UNICODE)


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户问题", min_length=1, max_length=500)
    generate_mode: str | None = Field(
        default=None,
        description="回答生成：auto / qwen / deepseek / extractive",
    )
    session_id: str = Field(
        default="default",
        description="会话 ID，用于多轮记忆",
        min_length=1,
        max_length=MAX_SESSION_ID_LEN,
    )

    @field_validator("session_id")
    @classmethod
    def _session_id_chars(cls, value: str) -> str:
        text = value.strip()
        if not text or not _SESSION_ID_RE.match(text):
            raise ValueError(
                "session_id 仅允许字母数字、下划线、点、短横线，长度 1–64"
            )
        return text

    @field_validator("generate_mode")
    @classmethod
    def _generate_mode_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        mode = value.strip().lower()
        allowed = VALID_GENERATE - {"off"}
        if mode not in allowed:
            raise ValueError(f"generate_mode 仅允许 {sorted(allowed)}")
        return mode

class ChatResponse(BaseModel):
    answer: str
    drawing: str | None = Field(
        default=None,
        description="本轮聚焦的图纸；证据跨多张图或属于全库提问时为 null",
    )
    sources: list[dict] = Field(description="命中的索引块，可回溯到 final JSON")
    history_turns: int
    backend: str = Field(description="检索后端：bm25 / dense / hybrid")
    generate_mode: str = Field(
        description="实际生成后端：off / extractive / deepseek / qwen"
    )
    generate_note: str | None = Field(
        default=None, description="生成降级说明，例如 API 失败后改用 extractive"
    )
    preview_url: str | None = Field(
        default=None, description="当前锁定图纸的渲染预览 URL"
    )
    intent: dict | None = Field(
        default=None, description="结构化意图：scope / kind / drawing / clause_code"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="本轮调用的只读工具名"
    )


def get_session(session_id: str) -> SessionState:
    raw = db.get_session(session_id, max_history=MAX_HISTORY_MESSAGES)
    return SessionState(
        drawing=raw.get("drawing"),
        history=list(raw.get("history") or []),
    )


def _persist_session(session_id: str, state: SessionState) -> None:
    db.save_session(
        session_id,
        drawing=state.drawing,
        history=state.history[-MAX_HISTORY_MESSAGES:],
        max_sessions=MAX_SESSIONS,
        max_history=MAX_HISTORY_MESSAGES,
    )


def _preview_url(drawing: str | None) -> str | None:
    if not drawing:
        return None
    path = IMAGE_DIR / f"{drawing}.png"
    if path.is_file():
        return f"/images/{drawing}.png"
    return None


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/images/{name:path}")
def drawing_image(name: str) -> FileResponse:
    """Serve rendered PNGs for the sources rail / drawing preview."""
    base = Path(name).name
    if not base.lower().endswith(".png"):
        raise HTTPException(status_code=404, detail="only .png")
    stem = base[:-4]
    if ".." in name or "/" in name.replace("\\", "/") or not _STEM_RE.match(stem):
        raise HTTPException(status_code=400, detail="invalid image name")
    path = (IMAGE_DIR / base).resolve()
    if not str(path).startswith(str(IMAGE_DIR.resolve())) or not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api")
def api_info() -> dict:
    return {
        "name": "CAD Agent (Engineering Knowledge Base)",
        "version": "0.3.0",
        "endpoints": [
            "/health",
            "/chat",
            "/session/{session_id}/clear-drawing",
            "/upload",
            "/ingest/{job_id}",
            "/drawings",
            "/images/{name}",
            "/graph",
            "/tools",
            "/ui",
            "/docs",
        ],
        "backend": rag.backend,
        "generate": GENERATE_RESOLVED,
        "graph": GRAPH_PATH.exists(),
        "store": "sqlite",
        "tip": (
            "Upload .dwg/.dxf via POST /upload; poll /ingest/{job_id}; "
            "CAD_RAG_BACKEND=bm25|dense|hybrid; "
            "CAD_GENERATE=auto|qwen|deepseek|extractive|off; "
            "CAD_GRAPH_HOPS=3; CAD_TOOLS=1; CAD_RERANK=1"
        ),
    }


@app.get("/tools")
def list_tools() -> dict:
    """Read-only tool catalog (Pydantic-validated; no CAD write-back)."""
    from app.tools import tool_catalog

    return {
        "ok": True,
        "readonly": True,
        "note": (
            "受限只读工具：由意图路由调用，参数经 Pydantic 校验。"
            "不是 AutoCAD 写回 / Function Calling 改图。"
        ),
        "tools": tool_catalog(),
    }


@app.get("/health")
def health() -> dict:
    graph_stats = {}
    if GRAPH_PATH.exists():
        try:
            graph_stats = json.loads(GRAPH_PATH.read_text(encoding="utf-8")).get(
                "stats", {}
            )
        except (OSError, json.JSONDecodeError):
            graph_stats = {}

    ocr_python = None
    ocr_python_error = None
    try:
        ocr_python = resolve_ocr_python()
    except Exception as exc:  # noqa: BLE001
        ocr_python_error = str(exc)

    embeddings_synced = None
    try:
        from app.embeddings import EMBED_META_PATH

        if EMBED_META_PATH.exists():
            meta = json.loads(EMBED_META_PATH.read_text(encoding="utf-8"))
            embeddings_synced = meta.get("chunk_count") == len(rag.chunks)
        else:
            embeddings_synced = False
    except Exception:  # noqa: BLE001
        embeddings_synced = False

    return {
        "ok": True,
        "chunks": len(rag.chunks),
        "drawings": len(rag.name_terms),
        "sessions": db.session_count(),
        "backend": rag.backend,
        "generate": GENERATE_RESOLVED,
        "generate_configured": GENERATE,
        "cloud_providers": configured_providers(),
        "vlm_default": resolve_vlm_mode("auto"),
        "ocr_python": ocr_python,
        "ocr_python_error": ocr_python_error,
        "embeddings_synced": embeddings_synced,
        "graph_nodes": graph_stats.get("node_count"),
        "graph_edges": graph_stats.get("edge_count"),
        "graph_hops": int(os.environ.get("CAD_GRAPH_HOPS", "3")),
        "store": "sqlite",
        "tip": (
            "generate 是配置目标；API 失败时单次 /chat 会降级为 extractive"
            if GENERATE_RESOLVED in {"deepseek", "qwen"}
            else None
        ),
    }


@app.get("/drawings")
def drawings() -> dict:
    items = []
    for drawing, terms in sorted(rag.name_terms.items()):
        items.append(
            {
                "id": drawing,
                "aliases": sorted(terms)[:12],
                "preview_url": _preview_url(drawing),
            }
        )
    return {"count": len(items), "drawings": items}


@app.get("/graph")
def graph_summary() -> dict:
    if not GRAPH_PATH.exists():
        return {"ok": False, "error": "graph missing; run py scripts/build_graph.py"}
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "stats": data.get("stats"),
        "sample_nodes": (data.get("nodes") or [])[:8],
        "sample_edges": (data.get("edges") or [])[:8],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    state = get_session(req.session_id)
    result = rag.answer(req.message, state, generate=req.generate_mode)
    del state.history[:-MAX_HISTORY_MESSAGES]
    _persist_session(req.session_id, state)
    drawing = result.get("drawing")
    sources = result.get("sources") or []
    for src in sources:
        if src.get("drawing") and not src.get("preview_url"):
            src["preview_url"] = _preview_url(src.get("drawing"))
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    intent = result.get("intent") or {}
    log.info(
        "chat session=%s backend=%s generate=%s note=%s drawing=%s "
        "intent=%s tools=%s sources=%s ms=%s q=%r",
        req.session_id,
        rag.backend,
        result.get("generate_mode") or GENERATE_RESOLVED,
        result.get("generate_note"),
        drawing,
        intent.get("kind"),
        result.get("tools_used"),
        len(sources),
        elapsed_ms,
        req.message[:80],
    )
    return ChatResponse(
        answer=result["answer"],
        drawing=drawing,
        sources=sources,
        history_turns=result.get("history_turns") or 0,
        backend=rag.backend,
        generate_mode=result.get("generate_mode") or GENERATE_RESOLVED,
        generate_note=result.get("generate_note"),
        preview_url=_preview_url(drawing) or _preview_url(state.drawing),
        intent=result.get("intent"),
        tools_used=list(result.get("tools_used") or []),
    )


@app.post("/session/{session_id}/clear-drawing")
def clear_session_drawing(session_id: str) -> dict:
    """Release the drawing lock while preserving the conversation history."""
    if not _SESSION_ID_RE.match(session_id) or len(session_id) > MAX_SESSION_ID_LEN:
        raise HTTPException(status_code=400, detail="invalid session_id")
    state = get_session(session_id)
    state.drawing = None
    _persist_session(session_id, state)
    return {"ok": True, "drawing": None}


@app.post("/upload")
async def upload(
    file: UploadFile = File(..., description="DWG 或 DXF 文件"),
    mode: str = Form(
        "auto",
        description="视觉模式：auto（有密钥则真实 VLM）/ qwen / heuristic",
    ),
    rebuild_embeddings: bool = Form(
        True,
        description="入库后是否重建 BGE 向量索引（默认 true，保持 dense/hybrid 同步）",
    ),
) -> dict:
    if mode not in {"auto", "heuristic", "qwen"}:
        return {"ok": False, "error": "mode must be auto|heuristic|qwen"}
    raw = await file.read()
    try:
        path = save_upload(file.filename or "upload.dxf", raw)
    except ValueError as exc:
        log.warning("upload rejected: %s", exc)
        return {"ok": False, "error": str(exc)}

    job = start_ingest_async(
        [path],
        mode=mode,
        rebuild_embeddings=rebuild_embeddings,
    )
    log.info(
        "upload job=%s file=%s mode=%s resolved=%s rebuild_emb=%s bytes=%s",
        job.id,
        path.name,
        mode,
        job.resolved_mode,
        rebuild_embeddings,
        len(raw),
    )
    return {
        "ok": True,
        "job_id": job.id,
        "file": path.name,
        "mode": mode,
        "resolved_mode": job.resolved_mode,
        "rebuild_embeddings": rebuild_embeddings,
        "status_url": f"/ingest/{job.id}",
        "tip": (
            "轮询 /ingest/{job_id}；完成后知识库自动重载。"
            "默认会重建向量索引；无密钥时 VLM 回退 heuristic。"
        ),
    }


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        return {"ok": False, "error": "unknown job_id"}
    if job.status == "done":
        rag.reload()
        log.info("ingest done job=%s stems=%s → rag.reload()", job_id, job.stems)
    elif job.status == "error":
        log.warning("ingest error job=%s error=%s", job_id, job.error)
    return {"ok": True, **job.to_dict()}


@app.get("/ingest")
def ingest_list() -> dict:
    return {"ok": True, "jobs": list_jobs()}
