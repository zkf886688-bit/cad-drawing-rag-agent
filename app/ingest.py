"""Ingest a new DWG/DXF into the knowledge base (upload pipeline).

Runs the same stages as the offline scripts, scoped to one or more stems,
then rebuilds the retrieval index, knowledge graph, and (by default) dense
embeddings so BM25 / dense / hybrid stay in sync.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Background ingest renders with matplotlib; force non-GUI backend early.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from _common import (  # noqa: E402
    DATA,
    FINAL_DIR,
    IMAGE_DIR,
    JSON_DIR,
    MERGED_DIR,
    OCR_DIR,
    RAW_DIR,
    VLM_DIR,
    dump_json,
    load_json,
    resolve_ocr_python,
)
from app.envfile import load_local_env  # noqa: E402
from app.logutil import setup_logging  # noqa: E402
from app import store as db  # noqa: E402
from vlm_understand import resolve_vlm_mode  # noqa: E402

load_local_env()
log = setup_logging("cad_agent.ingest")

__all__ = [
    "IngestJob",
    "get_job",
    "list_jobs",
    "rebuild_search_assets",
    "resolve_ocr_python",
    "resolve_vlm_mode",
    "run_ingest",
    "save_upload",
    "start_ingest_async",
]

DWG_DIR = DATA / "dwg"
INBOX_DIR = DATA / "inbox"
UPLOAD_MAX_BYTES = 80 * 1024 * 1024

_jobs_lock = threading.Lock()
_jobs: dict[str, "IngestJob"] = {}

# Live workers that crash (e.g. matplotlib/Tk) leave status=running forever.
STALE_STEP_SEC = 12 * 60
MAX_JOB_SEC = 30 * 60


@dataclass
class IngestJob:
    id: str
    status: str = "queued"  # queued | running | done | error
    files: list[str] = field(default_factory=list)
    stems: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    mode: str = "auto"  # requested VLM mode (auto|heuristic|qwen)
    resolved_mode: str | None = None  # concrete backend after auto resolve
    rebuild_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _maybe_fail_stale(job: IngestJob) -> IngestJob:
    """Mark abandoned / hung ingest jobs as error so the UI can stop polling."""
    if job.status not in {"queued", "running"}:
        return job
    now = time.time()
    age = now - float(job.created_at or now)
    last_at = float(job.created_at or now)
    if job.steps:
        last_at = float(job.created_at or now) + float(job.steps[-1].get("at") or 0)
    idle = now - last_at
    if age <= MAX_JOB_SEC and idle <= STALE_STEP_SEC:
        return job
    job.status = "error"
    job.error = (
        f"入库超时或中断（已运行 {int(age)}s，连续 {int(idle)}s 无新步骤）。"
        "服务可能曾崩溃或重启。请重新上传该图纸。"
    )
    job.finished_at = now
    _persist_job(job)
    log.warning("ingest stale→error job=%s age=%.0fs idle=%.0fs", job.id, age, idle)
    return job


def _persist_job(job: IngestJob) -> None:
    db.save_job(job.to_dict())


def get_job(job_id: str) -> IngestJob | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return _maybe_fail_stale(job)
    data = db.load_job(job_id)
    if data is None:
        return None
    job = IngestJob(**{k: data[k] for k in IngestJob.__dataclass_fields__ if k in data})
    # After process restart, DB may still say running but no worker exists.
    if job.status in {"queued", "running"}:
        job.status = "error"
        job.error = "入库进程已中断（服务重启或后台线程崩溃）。请重新上传该图纸。"
        job.finished_at = time.time()
        _persist_job(job)
        log.warning("ingest orphan→error job=%s", job_id)
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def list_jobs(limit: int = 20) -> list[dict]:
    return db.list_jobs(limit=limit)


def _step(job: IngestJob, name: str, detail: str = "", ok: bool = True) -> None:
    job.steps.append(
        {
            "name": name,
            "ok": ok,
            "detail": detail,
            "at": round(time.time() - job.created_at, 2),
        }
    )
    _persist_job(job)
    log.info(
        "ingest job=%s step=%s ok=%s detail=%s t+%.1fs",
        job.id,
        name,
        ok,
        (detail or "")[:160],
        time.time() - job.created_at,
    )

def _run_py(args: list[str], *, python: str | None = None) -> subprocess.CompletedProcess:
    exe = python or sys.executable
    return subprocess.run(
        [exe, *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in stem).strip()
    return cleaned or f"upload_{uuid.uuid4().hex[:8]}"


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _find_duplicate_upload(data: bytes, suffix: str) -> Path | None:
    """Return an existing dwg/raw file with the same bytes, if any."""
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    candidates: list[Path] = []
    if suffix == ".dwg":
        candidates.extend(sorted(DWG_DIR.glob("*.dwg")))
    else:
        candidates.extend(sorted(RAW_DIR.glob("*.dxf")))
    # Also scan inbox leftovers from interrupted uploads.
    if INBOX_DIR.is_dir():
        candidates.extend(sorted(INBOX_DIR.glob(f"*{suffix}")))
    for path in candidates:
        try:
            if path.stat().st_size != len(data):
                continue
            if _file_sha256(path) == digest:
                return path
        except OSError:
            continue
    return None


def save_upload(filename: str, data: bytes) -> Path:
    if not data:
        raise ValueError("empty upload")
    if len(data) > UPLOAD_MAX_BYTES:
        raise ValueError(f"file too large (max {UPLOAD_MAX_BYTES // (1024 * 1024)} MB)")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".dwg", ".dxf"}:
        raise ValueError("only .dwg / .dxf are accepted")
    # Cheap magic check: DWG starts with AC10xx; reject obvious mislabeled files.
    if suffix == ".dxf" and data.startswith(b"AC10"):
        raise ValueError("file looks like DWG but extension is .dxf")
    if suffix == ".dwg" and not data.startswith(b"AC10"):
        raise ValueError("file does not look like a DWG (missing AC10… magic)")

    dup = _find_duplicate_upload(data, suffix)
    if dup is not None:
        raise ValueError(
            f"相同内容的图纸已存在：{dup.name}（内容查重）。"
            "无需重复上传，直接提问该图即可。"
        )

    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(filename)
    dest = INBOX_DIR / f"{stem}{suffix}"
    # Avoid clobbering an existing drawing with the same stem.
    if dest.exists() or (RAW_DIR / f"{stem}.dxf").exists() or (DWG_DIR / f"{stem}.dwg").exists():
        stem = f"{stem}_{uuid.uuid4().hex[:6]}"
        dest = INBOX_DIR / f"{stem}{suffix}"
    dest.write_bytes(data)
    return dest


def _place_into_pipeline(path: Path) -> str:
    """Copy inbox file into dwg/ or raw/ and return the drawing stem."""
    suffix = path.suffix.lower()
    stem = path.stem
    src = path.resolve()
    if suffix == ".dwg":
        DWG_DIR.mkdir(parents=True, exist_ok=True)
        target = DWG_DIR / f"{stem}.dwg"
        if src != target.resolve():
            shutil.copy2(path, target)
        result = _run_py(
            ["scripts/convert_dwg.py", "--force", "--only", stem]
        )
        dxf = RAW_DIR / f"{stem}.dxf"
        if not dxf.exists():
            detail = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
            raise RuntimeError(
                f"DWG「{stem}」已保存，但未能转换成 DXF。"
                "请安装 ODA File Converter，或改上传 .dxf。"
                + (f"\n转换输出: {detail[-500:]}" if detail else "")
            )
        return stem
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / f"{stem}.dxf"
    if src != target.resolve():
        shutil.copy2(path, target)
    elif not target.exists():
        raise FileNotFoundError(f"missing DXF: {target}")
    return stem


def _parse_one(stem: str) -> None:
    from parse_dxf import parse_dxf

    path = RAW_DIR / f"{stem}.dxf"
    if not path.exists():
        raise FileNotFoundError(f"missing DXF: {path.name}")
    try:
        dump_json(JSON_DIR / f"{stem}.json", parse_dxf(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"解析 DXF「{stem}」失败: {exc}") from exc


def _render_one(stem: str) -> None:
    from render_dxf import render_dxf

    path = RAW_DIR / f"{stem}.dxf"
    try:
        render_dxf(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"渲染 DXF「{stem}」失败: {exc}") from exc

def _ocr_one(stem: str) -> None:
    """Always run OCR under Python 3.13 (RapidOCR); fall back to empty record."""
    png = IMAGE_DIR / f"{stem}.png"
    if not png.exists():
        raise FileNotFoundError(f"missing render: {png.name}")

    out = OCR_DIR / f"{stem}_ocr.json"
    note = ""
    try:
        py313 = resolve_ocr_python()
        result = _run_py(["scripts/ocr_images.py", "--only", stem], python=py313)
        if result.returncode == 0 and out.exists():
            return
        note = (
            f"OCR subprocess failed (exit {result.returncode}). "
            f"stdout={(result.stdout or '')[-300:]} "
            f"stderr={(result.stderr or '')[-300:]}"
        )
    except Exception as exc:  # noqa: BLE001
        note = f"OCR skipped: {exc}"

    from _common import rel

    dump_json(
        out,
        {
            "file_name": png.name,
            "source_image": rel(png),
            "image_size_px": [0, 0],
            "tile_count": 0,
            "whole_pass_texts": 0,
            "text_count": 0,
            "texts": [],
            "note": note or "OCR skipped",
        },
    )


def _merge_one(stem: str) -> None:
    from merge_ocr import merge_one

    ocr_path = OCR_DIR / f"{stem}_ocr.json"
    merged = merge_one(stem, ocr_path)
    dump_json(MERGED_DIR / f"{stem}_merged.json", merged)


def _vlm_one(stem: str, mode: str) -> None:
    from vlm_understand import heuristic_vlm, qwen_vlm

    merged = load_json(MERGED_DIR / f"{stem}_merged.json")
    if mode == "qwen":
        data = qwen_vlm(IMAGE_DIR / f"{stem}.png", merged, stem)
    else:
        data = heuristic_vlm(merged, stem)
    dump_json(VLM_DIR / f"{stem}_vlm.json", data)


def _final_one(stem: str) -> None:
    from build_final import build_one

    data = build_one(
        MERGED_DIR / f"{stem}_merged.json",
        VLM_DIR / f"{stem}_vlm.json",
    )
    dump_json(FINAL_DIR / f"{stem}_final.json", data)


def rebuild_search_assets(*, rebuild_embeddings: bool = True) -> tuple[bool, str]:
    """Rebuild chunks, standards, graph; by default also dense embeddings.

    Returns (embeddings_ok, detail). Chunks/graph always rebuild; embeddings
    failure is reported so callers can surface it without discarding the ingest.
    """
    import importlib.util

    def load_script(name: str):
        path = SCRIPTS / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    load_script("build_standards").ensure_standards_corpus()
    load_script("build_index").main()
    load_script("build_graph").build_and_save()

    if not rebuild_embeddings:
        return True, "embeddings skipped (rebuild_embeddings=false)"

    result = _run_py(["scripts/build_embeddings.py"])
    if result.returncode != 0:
        detail = (
            "embeddings rebuild failed: "
            + ((result.stderr or result.stdout or "")[-500:])
        )
        return False, detail
    return True, "chunks + graph + embeddings rebuilt"


def run_ingest(
    paths: list[Path],
    *,
    mode: str = "auto",
    rebuild_embeddings: bool = True,
    job: IngestJob | None = None,
) -> IngestJob:
    resolved = resolve_vlm_mode(mode)
    job = job or IngestJob(
        id=uuid.uuid4().hex[:12],
        files=[p.name for p in paths],
        mode=mode,
        resolved_mode=resolved,
        rebuild_embeddings=rebuild_embeddings,
    )
    job.mode = mode
    job.resolved_mode = resolved
    job.rebuild_embeddings = rebuild_embeddings
    job.status = "running"
    with _jobs_lock:
        _jobs[job.id] = job
    _persist_job(job)

    try:
        stems: list[str] = []
        for path in paths:
            stem = _place_into_pipeline(path)
            stems.append(stem)
            _step(job, "place", f"{path.name} → stem={stem}")

        job.stems = stems
        for stem in stems:
            _parse_one(stem)
            _step(job, "parse", stem)
            _render_one(stem)
            _step(job, "render", stem)
            _ocr_one(stem)
            ocr_note = ""
            ocr_path = OCR_DIR / f"{stem}_ocr.json"
            if ocr_path.exists():
                ocr_note = (load_json(ocr_path).get("note") or "").strip()
            _step(
                job,
                "ocr",
                f"{stem} via py3.13" + (f" ({ocr_note[:120]})" if ocr_note else ""),
                ok=not bool(ocr_note),
            )
            _merge_one(stem)
            _step(job, "merge", stem)
            try:
                _vlm_one(stem, resolved)
                _step(job, "vlm", f"{stem} mode={resolved}")
            except Exception as exc:  # noqa: BLE001
                _vlm_one(stem, "heuristic")
                _step(
                    job,
                    "vlm",
                    f"{stem} fallback heuristic ({exc})",
                    ok=False,
                )
            _final_one(stem)
            _step(job, "final", stem)

        emb_ok, emb_detail = rebuild_search_assets(
            rebuild_embeddings=rebuild_embeddings
        )
        _step(job, "index", emb_detail, ok=emb_ok)
        if rebuild_embeddings and not emb_ok:
            # Keep the drawing searchable via BM25; flag the job as error so UI
            # does not claim dense/hybrid are ready.
            job.status = "error"
            job.error = emb_detail
            job.finished_at = time.time()
            _persist_job(job)
            return job

        job.status = "done"
        job.finished_at = time.time()
        _persist_job(job)
        log.info(
            "ingest finished job=%s stems=%s elapsed=%.1fs",
            job.id,
            job.stems,
            job.finished_at - job.created_at,
        )
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)
        job.finished_at = time.time()
        _step(job, "error", str(exc), ok=False)
        _persist_job(job)
        log.exception("ingest failed job=%s: %s", job.id, exc)
    return job


def start_ingest_async(
    paths: list[Path],
    *,
    mode: str = "auto",
    rebuild_embeddings: bool = True,
) -> IngestJob:
    resolved = resolve_vlm_mode(mode)
    job = IngestJob(
        id=uuid.uuid4().hex[:12],
        files=[p.name for p in paths],
        mode=mode,
        resolved_mode=resolved,
        rebuild_embeddings=rebuild_embeddings,
    )
    with _jobs_lock:
        _jobs[job.id] = job
    _persist_job(job)

    def worker() -> None:
        os.environ["MPLBACKEND"] = "Agg"
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        run_ingest(
            paths,
            mode=mode,
            rebuild_embeddings=rebuild_embeddings,
            job=job,
        )

    threading.Thread(target=worker, daemon=True).start()
    return job
