"""Shared helpers for the dataset pipeline scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"

RAW_DIR = DATA / "raw"
JSON_DIR = DATA / "json"
IMAGE_DIR = DATA / "images"
OCR_DIR = DATA / "ocr"
MERGED_DIR = DATA / "merged"
VLM_DIR = DATA / "vlm"
FINAL_DIR = DATA / "final"
INDEX_DIR = DATA / "index"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dxf(path: Path):
    """Open a DXF with recover fallback for mildly corrupt files.

    Returns ``(doc, auditor_or_None, recovered: bool)``. Raises if both
    ``readfile`` and ``recover.readfile`` fail.
    """
    import ezdxf
    from ezdxf import recover

    try:
        return ezdxf.readfile(path), None, False
    except Exception as primary:
        try:
            doc, auditor = recover.readfile(str(path))
            return doc, auditor, True
        except Exception as secondary:
            raise RuntimeError(
                f"无法读取 DXF「{path.name}」: {secondary} "
                f"（严格读取失败: {primary}）"
            ) from secondary


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    """Project-relative POSIX path, used to record data provenance."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


def strip_suffix(path: Path, suffix: str) -> str:
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def resolve_ocr_python() -> str:
    """Python 3.13 executable for RapidOCR (not supported on 3.14 yet).

    Order: CAD_OCR_PYTHON → ``py -3.13`` → current interpreter if already 3.13.
    """
    override = (os.environ.get("CAD_OCR_PYTHON") or "").strip().strip('"')
    if override:
        path = Path(override)
        if path.is_file():
            return str(path)
        raise RuntimeError(f"CAD_OCR_PYTHON is set but not a file: {override}")

    try:
        result = subprocess.run(
            ["py", "-3.13", "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=str(ROOT),
        )
        exe = (result.stdout or "").strip()
        if result.returncode == 0 and exe:
            return exe
    except (OSError, subprocess.SubprocessError):
        pass

    if sys.version_info[:2] == (3, 13):
        return sys.executable

    raise RuntimeError(
        "OCR requires Python 3.13 (RapidOCR does not support 3.14). "
        "Install 3.13, or set CAD_OCR_PYTHON to that interpreter, e.g. "
        r"CAD_OCR_PYTHON=D:\python\python.exe"
    )


def run_batch(
    files: Sequence[Path],
    handler: Callable[[Path], str],
    *,
    out_dir: Path,
    empty_hint: str = "",
) -> None:
    """Apply handler to each file, reporting one status line per file.

    handler returns a short summary string, or raises to report a failure.
    """
    if not files:
        print(f"No input files found.{' ' + empty_hint if empty_hint else ''}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} input files\n")

    ok = 0
    for path in files:
        try:
            summary = handler(path)
            ok += 1
            print(f"[OK] {path.name} | {summary}")
        except Exception as exc:
            print(f"[FAIL] {path.name}: {exc}")

    print(f"\nDone: {ok}/{len(files)} succeeded")
    print(f"Output dir: {out_dir}")
