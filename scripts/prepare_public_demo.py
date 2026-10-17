"""Build a synthetic, vector-only demo in a fresh public checkout.

No private drawings, OCR outputs, API calls or downloaded models are used.
Refuses nonempty datasets to avoid mixing the demo with existing local data.
"""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / "datasets"
    if data.exists() and any(data.rglob("*")):
        raise SystemExit("datasets is not empty. Use a fresh checkout for this demo.")
    env = dict(os.environ, CAD_GENERATE="extractive", CAD_VLM_MODE="heuristic",
               CAD_RAG_BACKEND="bm25", PYTHONIOENCODING="utf-8")

    def run(script, *args):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                       cwd=ROOT, env=env, check=True)

    run("make_demo_engineering_dxf.py")
    run("parse_dxf.py")
    run("render_dxf.py")
    ocr = data / "ocr"
    ocr.mkdir(parents=True, exist_ok=True)
    (ocr / "demo_engineering_sheet_ocr.json").write_text(json.dumps({
        "texts": [], "mode": "skipped-vector-only-demo",
        "note": "OCR was not run; only native DXF text is used in this demo."
    }), encoding="utf-8")
    run("merge_ocr.py")
    run("vlm_understand.py", "--mode", "heuristic")
    run("build_final.py")
    run("build_standards.py")
    run("build_index.py")
    run("build_graph.py")
    run("validate_json.py")
    print("Synthetic demo ready. Start: python -m uvicorn app.main:app --port 8000")


if __name__ == "__main__":
    main()
