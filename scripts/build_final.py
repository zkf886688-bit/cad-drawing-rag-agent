"""Combine vector, OCR and visual understanding into one record per drawing."""

from __future__ import annotations

from pathlib import Path

from _common import (
    FINAL_DIR,
    MERGED_DIR,
    VLM_DIR,
    dump_json,
    load_json,
    rel,
    run_batch,
    strip_suffix,
)
from steel_specs import extract_specs

# Same filename cues as scripts/vlm_understand.py. Retrieval aliases depend on
# these Chinese labels (门 / 视口 / 表格 / 角钢…); vision models often return
# vague types like 其他 that break question routing if used as-is.
DRAWING_KINDS = (
    ("door", "门/块构件示意"),
    ("wipeout", "门/块构件示意"),
    ("steel", "型钢截面"),
    ("table", "表格/块引用"),
    ("text", "文字样式测试图"),
    ("viewport", "视口/图层测试图"),
    ("engineering", "工程示意平面图"),
    ("minimal", "最简 DXF 样例"),
)


def classify_stem(stem: str) -> str:
    lowered = stem.lower()
    for keyword, kind in DRAWING_KINDS:
        if keyword in lowered:
            return kind
    return "CAD 样例图"


def spec_summary(merged: dict) -> dict:
    """Read the steel designations out of the drawing's own text.

    They are derived here, from the merged record, rather than taken from the
    VLM stage: stage 3 can run as a heuristic or as a vision model, and whether
    the index gets its designation chunks must not depend on which one ran or on
    what free-form JSON a model chose to return.
    """
    texts = [
        item.get("content") or ""
        for key in ("ocr_texts", "vector_texts")
        for item in merged.get(key) or []
    ]
    return extract_specs(texts)


def stabilize_drawing_type(stem: str, vlm_block: dict, specs: dict) -> dict:
    """Pin drawing_type for RAG; keep vision summary / spatial / OCR notes.

    Member-spec sheets use the Chinese member name (角钢截面图); other sheets
    use the filename rule. The model's free-form type is kept only as
    drawing_type_model for audit and must not enter retrieval aliases.
    """
    members = specs.get("member_types") or []
    if members:
        stable = f"{members[0]['name']}截面图"
    else:
        stable = classify_stem(stem)

    model_type = (vlm_block.get("drawing_type") or "").strip()
    out = {**vlm_block, "drawing_type": stable}
    if model_type and model_type != stable:
        out["drawing_type_model"] = model_type
    return out


def build_one(merged_path: Path, vlm_path: Path) -> dict:
    merged = load_json(merged_path)
    vlm = load_json(vlm_path)
    specs = spec_summary(merged)
    stem = strip_suffix(merged_path, "_merged.json")

    return {
        "file_name": merged.get("file_name"),
        "dxf_version": merged.get("dxf_version"),
        "header": merged.get("header", {}),
        "stats": merged.get("stats", {}),
        "layers": merged.get("layers", []),
        "blocks": merged.get("blocks", []),
        "block_references": merged.get("block_references", []),
        "dimensions": merged.get("dimensions", []),
        "tables": merged.get("tables", []),
        "layouts": merged.get("layouts", []),
        "vector_texts": merged.get("vector_texts", []),
        "ocr_texts": merged.get("ocr_texts", []),
        "entities": merged.get("entities", []),
        "steel_specs": specs,
        "vlm": stabilize_drawing_type(
            stem,
            {
                key: vlm.get(key)
                for key in (
                    "mode",
                    "model",
                    "drawing_type",
                    "summary",
                    "main_objects",
                    "spatial_relations",
                    "ocr_hard_notes",
                    "risks",
                )
                if vlm.get(key) is not None
            },
            specs,
        ),
        "sources": {
            **(merged.get("sources") or {}),
            "vlm_json": rel(vlm_path),
        },
    }


def main() -> None:
    files = sorted(MERGED_DIR.glob("*_merged.json"))

    def handle(merged_path: Path) -> str:
        stem = strip_suffix(merged_path, "_merged.json")
        vlm_path = VLM_DIR / f"{stem}_vlm.json"
        if not vlm_path.exists():
            raise FileNotFoundError(f"missing VLM json: {vlm_path.name}")

        data = build_one(merged_path, vlm_path)
        dump_json(FINAL_DIR / f"{stem}_final.json", data)
        vlm = data["vlm"]
        return (
            f"type={vlm.get('drawing_type')} | mode={vlm.get('mode')} | "
            f"specs={data['steel_specs'].get('designation_count', 0)}"
        )

    run_batch(
        files,
        handle,
        out_dir=FINAL_DIR,
        empty_hint="Run: py scripts/merge_ocr.py",
    )


if __name__ == "__main__":
    main()
