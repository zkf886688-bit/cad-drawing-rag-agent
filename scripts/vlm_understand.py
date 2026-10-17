"""Visual understanding of the drawings (stage 3 of the pipeline).

Modes:
  auto       default for upload/ingest: qwen → heuristic（无密钥时离线兜底）
  heuristic  offline only — derives description from vector + OCR, does not look
             at the image. Prefer auto/qwen when a key is available.
  qwen       通义千问 VL (DashScope)，需 DASHSCOPE_API_KEY / QWEN_API_KEY

  py scripts/vlm_understand.py
  py scripts/vlm_understand.py --mode auto
  py scripts/vlm_understand.py --mode qwen --only wipeout_door
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from _common import (
    IMAGE_DIR,
    MERGED_DIR,
    ROOT,
    VLM_DIR,
    dump_json,
    load_json,
    run_batch,
    strip_suffix,
)
from steel_specs import extract_specs

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.envfile import load_local_env

load_local_env()

QWEN_VL_MODEL = os.environ.get("CAD_QWEN_VL_MODEL", "qwen-vl-plus")
DASHSCOPE_CHAT_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)

# Free-tier keys often return HTTP 429 if eight images are fired back-to-back.
# Wait between drawings, and retry with backoff when the API says to slow down.
API_GAP_SECONDS = 12
API_MAX_RETRIES = 6

PROMPT = """你是 CAD 图纸理解助手。请观察这张工程图纸截图，输出严格 JSON（不要 markdown）：
{
  "drawing_type": "图纸类型，如门块/型钢截面/文字测试/表格/视口测试/其他",
  "summary": "一两句话描述整张图在表达什么",
  "main_objects": ["主要对象1", "主要对象2"],
  "spatial_relations": ["对象之间的空间/连接关系"],
  "ocr_hard_notes": ["OCR 可能搞不定、但从图能看出的信息"],
  "risks": ["可能误判或看不清的地方"]
}
只输出 JSON。
"""

# Keyword in the file name -> (drawing type, what vision adds over OCR).
DRAWING_KINDS = (
    ("door", "门/块构件示意", "门的开启方向、墙厚遮罩(Wipeout)的叠压关系"),
    ("wipeout", "门/块构件示意", "门的开启方向、墙厚遮罩(Wipeout)的叠压关系"),
    ("steel", "型钢截面", "截面外轮廓形状与肢长方向"),
    ("table", "表格/块引用", "表格的行列归属与合并单元格"),
    ("text", "文字样式测试图", "旋转/倾斜文字的基线与对齐方式"),
    ("viewport", "视口/图层测试图", "视口裁剪范围与被冻结的图层"),
    ("engineering", "工程示意平面图", "图框标题栏、墙线与构件尺寸标注关系"),
    ("minimal", "最简 DXF 样例", "几何过少，视觉信息有限"),
)

VALID_VLM_MODES = frozenset({"auto", "heuristic", "qwen"})


def _dashscope_key() -> str | None:
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")


def resolve_vlm_mode(requested: str | None = None) -> str:
    """Pick a real vision backend when keys exist; heuristic only as fallback."""
    mode = (requested or os.environ.get("CAD_VLM_MODE", "auto")).strip().lower()
    if mode not in VALID_VLM_MODES:
        raise ValueError(f"VLM mode must be one of {sorted(VALID_VLM_MODES)}")
    if mode == "auto":
        if _dashscope_key():
            return "qwen"
        return "heuristic"
    return mode


def classify(stem: str) -> tuple[str, str]:
    lowered = stem.lower()
    for keyword, kind, note in DRAWING_KINDS:
        if keyword in lowered:
            return kind, note
    return "CAD 样例图", "旋转文字、符号含义与区域功能划分"


def heuristic_vlm(merged: dict, stem: str) -> dict:
    layers = [item.get("name", "") for item in merged.get("layers", [])]
    blocks = [item.get("name", "") for item in merged.get("blocks", [])]
    references = merged.get("block_references", [])
    dimensions = merged.get("dimensions", [])
    entities = merged.get("entities", [])
    ocr_texts = [item.get("content", "") for item in merged.get("ocr_texts", [])]
    vector_texts = [item.get("content", "") for item in merged.get("vector_texts", [])]
    stats = merged.get("stats", {})
    type_counts = Counter(item.get("type", "?") for item in entities)

    drawing_type, ocr_hard_note = classify(stem)

    # The file name says only "Structural_Steel L 2D"; the designations say
    # which member this actually is, which is how a question will refer to it.
    # build_final derives the designations again from the merged record, so the
    # index does not depend on this stage having run in heuristic mode.
    specs = extract_specs(ocr_texts + vector_texts)
    member_names = [item["name"] for item in specs["member_types"]]
    if member_names:
        drawing_type = f"{member_names[0]}截面图"

    main_objects = [f"块:{name}" for name in blocks[:5]]
    main_objects += [f"{name} x{count}" for name, count in type_counts.most_common(5)]
    if specs["designations"]:
        main_objects.append(
            f"规格标注 {specs['designation_count']} 种，例如 "
            + "、".join(specs["designations"][:3])
        )
    elif ocr_texts:
        main_objects.append(f"OCR文字样例:{ocr_texts[0]}")

    spatial = []
    if any("wall" in layer.lower() for layer in layers):
        spatial.append("存在墙体类图层，构件可能与墙线关联")
    if references:
        placed = ", ".join(
            f"{item.get('block_name')}(含{item.get('content_count')}实体)"
            for item in references[:3]
        )
        spatial.append(f"块被实际放置在模型空间: {placed}")
    elif blocks:
        spatial.append("存在块定义但未在模型空间插入，属符号库性质")
    if dimensions:
        spatial.append(f"含 {len(dimensions)} 处尺寸标注，可用于回答尺寸类问题")
    if not spatial:
        spatial.append("以模型空间几何为主，未检测到块引用或墙体图层关系")

    linked = stats.get("linked_text_count")
    link_note = ""
    if linked is not None and ocr_texts:
        link_note = f"其中 {linked} 条 OCR 文字已关联到矢量实体。"

    spec_note = ""
    if specs["designations"]:
        types_text = "、".join(
            f"{item['name']}({item['prefix']} 前缀 {item['count']} 处)"
            for item in specs["member_types"]
        )
        spec_note = (
            f"识别到 {specs['designation_count']} 种钢材规格型号，"
            f"构件类型: {types_text}。"
        )

    summary = (
        f"{drawing_type}。图层 {len(layers)} 个，实体 {len(entities)} 个，"
        f"矢量文字 {len(vector_texts)} 条，OCR 文字 {len(ocr_texts)} 条。"
        f"{link_note}{spec_note}"
    )

    return {
        "file_name": merged.get("file_name", stem),
        "mode": "heuristic",
        "drawing_type": drawing_type,
        "summary": summary,
        "main_objects": main_objects[:8],
        "spatial_relations": spatial,
        "ocr_hard_notes": [ocr_hard_note],
        "risks": [
            "启发式推断，非真实视觉模型；接 qwen 后可替换为真实图像理解",
        ],
        "evidence": {
            "layers": layers,
            "blocks": blocks,
            "block_reference_count": len(references),
            "dimension_count": len(dimensions),
            "entity_type_counts": dict(type_counts),
            "ocr_preview": ocr_texts[:8],
            "vector_text_preview": vector_texts[:8],
        },
    }


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def qwen_vlm(image_path: Path, merged: dict, stem: str) -> dict:
    api_key = _dashscope_key()
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY / QWEN_API_KEY not set. "
            "Get a key at https://bailian.console.aliyun.com/"
        )

    context = {
        "layers": [item.get("name") for item in merged.get("layers", [])],
        "blocks": [item.get("name") for item in merged.get("blocks", [])],
        "ocr_preview": [
            item.get("content") for item in merged.get("ocr_texts", [])
        ][:15],
    }
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    body = {
        "model": QWEN_VL_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            PROMPT
                            + "\n已知结构化摘要(可参考):\n"
                            + json.dumps(context, ensure_ascii=False)
                        ),
                    },
                ],
            }
        ],
        "temperature": 0.2,
    }
    payload_bytes = json.dumps(body).encode("utf-8")
    started = time.perf_counter()
    payload = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        request = urllib.request.Request(
            DASHSCOPE_CHAT_URL,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429 and attempt < API_MAX_RETRIES:
                wait = min(60, API_GAP_SECONDS * attempt)
                print(
                    f"    rate-limited on {stem} "
                    f"(try {attempt}/{API_MAX_RETRIES}), wait {wait}s…"
                )
                if attempt == 1:
                    print(f"    detail: {detail[:240]}")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Qwen HTTP {exc.code}: {detail}") from exc

    if payload is None:
        raise RuntimeError(f"Qwen gave no response for {stem}")

    try:
        text = (payload["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Qwen unexpected response: {str(payload)[:200]}") from exc

    parsed = json.loads(_strip_json_fence(text))
    parsed["file_name"] = merged.get("file_name", stem)
    parsed["mode"] = "qwen"
    parsed["model"] = QWEN_VL_MODEL
    parsed["latency_seconds"] = round(time.perf_counter() - started, 2)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["auto", "heuristic", "qwen"],
        default="auto",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只处理这些图纸 stem，例如 --only wipeout_door viewport_frozen_layers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使已有同 mode 的产物也重新请求",
    )
    args = parser.parse_args()
    mode = resolve_vlm_mode(args.mode)

    files = sorted(MERGED_DIR.glob("*_merged.json"))
    if args.only:
        wanted = set(args.only)
        files = [
            path
            for path in files
            if strip_suffix(path, "_merged.json") in wanted
        ]
        missing = wanted - {strip_suffix(path, "_merged.json") for path in files}
        if missing:
            raise SystemExit(f"unknown stems: {sorted(missing)}")

    print(f"Mode: {mode}" + (f" (from {args.mode})" if args.mode == "auto" else ""))
    if mode == "qwen":
        print(
            f"Rate limit: {API_GAP_SECONDS}s between drawings, "
            f"up to {API_MAX_RETRIES} retries on HTTP 429"
        )

    def handle(merged_path: Path) -> str:
        stem = strip_suffix(merged_path, "_merged.json")
        out_path = VLM_DIR / f"{stem}_vlm.json"

        if (
            mode == "qwen"
            and not args.force
            and out_path.exists()
            and load_json(out_path).get("mode") == mode
        ):
            data = load_json(out_path)
            return (
                f"type={data.get('drawing_type')} | "
                f"skipped(already {mode})"
            )

        merged = load_json(merged_path)

        if mode == "qwen":
            image_path = IMAGE_DIR / f"{stem}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"missing image: {image_path.name}")
            data = qwen_vlm(image_path, merged, stem)
            dump_json(out_path, data)
            time.sleep(API_GAP_SECONDS)
            return (
                f"type={data.get('drawing_type')} | "
                f"{data.get('latency_seconds')}s"
            )

        data = heuristic_vlm(merged, stem)
        dump_json(out_path, data)
        return f"type={data.get('drawing_type')}"

    run_batch(
        files,
        handle,
        out_dir=VLM_DIR,
        empty_hint="Run: py scripts/merge_ocr.py",
    )


if __name__ == "__main__":
    main()
