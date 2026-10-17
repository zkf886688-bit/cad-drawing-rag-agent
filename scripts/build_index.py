"""Turn the final records into retrievable text chunks."""

from __future__ import annotations

import json
import re

from _common import FINAL_DIR, INDEX_DIR, load_json

OUT_PATH = INDEX_DIR / "chunks.jsonl"

# Text chunks stay small so a match points at a specific part of the drawing
# instead of one giant blob per file.
TEXTS_PER_CHUNK = 12

# Small batches, because BM25 divides by document length: with 30 designations
# per chunk, a rare term such as "L4X4X1/2" occurring once was normalised down
# far enough to lose to a short unrelated summary chunk. Ten keeps each chunk
# near the average length so a single exact match still wins.
SPECS_PER_CHUNK = 10


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk(
    stem: str,
    file_name: str,
    kind: str,
    text: str,
    suffix: str = "",
    **extra: object,
) -> dict:
    return {
        "id": f"{stem}::{kind}{suffix}",
        "drawing": stem,
        "file_name": file_name,
        "kind": kind,
        "text": clean(text),
        **extra,
    }


def summary_chunk(doc: dict, stem: str, file_name: str) -> dict:
    vlm = doc.get("vlm") or {}
    parts = [
        f"图纸文件: {file_name}",
        f"类型: {vlm.get('drawing_type', '')}",
        f"摘要: {vlm.get('summary', '')}",
        "主要对象: " + "；".join(vlm.get("main_objects") or []),
        "空间关系: " + "；".join(vlm.get("spatial_relations") or []),
        "OCR难识别点: " + "；".join(vlm.get("ocr_hard_notes") or []),
    ]
    body = " | ".join(part for part in parts if part.strip(" :|"))
    # drawing_type is kept as its own field, not just inside the text, because
    # retrieval uses it as a deliberate Chinese alias for the drawing.
    return chunk(
        stem,
        file_name,
        "summary",
        body,
        drawing_type=vlm.get("drawing_type") or "",
    )


def vision_chunks(doc: dict, stem: str, file_name: str) -> list[dict]:
    """Dedicated chunks for VLM spatial / OCR-hard notes (stage-3 signal).

    Kept separate from the overview summary so questions about what the eye
    sees (叠压、开启方向、视口裁剪…) can hit without relying on summary BM25.
    """
    vlm = doc.get("vlm") or {}
    mode = vlm.get("mode") or "heuristic"
    out: list[dict] = []

    spatial = [s for s in (vlm.get("spatial_relations") or []) if str(s).strip()]
    if spatial:
        out.append(
            chunk(
                stem,
                file_name,
                "vision_spatial",
                f"{file_name} 视觉空间关系(mode={mode}): " + "；".join(spatial),
            )
        )

    notes = [s for s in (vlm.get("ocr_hard_notes") or []) if str(s).strip()]
    if notes:
        out.append(
            chunk(
                stem,
                file_name,
                "vision_ocr_notes",
                f"{file_name} OCR难识别但图上可见(mode={mode}): " + "；".join(notes),
            )
        )

    risks = [s for s in (vlm.get("risks") or []) if str(s).strip()]
    if risks and mode == "qwen":
        out.append(
            chunk(
                stem,
                file_name,
                "vision_risks",
                f"{file_name} 视觉理解风险点: " + "；".join(risks),
            )
        )
    return out


def text_lines(doc: dict) -> tuple[list[str], int]:
    """One line per label, noting the source and any vector/OCR link.

    Single-glyph OCR readings are left out: they are section outlines misread as
    characters, and indexing them lets a question about "L" match a pipe
    drawing. They stay in the merged and final JSON, and the count of what was
    held back is reported in the stats chunk.
    """
    lines = []
    for item in doc.get("vector_texts") or []:
        content = (item.get("content") or "").strip()
        if content:
            lines.append(f"[矢量|{item.get('layer', '')}] {content}")

    skipped = 0
    for item in doc.get("ocr_texts") or []:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if item.get("text_class") == "single_char":
            skipped += 1
            continue
        confidence = item.get("confidence")
        prefix = f"[OCR:{confidence}]" if confidence is not None else "[OCR]"
        link = item.get("link")
        if link:
            prefix += f"[关联矢量:{link.get('vector_handle')}]"
        lines.append(f"{prefix} {content}")

    return lines, skipped


def make_chunks(doc: dict, stem: str) -> list[dict]:
    file_name = doc.get("file_name", stem)
    chunks = [summary_chunk(doc, stem, file_name)]
    chunks.extend(vision_chunks(doc, stem, file_name))

    header = doc.get("header") or {}
    if header:
        extents = header.get("extents") or {}
        parts = [f"图纸单位: {header.get('units', '未指定')}"]
        if header.get("insunits_code") is not None:
            parts.append(f"$INSUNITS={header['insunits_code']}")
        if header.get("measurement"):
            parts.append(f"计量制: {header['measurement']}")
        if extents.get("min") and extents.get("max"):
            low = [round(v, 2) for v in extents["min"][:2]]
            high = [round(v, 2) for v in extents["max"][:2]]
            parts.append(f"图纸范围 X/Y: {low} 到 {high}")
        chunks.append(
            chunk(stem, file_name, "header", f"{file_name} " + "，".join(parts))
        )

    layers = doc.get("layers") or []
    if layers:
        # Layer state is spelled out because "which drawing has a frozen layer"
        # is a question about these flags, not about colours.
        def layer_text(item: dict) -> str:
            flags = [
                name
                for key, name in (("frozen", "冻结"), ("off", "关闭"), ("locked", "锁定"))
                if item.get(key)
            ]
            state = f"[{'/'.join(flags)}]" if flags else ""
            return f"{item.get('name')}(颜色{item.get('color')}){state}"

        body = "图层列表: " + ", ".join(layer_text(item) for item in layers)
        frozen = [item.get("name") for item in layers if item.get("frozen")]
        if frozen:
            body += f"；被冻结的图层: {', '.join(frozen)}"
        else:
            body += "；无被冻结的图层"
        chunks.append(chunk(stem, file_name, "layers", f"{file_name} {body}"))

    blocks = doc.get("blocks") or []
    if blocks:
        body = "块定义: " + ", ".join(
            f"{item.get('name')}(实体数{item.get('entity_count')})" for item in blocks
        )
        chunks.append(chunk(stem, file_name, "blocks", f"{file_name} {body}"))

    references = doc.get("block_references") or []
    if references:
        body = "块引用: " + ", ".join(
            f"{item.get('block_name')}@{[round(v, 1) for v in item.get('insert', [])[:2]]}"
            f"(含{item.get('content_count')}实体)"
            for item in references[:20]
        )
        chunks.append(chunk(stem, file_name, "block_refs", f"{file_name} {body}"))

    for index, table in enumerate(doc.get("tables") or []):
        rows = table.get("rows") or []
        parent_id = f"{stem}::tables::{index}"
        body = (
            f"表格: 检测到 {table.get('rows_detected')} 行、"
            f"{table.get('cell_count')} 个非空单元格"
            f"（文件声明 n_rows={table.get('declared_rows')}，"
            f"行数由单元格坐标推断）。内容按行: "
            + " | ".join(" / ".join(row) for row in rows[:20])
        )
        parent = chunk(
            stem, file_name, "tables", f"{file_name} {body}", suffix=f"::{index}"
        )
        parent["id"] = parent_id
        chunks.append(parent)
        # Child row chunks for precise cell/title lookups (hierarchical RAG).
        for row_index, row in enumerate(rows[:40]):
            cells = [str(cell).strip() for cell in row if str(cell).strip()]
            if not cells:
                continue
            row_text = f"{file_name} 表格第{row_index + 1}行: " + " / ".join(cells)
            child = chunk(
                stem,
                file_name,
                "table_row",
                row_text,
                suffix=f"::{index}::r{row_index}",
            )
            child["parent_id"] = parent_id
            child["row_index"] = row_index
            chunks.append(child)

    layouts = doc.get("layouts") or []
    for index, layout in enumerate(layouts):
        parts = [
            f"布局(图纸空间) {layout.get('name')}: "
            f"{layout.get('entity_count')} 个实体，"
            f"{len(layout.get('viewports') or [])} 个视口"
        ]
        for viewport in layout.get("viewports") or []:
            frozen = viewport.get("frozen_layers") or []
            note = f"冻结图层 {', '.join(frozen)}" if frozen else "无图层冻结覆盖"
            parts.append(
                f"视口{viewport.get('id')}("
                f"{round(viewport.get('width', 0), 1)}x"
                f"{round(viewport.get('height', 0), 1)}, {note})"
            )
        texts = [item.get("content", "") for item in layout.get("texts") or []]
        if texts:
            parts.append("布局文字: " + "；".join(texts))
        chunks.append(
            chunk(
                stem,
                file_name,
                "layouts",
                f"{file_name} " + "，".join(parts),
                suffix=f"::{index}",
            )
        )

    dimensions = doc.get("dimensions") or []
    if dimensions:
        labels = []
        for item in dimensions[:20]:
            text = str(item.get("text") or "").strip()
            # AutoCAD uses <> as "display measured value"; treat as empty.
            if not text or text in {"<>", " "}:
                text = item.get("measurement")
            labels.append(f"{text}@图层{item.get('layer')}")
        body = "尺寸标注(DIMENSION): " + ", ".join(labels)
        chunks.append(chunk(stem, file_name, "dimensions", f"{file_name} {body}"))

    # A dedicated chunk for steel designations, worded with the terms a question
    # would use ("规格", "型号"), since the raw OCR chunk only says "文字内容".
    specs = doc.get("steel_specs") or {}
    designations = specs.get("designations") or []
    if designations:
        member_types = specs.get("member_types") or []
        type_text = "、".join(
            f"{item['name']}({item['prefix']}系列, {item['note']})"
            for item in member_types
        )
        for batch_index in range(0, len(designations), SPECS_PER_CHUNK):
            batch = designations[batch_index : batch_index + SPECS_PER_CHUNK]
            part = batch_index // SPECS_PER_CHUNK
            head = (
                f"{file_name} 钢材规格型号列表"
                + (f"(构件类型: {type_text})" if part == 0 else f"(续{part})")
            )
            chunks.append(
                chunk(
                    stem,
                    file_name,
                    "steel_specs",
                    f"{head}: " + "、".join(batch),
                    suffix=f"::{part}",
                )
            )

        # The batches above answer "what is X"; this one answers "list them
        # all", which no single batch can. Both are kept because a short chunk
        # wins a lookup and only a complete one can be exhaustive.
        if len(designations) > SPECS_PER_CHUNK:
            chunks.append(
                chunk(
                    stem,
                    file_name,
                    "steel_specs_all",
                    f"{file_name} 全部规格型号共 {len(designations)} 个"
                    f"(构件类型: {type_text}): " + "、".join(designations),
                )
            )

    stats = doc.get("stats") or {}
    lines, skipped_noise = text_lines(doc)
    stat_parts = [
        f"图层{stats.get('layer_count')}",
        f"实体{stats.get('entity_count')}",
        f"矢量文字{stats.get('vector_text_count')}",
        f"OCR文字{stats.get('ocr_text_count')}",
        f"已关联文字{stats.get('linked_text_count')}",
        f"尺寸标注{stats.get('dimension_count')}",
        f"单位{stats.get('units') or '未指定'}",
    ]
    if stats.get("table_count"):
        stat_parts.append(
            f"表格{stats['table_count']}(单元格{stats.get('table_cell_count')})"
        )
    if stats.get("layout_count"):
        stat_parts.append(
            f"布局{stats['layout_count']}(实体{stats.get('paperspace_entity_count')}"
            f"、视口{stats.get('viewport_count')})"
        )
    if skipped_noise:
        stat_parts.append(f"OCR单字疑似噪声{skipped_noise}条(保留在数据里,未进检索)")
    chunks.append(
        chunk(stem, file_name, "stats", f"{file_name} 统计: " + " ".join(stat_parts))
    )

    for batch_index in range(0, len(lines), TEXTS_PER_CHUNK):
        batch = lines[batch_index : batch_index + TEXTS_PER_CHUNK]
        chunks.append(
            chunk(
                stem,
                file_name,
                "texts",
                f"{file_name} 文字内容: " + " || ".join(batch),
                suffix=f"::{batch_index // TEXTS_PER_CHUNK}",
            )
        )

    return chunks


def corpus_chunks(profiles: list[dict]) -> list[dict]:
    """Facts about the collection as a whole.

    Every other chunk describes one drawing, so questions like "how many
    drawings are there" or "which one has the most entities" had nothing to
    retrieve and were answered from whichever single drawing happened to rank
    first. These chunks carry no drawing id, which also keeps them out of
    results that are filtered to one drawing.
    """
    if not profiles:
        return []

    def make(kind: str, text: str) -> dict:
        return {
            "id": f"corpus::{kind}",
            "drawing": None,
            "file_name": "(全部图纸)",
            "kind": "corpus",
            "text": clean(text),
        }

    total_entities = sum(item["entity_count"] for item in profiles)
    total_texts = sum(item["text_count"] for item in profiles)
    with_dims = [item for item in profiles if item["dimension_count"]]
    with_tables = [item for item in profiles if item["table_count"]]
    with_layouts = [item for item in profiles if item["layout_count"]]
    frozen = [item for item in profiles if item["frozen_layers"]]

    unit_groups: dict[str, list[str]] = {}
    for item in profiles:
        unit_groups.setdefault(item["units"] or "未指定", []).append(item["file_name"])

    overview = (
        f"知识库总览: 共 {len(profiles)} 张图纸，"
        f"合计 {total_entities} 个实体、{total_texts} 条文字。"
        "图纸清单: "
        + "；".join(
            f"{item['file_name']}({item['drawing_type'] or '类型未定'}, "
            f"实体{item['entity_count']}, 图层{item['layer_count']}, "
            f"文字{item['text_count']}, 单位{item['units'] or '未指定'})"
            for item in profiles
        )
    )

    ranked = sorted(profiles, key=lambda item: item["entity_count"], reverse=True)
    ranking = (
        "按实体数量排序（多到少）: "
        + "；".join(
            f"{position}. {item['file_name']} {item['entity_count']}个实体"
            for position, item in enumerate(ranked, start=1)
        )
        + f"。实体最多的图纸是 {ranked[0]['file_name']}"
        f"({ranked[0]['entity_count']}个)，最少的是 {ranked[-1]['file_name']}"
        f"({ranked[-1]['entity_count']}个)。"
    )

    coverage = (
        "跨图纸情况: 含尺寸标注(DIMENSION)的图纸 "
        + (
            "、".join(item["file_name"] for item in with_dims)
            if with_dims
            else f"无（{len(profiles)} 个样本都没有 DIMENSION 实体）"
        )
        + "；含表格(ACAD_TABLE)的图纸 "
        + ("、".join(item["file_name"] for item in with_tables) if with_tables else "无")
        + "；含布局(图纸空间)的图纸 "
        + ("、".join(item["file_name"] for item in with_layouts) if with_layouts else "无")
        + "；含被冻结图层的图纸 "
        + (
            "、".join(
                f"{item['file_name']}({', '.join(item['frozen_layers'])})"
                for item in frozen
            )
            if frozen
            else "无"
        )
        + "。单位分布: "
        + "；".join(
            f"{unit}: {', '.join(names)}" for unit, names in unit_groups.items()
        )
    )

    return [
        make("overview", overview),
        make("ranking", ranking),
        make("coverage", coverage),
    ]


def profile(doc: dict, stem: str) -> dict:
    stats = doc.get("stats") or {}
    return {
        "drawing": stem,
        "file_name": doc.get("file_name", stem),
        "drawing_type": (doc.get("vlm") or {}).get("drawing_type", ""),
        "entity_count": stats.get("entity_count") or 0,
        "layer_count": stats.get("layer_count") or 0,
        "text_count": (stats.get("vector_text_count") or 0)
        + (stats.get("ocr_text_count") or 0),
        "dimension_count": stats.get("dimension_count") or 0,
        "table_count": stats.get("table_count") or 0,
        "layout_count": stats.get("layout_count") or 0,
        "frozen_layers": stats.get("frozen_layers") or [],
        "units": stats.get("units"),
    }


def main() -> None:
    files = sorted(FINAL_DIR.glob("*_final.json"))
    if not files:
        print(f"No final json found. Run: py scripts/build_final.py")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    profiles: list[dict] = []

    for path in files:
        stem = path.name.replace("_final.json", "")
        doc = load_json(path)
        chunks = make_chunks(doc, stem)
        all_chunks.extend(chunks)
        profiles.append(profile(doc, stem))
        print(f"[OK] {stem}: {len(chunks)} chunks")

    corpus = corpus_chunks(profiles)
    all_chunks.extend(corpus)
    print(f"[OK] 语料级: {len(corpus)} chunks")

    try:
        from build_standards import ensure_standards_corpus, standard_chunks

        ensure_standards_corpus()
        standards = standard_chunks()
        all_chunks.extend(standards)
        print(f"[OK] 工程标准: {len(standards)} chunks")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] standards skipped: {exc}")

    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for item in all_chunks:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Index file: {OUT_PATH}")
    print(
        "Next: py scripts/build_graph.py"
        " && (optional) py scripts/build_embeddings.py && py scripts/eval_embeddings.py"
    )


if __name__ == "__main__":
    main()
