"""Link OCR results to vector entities (stage 2 of the pipeline).

OCR gives pixel boxes; stage 1 gives drawing coordinates. Using the transform
recorded at render time, each OCR box is projected back into drawing space and
matched against nearby vector text, so the two sources become one linked record
rather than two parallel lists.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from _common import (
    IMAGE_DIR,
    JSON_DIR,
    MERGED_DIR,
    OCR_DIR,
    dump_json,
    load_json,
    rel,
    run_batch,
    strip_suffix,
)
from steel_specs import match_spec

# A vector text anchor sits at the entity's corner while an OCR box centre sits
# in the middle of the rendered glyphs, and a multi-line MTEXT is split into
# several OCR lines. Distance alone therefore cannot decide a match, so spatial
# proximity is combined with how well the strings agree.
SPATIAL_ONLY_RATIO = 0.02
TEXT_MATCH_RATIO = 0.30
MIN_TEXT_AGREEMENT = 0.5

# A one- or two-character reading is contained in almost any sentence, so
# containment on its own would link OCR noise -- the 'O' read off a pipe
# section, the '2' read off a table -- to whatever text happens to fall inside
# the generous text-match radius. Require the shared part to be long enough to
# identify the text by itself.
MIN_CONTAINMENT_CHARS = 4

# A single glyph carries no engineering meaning on these sheets and is the
# signature of a profile outline being read as a character: the channel section
# reads as 'L', the round pipe as 'O'. Table row numbers are also single
# characters, but the table's real content now comes from the DXF itself, so
# nothing is lost by treating them the same way.
NOISE_MAX_LEN = 1


def classify_text(content: str) -> str:
    """Label an OCR reading so downstream stages can decide what to trust."""
    text = content.strip()
    if not text:
        return "empty"
    if match_spec(text) is not None:
        return "spec"
    if len(text) <= NOISE_MAX_LEN:
        return "single_char"
    return "other"


class PixelToDrawing:
    """Linear map from image pixels to drawing coordinates."""

    def __init__(self, transform: dict) -> None:
        width_px, height_px = transform["image_size_px"]
        extent = transform["drawing_extent"]
        self.width_px = width_px
        self.height_px = height_px
        self.x_min = extent["x_min"]
        self.y_min = extent["y_min"]
        self.x_span = extent["x_max"] - extent["x_min"]
        self.y_span = extent["y_max"] - extent["y_min"]

    @property
    def diagonal(self) -> float:
        return math.hypot(self.x_span, self.y_span)

    def point(self, px: float, py: float) -> list[float]:
        x = self.x_min + (px / self.width_px) * self.x_span
        # Pixel y axis points down, drawing y axis points up.
        y = self.y_min + (1 - py / self.height_px) * self.y_span
        return [x, y]

    def box_centre(self, bbox: list[list[float]]) -> list[float]:
        px = sum(p[0] for p in bbox) / len(bbox)
        py = sum(p[1] for p in bbox) / len(bbox)
        return self.point(px, py)


def normalise(text: str) -> str:
    return "".join(text.split()).lower()


def match_score(ocr_text: str, vector_text: str) -> float:
    """Crude text agreement in [0, 1]; OCR often mangles a few characters."""
    a, b = normalise(ocr_text), normalise(vector_text)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # A multi-line MTEXT comes back as several OCR lines, so one string being
    # part of the other is real evidence -- but only once the shared part is
    # substantial. Shorter overlaps fall through to the character count below,
    # which scores them on their own merits.
    if (a in b or b in a) and min(len(a), len(b)) >= MIN_CONTAINMENT_CHARS:
        return 0.8
    common = sum((min(a.count(c), b.count(c)) for c in set(a)))
    return common / max(len(a), len(b))


def best_vector_match(
    ocr_item: dict,
    centre: list[float],
    vector_texts: list[dict],
    diagonal: float,
) -> tuple[dict | None, float, float, str]:
    """Pick the vector text that best explains this OCR box.

    Returns the candidate plus the distance, text agreement and which rule
    accepted it, so the provenance of every link stays inspectable.
    """
    spatial_limit = diagonal * SPATIAL_ONLY_RATIO
    text_limit = diagonal * TEXT_MATCH_RATIO
    ocr_content = ocr_item.get("content", "")

    best: dict | None = None
    best_rank = (-1.0, float("inf"))
    best_distance = float("inf")
    best_agreement = 0.0
    best_rule = ""

    for candidate in vector_texts:
        position = candidate.get("position") or []
        if len(position) < 2:
            continue

        distance = math.hypot(position[0] - centre[0], position[1] - centre[1])
        agreement = match_score(ocr_content, candidate.get("content", ""))

        if agreement >= MIN_TEXT_AGREEMENT and distance <= text_limit:
            rule = "text+spatial"
        elif distance <= spatial_limit:
            rule = "spatial"
        else:
            continue

        # Prefer stronger text agreement, then the closer candidate.
        rank = (agreement, -distance)
        if rank > best_rank:
            best_rank = rank
            best = candidate
            best_distance = distance
            best_agreement = agreement
            best_rule = rule

    return best, best_distance, best_agreement, best_rule


def link_ocr_to_vector(
    ocr_texts: list[dict],
    vector_texts: list[dict],
    mapper: PixelToDrawing | None,
    *,
    layout_mappers: dict[str, PixelToDrawing] | None = None,
) -> tuple[list[dict], int]:
    """Attach a drawing-space position and best vector match to each OCR item."""
    linked: list[dict] = []
    matched = 0
    layout_mappers = layout_mappers or {}

    for item in ocr_texts:
        record = dict(item)
        record["text_class"] = classify_text(item.get("content", ""))

        active_mapper = mapper
        layout_key = item.get("layout_key")
        source = str(item.get("source") or "")
        if layout_key and layout_key in layout_mappers:
            active_mapper = layout_mappers[layout_key]
        elif source.startswith("paperspace:"):
            # Prefer the layout transform; do not map paper pixels into model space.
            key = source.split("/", 1)[0].removeprefix("paperspace:")
            active_mapper = layout_mappers.get(key)

        if active_mapper is None or not item.get("bbox"):
            record["drawing_position"] = None
            record["link"] = None
            linked.append(record)
            continue

        centre = active_mapper.box_centre(item["bbox"])
        record["drawing_position"] = centre
        if layout_key or source.startswith("paperspace:"):
            record["space"] = "paperspace"
            # Paperspace OCR is not matched against model-space TEXT entities.
            record["link"] = None
            linked.append(record)
            continue

        best, distance, agreement, rule = best_vector_match(
            item, centre, vector_texts, active_mapper.diagonal
        )

        if best is None:
            record["link"] = None
        else:
            record["link"] = {
                "vector_handle": best.get("handle"),
                "vector_content": best.get("content"),
                "layer": best.get("layer"),
                "distance": round(distance, 3),
                "text_agreement": round(agreement, 3),
                "matched_by": rule,
            }
            matched += 1

        linked.append(record)

    return linked, matched


def merge_one(stem: str, ocr_path: Path) -> dict:
    vector = load_json(JSON_DIR / f"{stem}.json")
    ocr = load_json(ocr_path)

    transform_path = IMAGE_DIR / f"{stem}_transform.json"
    mapper = None
    if transform_path.exists():
        mapper = PixelToDrawing(load_json(transform_path))

    layout_mappers: dict[str, PixelToDrawing] = {}
    for path in sorted(IMAGE_DIR.glob(f"{stem}__ps_*_transform.json")):
        # `{stem}__ps_{key}_transform.json` → key
        suffix = path.name[len(f"{stem}__ps_") :]
        key = suffix[: -len("_transform.json")]
        layout_mappers[key] = PixelToDrawing(load_json(path))

    vector_texts = vector.get("texts", [])
    linked_ocr, matched = link_ocr_to_vector(
        ocr.get("texts", []),
        vector_texts,
        mapper,
        layout_mappers=layout_mappers,
    )

    # Reported separately: "text+spatial" means the strings agree, while
    # "spatial" only means nothing else was nearby. Summing them into one
    # link rate overstates how much of the OCR output is genuinely tied to a
    # vector entity.
    rule_counts = Counter(
        item["link"]["matched_by"] for item in linked_ocr if item.get("link")
    )
    class_counts = Counter(item["text_class"] for item in linked_ocr)

    vector_stats = vector.get("stats", {})
    sources = {
        "vector_json": rel(JSON_DIR / f"{stem}.json"),
        "ocr_json": rel(ocr_path),
    }
    if transform_path.exists():
        sources["transform_json"] = rel(transform_path)

    header = vector.get("header", {})

    return {
        "file_name": vector.get("file_name", f"{stem}.dxf"),
        "dxf_version": vector.get("dxf_version"),
        "header": header,
        "stats": {
            "vector_text_count": len(vector_texts),
            "ocr_text_count": len(linked_ocr),
            "linked_text_count": matched,
            "linked_by_rule": dict(rule_counts),
            "ocr_class_counts": dict(class_counts),
            "ocr_suspect_count": class_counts.get("single_char", 0),
            "layer_count": vector_stats.get("layer_count"),
            "entity_count": vector_stats.get("entity_count"),
            "dimension_count": vector_stats.get("dimension_count"),
            "units": header.get("units"),
            "table_count": vector_stats.get("table_count"),
            "table_cell_count": vector_stats.get("table_cell_count"),
            "layout_count": vector_stats.get("layout_count"),
            "paperspace_entity_count": vector_stats.get("paperspace_entity_count"),
            "viewport_count": vector_stats.get("viewport_count"),
            "frozen_layers": vector_stats.get("frozen_layers", []),
            "viewport_frozen_layers": vector_stats.get("viewport_frozen_layers", []),
        },
        "layers": vector.get("layers", []),
        "blocks": vector.get("blocks", []),
        "block_references": vector.get("block_references", []),
        "dimensions": vector.get("dimensions", []),
        "tables": vector.get("tables", []),
        "layouts": vector.get("layouts", []),
        "vector_texts": vector_texts,
        "ocr_texts": linked_ocr,
        "entities": vector.get("entities", []),
        "sources": sources,
    }


def main() -> None:
    files = sorted(OCR_DIR.glob("*_ocr.json"))

    def handle(ocr_path: Path) -> str:
        stem = strip_suffix(ocr_path, "_ocr.json")
        vector_path = JSON_DIR / f"{stem}.json"
        if not vector_path.exists():
            raise FileNotFoundError(f"missing vector json: {vector_path.name}")

        merged = merge_one(stem, ocr_path)
        dump_json(MERGED_DIR / f"{stem}_merged.json", merged)
        stats = merged["stats"]
        rules = stats["linked_by_rule"]
        return (
            f"vector {stats['vector_text_count']} | "
            f"ocr {stats['ocr_text_count']} | "
            f"linked {stats['linked_text_count']} "
            f"(text+spatial {rules.get('text+spatial', 0)}, "
            f"spatial {rules.get('spatial', 0)})"
        )

    run_batch(
        files,
        handle,
        out_dir=MERGED_DIR,
        empty_hint="Run: py -3.13 scripts/ocr_images.py",
    )


if __name__ == "__main__":
    main()
