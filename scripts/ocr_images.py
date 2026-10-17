"""Run OCR over the rendered PNGs (stage 2 of the pipeline).

The engine downscales any input to a fixed maximum side length, so feeding it a
whole high-resolution sheet defeats the resolution: on the AISC channel sheet,
whole-image OCR returned 'csng' and 'cpaq' where the labels read C15X50 and
C15X40, and raising the render to 9600px made it worse, not better.

Tiling into roughly square patches keeps each label large relative to its tile.
Same sheet, same engine, exact label matches:

  whole image, 2400px    0/3
  whole image, 9600px    0/3
  tiled 1200px, 9600px   3/3

Tiling alone trades one failure for another: a sentence crossing a seam comes
back in halves ('ayer 3 is fro' + 'rozen' for "Layer 3 is frozen"), which the
whole-image pass reads correctly. So both passes run and their results merge,
preferring the more complete reading where they overlap.

Tile boxes are offset back into full-image coordinates so stage 3 can still map
them onto drawing coordinates.

RapidOCR does not support Python 3.14 yet, so run this step with 3.13:
  py -3.13 scripts/ocr_images.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from rapidocr import RapidOCR

from _common import IMAGE_DIR, OCR_DIR, dump_json, rel, run_batch

# Square-ish patch that keeps labels legible after the engine's own downscale.
TILE_PX = 1200
# Enough overlap that a label straddling a seam appears whole in one tile.
TILE_OVERLAP_PX = 150
# Below this, the whole image already fits the engine's input size.
TILE_THRESHOLD_PX = 1600

# Two readings are treated as the same piece of text when this much of the
# smaller box lies inside the larger one.
SAME_TEXT_OVERLAP = 0.5

# Guard against decompression bombs. Render budget is ~12M px; allow headroom
# for paperspace / odd aspect ratios, but reject pathological inputs.
MAX_OCR_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_OCR_PIXELS


def tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Cover the image with overlapping tiles."""
    if max(width, height) <= TILE_THRESHOLD_PX:
        return [(0, 0, width, height)]

    step = TILE_PX - TILE_OVERLAP_PX
    boxes = []
    for top in range(0, height, step):
        for left in range(0, width, step):
            right, bottom = min(left + TILE_PX, width), min(top + TILE_PX, height)
            if right - left < 32 or bottom - top < 32:
                continue
            boxes.append((left, top, right, bottom))
            if right == width:
                break
        if min(top + TILE_PX, height) == height:
            break
    return boxes


def centre(bbox: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def extent(bbox: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def same_place(a: dict, b: dict) -> bool:
    """True when two readings cover substantially the same area."""
    ax0, ay0, ax1, ay1 = extent(a["bbox"])
    bx0, by0, bx1, by1 = extent(b["bbox"])

    overlap_w = min(ax1, bx1) - max(ax0, bx0)
    overlap_h = min(ay1, by1) - max(ay0, by0)
    if overlap_w <= 0 or overlap_h <= 0:
        return False

    overlap = overlap_w * overlap_h
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return smaller > 0 and overlap / smaller >= SAME_TEXT_OVERLAP


def better(a: dict, b: dict) -> dict:
    """Pick the more complete of two readings of the same text."""
    a_text, b_text = a["content"], b["content"]
    # A fragment from a cut-off tile is a substring of the full reading.
    if a_text != b_text:
        if a_text in b_text:
            return b
        if b_text in a_text:
            return a
    return a if a["confidence"] >= b["confidence"] else b


def merge_readings(items: list[dict]) -> list[dict]:
    """Collapse readings of the same text from different passes and tiles."""
    kept: list[dict] = []
    # Longest first, so a full sentence is in place before its fragments arrive.
    for item in sorted(items, key=lambda i: (-len(i["content"]), -i["confidence"])):
        for index, existing in enumerate(kept):
            if same_place(existing, item):
                kept[index] = better(existing, item)
                break
        else:
            kept.append(item)
    return kept


def read_region(
    engine: RapidOCR,
    image: Image.Image,
    box: tuple[int, int, int, int],
    source: str,
) -> list[dict]:
    """OCR one region, returning boxes in full-image pixel coordinates."""
    left, top, _, _ = box
    region = image if box == (0, 0, image.width, image.height) else image.crop(box)

    result = engine(region)
    texts = getattr(result, "txts", None) or []
    scores = getattr(result, "scores", None) or []
    result_boxes = getattr(result, "boxes", None)
    if result_boxes is None or not len(texts):
        return []

    items = []
    for result_box, text, score in zip(result_boxes, texts, scores):
        content = str(text).strip()
        if not content:
            continue
        items.append(
            {
                "content": content,
                "confidence": float(score),
                "bbox": [[float(x) + left, float(y) + top] for x, y in result_box],
                "source": source,
            }
        )
    return items


def ocr_image(engine: RapidOCR, path: Path, *, source_prefix: str = "") -> dict:
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    width, height = image.size
    if width * height > MAX_OCR_PIXELS:
        raise ValueError(
            f"image too large for OCR: {width}x{height} "
            f"(limit {MAX_OCR_PIXELS} pixels)"
        )
    whole = (0, 0, width, height)
    tiles = tile_boxes(width, height)

    # The whole-image pass catches long text that tile seams would split.
    def tag(source: str) -> str:
        return f"{source_prefix}{source}" if source_prefix else source

    items = read_region(engine, image, whole, tag("whole"))
    whole_count = len(items)

    tiled = tiles != [whole]
    if tiled:
        for box in tiles:
            items.extend(read_region(engine, image, box, tag("tile")))

    items = merge_readings(items)
    items.sort(key=lambda i: (centre(i["bbox"])[1], centre(i["bbox"])[0]))
    image.close()

    return {
        "file_name": path.name,
        "source_image": rel(path),
        "image_size_px": [width, height],
        "tile_count": len(tiles) if tiled else 0,
        "whole_pass_texts": whole_count,
        "text_count": len(items),
        "texts": items,
    }


def paperspace_pngs(stem: str) -> list[Path]:
    """Layout renders written by render_dxf.py as `{stem}__ps_{layout}.png`."""
    return sorted(
        path
        for path in IMAGE_DIR.glob(f"{stem}__ps_*.png")
        if "__ps_" in path.stem and not path.stem.endswith("_transform")
    )


def ocr_drawing(engine: RapidOCR, model_png: Path) -> dict:
    """OCR model space, then fold in any paperspace layout renders."""
    data = ocr_image(engine, model_png)
    layout_passes: list[dict] = []
    for ps_path in paperspace_pngs(model_png.stem):
        # Layout name sits after `__ps_`; keep boxes in that image's pixel space
        # and tag the source so merge can pick the matching transform.
        layout_key = ps_path.stem.split("__ps_", 1)[-1]
        part = ocr_image(engine, ps_path, source_prefix=f"paperspace:{layout_key}/")
        for item in part["texts"]:
            item["layout_image"] = rel(ps_path)
            item["layout_key"] = layout_key
            data["texts"].append(item)
        layout_passes.append(
            {
                "layout_key": layout_key,
                "source_image": rel(ps_path),
                "text_count": part["text_count"],
                "tile_count": part["tile_count"],
            }
        )

    data["texts"] = merge_readings(data["texts"])
    data["texts"].sort(key=lambda i: (centre(i["bbox"])[1], centre(i["bbox"])[0]))
    data["text_count"] = len(data["texts"])
    data["paperspace_passes"] = layout_passes
    return data


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只处理这些图纸 stem，例如 --only demo_engineering_sheet",
    )
    args = parser.parse_args()

    # Model-space PNGs only; `__ps_` layout renders are folded into the same
    # drawing's OCR json so merge_ocr keeps a 1:1 stem mapping.
    files = sorted(
        path
        for path in IMAGE_DIR.glob("*.png")
        if "__ps_" not in path.stem
    )
    if args.only:
        wanted = set(args.only)
        files = [path for path in files if path.stem in wanted]
        missing = wanted - {path.stem for path in files}
        if missing:
            raise SystemExit(f"missing renders for: {sorted(missing)}")

    if not files:
        print("No PNG files found. Run: py scripts/render_dxf.py")
        return

    if sys.version_info[:2] != (3, 13):
        print(
            f"Warning: running on Python {sys.version_info.major}."
            f"{sys.version_info.minor}; RapidOCR expects 3.13.",
            file=sys.stderr,
        )

    print("Loading OCR model...")
    engine = RapidOCR()

    def handle(path: Path) -> str:
        data = ocr_drawing(engine, path)
        dump_json(OCR_DIR / f"{path.stem}_ocr.json", data)
        ps = len(data.get("paperspace_passes") or [])
        return (
            f"tiles {data['tile_count']} | whole-pass {data['whole_pass_texts']} "
            f"| merged {data['text_count']}"
            + (f" | paperspace {ps}" if ps else "")
        )

    run_batch(files, handle, out_dir=OCR_DIR)


if __name__ == "__main__":
    main()
