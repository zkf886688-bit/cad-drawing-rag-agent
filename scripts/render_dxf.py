"""Render DXF files to PNG, recording the pixel <-> drawing coordinate mapping.

The transform files are what allow stage 2 to map OCR pixel boxes back onto
drawing coordinates, so OCR text can be linked to vector entities.

Model space is always rendered to `{stem}.png`. Paper-space layouts with
entities are also rendered to `{stem}__ps_{layout}.png` so viewport / layout
annotations become visible to OCR (model-space-only renders miss them).

Two render choices are driven by OCR accuracy on stroke-font labels, which are
drawn as line work rather than TEXT entities and are therefore only readable
from the image. Measured on the AISC channel sheet, exact label matches were:

  dark background, 4800px long edge   1/3
  white background, 4800px long edge  2/3
  white background, 9600px long edge  3/3

So: white background, and a pixel budget generous enough that thin strokes
survive. Resolution alone is not sufficient -- see ocr_images.py, which must
tile the image because the OCR engine downscales whole large inputs.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

# Must be set before pyplot import: ingest runs render in a background thread,
# and the default TkAgg backend deadlocks / crashes ("Tcl_AsyncDelete").
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

from ezdxf.addons.drawing import Frontend, RenderContext, matplotlib as draw_mpl
from ezdxf.addons.drawing.properties import LayoutProperties
from matplotlib import pyplot as plt

from _common import IMAGE_DIR, RAW_DIR, dump_json, load_dxf, rel, run_batch

DPI = 150

# Spend a fixed pixel budget rather than a fixed long edge: a wide strip then
# gets ~9600x1200 while a squarer sheet gets ~4000x3000, instead of an
# unusable 9600x7200.
TARGET_PIXELS = 12_000_000
MAX_LONG_EDGE = 9600
BACKGROUND = "#FFFFFF"


def target_size(width_px: int, height_px: int) -> tuple[int, int]:
    """Scale a canvas to the pixel budget, preserving aspect ratio."""
    aspect = height_px / width_px
    width = math.sqrt(TARGET_PIXELS / aspect)
    height = width * aspect
    if max(width, height) > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / max(width, height)
        width, height = width * scale, height * scale
    return max(int(width), 1), max(int(height), 1)


def safe_layout_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return cleaned or "layout"


def render_layout(
    doc,
    layout,
    png_path: Path,
    transform_path: Path,
    *,
    file_name: str,
    space: str,
    layout_name: str,
) -> dict:
    """Render one layout (model or paper) and write PNG + transform."""
    layout_properties = LayoutProperties.from_layout(layout)
    layout_properties.set_colors(BACKGROUND)

    fig = plt.figure(dpi=DPI)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        Frontend(RenderContext(doc), draw_mpl.MatplotlibBackend(ax)).draw_layout(
            layout,
            finalize=True,
            layout_properties=layout_properties,
        )

        base_width, base_height = fig.canvas.get_width_height()
        width_px, height_px = target_size(base_width, base_height)
        fig.set_size_inches(width_px / DPI, height_px / DPI)

        fig.savefig(png_path, dpi=DPI, facecolor=BACKGROUND)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        width_px, height_px = fig.canvas.get_width_height()
    finally:
        plt.close(fig)

    transform = {
        "file_name": file_name,
        "image": rel(png_path),
        "dpi": DPI,
        "background": BACKGROUND,
        "space": space,
        "layout_name": layout_name,
        "image_size_px": [width_px, height_px],
        "drawing_extent": {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "note": (
            "Axes fill the whole canvas, so pixel->drawing is a linear map. "
            "Pixel y grows downward while drawing y grows upward. "
            f"space={space}, layout={layout_name}."
        ),
    }
    dump_json(transform_path, transform)
    return transform


def render_dxf(path: Path) -> dict:
    doc, _auditor, recovered = load_dxf(path)
    msp = doc.modelspace()

    model_png = IMAGE_DIR / f"{path.stem}.png"
    model_transform = IMAGE_DIR / f"{path.stem}_transform.json"
    model = render_layout(
        doc,
        msp,
        model_png,
        model_transform,
        file_name=path.name,
        space="model",
        layout_name="Model",
    )
    if recovered:
        model["recovered"] = True

    paperspace: list[dict] = []
    for layout in doc.layouts:
        if layout.name.lower() == "model":
            continue
        entities = list(layout)
        if not entities:
            continue
        safe = safe_layout_name(layout.name)
        ps_png = IMAGE_DIR / f"{path.stem}__ps_{safe}.png"
        ps_transform = IMAGE_DIR / f"{path.stem}__ps_{safe}_transform.json"
        transform = render_layout(
            doc,
            layout,
            ps_png,
            ps_transform,
            file_name=path.name,
            space="paperspace",
            layout_name=layout.name,
        )
        paperspace.append(
            {
                "layout_name": layout.name,
                "image": transform["image"],
                "transform": rel(ps_transform),
                "image_size_px": transform["image_size_px"],
                "entity_count": len(entities),
            }
        )

    model["paperspace_renders"] = paperspace
    dump_json(model_transform, model)
    return model

def main() -> None:
    files = sorted(RAW_DIR.glob("*.dxf"))

    def handle(path: Path) -> str:
        transform = render_dxf(path)
        w, h = transform["image_size_px"]
        ps = transform.get("paperspace_renders") or []
        extra = f" + {len(ps)} paperspace" if ps else ""
        return f"{path.stem}.png ({w}x{h}px){extra}"

    run_batch(
        files,
        handle,
        out_dir=IMAGE_DIR,
        empty_hint=f"Put DXF files in {RAW_DIR}",
    )


if __name__ == "__main__":
    main()
