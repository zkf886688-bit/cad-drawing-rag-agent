"""Create a small engineering-like DXF demo with layers, title block and dims.

Public CC0 steel sheets lack title blocks / mixed annotations. This sample is
generated locally so the knowledge base demonstrates a more project-like sheet
without redistributing proprietary drawings.

  py scripts/make_demo_engineering_dxf.py
  # then ingest via UI upload or:
  py -c "from pathlib import Path; from app.ingest import run_ingest; run_ingest([Path('datasets/raw/demo_engineering_sheet.dxf')], mode='heuristic')"
"""

from __future__ import annotations

import sys
from pathlib import Path

import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import RAW_DIR  # noqa: E402

OUT = RAW_DIR / "demo_engineering_sheet.dxf"


def main() -> None:
    doc = ezdxf.new("R2010")
    doc.units = units.MM
    doc.header["$INSUNITS"] = 4  # millimeters
    doc.header["$MEASUREMENT"] = 1  # metric

    doc.layers.add("TITLE", color=7)
    doc.layers.add("WALL", color=8)
    doc.layers.add("STEEL", color=1)
    doc.layers.add("DIMS", color=3)
    doc.layers.add("ANNOTATION", color=5)
    doc.layers.add("BORDER", color=8)

    msp = doc.modelspace()

    # Outer title-block / 图框 frame.
    msp.add_lwpolyline(
        [(-500, -800), (9000, -800), (9000, 6200), (-500, 6200)],
        close=True,
        dxfattribs={"layer": "BORDER"},
    )

    # Simple plan: rectangle wall outline + a steel member as thick line.
    msp.add_lwpolyline(
        [(0, 0), (8000, 0), (8000, 5000), (0, 5000)],
        close=True,
        dxfattribs={"layer": "WALL"},
    )
    msp.add_line((1000, 2500), (7000, 2500), dxfattribs={"layer": "STEEL"})

    # Title block fields (图框标题栏) — used by DIMENSION / 图框 regression cases.
    title_lines = [
        ((200, 5600), "工程示意平面图", 220),
        ((200, 5300), "图名: 工程示意平面图", 120),
        ((200, 5100), "图号: DEMO-ENG-001", 120),
        ((4500, 5100), "比例: 1:100", 120),
        ((6500, 5100), "单位: 毫米", 120),
        ((200, 4800), "构件: 槽钢示意 C150×75（教学用，非真实项目）", 110),
        (
            (200, 4550),
            "图层: TITLE/WALL/STEEL/DIMS/ANNOTATION/BORDER | 含 DIMENSION 梁长标注",
            100,
        ),
    ]
    for (x, y), content, height in title_lines:
        layer = "TITLE" if "工程示意平面图" in content or content.startswith("图") else "ANNOTATION"
        if content.startswith("比例") or content.startswith("单位"):
            layer = "TITLE"
        msp.add_text(
            content,
            height=height,
            dxfattribs={"layer": layer},
        ).set_placement((x, y), align=TextEntityAlignment.LEFT)

    # Linear dimension spanning the steel line — explicit text for retrieval.
    dim = msp.add_linear_dim(
        base=(4000, 3000),
        p1=(1000, 2500),
        p2=(7000, 2500),
        dimstyle="EZDXF",
        override={"dimtxsty": "Standard"},
        dxfattribs={"layer": "DIMS"},
    )
    dim.set_text("梁长 6000")
    dim.render()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    doc.saveas(OUT)
    print(f"[OK] wrote {OUT}")
    print("Next: upload in /ui, or run the ingest pipeline on this DXF.")


if __name__ == "__main__":
    main()
