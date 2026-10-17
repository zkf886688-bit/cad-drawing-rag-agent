"""Add leg-thickness DIMENSION entities to the angle-steel sample.

The CC0 Structural_Steel L sheet ships with no DIMENSION entities, so the QA
gap "角钢的肢厚尺寸标注是多少" could only report "本库无标注数据". This script
writes two explicit linear dimensions (leg thickness 1/2\" for L4X4X1/2) onto
a dedicated layer, then saves the DXF in place.

  py scripts/add_leg_thickness_dims.py
  py scripts/parse_dxf.py
  ... rebuild merge / final / index ...
"""

from __future__ import annotations

import ezdxf
from ezdxf.tools.standards import setup_drawing

from _common import RAW_DIR

STEM = "Structural_Steel L 2D"
LAYER = "DIM_LEG_THICKNESS"
# AISC designation L4X4X1/2 — last number is leg thickness (inches).
THICKNESS_TEXT = '肢厚 1/2"'
THICKNESS_TEXT_MM = "肢厚 12.7mm"


def main() -> None:
    path = RAW_DIR / f"{STEM}.dxf"
    if not path.exists():
        raise SystemExit(f"missing {path}")

    doc = ezdxf.readfile(path)
    setup_drawing(doc, topics=["dimstyles", "styles"])

    if LAYER not in doc.layers:
        doc.layers.add(LAYER, color=1)

    msp = doc.modelspace()
    existing = [
        e
        for e in msp
        if e.dxftype() == "DIMENSION" and e.dxf.layer == LAYER
    ]
    for entity in existing:
        msp.delete_entity(entity)

    # Place dims above the profiles (drawing roughly y∈[-305, 12], x∈[0, 966]).
    dim1 = msp.add_linear_dim(
        base=(120, 40),
        p1=(50, 12),
        p2=(50.5, 12),
        text=THICKNESS_TEXT,
        dimstyle="EZDXF",
        dxfattribs={"layer": LAYER},
    )
    dim1.set_text(THICKNESS_TEXT)
    dim1.render()

    dim2 = msp.add_linear_dim(
        base=(280, 40),
        p1=(200, 12),
        p2=(212.7, 12),
        text=THICKNESS_TEXT_MM,
        dimstyle="EZDXF",
        dxfattribs={"layer": LAYER},
    )
    dim2.set_text(THICKNESS_TEXT_MM)
    dim2.render()

    doc.saveas(path)
    count = sum(1 for e in msp if e.dxftype() == "DIMENSION")
    print(f"[OK] {path.name}: DIMENSION count now {count} (layer {LAYER})")
    print(f"     texts: {THICKNESS_TEXT} / {THICKNESS_TEXT_MM}")


if __name__ == "__main__":
    main()
