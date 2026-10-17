"""Create a text/block/dimension-bearing DXF for ODA round-trip fidelity tests.

The three CC0 steel DWG pairs are geometry-only (labels are stroke polylines),
so text_preserved_ratio is always n/a. This sample exists so compare_conversion
can measure TEXT / MTEXT / INSERT / DIMENSION survival across:

  native DXF → (ODA) DWG → (ODA) DXF

  py scripts/make_fidelity_text_dxf.py
  py scripts/roundtrip_text_fidelity.py   # needs ODA File Converter
  py scripts/compare_conversion.py
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

from scripts._common import DATA  # noqa: E402

NATIVE_DIR = DATA / "native_dxf"
STEM = "fidelity_text_blocks_dims"
OUT = NATIVE_DIR / f"{STEM}.dxf"


def main() -> None:
    doc = ezdxf.new("R2010")
    doc.units = units.MM
    doc.header["$INSUNITS"] = 4

    doc.layers.add("TITLE", color=7)
    doc.layers.add("GEOM", color=1)
    doc.layers.add("DIMS", color=3)
    doc.layers.add("TEXT", color=5)

    # A reusable block with geometry + attribute-like text inside.
    block = doc.blocks.new("FIDELITY_MARK")
    block.add_circle((0, 0), radius=50, dxfattribs={"layer": "GEOM"})
    block.add_text(
        "MARK",
        height=20,
        dxfattribs={"layer": "TEXT"},
    ).set_placement((-30, -10), align=TextEntityAlignment.LEFT)

    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (400, 0), (400, 200), (0, 200)],
        close=True,
        dxfattribs={"layer": "GEOM"},
    )
    msp.add_text(
        "保真度文字样本 TITLE",
        height=25,
        dxfattribs={"layer": "TITLE"},
    ).set_placement((10, 220), align=TextEntityAlignment.LEFT)
    msp.add_mtext(
        "MTEXT多行\\P第二行 DIM/块/文字往返测试",
        dxfattribs={"layer": "TEXT", "char_height": 18},
    ).set_location((10, 160))
    msp.add_blockref("FIDELITY_MARK", insert=(100, 80), dxfattribs={"layer": "GEOM"})
    msp.add_blockref("FIDELITY_MARK", insert=(300, 80), dxfattribs={"layer": "GEOM"})

    dim = msp.add_linear_dim(
        base=(200, 120),
        p1=(0, 0),
        p2=(400, 0),
        dimstyle="EZDXF",
        dxfattribs={"layer": "DIMS"},
    )
    dim.set_text("宽 400")
    dim.render()

    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    doc.saveas(OUT)
    print(f"[OK] wrote {OUT}")
    print("Next: py scripts/roundtrip_text_fidelity.py")


if __name__ == "__main__":
    main()
