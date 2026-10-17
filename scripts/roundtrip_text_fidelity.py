"""ODA round-trip for the text-bearing fidelity sample.

Flow:
  datasets/native_dxf/fidelity_text_blocks_dims.dxf
    → (ODA) datasets/dwg/fidelity_text_blocks_dims.dwg
    → (ODA) datasets/raw/fidelity_text_blocks_dims.dxf
    → compare_conversion.py reports text_preserved_ratio

Requires ODA File Converter. Without it, the script exits 0 after printing
how to install, so CI without ODA still passes.

  py scripts/make_fidelity_text_dxf.py
  py scripts/roundtrip_text_fidelity.py
  py scripts/compare_conversion.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import ezdxf
from ezdxf.addons import odafc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.envfile import load_local_env  # noqa: E402
from scripts._common import DATA, RAW_DIR  # noqa: E402
from scripts.convert_dwg import check_oda_available  # noqa: E402

load_local_env()

DWG_DIR = DATA / "dwg"
STEM = "fidelity_text_blocks_dims"
NATIVE_DXF = DATA / "native_dxf" / f"{STEM}.dxf"


def main() -> None:
    if not NATIVE_DXF.exists():
        from scripts.make_fidelity_text_dxf import main as make_native

        make_native()

    oda = check_oda_available()
    if not oda:
        print(
            "ODA File Converter not found — skip text round-trip. "
            "Install from https://www.opendesign.com/guestfiles/oda_file_converter "
            "then re-run this script to measure TEXT/INSERT/DIMENSION fidelity."
        )
        return

    print(f"ODA: {oda}")
    DWG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dwg_path = DWG_DIR / f"{STEM}.dwg"
    raw_path = RAW_DIR / f"{STEM}.dxf"

    with tempfile.TemporaryDirectory(prefix="cad_fidelity_") as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / NATIVE_DXF.name
        shutil.copy2(NATIVE_DXF, src)
        print(f"DXF→DWG: {NATIVE_DXF.name}")
        odafc.convert(str(src), str(tmp_dir / f"{STEM}.dwg"), version="R2018")
        produced_dwg = tmp_dir / f"{STEM}.dwg"
        if not produced_dwg.exists():
            candidates = list(tmp_dir.glob("*.dwg"))
            if not candidates:
                raise SystemExit("ODA did not produce a DWG")
            produced_dwg = candidates[0]
        shutil.copy2(produced_dwg, dwg_path)

        print(f"DWG→DXF: {dwg_path.name}")
        out_dxf = tmp_dir / "out" / f"{STEM}.dxf"
        out_dxf.parent.mkdir(parents=True, exist_ok=True)
        try:
            odafc.convert(str(dwg_path), str(out_dxf), version="R2018")
        except Exception:
            out_dxf = Path()
        if not out_dxf.exists():
            doc = odafc.readfile(str(dwg_path))
            doc.saveas(raw_path)
        else:
            shutil.copy2(out_dxf, raw_path)

    ezdxf.readfile(raw_path)
    print(f"[OK] round-trip DXF → {raw_path}")
    print("Next: py scripts/compare_conversion.py")


if __name__ == "__main__":
    main()
