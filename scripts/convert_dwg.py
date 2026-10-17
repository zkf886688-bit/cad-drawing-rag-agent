"""Convert DWG source files to DXF (stage 0 of the pipeline).

DWG is Autodesk's closed binary format and no Python library can parse it
directly -- ezdxf itself states it "can not convert DXF files to other CAD
formats such as DWG". The accepted route, and the one ezdxf ships an add-on
for, is to convert DWG to DXF with the ODA File Converter and parse the DXF.

This script makes that conversion an explicit, measured pipeline step so the
DWG -> DXF success rate can be reported rather than assumed.

Prerequisite: install the free ODA File Converter
  https://www.opendesign.com/guestfiles/oda_file_converter

  py scripts/convert_dwg.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import ezdxf
from ezdxf.addons import odafc

from _common import DATA, RAW_DIR, ROOT, dump_json, rel, run_batch

# Allow ODA_FILE_CONVERTER from project .env
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.envfile import load_local_env

load_local_env()

DWG_DIR = DATA / "dwg"
REPORT_PATH = DATA / "dwg" / "conversion_report.json"

# Target DXF version. R2018 keeps the most entity detail; older readers may
# need R2013 or earlier.
TARGET_VERSION = "R2018"

# Override with ODA_FILE_CONVERTER / ODAFC_PATH pointing at the executable
# (or its directory). Falls back to common Windows install roots.
INSTALL_ROOTS = (
    Path(r"C:\Program Files\ODA"),
    Path(r"C:\Program Files (x86)\ODA"),
)


def _resolve_exec_from_path(raw: str) -> Path | None:
    """Accept an .exe path or a directory that contains ODAFileConverter*."""
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.glob("ODAFileConverter*/ODAFileConverter.exe"))
        if matches:
            return matches[-1]
        direct = path / "ODAFileConverter.exe"
        if direct.is_file():
            return direct
    return None


def check_oda_available() -> str | None:
    """Return the ODA File Converter path, or None when it is not installed."""
    for env_key in ("ODA_FILE_CONVERTER", "ODAFC_PATH"):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        found = _resolve_exec_from_path(raw)
        if found is None:
            print(f"Warning: {env_key}={raw!r} is not a usable ODA path.")
            continue
        ezdxf.options.set("odafc-addon", "win_exec_path", str(found))
        if odafc.is_installed():
            return str(found)

    if odafc.is_installed():
        return odafc.get_win_exec_path()

    candidates = [
        exe
        for root in INSTALL_ROOTS
        if root.is_dir()
        for exe in root.glob("ODAFileConverter*/ODAFileConverter.exe")
    ]
    if not candidates:
        return None

    # Newest version last by name, which sorts correctly for these directories.
    found = sorted(candidates)[-1]
    ezdxf.options.set("odafc-addon", "win_exec_path", str(found))
    return str(found) if odafc.is_installed() else None


class OutputExists(Exception):
    """Raised when converting would overwrite a DXF this script did not create."""


def convert_one(dwg_path: Path, *, force: bool = False) -> dict:
    """Convert a single DWG and report what survived the conversion."""
    dxf_path = RAW_DIR / f"{dwg_path.stem}.dxf"

    # A same-named DXF may be an original supplied alongside the DWG; silently
    # replacing it would destroy the only baseline for judging conversion loss.
    if dxf_path.exists() and not force:
        raise OutputExists(rel(dxf_path))

    started = time.perf_counter()
    odafc.convert(dwg_path, dxf_path, version=TARGET_VERSION, replace=True)
    elapsed = time.perf_counter() - started

    if not dxf_path.exists():
        raise RuntimeError("converter reported success but produced no DXF")

    # Reload the result so the report reflects what is actually parseable,
    # not merely that a file appeared on disk.
    from _common import load_dxf

    doc, _auditor, recovered = load_dxf(dxf_path)
    entity_count = len(list(doc.modelspace()))

    result = {
        "source_dwg": rel(dwg_path),
        "output_dxf": rel(dxf_path),
        "dwg_size_bytes": dwg_path.stat().st_size,
        "dxf_size_bytes": dxf_path.stat().st_size,
        "dxf_version": doc.dxfversion,
        "layer_count": len(doc.layers),
        "entity_count": entity_count,
        "seconds": round(elapsed, 3),
    }
    if recovered:
        result["recovered"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing DXF files with the same name",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEM",
        help="only convert these DWG stems (file name without .dwg)",
    )
    args = parser.parse_args()

    DWG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    exec_path = check_oda_available()
    if exec_path is None:
        print("ODA File Converter not found.")
        print("Install it (free) from:")
        print("  https://www.opendesign.com/guestfiles/oda_file_converter")
        print()
        print("Without it, DWG files cannot be read: DWG is a closed binary")
        print("format and ezdxf has no DWG parser. Existing DXF files in")
        print(f"{RAW_DIR} still work for every later stage.")
        return

    print(f"ODA File Converter: {exec_path}")

    files = sorted(DWG_DIR.glob("*.dwg"))
    if args.only:
        wanted = {stem.lower() for stem in args.only}
        files = [path for path in files if path.stem.lower() in wanted]
        missing = wanted - {path.stem.lower() for path in files}
        if missing:
            print(f"Warning: --only stems not found in {DWG_DIR}: {sorted(missing)}")
        if not files:
            print(f"No matching DWG for --only {args.only}. Nothing to convert.")
            return

    results: list[dict] = []

    skipped: list[str] = []

    def handle(dwg_path: Path) -> str:
        try:
            result = convert_one(dwg_path, force=args.force)
        except OutputExists as exc:
            skipped.append(str(exc))
            return f"skipped, {exc} already exists (use --force to overwrite)"
        results.append(result)
        return (
            f"-> {Path(result['output_dxf']).name} | "
            f"{result['dxf_version']} | "
            f"layers {result['layer_count']} | "
            f"entities {result['entity_count']} | "
            f"{result['seconds']}s"
        )

    run_batch(
        files,
        handle,
        out_dir=RAW_DIR,
        empty_hint=f"Put DWG files in {DWG_DIR}",
    )

    # run_batch counts a skip as a completed call, so state the real tally.
    print(f"\nConverted {len(results)}, skipped {len(skipped)}, of {len(files)} DWG.")
    if skipped:
        print("Skips avoided overwriting an existing DXF; use --force to replace.")

    if results:
        dump_json(
            REPORT_PATH,
            {
                "target_version": TARGET_VERSION,
                "converter": str(exec_path),
                "attempted": len(files),
                "succeeded": len(results),
                "skipped": skipped,
                "only": list(args.only) if args.only else None,
                "conversions": results,
            },
        )
        print(f"Conversion report: {REPORT_PATH}")


if __name__ == "__main__":
    main()