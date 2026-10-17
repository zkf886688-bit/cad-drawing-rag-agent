"""Compare DWG-converted DXF against the publisher's original DXF.

The pipeline reads DWG only by way of the ODA File Converter, so the honest
question is what that hop costs. Where a drawing ships as both DWG and DXF,
both routes can be parsed and their structure compared directly.

  datasets/native_dxf/x.dxf   original DXF from the publisher   (baseline)
  datasets/raw/x.dxf          produced by convert_dwg.py        (comparison)

  py scripts/compare_conversion.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import ezdxf

from _common import DATA, RAW_DIR, dump_json, rel

NATIVE_DIR = DATA / "native_dxf"
REPORT_PATH = DATA / "dwg" / "fidelity_report.json"


def profile(path: Path) -> dict:
    """Summarise a DXF's structure for comparison."""
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    entity_types: Counter[str] = Counter()
    text_values: list[str] = []
    dimension_texts: list[str] = []
    insert_names: list[str] = []
    for entity in msp:
        dxftype = entity.dxftype()
        entity_types[dxftype] += 1
        if dxftype in ("TEXT", "MTEXT", "ATTRIB"):
            raw = entity.plain_text() if dxftype == "MTEXT" else entity.dxf.text
            cleaned = str(raw).strip()
            if cleaned:
                text_values.append(cleaned)
        elif dxftype == "INSERT":
            insert_names.append(str(entity.dxf.name))
        elif dxftype == "DIMENSION":
            label = ""
            if entity.dxf.hasattr("text"):
                label = str(entity.dxf.text or "").strip()
            if not label or label in {"<>", " "}:
                try:
                    label = str(entity.get_measurement())
                except Exception:
                    label = ""
            if label:
                dimension_texts.append(label)

    return {
        "dxf_version": doc.dxfversion,
        "size_bytes": path.stat().st_size,
        "layers": sorted(layer.dxf.name for layer in doc.layers),
        "block_definitions": sum(1 for b in doc.blocks if not b.name.startswith("*")),
        "entity_total": sum(entity_types.values()),
        "entity_types": dict(sorted(entity_types.items())),
        "text_count": len(text_values),
        "texts": sorted(set(text_values)),
        "insert_names": sorted(insert_names),
        "insert_count": len(insert_names),
        "dimension_texts": sorted(set(dimension_texts)),
        "dimension_count": entity_types.get("DIMENSION", 0),
    }


def compare(native: dict, converted: dict) -> dict:
    """Diff two profiles, focusing on what a drawing would lose."""
    native_types = native["entity_types"]
    converted_types = converted["entity_types"]

    type_diff = {
        dxftype: {
            "native": native_types.get(dxftype, 0),
            "converted": converted_types.get(dxftype, 0),
        }
        for dxftype in sorted(set(native_types) | set(converted_types))
        if native_types.get(dxftype, 0) != converted_types.get(dxftype, 0)
    }

    native_texts = set(native["texts"])
    converted_texts = set(converted["texts"])
    native_inserts = set(native.get("insert_names") or [])
    converted_inserts = set(converted.get("insert_names") or [])
    native_dims = set(native.get("dimension_texts") or [])
    converted_dims = set(converted.get("dimension_texts") or [])

    return {
        "entity_total": {
            "native": native["entity_total"],
            "converted": converted["entity_total"],
            "delta": converted["entity_total"] - native["entity_total"],
        },
        "layers_only_in_native": sorted(set(native["layers"]) - set(converted["layers"])),
        "layers_only_in_converted": sorted(set(converted["layers"]) - set(native["layers"])),
        "entity_type_differences": type_diff,
        "texts_lost": sorted(native_texts - converted_texts),
        "texts_added": sorted(converted_texts - native_texts),
        "text_preserved_ratio": (
            round(len(native_texts & converted_texts) / len(native_texts), 4)
            if native_texts
            else None
        ),
        "inserts_lost": sorted(native_inserts - converted_inserts),
        "insert_preserved_ratio": (
            round(len(native_inserts & converted_inserts) / len(native_inserts), 4)
            if native_inserts
            else None
        ),
        "dimensions_lost": sorted(native_dims - converted_dims),
        "dimension_preserved_ratio": (
            round(len(native_dims & converted_dims) / len(native_dims), 4)
            if native_dims
            else None
        ),
    }


def main() -> None:
    if not NATIVE_DIR.is_dir():
        print(f"No baseline directory: {NATIVE_DIR}")
        print("Put publisher-provided original DXF files there to compare against.")
        return

    pairs = [
        (native, RAW_DIR / native.name)
        for native in sorted(NATIVE_DIR.glob("*.dxf"))
        if (RAW_DIR / native.name).exists()
    ]
    if not pairs:
        print(f"No matching name found in both {NATIVE_DIR} and {RAW_DIR}.")
        return

    print(f"Comparing {len(pairs)} drawing(s)\n")
    comparisons = []

    for native_path, converted_path in pairs:
        native = profile(native_path)
        converted = profile(converted_path)
        diff = compare(native, converted)

        comparisons.append(
            {
                "drawing": native_path.stem,
                "native_file": rel(native_path),
                "converted_file": rel(converted_path),
                "native": native,
                "converted": converted,
                "diff": diff,
            }
        )

        total = diff["entity_total"]
        ratio = diff["text_preserved_ratio"]
        insert_ratio = diff.get("insert_preserved_ratio")
        dim_ratio = diff.get("dimension_preserved_ratio")
        print(f"{native_path.stem}")
        print(f"  entities  native {total['native']} -> converted {total['converted']}")
        print(f"  layers    native {len(native['layers'])} / converted {len(converted['layers'])}")
        print(f"  text kept {ratio if ratio is not None else 'n/a (no text)'}")
        print(
            f"  inserts   {insert_ratio if insert_ratio is not None else 'n/a'} "
            f"(native {native.get('insert_count', 0)})"
        )
        print(
            f"  dims      {dim_ratio if dim_ratio is not None else 'n/a'} "
            f"(native {native.get('dimension_count', 0)})"
        )
        if diff["entity_type_differences"]:
            for dxftype, counts in diff["entity_type_differences"].items():
                print(f"    {dxftype}: {counts['native']} -> {counts['converted']}")
        if diff["texts_lost"]:
            print(f"    text lost: {diff['texts_lost'][:5]}")
        if diff.get("inserts_lost"):
            print(f"    inserts lost: {diff['inserts_lost'][:5]}")
        if diff.get("dimensions_lost"):
            print(f"    dims lost: {diff['dimensions_lost'][:5]}")
        print()

    dump_json(REPORT_PATH, {"compared": len(comparisons), "drawings": comparisons})
    print(f"Fidelity report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
