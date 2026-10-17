"""Validate the pipeline's JSON against the schemas.

A schema that nothing checks is just a comment: this script is what makes
schemas/*.json binding, and it exits non-zero so a broken change is noticed.

  py scripts/validate_json.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import FINAL_DIR, INDEX_DIR, JSON_DIR, MERGED_DIR, VLM_DIR, load_json

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
CHUNK_PATH = INDEX_DIR / "chunks.jsonl"

# Chunk records are checked here rather than with a schema: the interesting
# properties are relational (ids unique, drawing ids known), which JSON Schema
# cannot express across records.
REQUIRED_CHUNK_KEYS = ("id", "drawing", "file_name", "kind", "text")


def check_stage(name: str, files: list[Path], schema: dict) -> list[str]:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    problems: list[str] = []

    for path in files:
        errors = sorted(validator.iter_errors(load_json(path)), key=lambda e: e.path)
        if not errors:
            print(f"  [OK] {path.name}")
            continue
        print(f"  [FAIL] {path.name}: {len(errors)} problem(s)")
        for error in errors[:5]:
            location = "/".join(str(part) for part in error.absolute_path) or "(root)"
            print(f"         {location}: {error.message[:160]}")
            problems.append(f"{name}/{path.name}: {location}: {error.message[:160]}")
    return problems


def check_chunks() -> list[str]:
    if not CHUNK_PATH.exists():
        return [f"missing index: {CHUNK_PATH.name} (run scripts/build_index.py)"]

    problems: list[str] = []
    seen: set[str] = set()
    drawings = {path.name.replace("_final.json", "") for path in FINAL_DIR.glob("*_final.json")}
    kinds: dict[str, int] = {}

    with CHUNK_PATH.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            missing = [key for key in REQUIRED_CHUNK_KEYS if key not in record]
            if missing:
                problems.append(f"chunk line {number}: missing keys {missing}")
                continue
            if record["id"] in seen:
                problems.append(f"chunk line {number}: duplicate id {record['id']}")
            seen.add(record["id"])
            if not record["text"].strip():
                problems.append(f"chunk line {number}: empty text ({record['id']})")
            # A None drawing marks a corpus-level chunk; anything else must name
            # a drawing that actually exists, or filtering silently drops it.
            if record["drawing"] is not None and record["drawing"] not in drawings:
                problems.append(
                    f"chunk line {number}: unknown drawing {record['drawing']!r}"
                )
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1

    print(f"  {len(seen)} chunks, kinds: " + ", ".join(
        f"{kind}={count}" for kind, count in sorted(kinds.items())
    ))
    if "corpus" not in kinds:
        problems.append("no corpus-level chunk: cross-drawing questions cannot be answered")
    return problems


def main() -> int:
    try:
        import jsonschema  # noqa: F401
    except ModuleNotFoundError:
        print("jsonschema is not installed. Run: py -m pip install jsonschema")
        return 2

    problems: list[str] = []

    print("stage 1 (datasets/json vs schemas/drawing.schema.json):")
    problems += check_stage(
        "drawing",
        sorted(JSON_DIR.glob("*.json")),
        load_json(SCHEMA_DIR / "drawing.schema.json"),
    )

    print("\nstage 2 (datasets/merged vs schemas/merged.schema.json):")
    problems += check_stage(
        "merged",
        sorted(MERGED_DIR.glob("*_merged.json")),
        load_json(SCHEMA_DIR / "merged.schema.json"),
    )

    print("\nstage 3 (datasets/vlm vs schemas/vlm.schema.json):")
    problems += check_stage(
        "vlm",
        sorted(VLM_DIR.glob("*_vlm.json")),
        load_json(SCHEMA_DIR / "vlm.schema.json"),
    )

    print("\nstage 4 (datasets/final vs schemas/final.schema.json):")
    problems += check_stage(
        "final",
        sorted(FINAL_DIR.glob("*_final.json")),
        load_json(SCHEMA_DIR / "final.schema.json"),
    )

    print("\nindex (datasets/index/chunks.jsonl):")
    problems += check_chunks()

    print()
    if problems:
        print(f"FAILED: {len(problems)} problem(s)")
        return 1
    print("All JSON conforms to the schemas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
