"""End-to-end consistency checks over the pipeline's outputs.

Complements the two other checkers instead of repeating them:
`validate_json.py` checks each file's shape, `eval_qa.py` checks retrieval
quality, and this one checks that the stages agree with each other -- that no
drawing was silently dropped between stages and that numbers carried forward
still match their source. Exits non-zero on any failure.

  py scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _common import FINAL_DIR, JSON_DIR, MERGED_DIR, OCR_DIR, VLM_DIR, load_json

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  [{'OK' if condition else 'FAIL'}] {message}")
    if not condition:
        FAILURES.append(message)


def stems(directory: Path, suffix: str) -> set[str]:
    return {
        path.name[: -len(suffix)]
        for path in directory.glob(f"*{suffix}")
        if path.name.endswith(suffix)
    }


def check_stage_coverage() -> None:
    print("stage coverage:")
    vector = stems(JSON_DIR, ".json")
    ocr = stems(OCR_DIR, "_ocr.json")
    merged = stems(MERGED_DIR, "_merged.json")
    vlm = stems(VLM_DIR, "_vlm.json")
    final = stems(FINAL_DIR, "_final.json")

    check(bool(vector), f"stage 1 produced {len(vector)} records")
    check(ocr == vector, f"stage 2 covers every drawing (missing: {sorted(vector - ocr)})")
    check(merged == vector, f"stage 2b merged every drawing (missing: {sorted(vector - merged)})")
    check(vlm == vector, f"stage 3 covers every drawing (missing: {sorted(vector - vlm)})")
    check(final == vector, f"stage 4 covers every drawing (missing: {sorted(vector - final)})")


def check_carried_numbers() -> None:
    print("\nnumbers carried between stages:")
    for path in sorted(FINAL_DIR.glob("*_final.json")):
        stem = path.name[: -len("_final.json")]
        final = load_json(path)
        vector = load_json(JSON_DIR / f"{stem}.json")

        check(
            final["stats"]["entity_count"] == vector["stats"]["entity_count"],
            f"{stem}: entity_count matches stage 1 "
            f"({final['stats']['entity_count']})",
        )
        check(
            len(final["entities"]) == vector["stats"]["entity_count"],
            f"{stem}: entity list length matches its own count",
        )
        check(
            final["stats"]["linked_text_count"] <= final["stats"]["ocr_text_count"],
            f"{stem}: linked OCR ({final['stats']['linked_text_count']}) "
            f"<= total OCR ({final['stats']['ocr_text_count']})",
        )
        check(
            final["steel_specs"]["designation_count"]
            == len(final["steel_specs"]["designations"]),
            f"{stem}: designation_count matches the list",
        )
        check(
            all(item.get("text_class") for item in final["ocr_texts"]),
            f"{stem}: every OCR reading is classified",
        )
        check(
            bool(final["sources"]),
            f"{stem}: provenance recorded ({len(final['sources'])} sources)",
        )


def check_retrieval() -> None:
    print("\nretrieval API:")
    from app.main import app, get_session
    from app.rag import CadRAG, SessionState

    routes = {route.path for route in app.routes}
    check(
        {"/ui", "/health", "/chat"} <= routes,
        f"routes present: {sorted(routes)}",
    )

    rag = CadRAG(generate="extractive")
    check(len(rag.chunks) > 0, f"index loaded ({len(rag.chunks)} chunks)")
    check(
        any(chunk.get("drawing") is None for chunk in rag.chunks),
        "corpus-level chunk present",
    )

    state = SessionState()
    result = rag.answer("门图有哪些图层", state)
    check(result["drawing"] == "wipeout_door", f"drawing resolved: {result['drawing']}")
    check(bool(result["sources"]), "sources returned")
    check(
        result.get("generate_mode") == "extractive",
        f"generate_mode={result.get('generate_mode')}",
    )
    check("—— 依据" in result.get("answer", ""), "answer keeps evidence section")

    follow_up = rag.answer("它有几个块", state)
    check(follow_up["drawing"] == "wipeout_door", "follow-up keeps the same drawing")

    corpus = rag.answer("总共有几张图纸", state)
    check(corpus["drawing"] is None, "corpus question does not claim one drawing")
    check(state.drawing == "wipeout_door", "corpus question preserves the session subject")

    empty = rag.answer("完全无关的内容 zzzz", SessionState())
    check(isinstance(empty["answer"], str) and empty["answer"], "unknown question still answers")

    # Persistence is SQLite-backed. Keep this smoke check read-only so running
    # the suite never pollutes or evicts a developer's real browser sessions.
    from app import store as db
    from app.main import MAX_SESSIONS

    count = db.session_count()
    check(count <= MAX_SESSIONS, f"session store bounded ({count})")


def main() -> int:
    check_stage_coverage()
    check_carried_numbers()
    check_retrieval()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for message in FAILURES:
            print(f"  - {message}")
        return 1
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
