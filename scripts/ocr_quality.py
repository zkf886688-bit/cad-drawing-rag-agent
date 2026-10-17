"""Sort OCR output into plausible annotations and likely misreads.

Rendering a drawing and running OCR over it recovers text that exists only as
line work, which vector parsing cannot see. The same pass also reports letters
for shapes that merely look like letters -- a channel profile reads as "L" or
"C" -- so a raw OCR count overstates what was actually recovered.

This script separates the two so the recovery rate can be quoted honestly.

  py scripts/ocr_quality.py
"""

from __future__ import annotations

from _common import DATA, dump_json, load_json

# The pipeline stage owns the definition; this report reuses it so the two can
# never drift apart and disagree about what counts as noise.
from merge_ocr import classify_text as classify

OCR_DIR = DATA / "ocr"
REPORT_PATH = DATA / "ocr" / "quality_report.json"


def main() -> None:
    files = sorted(OCR_DIR.glob("*_ocr.json"))
    if not files:
        print(f"No OCR output in {OCR_DIR}")
        return

    drawings = []
    for path in files:
        data = load_json(path)
        texts = data.get("texts", [])

        buckets: dict[str, list[dict]] = {}
        for item in texts:
            buckets.setdefault(classify(item.get("content", "")), []).append(item)

        def sample(kind: str, n: int = 8) -> list[str]:
            return [t["content"] for t in buckets.get(kind, [])[:n]]

        def mean_conf(kind: str) -> float | None:
            vals = [t["confidence"] for t in buckets.get(kind, [])]
            return round(sum(vals) / len(vals), 4) if vals else None

        spec_count = len(buckets.get("spec", []))
        entry = {
            "drawing": path.stem.removesuffix("_ocr"),
            "ocr_total": len(texts),
            "spec_like": spec_count,
            "single_char": len(buckets.get("single_char", [])),
            "other": len(buckets.get("other", [])),
            "spec_ratio": round(spec_count / len(texts), 4) if texts else None,
            "mean_confidence": {
                "spec": mean_conf("spec"),
                "single_char": mean_conf("single_char"),
                "other": mean_conf("other"),
            },
            "samples": {
                "spec": sample("spec"),
                "single_char": sample("single_char"),
                "other": sample("other"),
            },
        }
        drawings.append(entry)

        print(f"{entry['drawing']}")
        print(
            f"  total {entry['ocr_total']} | spec {spec_count} "
            f"| 1-char {entry['single_char']} | other {entry['other']}"
        )
        if entry["samples"]["spec"]:
            print(f"  spec:   {entry['samples']['spec']}")
        if entry["samples"]["single_char"]:
            print(f"  1-char: {entry['samples']['single_char']}")
        if entry["samples"]["other"]:
            print(f"  other:  {entry['samples']['other']}")
        print()

    dump_json(REPORT_PATH, {"drawings": drawings})
    print(f"Quality report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
