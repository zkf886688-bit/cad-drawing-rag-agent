"""Compare BM25 / dense (BGE) / hybrid / TF-IDF on the fixed question set.

Produces the numbers for requirement 5 ("所选模型性能如何"):
  - drawing hit rate (top answer's drawing matches expected)
  - text hit rate (expected substring appears in the answer)
  - recall@4 (expected drawing appears in the top-4 sources)
  - mean latency per query
  - index size

  py scripts/build_embeddings.py   # once
  py scripts/eval_embeddings.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import CadRAG, SessionState  # noqa: E402
from _common import dump_json  # noqa: E402

# eval_qa lives as a script, not a package module.
import importlib.util

_spec = importlib.util.spec_from_file_location("eval_qa", ROOT / "scripts" / "eval_qa.py")
_eval_qa = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eval_qa)
CASES = _eval_qa.CASES
NO_LOCK = _eval_qa.NO_LOCK

INDEX_DIR = ROOT / "datasets" / "index"
REPORT_PATH = INDEX_DIR / "embedding_eval.json"
TFIDF_PATH = INDEX_DIR / "tfidf.npz"


def evaluate_backend(backend: str) -> dict:
    # Keep generation offline so this script measures retrieval, not API quota.
    rag = CadRAG(backend=backend, generate="extractive")
    drawing_hits = 0
    text_hits = 0
    recall_hits = 0
    scored = 0
    latencies: list[float] = []
    rows: list[dict] = []

    for question, expect_drawing, expect_text in CASES:
        started = time.perf_counter()
        result = rag.answer(question, SessionState())
        latency = time.perf_counter() - started
        latencies.append(latency)

        drawing = result.get("drawing")
        answer = result.get("answer", "")
        source_drawings = {
            item.get("drawing") for item in result.get("sources") or [] if item.get("drawing")
        }

        if expect_drawing == NO_LOCK:
            drawing_ok = drawing is None
            recall_ok = True  # corpus questions are not scored on recall@drawing
        else:
            drawing_ok = expect_drawing is None or drawing == expect_drawing
            recall_ok = expect_drawing is None or expect_drawing in source_drawings

        text_ok = expect_text is None or expect_text.lower() in answer.lower()
        scored += 1
        drawing_hits += drawing_ok
        text_hits += text_ok
        recall_hits += recall_ok

        rows.append(
            {
                "question": question,
                "expect_drawing": expect_drawing,
                "got_drawing": drawing,
                "drawing_ok": drawing_ok,
                "text_ok": text_ok,
                "recall_ok": recall_ok,
                "latency_ms": round(latency * 1000, 1),
            }
        )

    return {
        "backend": backend,
        "n": scored,
        "drawing_accuracy": round(drawing_hits / scored, 4),
        "text_hit_rate": round(text_hits / scored, 4),
        "recall_at_4": round(recall_hits / scored, 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies) * 1000, 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000, 1),
        "cases": rows,
    }


def evaluate_tfidf() -> dict:
    """TF-IDF cosine ranking without the CadRAG answer wrapper.

    Drawing is taken from the top-1 chunk so the metric stays comparable.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not TFIDF_PATH.exists():
        return {"backend": "tfidf", "error": "missing tfidf.npz — run build_embeddings.py"}

    data = np.load(TFIDF_PATH)
    vectors = data["vectors"].astype(np.float32)
    idf = data["idf"].astype(np.float32)
    meta = json.loads((INDEX_DIR / "tfidf_meta.json").read_text(encoding="utf-8"))
    ids = [str(item) for item in meta["ids"]]
    vocabulary = [str(item) for item in meta["vocabulary"]]

    chunks = []
    with (INDEX_DIR / "chunks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    id_to_chunk = {chunk["id"]: chunk for chunk in chunks}

    # Rebuild an identical vectorizer from the saved vocabulary/idf so queries
    # land in the same space without pickling the whole sklearn object.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        vocabulary={term: index for index, term in enumerate(vocabulary)},
    )
    # idf_ must be set after a dummy fit on empty — sklearn needs the attribute.
    vectorizer.fit(["placeholder"])
    vectorizer.idf_ = idf

    drawing_hits = text_hits = recall_hits = scored = 0
    latencies: list[float] = []
    rows: list[dict] = []

    for question, expect_drawing, expect_text in CASES:
        started = time.perf_counter()
        query_vec = vectorizer.transform([question]).toarray().astype(np.float32)[0]
        norm = float(np.linalg.norm(query_vec)) or 1e-12
        query_vec = query_vec / norm
        scores = vectors @ query_vec
        order = np.argsort(-scores)[:4]
        hits = [id_to_chunk[ids[index]] for index in order if ids[index] in id_to_chunk]
        latency = time.perf_counter() - started
        latencies.append(latency)

        top = hits[0] if hits else {}
        drawing = top.get("drawing")
        answer = "\n".join(hit.get("text", "") for hit in hits)
        source_drawings = {hit.get("drawing") for hit in hits if hit.get("drawing")}

        if expect_drawing == NO_LOCK:
            # TF-IDF alone has no corpus-scope detector; score as drawing is None
            # only when top chunk is a corpus chunk (drawing is null).
            drawing_ok = drawing is None
            recall_ok = True
        else:
            drawing_ok = expect_drawing is None or drawing == expect_drawing
            recall_ok = expect_drawing is None or expect_drawing in source_drawings
        text_ok = expect_text is None or expect_text.lower() in answer.lower()

        scored += 1
        drawing_hits += drawing_ok
        text_hits += text_ok
        recall_hits += recall_ok
        rows.append(
            {
                "question": question,
                "expect_drawing": expect_drawing,
                "got_drawing": drawing,
                "drawing_ok": drawing_ok,
                "text_ok": text_ok,
                "recall_ok": recall_ok,
                "latency_ms": round(latency * 1000, 1),
            }
        )

    return {
        "backend": "tfidf",
        "n": scored,
        "drawing_accuracy": round(drawing_hits / scored, 4),
        "text_hit_rate": round(text_hits / scored, 4),
        "recall_at_4": round(recall_hits / scored, 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies) * 1000, 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000, 1),
        "cases": rows,
    }


def main() -> int:
    dense_meta = {}
    meta_path = INDEX_DIR / "embeddings_meta.json"
    if meta_path.exists():
        dense_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        print("Dense index missing. Run: py scripts/build_embeddings.py")
        return 2

    results = []
    for backend in ("bm25", "dense", "hybrid"):
        print(f"\n=== {backend} ===")
        result = evaluate_backend(backend)
        results.append(result)
        print(
            f"drawing={result['drawing_accuracy']:.0%}  "
            f"text={result['text_hit_rate']:.0%}  "
            f"recall@4={result['recall_at_4']:.0%}  "
            f"mean={result['mean_latency_ms']}ms"
        )

    print("\n=== tfidf ===")
    tfidf = evaluate_tfidf()
    results.append(tfidf)
    if "error" in tfidf:
        print(tfidf["error"])
    else:
        print(
            f"drawing={tfidf['drawing_accuracy']:.0%}  "
            f"text={tfidf['text_hit_rate']:.0%}  "
            f"recall@4={tfidf['recall_at_4']:.0%}  "
            f"mean={tfidf['mean_latency_ms']}ms"
        )

    report = {
        "model": dense_meta.get("model"),
        "dim": dense_meta.get("dim"),
        "chunk_count": dense_meta.get("chunk_count"),
        "dense_index_bytes": dense_meta.get("bytes"),
        "encode_seconds": dense_meta.get("encode_seconds"),
        "results": [
            {key: value for key, value in item.items() if key != "cases"}
            for item in results
        ],
        "details": results,
    }
    dump_json(REPORT_PATH, report)

    print("\n--- summary ---")
    print(f"{'backend':8} {'draw':>6} {'text':>6} {'r@4':>6} {'ms':>8}")
    for item in results:
        if "error" in item:
            continue
        print(
            f"{item['backend']:8} "
            f"{item['drawing_accuracy']:>6.0%} "
            f"{item['text_hit_rate']:>6.0%} "
            f"{item['recall_at_4']:>6.0%} "
            f"{item['mean_latency_ms']:>8.1f}"
        )
    print(f"\nReport: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
