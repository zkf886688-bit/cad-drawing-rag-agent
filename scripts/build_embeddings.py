"""Build a dense embedding index over chunks.jsonl (requirement 5).

Default model: BAAI/bge-small-zh-v1.5 (Chinese retrieval). First run downloads
the weights (~100MB). Also writes a TF-IDF baseline matrix for comparison.

  py scripts/build_embeddings.py
  py scripts/build_embeddings.py --model BAAI/bge-small-zh-v1.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import (  # noqa: E402
    BGE_QUERY_INSTRUCTION,
    EMBED_META_PATH,
    EMBED_PATH,
    INDEX_DIR,
    l2_normalize,
)
from _common import dump_json  # noqa: E402

CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
TFIDF_PATH = INDEX_DIR / "tfidf.npz"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def load_chunks(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dense(chunks: list[dict], model_name: str, batch_size: int) -> dict:
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    started = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_seconds = round(time.perf_counter() - started, 2)

    texts = [chunk.get("text", "") for chunk in chunks]
    print(f"Encoding {len(texts)} chunks (batch_size={batch_size})…")
    encode_started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    encode_seconds = round(time.perf_counter() - encode_started, 2)
    vectors = np.asarray(vectors, dtype=np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # ids stay in the meta JSON so np.load does not need allow_pickle.
    np.savez_compressed(EMBED_PATH, vectors=vectors)

    meta = {
        "model": model_name,
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "chunk_count": len(chunks),
        "dim": int(vectors.shape[1]),
        "ids": [chunk["id"] for chunk in chunks],
        "load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "bytes": EMBED_PATH.stat().st_size,
        "path": str(EMBED_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    dump_json(EMBED_META_PATH, meta)
    print(
        f"[OK] dense index: {vectors.shape[0]} x {vectors.shape[1]} "
        f"in {encode_seconds}s → {EMBED_PATH.name}"
    )
    return meta


def build_tfidf(chunks: list[dict]) -> dict:
    """Sparse TF-IDF vectors: a cheap second baseline for the comparison table."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [chunk.get("text", "") for chunk in chunks]
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(texts)
    # Store as dense float32 for a simple cosine matmul in the eval script;
    # 108 chunks x ~20k is fine. Normalize rows for cosine = dot product.
    dense = matrix.toarray().astype(np.float32)
    dense = l2_normalize(dense)
    encode_seconds = round(time.perf_counter() - started, 2)

    ids = [chunk["id"] for chunk in chunks]
    np.savez_compressed(TFIDF_PATH, vectors=dense, idf=vectorizer.idf_.astype(np.float32))
    meta = {
        "model": "sklearn-TfidfVectorizer(char_wb 2-4)",
        "chunk_count": len(chunks),
        "dim": int(dense.shape[1]),
        "ids": ids,
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
        "encode_seconds": encode_seconds,
        "bytes": TFIDF_PATH.stat().st_size,
        "path": str(TFIDF_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    dump_json(INDEX_DIR / "tfidf_meta.json", meta)
    print(
        f"[OK] tfidf index: {dense.shape[0]} x {dense.shape[1]} "
        f"in {encode_seconds}s → {TFIDF_PATH.name}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-tfidf", action="store_true")
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing {CHUNKS_PATH}. Run: py scripts/build_index.py")

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Chunks: {len(chunks)}")

    meta_path = EMBED_META_PATH
    if meta_path.exists() and not args.skip_dense:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old_count = old.get("chunk_count")
        if old_count is not None and old_count != len(chunks):
            print(
                f"Note: previous dense index had {old_count} chunks; "
                f"rebuilding for {len(chunks)}."
            )

    if not args.skip_dense:
        build_dense(chunks, args.model, args.batch_size)
    if not args.skip_tfidf:
        build_tfidf(chunks)

    print("Done. Next: py scripts/eval_embeddings.py")


if __name__ == "__main__":
    main()
