"""Dense embedding index for CAD chunk retrieval (requirement 5).

Built by scripts/build_embeddings.py. CadRAG can search with backend=
  bm25 | dense | hybrid
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "datasets" / "index"
EMBED_PATH = INDEX_DIR / "embeddings.npz"
EMBED_META_PATH = INDEX_DIR / "embeddings_meta.json"

# BGE retrieval queries need this instruction; passages do not.
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索："

# Small in-process LRU for repeated demo questions (avoids FAISS for tiny corpora).
QUERY_CACHE_SIZE = int(os.environ.get("CAD_EMBED_QUERY_CACHE", "64"))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


class EmbeddingIndex:
    """Matrix of chunk vectors + the sentence-transformers model that made them."""

    def __init__(self, path: Path = EMBED_PATH, meta_path: Path = EMBED_META_PATH) -> None:
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Embedding index missing. Run: py scripts/build_embeddings.py"
            )
        data = np.load(path)
        self.vectors = data["vectors"].astype(np.float32)
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.ids = [str(item) for item in self.meta["ids"]]
        self.model_name = self.meta.get("model", "")
        self._model = None
        self._id_to_row = {chunk_id: index for index, chunk_id in enumerate(self.ids)}
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()
        if len(self.ids) != len(self.vectors):
            raise RuntimeError(
                f"Embedding id count {len(self.ids)} != vector rows {len(self.vectors)}"
            )

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        key = query.strip()
        if QUERY_CACHE_SIZE > 0:
            with self._cache_lock:
                cached = self._query_cache.get(key)
                if cached is not None:
                    self._query_cache.move_to_end(key)
                    return cached.copy()

        model = self._load_model()
        text = BGE_QUERY_INSTRUCTION + query
        vector = model.encode([text], normalize_embeddings=True)
        encoded = np.asarray(vector, dtype=np.float32)[0]

        if QUERY_CACHE_SIZE > 0:
            with self._cache_lock:
                self._query_cache[key] = encoded
                self._query_cache.move_to_end(key)
                while len(self._query_cache) > QUERY_CACHE_SIZE:
                    self._query_cache.popitem(last=False)
        return encoded.copy()

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity to every chunk (vectors are L2-normalised)."""
        query_vec = self.encode_query(query)
        return self.vectors @ query_vec

    def top_indices(self, query: str, top_k: int = 4) -> list[tuple[int, float]]:
        scores = self.scores(query)
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            # argpartition is cheaper than full sort for small top_k.
            part = np.argpartition(-scores, top_k)[:top_k]
            order = part[np.argsort(-scores[part])]
        return [(int(index), float(scores[index])) for index in order]
