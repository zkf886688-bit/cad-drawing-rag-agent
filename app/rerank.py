"""Lightweight rerank over retrieval candidates (no extra model download).

Combines original rank score with lexical overlap and kind priors so hybrid
top-N can be trimmed to a sharper top-K for generation.
"""

from __future__ import annotations

import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z0-9_./\-]+|[\u4e00-\u9fff]", re.UNICODE)

KIND_PRIOR = {
    "clause": 1.25,
    "standard": 1.15,
    "steel_specs": 1.12,
    "steel_specs_all": 1.1,
    "dimensions": 1.1,
    "layers": 1.08,
    "tool": 1.05,
    "graph": 1.0,
    "summary": 0.95,
    "stats": 0.9,
}


def _tokens(text: str) -> Counter[str]:
    parts = WORD_RE.findall(text or "")
    out: list[str] = []
    chinese: list[str] = []

    def flush() -> None:
        if not chinese:
            return
        out.extend(chinese)
        for i in range(len(chinese) - 1):
            out.append(chinese[i] + chinese[i + 1])
        chinese.clear()

    for part in parts:
        if "\u4e00" <= part[0] <= "\u9fff":
            chinese.append(part)
        else:
            flush()
            out.append(part.lower())
    flush()
    return Counter(out)


def rerank_hits(
    query: str,
    hits: list[dict],
    *,
    top_k: int = 4,
    clause_code: str | None = None,
) -> list[dict]:
    if not hits:
        return []
    if len(hits) == 1:
        return hits[:top_k]

    q = _tokens(query)
    q_terms = set(q)
    ranked: list[tuple[float, int, dict]] = []

    for index, hit in enumerate(hits):
        text = hit.get("text") or ""
        doc = _tokens(text)
        overlap = sum(min(q[t], doc[t]) for t in q_terms if t in doc)
        coverage = overlap / max(sum(q.values()), 1)
        base = float(hit.get("score") or 0.0)
        # Normalise disparate backends roughly into a positive band.
        base_n = base if base <= 2.5 else 2.5
        prior = KIND_PRIOR.get(hit.get("kind") or "", 1.0)
        bonus = 0.0
        if clause_code:
            code = clause_code.lower()
            if (hit.get("clause_code") or "").lower() == code:
                bonus += 1.5
            elif code in text.lower():
                bonus += 0.8
        # Exact designation-style tokens (e.g. L4X4X1/2, C15X50)
        for term in q_terms:
            if len(term) >= 4 and term in text.lower():
                bonus += 0.35
                break
        score = (0.55 * base_n) + (0.35 * coverage) + bonus
        score *= prior
        ranked.append((score, index, hit))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    out: list[dict] = []
    for score, _, hit in ranked[:top_k]:
        item = dict(hit)
        item["rerank_score"] = round(score, 6)
        out.append(item)
    return out
