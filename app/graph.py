"""Lightweight knowledge graph over drawings, steel specs and standards.

Not a full GraphRAG stack: nodes/edges are materialised to JSON and queried by
entity matching + multi-hop expansion (default 3), then fused into CadRAG hits.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "datasets" / "index"
GRAPH_PATH = INDEX_DIR / "knowledge_graph.json"
FINAL_DIR = ROOT / "datasets" / "final"
STANDARDS_PATH = ROOT / "datasets" / "standards" / "engineering_standards.json"

WORD_RE = re.compile(r"[A-Za-z0-9_/.\-]+|[\u4e00-\u9fff]+", re.UNICODE)
DEFAULT_GRAPH_HOPS = int(os.environ.get("CAD_GRAPH_HOPS", "3"))
MAX_GRAPH_EDGES = int(os.environ.get("CAD_GRAPH_MAX_EDGES", "40"))

# Prefer paths that end at standards / citable clauses when ranking facts.
PRIORITY_KINDS = {"clause": 0, "standard": 1, "spec": 2, "member_type": 3}


def _node_id(kind: str, key: str) -> str:
    return f"{kind}:{key}"


def build_graph() -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(kind: str, key: str, **props: object) -> str:
        nid = _node_id(kind, key)
        if nid not in nodes:
            nodes[nid] = {"id": nid, "kind": kind, "key": key, **props}
        else:
            nodes[nid].update({k: v for k, v in props.items() if v is not None})
        return nid

    def add_edge(src: str, rel: str, dst: str, **props: object) -> None:
        edges.append({"source": src, "rel": rel, "target": dst, **props})

    standards: list[dict] = []
    if STANDARDS_PATH.exists():
        standards = json.loads(STANDARDS_PATH.read_text(encoding="utf-8")).get(
            "items", []
        )

    std_by_prefix: dict[str, list[str]] = defaultdict(list)
    std_by_apply: dict[str, list[str]] = defaultdict(list)
    for item in standards:
        sid = add_node(
            "standard",
            item["id"],
            title=item.get("title"),
            jurisdiction=item.get("jurisdiction"),
            text=item.get("text"),
        )
        for prefix in item.get("prefixes") or []:
            std_by_prefix[prefix.upper()].append(sid)
        for name in item.get("applies_to") or []:
            std_by_apply[name].append(sid)
            mid = add_node("member_type", name, name=name)
            add_edge(mid, "GOVERNED_BY", sid)
        for clause in item.get("clauses") or []:
            code = clause.get("code") or "clause"
            cid = add_node(
                "clause",
                f"{item['id']}/{code}",
                code=code,
                title=clause.get("title"),
                text=clause.get("text"),
                citation=clause.get("source") or item.get("jurisdiction"),
                standard_id=item["id"],
            )
            add_edge(cid, "CLAUSE_OF", sid)
            add_edge(sid, "HAS_CLAUSE", cid)
            # Cite clauses from steel member types only — avoid wiring every
            # CAD convention label (图层/单位/尺寸…) to clauses, which drowns
            # ordinary drawing questions in CITED_BY facts.
            if item.get("prefixes"):
                for name in item.get("applies_to") or []:
                    mid = add_node("member_type", name, name=name)
                    add_edge(mid, "CITED_BY", cid)

    for path in sorted(FINAL_DIR.glob("*_final.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        stem = path.name.replace("_final.json", "")
        file_name = doc.get("file_name") or f"{stem}.dxf"
        drawing_type = (doc.get("vlm") or {}).get("drawing_type") or ""
        did = add_node(
            "drawing",
            stem,
            file_name=file_name,
            drawing_type=drawing_type,
            units=(doc.get("header") or {}).get("units"),
        )

        for layer in doc.get("layers") or []:
            name = layer.get("name") or ""
            if not name:
                continue
            lid = add_node(
                "layer",
                f"{stem}/{name}",
                name=name,
                frozen=bool(layer.get("frozen")),
                drawing=stem,
            )
            add_edge(did, "HAS_LAYER", lid)

        for block in doc.get("blocks") or []:
            name = block.get("name") or ""
            if not name:
                continue
            bid = add_node("block", f"{stem}/{name}", name=name, drawing=stem)
            add_edge(did, "DEFINES_BLOCK", bid)

        specs = doc.get("steel_specs") or {}
        for member in specs.get("member_types") or []:
            mname = member.get("name") or "型钢"
            mid = add_node(
                "member_type",
                mname,
                name=mname,
                prefix=member.get("prefix"),
                note=member.get("note"),
            )
            add_edge(did, "SHOWS_MEMBER", mid)
            prefix = (member.get("prefix") or "").upper()
            for sid in std_by_prefix.get(prefix, []):
                add_edge(mid, "GOVERNED_BY", sid)
            for sid in std_by_apply.get(mname, []):
                add_edge(mid, "GOVERNED_BY", sid)

        for designation in specs.get("designations") or []:
            spid = add_node("spec", designation, designation=designation)
            add_edge(spid, "APPEARS_IN", did)
            for member in specs.get("member_types") or []:
                prefix = (member.get("prefix") or "").upper()
                if designation.upper().startswith(prefix):
                    mid = _node_id("member_type", member.get("name") or "型钢")
                    add_edge(spid, "IS_TYPE", mid)
                    for sid in std_by_prefix.get(prefix, []):
                        add_edge(spid, "GOVERNED_BY", sid)

        if (doc.get("stats") or {}).get("dimension_count"):
            for sid in std_by_apply.get("尺寸", []):
                add_edge(did, "USES_CONVENTION", sid)
            for sid in std_by_apply.get("标注", []):
                add_edge(did, "USES_CONVENTION", sid)

        if (doc.get("stats") or {}).get("viewport_count"):
            for sid in std_by_apply.get("视口", []):
                add_edge(did, "USES_CONVENTION", sid)

    seen = set()
    unique_edges = []
    for edge in edges:
        key = (edge["source"], edge["rel"], edge["target"])
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(edge)

    return {
        "nodes": list(nodes.values()),
        "edges": unique_edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(unique_edges),
            "kinds": {
                kind: sum(1 for n in nodes.values() if n["kind"] == kind)
                for kind in sorted({n["kind"] for n in nodes.values()})
            },
        },
    }


class KnowledgeGraph:
    def __init__(self, path: Path = GRAPH_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Knowledge graph missing. Run: py scripts/build_graph.py"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.nodes = {n["id"]: n for n in data.get("nodes") or []}
        self.edges = data.get("edges") or []
        self.out: dict[str, list[dict]] = defaultdict(list)
        self.inn: dict[str, list[dict]] = defaultdict(list)
        for edge in self.edges:
            self.out[edge["source"]].append(edge)
            self.inn[edge["target"]].append(edge)
        self._aliases = self._build_aliases()

    def _build_aliases(self) -> dict[str, set[str]]:
        aliases: dict[str, set[str]] = defaultdict(set)
        for node in self.nodes.values():
            keys = {
                str(node.get("key") or "").lower(),
                str(node.get("name") or "").lower(),
                str(node.get("title") or "").lower(),
                str(node.get("designation") or "").lower(),
                str(node.get("file_name") or "").lower(),
                str(node.get("drawing_type") or "").lower(),
                str(node.get("code") or "").lower(),
                str(node.get("citation") or "").lower(),
            }
            for key in keys:
                if key:
                    aliases[key].add(node["id"])
            for token in WORD_RE.findall(str(node.get("name") or "")):
                if len(token) >= 2:
                    aliases[token.lower()].add(node["id"])
            for token in WORD_RE.findall(str(node.get("code") or "")):
                if len(token) >= 2:
                    aliases[token.lower()].add(node["id"])
        return aliases

    def match_nodes(self, query: str) -> list[str]:
        q = query.lower()
        hits: set[str] = set()
        for alias, ids in sorted(self._aliases.items(), key=lambda x: -len(x[0])):
            if len(alias) >= 2 and alias in q:
                hits.update(ids)
        for token in WORD_RE.findall(query):
            ids = self._aliases.get(token.lower())
            if ids:
                hits.update(ids)
        return list(hits)

    def expand(self, node_ids: list[str], hops: int | None = None) -> list[dict]:
        hops = DEFAULT_GRAPH_HOPS if hops is None else hops
        frontier = set(node_ids)
        seen_edges: list[dict] = []
        visited = set(frontier)
        for _ in range(hops):
            nxt: set[str] = set()
            for nid in frontier:
                for edge in self.out.get(nid, []) + self.inn.get(nid, []):
                    seen_edges.append(edge)
                    other = edge["target"] if edge["source"] == nid else edge["source"]
                    if other not in visited:
                        visited.add(other)
                        nxt.add(other)
            frontier = nxt
            if not frontier:
                break
            if len(seen_edges) >= MAX_GRAPH_EDGES * 3:
                break
        uniq = []
        keys = set()
        for edge in seen_edges:
            key = (edge["source"], edge["rel"], edge["target"])
            if key in keys:
                continue
            keys.add(key)
            uniq.append(edge)
            if len(uniq) >= MAX_GRAPH_EDGES:
                break
        return uniq

    def _edge_priority(self, edge: dict) -> tuple[int, int]:
        src = self.nodes.get(edge["source"], {})
        dst = self.nodes.get(edge["target"], {})
        kind_score = min(
            PRIORITY_KINDS.get(src.get("kind"), 9),
            PRIORITY_KINDS.get(dst.get("kind"), 9),
        )
        rel_bonus = (
            0
            if edge.get("rel")
            in {"GOVERNED_BY", "CLAUSE_OF", "HAS_CLAUSE", "CITED_BY"}
            else 1
        )
        return (kind_score, rel_bonus)

    def facts_for_query(
        self, query: str, limit: int = 6, hops: int | None = None
    ) -> list[dict]:
        """Return retrievable fact chunks derived from graph neighbourhood."""
        seeds = self.match_nodes(query)
        if not seeds:
            return []
        edges = sorted(self.expand(seeds, hops=hops), key=self._edge_priority)
        facts: list[dict] = []
        for edge in edges:
            src = self.nodes.get(edge["source"], {})
            dst = self.nodes.get(edge["target"], {})
            text = (
                f"知识图谱: {src.get('kind')}【{src.get('key')}】"
                f" —{edge['rel']}→ "
                f"{dst.get('kind')}【{dst.get('key')}】"
            )
            for node in (src, dst):
                if node.get("kind") == "standard":
                    text += f"。标准要点: {(node.get('text') or '')[:160]}"
                if node.get("kind") == "clause":
                    cite = node.get("citation") or node.get("code")
                    text += (
                        f"。可引用条文 {node.get('code')}: "
                        f"{(node.get('text') or '')[:140]}"
                        f"（来源: {cite}）"
                    )
            drawing = None
            if src.get("kind") == "drawing":
                drawing = src.get("key")
            elif dst.get("kind") == "drawing":
                drawing = dst.get("key")
            elif src.get("drawing"):
                drawing = src.get("drawing")
            elif dst.get("drawing"):
                drawing = dst.get("drawing")
            facts.append(
                {
                    "id": f"graph::{edge['source']}::{edge['rel']}::{edge['target']}",
                    "drawing": drawing,
                    "file_name": "(知识图谱)",
                    "kind": "graph",
                    "text": text,
                    "score": 1.0,
                    "backend": "graph",
                }
            )
            if len(facts) >= limit:
                break
        return facts
