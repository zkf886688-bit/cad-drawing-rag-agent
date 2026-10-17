"""Build the CAD knowledge graph (drawings × specs × standards)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph import GRAPH_PATH, build_graph  # noqa: E402
from build_standards import ensure_standards_corpus  # noqa: E402
from _common import dump_json  # noqa: E402


def build_and_save() -> dict:
    ensure_standards_corpus()
    graph = build_graph()
    dump_json(GRAPH_PATH, graph)
    stats = graph["stats"]
    print(
        f"[OK] graph: {stats['node_count']} nodes, {stats['edge_count']} edges "
        f"→ {GRAPH_PATH.name}"
    )
    print("     kinds:", stats["kinds"])
    return graph


def main() -> None:
    build_and_save()


if __name__ == "__main__":
    main()
