"""Read-only CAD knowledge tools with strict Pydantic parameter validation.

These are not AutoCAD write-back APIs. The agent may only query the indexed
knowledge base / final records. Invalid LLM-style args raise ValidationError.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "datasets" / "final"
STANDARDS_PATH = ROOT / "datasets" / "standards" / "engineering_standards.json"

TOOL_NAMES = (
    "search_chunks",
    "get_drawing_stats",
    "list_layers",
    "list_dimensions",
    "lookup_standard",
)


class SearchChunksArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    drawing: str | None = Field(default=None, max_length=200)
    top_k: int = Field(default=4, ge=1, le=20)


class DrawingArgs(BaseModel):
    drawing: str = Field(..., min_length=1, max_length=200)

    @field_validator("drawing")
    @classmethod
    def _strip(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("drawing must be non-empty")
        return text


class LookupStandardArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    clause_code: str | None = Field(default=None, max_length=80)


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "search_chunks": SearchChunksArgs,
    "get_drawing_stats": DrawingArgs,
    "list_layers": DrawingArgs,
    "list_dimensions": DrawingArgs,
    "lookup_standard": LookupStandardArgs,
}


def tool_catalog() -> list[dict[str, Any]]:
    """JSON-serialisable tool list for /tools and答辩材料."""
    return [
        {
            "name": name,
            "description": {
                "search_chunks": "在知识库索引中检索相关 chunk",
                "get_drawing_stats": "读取某张图的统计信息（实体/图层/文字等）",
                "list_layers": "列出图纸图层及冻结/关闭/锁定状态",
                "list_dimensions": "列出图纸 DIMENSION 尺寸标注",
                "lookup_standard": "检索工程标准/可引用条文",
            }[name],
            "parameters": TOOL_SCHEMAS[name].model_json_schema(),
            "readonly": True,
        }
        for name in TOOL_NAMES
    ]


def _load_final(drawing: str) -> dict | None:
    path = FINAL_DIR / f"{drawing}_final.json"
    if not path.is_file():
        # Allow file_name stem variants with spaces already in drawing id.
        matches = sorted(FINAL_DIR.glob(f"*_final.json"))
        for candidate in matches:
            stem = candidate.name.replace("_final.json", "")
            if stem == drawing or stem.lower() == drawing.lower():
                path = candidate
                break
        else:
            return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_hit(
    *,
    tool: str,
    text: str,
    drawing: str | None = None,
    file_name: str | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "id": f"tool::{tool}::{drawing or 'corpus'}::{abs(hash(text)) % 10_000_000}",
        "drawing": drawing,
        "file_name": file_name or (f"{drawing}.dxf" if drawing else "(工具)"),
        "kind": "tool",
        "text": text,
        "score": 1.0,
        "backend": "tool",
        "tool": tool,
    }
    if extra:
        payload.update(extra)
    return payload


class ToolRuntime:
    """Execute validated read-only tools against a CadRAG instance + final JSON."""

    def __init__(self, rag: Any) -> None:
        self.rag = rag

    def validate(self, name: str, arguments: dict) -> BaseModel:
        schema = TOOL_SCHEMAS.get(name)
        if schema is None:
            raise ValueError(f"unknown tool: {name}; allowed={list(TOOL_NAMES)}")
        return schema.model_validate(arguments)

    def call(self, name: str, arguments: dict) -> list[dict]:
        try:
            args = self.validate(name, arguments)
        except ValidationError as exc:
            return [
                _as_hit(
                    tool=name,
                    text=f"工具参数校验失败: {exc.errors()[:3]}",
                    extra={"ok": False},
                )
            ]
        handler: Callable[[BaseModel], list[dict]] = {
            "search_chunks": self._search_chunks,
            "get_drawing_stats": self._get_drawing_stats,
            "list_layers": self._list_layers,
            "list_dimensions": self._list_dimensions,
            "lookup_standard": self._lookup_standard,
        }[name]
        return handler(args)

    def _search_chunks(self, args: SearchChunksArgs) -> list[dict]:
        hits = self.rag.search(args.query, top_k=args.top_k, drawing=args.drawing)
        for hit in hits:
            hit["tool"] = "search_chunks"
        return hits

    def _get_drawing_stats(self, args: DrawingArgs) -> list[dict]:
        doc = _load_final(args.drawing)
        if not doc:
            return [
                _as_hit(
                    tool="get_drawing_stats",
                    drawing=args.drawing,
                    text=f"未找到图纸记录: {args.drawing}",
                    extra={"ok": False},
                )
            ]
        stats = doc.get("stats") or {}
        header = doc.get("header") or {}
        text = (
            f"{doc.get('file_name') or args.drawing} 统计: "
            f"实体{stats.get('entity_count')} 图层{stats.get('layer_count')} "
            f"矢量文字{stats.get('vector_text_count')} OCR文字{stats.get('ocr_text_count')} "
            f"尺寸{stats.get('dimension_count')} 单位{header.get('units') or stats.get('units')}"
        )
        return [
            _as_hit(
                tool="get_drawing_stats",
                drawing=args.drawing,
                file_name=doc.get("file_name"),
                text=text,
                extra={"ok": True, "stats": stats},
            )
        ]

    def _list_layers(self, args: DrawingArgs) -> list[dict]:
        doc = _load_final(args.drawing)
        if not doc:
            return [
                _as_hit(
                    tool="list_layers",
                    drawing=args.drawing,
                    text=f"未找到图纸记录: {args.drawing}",
                    extra={"ok": False},
                )
            ]
        layers = doc.get("layers") or []
        parts = []
        for item in layers:
            flags = [
                name
                for key, name in (("frozen", "冻结"), ("off", "关闭"), ("locked", "锁定"))
                if item.get(key)
            ]
            state = f"[{'/'.join(flags)}]" if flags else ""
            parts.append(f"{item.get('name')}{state}")
        text = f"{doc.get('file_name') or args.drawing} 图层列表: " + ", ".join(parts)
        frozen = [item.get("name") for item in layers if item.get("frozen")]
        if frozen:
            text += f"；被冻结的图层: {', '.join(frozen)}"
        return [
            _as_hit(
                tool="list_layers",
                drawing=args.drawing,
                file_name=doc.get("file_name"),
                text=text,
                extra={"ok": True, "layer_count": len(layers)},
            )
        ]

    def _list_dimensions(self, args: DrawingArgs) -> list[dict]:
        doc = _load_final(args.drawing)
        if not doc:
            return [
                _as_hit(
                    tool="list_dimensions",
                    drawing=args.drawing,
                    text=f"未找到图纸记录: {args.drawing}",
                    extra={"ok": False},
                )
            ]
        dims = doc.get("dimensions") or []
        if not dims:
            text = f"{doc.get('file_name') or args.drawing} 无 DIMENSION 尺寸标注"
        else:
            labels = []
            for item in dims[:30]:
                label = str(item.get("text") or "").strip()
                if not label or label in {"<>", " "}:
                    label = str(item.get("measurement"))
                labels.append(f"{label}@图层{item.get('layer')}")
            text = (
                f"{doc.get('file_name') or args.drawing} 尺寸标注(DIMENSION): "
                + ", ".join(labels)
            )
        return [
            _as_hit(
                tool="list_dimensions",
                drawing=args.drawing,
                file_name=doc.get("file_name"),
                text=text,
                extra={"ok": True, "dimension_count": len(dims)},
            )
        ]

    def _lookup_standard(self, args: LookupStandardArgs) -> list[dict]:
        # Prefer indexed standard/clause chunks via search, then exact clause code.
        hits = [
            hit
            for hit in self.rag.search(args.query, top_k=8, drawing=None)
            if hit.get("kind") in {"standard", "clause"}
        ]
        if args.clause_code:
            code = args.clause_code.lower()
            exact = [
                dict(chunk, score=1.5, backend="tool", tool="lookup_standard")
                for chunk in self.rag.chunks
                if chunk.get("kind") in {"standard", "clause"}
                and (
                    (chunk.get("clause_code") or "").lower() == code
                    or code in (chunk.get("text") or "").lower()
                )
            ]
            if exact:
                hits = exact + hits
        # Dedup
        seen: set[str] = set()
        out: list[dict] = []
        for hit in hits:
            hid = hit.get("id")
            if hid in seen:
                continue
            seen.add(hid)
            hit = dict(hit)
            hit["tool"] = "lookup_standard"
            out.append(hit)
            if len(out) >= 6:
                break
        if out:
            return out
        # Offline fallback from standards JSON
        if STANDARDS_PATH.exists():
            items = json.loads(STANDARDS_PATH.read_text(encoding="utf-8")).get(
                "items", []
            )
            q = args.query.lower()
            for item in items:
                blob = json.dumps(item, ensure_ascii=False).lower()
                if q in blob or (args.clause_code and args.clause_code.lower() in blob):
                    return [
                        _as_hit(
                            tool="lookup_standard",
                            text=(
                                f"工程标准【{item.get('title')}】"
                                f"（{item.get('jurisdiction')}）: {item.get('text')}"
                            ),
                            file_name=item.get("title"),
                            extra={"ok": True, "standard_id": item.get("id")},
                        )
                    ]
        return [
            _as_hit(
                tool="lookup_standard",
                text="未找到匹配的工程标准/条文",
                extra={"ok": False},
            )
        ]


def tools_for_intent(intent: Any, query: str) -> list[tuple[str, dict]]:
    """Deterministic tool plan from structured intent (no LLM planner required)."""
    plan: list[tuple[str, dict]] = []
    kind = getattr(intent, "kind", "search")
    drawing = getattr(intent, "drawing", None)
    clause = getattr(intent, "clause_code", None)

    if kind == "help":
        return []
    if kind == "standard" or clause:
        plan.append(
            (
                "lookup_standard",
                {"query": query, "clause_code": clause},
            )
        )
    if kind == "layers" and drawing:
        plan.append(("list_layers", {"drawing": drawing}))
    if kind == "dimensions" and drawing:
        plan.append(("list_dimensions", {"drawing": drawing}))
    if kind == "stats" and drawing:
        plan.append(("get_drawing_stats", {"drawing": drawing}))
    if kind in {"search", "list", "general"} or not plan:
        plan.append(
            (
                "search_chunks",
                {"query": query, "drawing": drawing, "top_k": 6},
            )
        )
    return plan
