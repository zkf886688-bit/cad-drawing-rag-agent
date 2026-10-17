"""Structured query intent for the CAD knowledge-base agent.

Rule-based classifier (no extra LLM call). CadRAG uses the result to pick
scope / tools; phrase tables remain the source of truth and act as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Scope = Literal["drawing", "corpus", "standard", "followup", "help"]
IntentKind = Literal[
    "search",
    "stats",
    "layers",
    "dimensions",
    "standard",
    "list",
    "general",
    "help",
]


@dataclass(frozen=True)
class QueryIntent:
    scope: Scope
    kind: IntentKind
    drawing: str | None = None
    clause_code: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "kind": self.kind,
            "drawing": self.drawing,
            "clause_code": self.clause_code,
            "notes": list(self.notes),
        }


# Clause codes from the teaching standards corpus (exact-match boost).
KNOWN_CLAUSE_CODES = (
    "AISC-C-命名",
    "AISC-C-用途",
    "AISC-L-命名",
    "ASTM-PIPE-壁厚",
    "GB/T 707-对照",
    "GB/T 706-书写",
    "CAD-VP-冻结",
    "CAD-DIM-覆盖文字",
    "DXF-$INSUNITS",
)

# Product / UI FAQ — must not be answered from drawing RAG chunks.
HELP_CAPABILITY_TERMS = (
    "你能做什么",
    "你会什么",
    "有什么功能",
    "支持什么功能",
    "怎么用",
    "如何使用",
    "可以做什么",
    "能干嘛",
    "能干什么",
)
HELP_UPLOAD_CUES = ("上传", "入库", "导入", "附加图纸", "加文件", "附加文件")
HELP_UPLOAD_CONTEXT = (
    "可以",
    "能不能",
    "能否",
    "支持",
    "吗",
    "么",
    "怎么",
    "如何",
    "新的",
    "新",
    "dwg",
    "dxf",
    "文件",
    "图纸",
    "+",
)


def _has_any(query: str, terms: tuple[str, ...]) -> bool:
    return any(term in query for term in terms)


def is_help_query(query: str) -> bool:
    """True for product/capability/upload FAQ questions (not drawing facts)."""
    text = (query or "").strip()
    if not text:
        return False
    if _has_any(text, HELP_CAPABILITY_TERMS):
        return True
    if _has_any(text, HELP_UPLOAD_CUES) and _has_any(text, HELP_UPLOAD_CONTEXT):
        return True
    return False


# Fixed product facts for help intent (not from drawing index).
HELP_ANSWER = """可以。点输入框左侧「+」选择 .dwg / .dxf，再发送即可自动入库；等状态显示「入库完成」后，就能对新图提问。

说明：
1. 支持格式：.dwg、.dxf
2. DWG 需本机安装 ODA File Converter；没有的话请直接传 DXF
3. 也可调用 POST /upload，再轮询 GET /ingest/{job_id}

本系统是图纸知识库问答（检索 + 可追溯依据），不是操控 AutoCAD 写回图纸。"""

HELP_EVIDENCE = (
    {
        "id": "help:upload",
        "kind": "help",
        "file_name": "(产品说明)",
        "drawing": None,
        "score": 1.0,
        "text": (
            "网页 /ui 输入区「+」可上传 .dwg/.dxf；发送后走 POST /upload 自动入库，"
            "轮询 /ingest/{job_id} 至 done 后即可问答。DWG 需 ODA，否则请传 DXF。"
        ),
    },
    {
        "id": "help:scope",
        "kind": "help",
        "file_name": "(产品说明)",
        "drawing": None,
        "score": 0.9,
        "text": (
            "本系统是 CAD 图纸知识库 RAG 问答（多轮对话 + 依据），"
            "不是 AutoCAD 命令写回或三维碰撞检测 Agent。"
        ),
    },
)


def classify_intent(
    query: str,
    *,
    corpus_scope: bool,
    standard_intent: bool,
    identified: str | None,
    session_drawing: str | None,
    is_followup: bool,
    list_intent: bool,
) -> QueryIntent:
    """Map existing CadRAG flags + light keyword cues into a structured intent."""
    notes: list[str] = []
    if is_help_query(query):
        return QueryIntent(
            scope="help",
            kind="help",
            drawing=None,
            notes=("product_faq",),
        )

    clause_code = None
    for code in KNOWN_CLAUSE_CODES:
        if code.lower() in query.lower() or code in query:
            clause_code = code
            notes.append(f"clause:{code}")
            break

    if corpus_scope:
        scope: Scope = "corpus"
    elif standard_intent or clause_code:
        scope = "standard"
    elif is_followup and session_drawing and not identified:
        scope = "followup"
    else:
        scope = "drawing"

    drawing = None if scope == "corpus" else (identified or session_drawing)

    if clause_code or standard_intent:
        kind: IntentKind = "standard"
    elif list_intent:
        kind = "list"
    elif _has_any(query, ("图层", "冻结", "layer", "layers")):
        kind = "layers"
    elif _has_any(query, ("尺寸", "标注", "DIMENSION", "肢厚", "梁长")):
        kind = "dimensions"
    elif _has_any(query, ("实体", "统计", "多少", "几个", "几张", "总数")):
        kind = "stats"
    elif scope in {"drawing", "followup", "corpus"}:
        kind = "search"
    else:
        kind = "general"

    if is_followup:
        notes.append("followup")
    return QueryIntent(
        scope=scope,
        kind=kind,
        drawing=drawing,
        clause_code=clause_code,
        notes=tuple(notes),
    )
