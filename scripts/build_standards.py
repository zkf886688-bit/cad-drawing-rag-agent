"""Engineering standards corpus for the CAD knowledge base.

These are condensed reference notes (not full copyrighted standard text) that
link steel member types / drawing conventions to common GB and AISC practice,
so GraphRAG and RAG can answer "C15X50 对应什么标准/构件" style questions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _common import DATA, dump_json  # noqa: E402

STANDARDS_DIR = DATA / "standards"

# Curated teaching corpus: short factual notes for retrieval + graph edges.
# Each item may include `clauses` — citable sub-articles used as separate
# retrieval chunks and graph nodes (kind=clause).
STANDARDS: list[dict] = [
    {
        "id": "aisc-channel-c",
        "title": "AISC 标准槽钢 C 形截面",
        "jurisdiction": "AISC",
        "applies_to": ["槽钢", "C"],
        "prefixes": ["C", "MC"],
        "text": (
            "AISC 槽钢截面以 C 或 MC 前缀标识，例如 C15X50 表示名义深度约 15 in、"
            "单位长度重量约 50 lb/ft 的标准槽钢。常用于檩条、边缘梁与支撑。"
            "本知识库钢材样例图 Structural_Steel C 2D 即按 AISC 截面表绘制。"
        ),
        "clauses": [
            {
                "code": "AISC-C-命名",
                "title": "C 形截面命名",
                "text": "书写形式 C{深度}X{线密度}，深度单位英寸，线密度 lb/ft；MC 为杂项槽钢。",
                "source": "AISC Manual 教学摘要",
            },
            {
                "code": "AISC-C-用途",
                "title": "典型用途",
                "text": "檩条、边缘梁、支撑与楼梯斜梁等次结构构件。",
                "source": "AISC Manual 教学摘要",
            },
        ],
    },
    {
        "id": "aisc-angle-l",
        "title": "AISC 角钢 L 形截面",
        "jurisdiction": "AISC",
        "applies_to": ["角钢", "L"],
        "prefixes": ["L"],
        "text": (
            "AISC 角钢以 L 前缀标识，如 L4X4X1/2 表示肢长 4 in×4 in、肢厚 1/2 in。"
            "等边与不等边角钢广泛用于桁架腹杆、连接角钢与加劲肋。"
            "样例图 Structural_Steel L 2D 收录多种 L 规格。"
        ),
        "clauses": [
            {
                "code": "AISC-L-命名",
                "title": "角钢命名",
                "text": "L{肢长}X{肢长}X{肢厚}；不等边写成长肢×短肢×厚。",
                "source": "AISC Manual 教学摘要",
            }
        ],
    },
    {
        "id": "aisc-pipe",
        "title": "AISC/ASTM 圆管 Pipe",
        "jurisdiction": "AISC/ASTM",
        "applies_to": ["管材", "PIPE"],
        "prefixes": ["PIPE"],
        "text": (
            "圆管规格常写作 Pipe + 名义口径 + 壁厚等级（如 STD、XS、XXS），"
            "例如 Pipe10STD。用于柱、栏杆与工艺配管示意。"
            "样例图 Structural_Steel PIPE 2D 为圆管截面族。"
        ),
        "clauses": [
            {
                "code": "ASTM-PIPE-壁厚",
                "title": "壁厚等级",
                "text": "STD / XS / XXS 对应不同壁厚；名义口径不等于外径精确值。",
                "source": "ASTM/AISC 教学摘要",
            }
        ],
    },
    {
        "id": "aisc-hss",
        "title": "AISC 空心结构管 HSS",
        "jurisdiction": "AISC",
        "applies_to": ["方矩管", "HSS"],
        "prefixes": ["HSS"],
        "text": (
            "HSS（Hollow Structural Sections）为方形/矩形空心管，"
            "命名形如 HSS6X6X1/2。多用于柱与空间桁架。"
        ),
        "clauses": [],
    },
    {
        "id": "aisc-w-shape",
        "title": "AISC 宽翼缘工字钢 W",
        "jurisdiction": "AISC",
        "applies_to": ["宽翼缘工字钢", "W"],
        "prefixes": ["W", "S", "M", "HP", "WT", "MT", "ST"],
        "text": (
            "W 形（宽翼缘）与 S/M 形工字钢是北美建筑钢结构主梁柱常用截面；"
            "HP 用于桩基，WT/MT/ST 为剖分 T 型钢。"
        ),
        "clauses": [],
    },
    {
        "id": "gb-channel",
        "title": "GB/T 707 热轧槽钢（对照说明）",
        "jurisdiction": "GB",
        "applies_to": ["槽钢", "C"],
        "prefixes": ["C"],
        "text": (
            "中国热轧槽钢按 GB/T 707 等标准以号数命名（如 10#、16a），"
            "与 AISC 的 C 深度×重量 命名不同。跨标准对照时需查截面几何表，"
            "不可仅凭数字相等认为等价。"
        ),
        "clauses": [
            {
                "code": "GB/T 707-对照",
                "title": "与 AISC 对照原则",
                "text": "号数制与英制深度×重量不可直接互换；需按截面积、惯性矩等几何特性对照。",
                "source": "GB/T 707 教学摘要（非全文）",
            }
        ],
    },
    {
        "id": "gb-angle",
        "title": "GB/T 706 热轧角钢（对照说明）",
        "jurisdiction": "GB",
        "applies_to": ["角钢", "L"],
        "prefixes": ["L"],
        "text": (
            "中国热轧等边/不等边角钢按 GB/T 706，书写如 ∠50×50×5。"
            "与 AISC L4X4X1/2 对照时需换算英寸与毫米，并核对截面特性。"
        ),
        "clauses": [
            {
                "code": "GB/T 706-书写",
                "title": "角钢书写",
                "text": "等边∠a×a×t，不等边∠a×b×t；单位毫米。",
                "source": "GB/T 706 教学摘要（非全文）",
            }
        ],
    },
    {
        "id": "cad-layer-convention",
        "title": "CAD 图层与冻结约定",
        "jurisdiction": "制图惯例",
        "applies_to": ["图层", "视口", "冻结"],
        "prefixes": [],
        "text": (
            "工程 CAD 常用图层区分墙、结构、标注与参考。图层可冻结/关闭/锁定；"
            "图纸空间视口可单独冻结图层，用于同一布局展示不同专业内容。"
            "样例 viewport_frozen_layers 演示视口级图层冻结。"
        ),
        "clauses": [
            {
                "code": "CAD-VP-冻结",
                "title": "视口级冻结",
                "text": "同一图层可在模型空间显示、在某布局视口中冻结，不影响其他视口。",
                "source": "CAD 制图惯例摘要",
            }
        ],
    },
    {
        "id": "cad-block-wipeout",
        "title": "块引用与 Wipeout",
        "jurisdiction": "制图惯例",
        "applies_to": ["块", "门", "Wipeout"],
        "prefixes": [],
        "text": (
            "块（BLOCK）封装可重复构件；INSERT 为块引用。"
            "Wipeout 用于遮挡底层图形以表达洞口或门洞与墙厚叠压。"
            "样例 wipeout_door 演示门块与 Wipeout。"
        ),
        "clauses": [],
    },
    {
        "id": "cad-dimension",
        "title": "DIMENSION 尺寸标注",
        "jurisdiction": "制图惯例",
        "applies_to": ["尺寸", "标注", "DIMENSION"],
        "prefixes": [],
        "text": (
            "DIMENSION 实体携带测量值或覆盖文字，是尺寸类问答的主要来源。"
            "解析阶段读取 measurement / text；样例 example_r14 与角钢肢厚标注可供检索。"
        ),
        "clauses": [
            {
                "code": "CAD-DIM-覆盖文字",
                "title": "尺寸覆盖文字",
                "text": "DXF 中 text=<> 表示显示测量值；非空 text 为覆盖标注（如「梁长 6000」）。",
                "source": "CAD 制图惯例摘要",
            }
        ],
    },
    {
        "id": "cad-units",
        "title": "DXF 单位 $INSUNITS",
        "jurisdiction": "DXF",
        "applies_to": ["单位", "英寸", "米", "毫米"],
        "prefixes": [],
        "text": (
            "DXF 头变量 $INSUNITS 给出插入单位：1=英寸，4=毫米，6=米等。"
            "AISC 钢材样例多为英寸；表格/视口/文字测试样例多为米。"
            "回答尺寸前必须先确认单位，避免英制公制混用。"
        ),
        "clauses": [
            {
                "code": "DXF-$INSUNITS",
                "title": "$INSUNITS 常用值",
                "text": "0=无单位，1=英寸，4=毫米，6=米；与 $MEASUREMENT（0英制/1公制）一并解读。",
                "source": "DXF 参考摘要",
            }
        ],
    },
]


def ensure_standards_corpus() -> Path:
    STANDARDS_DIR.mkdir(parents=True, exist_ok=True)
    out = STANDARDS_DIR / "engineering_standards.json"
    dump_json(
        out,
        {
            "description": (
                "Condensed engineering/CAD reference notes for RAG + GraphRAG. "
                "clauses are citable teaching excerpts, not full copyrighted standards."
            ),
            "count": len(STANDARDS),
            "items": STANDARDS,
        },
    )
    return out


def load_standards() -> list[dict]:
    path = ensure_standards_corpus()
    return __import__("json").loads(path.read_text(encoding="utf-8")).get("items", [])


def standard_chunks() -> list[dict]:
    """Corpus-level chunks (drawing=None) for the retrieval index.

    Parent standard notes and child clause excerpts form a lightweight
    hierarchical RAG: clauses carry ``parent_id`` pointing at ``standard::<id>``.
    """
    chunks = []
    for item in load_standards():
        applies = "、".join(item.get("applies_to") or [])
        prefixes = "、".join(item.get("prefixes") or []) or "无"
        parent_id = f"standard::{item['id']}"
        text = (
            f"工程标准/惯例【{item['title']}】（{item['jurisdiction']}）"
            f"适用: {applies}；截面前缀: {prefixes}。{item['text']}"
        )
        clause_codes = [
            (clause.get("code") or "clause") for clause in (item.get("clauses") or [])
        ]
        if clause_codes:
            text += "；下属可引用条文: " + "、".join(clause_codes)
        chunks.append(
            {
                "id": parent_id,
                "drawing": None,
                "file_name": item["title"],
                "kind": "standard",
                "text": text,
                "standard_id": item["id"],
                "jurisdiction": item["jurisdiction"],
                "title": item["title"],
                "child_clause_codes": clause_codes,
            }
        )
        for clause in item.get("clauses") or []:
            code = clause.get("code") or "clause"
            cite = clause.get("source") or item["jurisdiction"]
            clause_text = (
                f"可引用条文【{code}】{clause.get('title') or ''} "
                f"（隶属 {item['title']} / {item['jurisdiction']}；来源: {cite}；"
                f"parent={parent_id}）。"
                f"{clause.get('text') or ''}"
            )
            chunks.append(
                {
                    "id": f"standard::{item['id']}::{code}",
                    "drawing": None,
                    "file_name": f"{item['title']} · {code}",
                    "kind": "clause",
                    "text": clause_text,
                    "standard_id": item["id"],
                    "clause_code": code,
                    "parent_id": parent_id,
                    "jurisdiction": item["jurisdiction"],
                    "title": clause.get("title") or code,
                    "citation": cite,
                }
            )
    return chunks

def main() -> None:
    path = ensure_standards_corpus()
    n_clauses = sum(len(item.get("clauses") or []) for item in STANDARDS)
    print(f"Wrote {len(STANDARDS)} standards + {n_clauses} clauses → {path}")


if __name__ == "__main__":
    main()
