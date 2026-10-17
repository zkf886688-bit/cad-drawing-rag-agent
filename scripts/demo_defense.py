"""答辩 Demo：固定 5 类问题（单图 / 全库 / 标准 / 多轮 / 上传说明）。

离线可跑（默认 extractive + bm25），用于演示与截图：

  set CAD_GENERATE=extractive
  py scripts/demo_defense.py

可选：把问答摘要写到 docs/demo_transcript.md（不含密钥）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.envfile import load_local_env
from app.logutil import setup_logging
from app.rag import CadRAG, SessionState

load_local_env()
log = setup_logging("demo_defense")

# (标签, 问题列表, 期望片段或 None)
SCENARIOS: list[tuple[str, list[str], str | None]] = [
    (
        "1. 单图问答",
        ["门图里的块引用是什么"],
        "Wipeout",
    ),
    (
        "2. 全库范围",
        ["总共有几张图纸"],
        "张图纸",
    ),
    (
        "3. 工程标准 / 条文",
        ["C15X50 对应什么标准"],
        "AISC",
    ),
    (
        "4. 多轮锁定",
        ["角钢图有哪些规格型号", "它用的是什么单位"],
        "英寸",
    ),
    (
        "5. 上传入库（FAQ + 操作说明）",
        ["我上传新的dwg给你，你也可以做到吗"],
        "入库",
    ),
]

UPLOAD_TIP = """上传演示（浏览器）：
  1. 打开 http://127.0.0.1:8000/ui
  2. 点击附件，选择 datasets/raw 下任一张 .dxf（或自备 DWG）
  3. 发送任意一句（如「这张图有哪些图层」），等待入库完成
  4. 截图：入库状态条 + 回答依据栏预览

命令行等价：POST /upload → 轮询 /ingest/{job_id} → POST /chat
详见 README「加入新图纸」
"""


def _short(text: str, limit: int = 160) -> str:
    line = (text or "").split("——", 1)[0].strip().replace("\n", " ")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def run_scenarios(*, write_transcript: Path | None) -> int:
    rag = CadRAG(generate="extractive")
    state = SessionState()
    lines: list[str] = [
        "# 答辩 Demo 问答摘要",
        "",
        "由 `py scripts/demo_defense.py --write-transcript` 生成；默认 extractive，不含 API 密钥。",
        "",
    ]
    failed = 0

    print("=" * 56)
    print("CAD 知识库 · 答辩 Demo（extractive / bm25）")
    print("=" * 56)

    for title, questions, expect in SCENARIOS:
        print(f"\n## {title}")
        lines.append(f"## {title}")
        lines.append("")

        # Fresh session for multi-turn group only keeps history within the group.
        if title.startswith("4."):
            state = SessionState()

        for index, question in enumerate(questions):
            started = time.perf_counter()
            result = rag.answer(question, state)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            answer = result.get("answer") or ""
            intent = result.get("intent") or {}
            tools = result.get("tools_used") or []
            drawing = result.get("drawing")
            # Expectation applies to the last turn of the scenario only.
            check_expect = expect is not None and index == len(questions) - 1
            ok = True if not check_expect else expect in answer

            log.info(
                "demo q=%r drawing=%s intent=%s tools=%s ms=%s ok=%s",
                question,
                drawing,
                intent.get("kind"),
                tools,
                elapsed_ms,
                ok,
            )
            print(f"Q: {question}")
            print(f"A: {_short(answer)}")
            print(
                f"   drawing={drawing}  intent={intent.get('kind')}  "
                f"tools={tools}  {elapsed_ms}ms"
                + ("" if ok else f"  !! 未命中期望「{expect}」")
            )
            lines.append(f"- **Q:** {question}")
            lines.append(f"  - A: {_short(answer, 240)}")
            lines.append(
                f"  - drawing=`{drawing}` intent=`{intent.get('kind')}` "
                f"tools=`{tools}` {elapsed_ms}ms"
            )
            lines.append("")
            if not ok:
                failed += 1

        if title.startswith("5."):
            print(UPLOAD_TIP)
            lines.append("```")
            lines.append(UPLOAD_TIP.strip())
            lines.append("```")
            lines.append("")

    print("\n" + "=" * 56)
    if failed:
        print(f"Demo 完成，但有 {failed} 处未命中期望片段。")
    else:
        print("Demo 完成：5 类问答应答正常（含上传 FAQ）。")
    print("建议截图位置：/ui 对话区、右侧依据栏、侧栏状态、上传进度。")

    if write_transcript is not None:
        write_transcript.parent.mkdir(parents=True, exist_ok=True)
        write_transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已写入摘要: {write_transcript}")

    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-transcript",
        action="store_true",
        help="写入 docs/demo_transcript.md",
    )
    args = parser.parse_args()
    path = ROOT / "docs" / "demo_transcript.md" if args.write_transcript else None
    raise SystemExit(run_scenarios(write_transcript=path))


if __name__ == "__main__":
    main()
