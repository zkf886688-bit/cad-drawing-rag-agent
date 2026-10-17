"""CLI multi-turn chat demo for CAD Agent (Engineering Knowledge Base).

Usage:
  py scripts/chat_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.envfile import load_local_env
from app.rag import CadRAG, SessionState

load_local_env()


def main() -> None:
    from app.generate import resolve_mode

    rag = CadRAG()
    state = SessionState()
    print("CAD Agent (Engineering Knowledge Base) Chat Demo")
    print(f"retrieve={rag.backend}  generate={resolve_mode(rag.generate)}")
    print("Type a question. Commands: /reset  /quit")
    print("Examples:")
    print("  这张门图有哪些图层？")
    print("  型钢图是什么类型？")
    print("  文字测试图 OCR 看到了什么？")
    print("Set DASHSCOPE_API_KEY for 通义千问 or DEEPSEEK_API_KEY for DeepSeek.")
    print("-" * 40)

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            continue
        if q in {"/quit", "/exit", "quit", "exit"}:
            print("Bye.")
            break
        if q == "/reset":
            state = SessionState()
            print("Session reset.\n")
            continue

        result = rag.answer(q, state)
        print(
            f"\nBot> drawing={result['drawing']}  "
            f"generate={result.get('generate_mode')}"
        )
        print(result["answer"])
        print("-" * 40)


if __name__ == "__main__":
    main()
