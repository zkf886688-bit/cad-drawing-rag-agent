"""Fixed question set for checking retrieval after pipeline changes.

Running the same questions before and after a change turns "seems better" into
something checkable.

  py scripts/eval_qa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import CadRAG, SessionState

# Expected drawing for a question that must not settle on any single drawing.
NO_LOCK = "<no single drawing>"

# (question, drawing the answer should come from, string the answer must contain)
CASES: list[tuple[str, str | None, str | None]] = [
    ("C 型槽钢图上有哪些规格", "Structural_Steel C 2D", "C15X50"),
    ("角钢图有哪些规格型号", "Structural_Steel L 2D", "角钢"),
    ("管材图有哪些型号", "Structural_Steel PIPE 2D", "Pipe10STD"),
    ("Structural_Steel C 2D 有哪些图层", "Structural_Steel C 2D", "AISC_STEEL"),
    # Asked about "型钢" all three steel sheets qualify, so picking one was only
    # ever an accident of ranking. Name the member to make the question answerable.
    ("槽钢图一共有多少个实体", "Structural_Steel C 2D", None),
    ("门图里的块引用是什么", "wipeout_door", "Wipeout"),
    ("哪张图有被冻结的图层", "viewport_frozen_layers", "frozen"),
    ("文字测试图的倾斜角度是多少", "text_oblique_rotate", "OBLIQUE"),
    # A corpus-wide question has no single subject; answering as though it did
    # is what used to poison the following turns.
    ("总共有几张图纸", NO_LOCK, None),
    ("哪张图的实体最多", NO_LOCK, "Structural_Steel L 2D"),
    ("哪些图纸有尺寸标注", NO_LOCK, "DIMENSION"),
    # Was previously mis-routed to the table sample: 规格 and 表格 share the
    # character 格, which was being treated as a drawing alias.
    ("L4X4X1/2 是什么规格", "Structural_Steel L 2D", "L4X4X1/2"),
    # A designation alone must identify its drawing, with no other clue given.
    ("C15X50 是什么构件", "Structural_Steel C 2D", "C15X50"),
    ("C15X50 对应什么标准", "Structural_Steel C 2D", "AISC"),
    ("Pipe10STD 出现在哪张图", "Structural_Steel PIPE 2D", "Pipe10STD"),
    ("角钢图的全部规格型号列出来", "Structural_Steel L 2D", "L4X4X1/2"),
    # Answerable only from the parsed header and paper-space layout.
    ("表格图的单位是什么", "acad_table_with_blk_ref", "米"),
    ("角钢图用的是什么单位", "Structural_Steel L 2D", "英寸"),
    ("表格里有几行内容", "acad_table_with_blk_ref", "10 行"),
    ("视口测试图的布局里有几个视口", "viewport_frozen_layers", "4 个视口"),
    ("视口里冻结了哪些图层", "viewport_frozen_layers", "Layer1"),
    # Closes the former library-wide DIMENSION gap (LibreDWG example_r14).
    ("example_r14 有哪些尺寸标注", "example_r14", "尺寸标注"),
    ("example_r14 尺寸里有没有 288", "example_r14", "288"),
    # Angle-steel leg thickness: DIMENSION entities added on Structural_Steel L 2D.
    ("角钢的肢厚尺寸标注是多少", "Structural_Steel L 2D", "肢厚"),
    ("角钢肢厚有没有 12.7mm", "Structural_Steel L 2D", "12.7mm"),
    # Project-like demo sheet: title block / 图框 / DIMENSION / units / layers.
    ("工程示意平面图有哪些图层", "demo_engineering_sheet", "TITLE"),
    ("工程示意平面图用的是什么单位", "demo_engineering_sheet", "毫米"),
    ("工程示意平面图的图号是什么", "demo_engineering_sheet", "DEMO-ENG-001"),
    ("工程示意平面图梁长尺寸是多少", "demo_engineering_sheet", "6000"),
    ("工程示意平面图有没有尺寸标注", "demo_engineering_sheet", "DIMENSION"),
    # Table cells / title text.
    ("表格图标题是什么", "acad_table_with_blk_ref", "Title"),
    ("表格里有没有 merge", "acad_table_with_blk_ref", "merge"),
    # Citable standards clauses (teaching excerpts).
    ("可引用条文 AISC-C-命名 说什么", NO_LOCK, "AISC-C-命名"),
    ("可引用条文 DXF-$INSUNITS 说什么", NO_LOCK, "毫米"),
]

# (turns, drawing the session must be on at the end). Guards the follow-up
# behaviour: a general question in the middle must not redirect the session.
SESSION_CASES: list[tuple[list[str], str]] = [
    (["门图有哪些图层", "它有几个块"], "wipeout_door"),
    (["总共有几张图纸", "槽钢图有哪些规格"], "Structural_Steel C 2D"),
    # A corpus question in the middle must answer at corpus scope without
    # dropping the drawing the session was already on.
    (["角钢图有哪些规格", "总共有几张图纸", "它有几个图层"], "Structural_Steel L 2D"),
]

# (turns, expected substring in the last answer). Checks the scope of the reply
# itself, not just which drawing the session ended on.
SESSION_TEXT_CASES: list[tuple[list[str], str]] = [
    # Locked on one drawing, a corpus question must still answer across all of
    # them instead of reporting the locked drawing's numbers.
    (["角钢图有哪些规格", "总共有几张图纸"], "共 11 张图纸"),
    (["表格图有哪些图层", "这张图用的是什么单位"], "米"),
    (["工程示意平面图有哪些图层", "它的图号是什么"], "DEMO-ENG-001"),
]

# Empty when the sample set covers every previously tracked limitation.
KNOWN_GAPS: list[tuple[str, str | None, str, str]] = []


def main() -> None:
    # Force extractive synthesis so this suite stays offline, deterministic, and
    # does not burn cloud-model quota. Evidence chunks are still in the answer, so
    # substring checks keep working.
    rag = CadRAG(generate="extractive")
    passed = 0

    for question, expect_drawing, expect_text in CASES:
        state = SessionState()
        result = rag.answer(question, state)
        drawing = result.get("drawing")
        answer = result.get("answer", "")

        if expect_drawing == NO_LOCK:
            drawing_ok = drawing is None
        else:
            drawing_ok = expect_drawing is None or drawing == expect_drawing
        text_ok = expect_text is None or expect_text.lower() in answer.lower()
        ok = drawing_ok and text_ok
        passed += ok

        print(f"[{'PASS' if ok else 'FAIL'}] {question}")
        print(f"       drawing: {drawing}" + ("" if drawing_ok else f"  (want {expect_drawing})"))
        if not text_ok:
            print(f"       missing expected text: {expect_text!r}")
        first_line = answer.strip().splitlines()[0] if answer.strip() else "(empty)"
        print(f"       answer[0]: {first_line[:100]}")
        print()

    session_passed = 0
    for turns, expect_drawing in SESSION_CASES:
        state = SessionState()
        for turn in turns:
            rag.answer(turn, state)
        ok = state.drawing == expect_drawing
        session_passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] 多轮: {' -> '.join(turns)}")
        print(f"       session drawing: {state.drawing}" + ("" if ok else f"  (want {expect_drawing})"))
        print()

    text_passed = 0
    for turns, expect_text in SESSION_TEXT_CASES:
        state = SessionState()
        answer = ""
        for turn in turns:
            answer = rag.answer(turn, state).get("answer", "")
        ok = expect_text.lower() in answer.lower()
        text_passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] 多轮范围: {' -> '.join(turns)}")
        print(f"       末轮答案含 {expect_text!r}: {ok}")
        print(f"       answer[0]: {answer.strip().splitlines()[0][:100]}")
        print()

    fixed = 0
    for question, expect_drawing, expect_text, why in KNOWN_GAPS:
        result = rag.answer(question, SessionState())
        drawing_ok = expect_drawing is None or result.get("drawing") == expect_drawing
        text_ok = expect_text.lower() in result.get("answer", "").lower()
        now_passes = drawing_ok and text_ok
        fixed += now_passes
        print(f"[{'FIXED' if now_passes else 'GAP'}] {question}")
        print(f"       drawing: {result.get('drawing')}, 命中 {expect_text!r}: {text_ok}")
        print(f"       {why}")
        print()

    print(f"单轮 {passed}/{len(CASES)} passed")
    print(f"多轮会话 {session_passed}/{len(SESSION_CASES)} passed")
    print(f"多轮范围 {text_passed}/{len(SESSION_TEXT_CASES)} passed")
    print(f"已知缺口 {len(KNOWN_GAPS) - fixed}/{len(KNOWN_GAPS)} 仍未修复")

    total = len(CASES) + len(SESSION_CASES) + len(SESSION_TEXT_CASES)
    if passed + session_passed + text_passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
