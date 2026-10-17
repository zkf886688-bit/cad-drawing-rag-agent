"""Read AISC steel designations out of OCR text.

The steel sheets label each profile with a designation such as C15X50 or
L4X4X1/2, drawn as line work rather than as TEXT entities, so these strings
exist only in the OCR output. Recognising them turns a loose bag of strings
into something a question can address: the prefix identifies the member type,
which is what a Chinese-language question actually names ("angle", "channel"),
while the file name only ever says "Structural_Steel L 2D".
"""

from __future__ import annotations

import re
from collections import Counter

# Longest prefixes first so HSS is not read as H, and WT not as W.
PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("PIPE", "管材", "圆管 / 钢管"),
    ("HSS", "方矩管", "空心结构管，方形或矩形截面"),
    ("WT", "T型钢", "由宽翼缘工字钢剖分而成"),
    ("MT", "T型钢", "由轻型工字钢剖分而成"),
    ("ST", "T型钢", "由标准工字钢剖分而成"),
    ("HP", "H型桩", "承压桩用 H 型截面"),
    ("MC", "槽钢", "轻型槽钢"),
    ("C", "槽钢", "标准槽钢，C 形截面"),
    ("L", "角钢", "L 形截面，等边或不等边"),
    ("W", "宽翼缘工字钢", "工字形截面，翼缘较宽"),
    ("S", "工字钢", "标准工字梁"),
    ("M", "轻型工字钢", "轻型工字截面"),
)

# A designation is a prefix followed by a dimension, e.g. C15X50, Pipe10STD,
# L4X4X1/2. Requiring a digit right after the prefix keeps stray letters out.
SPEC_PATTERN = re.compile(
    r"^(?P<prefix>PIPE|HSS|WT|MT|ST|HP|MC|C|L|W|S|M)\s*(?P<size>\d[\w./\-]*)$",
    re.IGNORECASE,
)


def match_spec(text: str) -> tuple[str, str] | None:
    """Return (prefix, designation) when text is a steel designation."""
    cleaned = text.strip().replace(" ", "")
    match = SPEC_PATTERN.match(cleaned)
    if not match:
        return None
    return match.group("prefix").upper(), cleaned


def describe_prefix(prefix: str) -> tuple[str, str]:
    """Map a designation prefix to its Chinese member name and a note."""
    for key, chinese, note in PREFIXES:
        if prefix.upper() == key:
            return chinese, note
    return "型钢", "未识别的截面类型"


def extract_specs(texts: list[str]) -> dict:
    """Summarise the steel designations present in a list of OCR strings."""
    designations: list[str] = []
    prefixes: Counter[str] = Counter()

    for text in texts:
        matched = match_spec(text)
        if matched is None:
            continue
        prefix, designation = matched
        if designation not in designations:
            designations.append(designation)
        prefixes[prefix] += 1

    member_types = []
    for prefix, count in prefixes.most_common():
        chinese, note = describe_prefix(prefix)
        if chinese not in [m["name"] for m in member_types]:
            member_types.append(
                {"name": chinese, "prefix": prefix, "count": count, "note": note}
            )

    return {
        "designations": sorted(designations),
        "designation_count": len(designations),
        "member_types": member_types,
        "prefix_counts": dict(prefixes),
    }
