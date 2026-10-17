"""Answer synthesis for requirement 4.

Retrieval finds the evidence; this module turns it into a short Chinese reply.
Backends:
  off         raw evidence list only (old behaviour)
  extractive  rule-based short answer from the top chunks (no API)
  deepseek    DeepSeek OpenAI-compatible API; fall back to extractive on failure
  qwen        通义千问 (DashScope); fall back to extractive on failure
  auto        qwen → deepseek → extractive (skip missing keys)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from app.logutil import setup_logging

log = setup_logging("cad_agent.generate")

QWEN_MODEL_DEFAULT = "qwen-plus"
DEEPSEEK_MODEL_DEFAULT = "deepseek-flash"
DASHSCOPE_CHAT_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
VALID_GENERATE = frozenset(
    {"off", "extractive", "deepseek", "qwen", "auto"}
)
AUTO_PROVIDER_ORDER = ("qwen", "deepseek")

SYSTEM_PROMPT = """你是 CAD 图纸知识库助手。只根据下面的【检索证据】回答用户问题。
要求：
1. 用简洁中文直接回答，不要复述整段证据。
2. 证据里若出现工程标准/惯例块（kind 含 standard/clause）或知识图谱 GOVERNED_BY / 标准要点，
   必须据此作答（例如指出 AISC / GB/T 及含义），不要说「无法确定」。
3. 只有证据完全未提及相关规格、标准或图纸事实时，才说「根据现有知识库无法确定」。
4. 不要编造证据中没有的数字或型号。
5. 提到具体图纸时用文件名或中文类型（如角钢图、门图）。
6. 在句末用 [1][2] 标出依据编号（对应证据列表）。
7. 若证据含「可引用条文」或条文代号（如 GB/T 707-对照、AISC-C-命名），回答中应写出该代号。
只输出答案正文，不要输出「根据证据」这类套话标题。"""


def _dashscope_key() -> str | None:
    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")):
        from app.envfile import load_local_env

        load_local_env()
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")


def _deepseek_key() -> str | None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        from app.envfile import load_local_env

        load_local_env()
    return os.environ.get("DEEPSEEK_API_KEY")


def configured_providers() -> list[str]:
    """Return configured cloud providers in automatic failover order."""
    available = {
        "qwen": bool(_dashscope_key()),
        "deepseek": bool(_deepseek_key()),
    }
    return [name for name in AUTO_PROVIDER_ORDER if available[name]]


def resolve_mode(requested: str | None = None) -> str:
    mode = (requested or os.environ.get("CAD_GENERATE", "auto")).strip().lower()
    if mode not in VALID_GENERATE:
        raise ValueError(f"CAD_GENERATE must be one of {sorted(VALID_GENERATE)}")
    if mode == "auto":
        providers = configured_providers()
        return providers[0] if providers else "extractive"
    return mode


def format_evidence(hits: list[dict]) -> str:
    lines = []
    for rank, hit in enumerate(hits, 1):
        lines.append(
            f"[{rank}] ({hit.get('kind')}, {hit.get('file_name')}) "
            f"{hit.get('text', '')}"
        )
    return "\n".join(lines)


def evidence_block(hits: list[dict], scope: str, backend: str) -> str:
    """Keep the raw chunks in the reply so answers stay auditable."""
    lines = [
        "",
        f"—— 依据（{scope}；检索={backend}）——",
    ]
    for rank, hit in enumerate(hits, 1):
        lines.append(
            f"[{rank}] ({hit.get('kind')}, score={hit.get('score')}) "
            f"{hit.get('file_name')}"
        )
        lines.append(hit.get("text", ""))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def retrieval_only_answer(hits: list[dict], scope: str) -> str:
    if not hits:
        return (
            "没有在知识库中找到相关内容。"
            "可以换个问法，或指定图纸名称（如 door / steel / viewport）。"
        )
    lines = [f"基于知识库检索（{scope}）：", ""]
    for rank, hit in enumerate(hits, 1):
        lines.append(
            f"[{rank}] ({hit.get('kind')}, score={hit.get('score')}) "
            f"{hit.get('file_name')}"
        )
        lines.append(hit.get("text", ""))
        lines.append("")
    lines.append("说明: 答案来自 final 索引的检索结果，可追溯来源。")
    return "\n".join(lines)


def extractive_answer(query: str, hits: list[dict], scope: str) -> str:
    """Build a short reply without an LLM."""
    if not hits:
        return (
            "没有在知识库中找到相关内容。"
            "可以换个问法，或指定图纸名称（如 door / steel / viewport）。"
        )

    # Prefer standard / graph facts when the user asks about 标准.
    preferred = [
        hit
        for hit in hits
        if hit.get("kind") in {"standard", "clause", "graph", "tool"}
        or "标准" in (hit.get("text") or "")
        or "可引用条文" in (hit.get("text") or "")
    ]
    top = preferred[0] if preferred else hits[0]
    drawing = top.get("drawing") or top.get("file_name") or "相关图纸"
    text = (top.get("text") or "").strip()
    snippet = text if len(text) <= 280 else text[:280].rstrip() + "…"
    more = f" 另有 {len(hits) - 1} 条相关证据见下方依据。" if len(hits) > 1 else ""
    return (
        f"针对「{query}」，知识库在【{drawing}】上最相关的内容是：{snippet}"
        f"{more}"
        f"（范围：{scope}）[1]"
    )


def format_history(history: list[dict] | None, max_turns: int = 4) -> str:
    """Recent user/assistant turns for multi-turn LLM context."""
    if not history:
        return ""
    # history is [user, assistant, user, assistant, ...]; keep the last N turns.
    pairs = max_turns * 2
    recent = history[-pairs:] if len(history) > pairs else history
    lines = []
    for item in recent:
        role = "用户" if item.get("role") == "user" else "助手"
        content = (item.get("content") or "").strip()
        if not content:
            continue
        # Evidence blocks are long; keep only the short answer line for context.
        if role == "助手" and "——" in content:
            content = content.split("——", 1)[0].strip()
        if len(content) > 400:
            content = content[:400].rstrip() + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _llm_user_prompt(
    query: str,
    hits: list[dict],
    scope: str,
    history: list[dict] | None = None,
) -> str:
    history_block = format_history(history)
    history_section = (
        f"【最近对话】\n{history_block}\n\n" if history_block else ""
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"【提问范围】{scope}\n"
        f"{history_section}"
        f"【用户问题】{query}\n\n"
        f"【检索证据】\n{format_evidence(hits)}\n"
    )


def _failure_note(provider: str, exc: BaseException) -> str:
    text = str(exc)
    low = text.lower()
    if "not set" in low:
        return f"{provider} 未配置"
    if "401" in text or "invalid_api_key" in low or "authentication" in low:
        return f"{provider} 密钥无效"
    if "429" in text or "quota" in low or "rate" in low or "限流" in text:
        return f"{provider} 配额不足或触发限流"
    return f"{provider} 暂时不可用"


def _dashscope_chat(
    messages: list[dict],
    *,
    model: str,
    timeout: int = 60,
) -> str:
    api_key = _dashscope_key()
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY / QWEN_API_KEY not set. "
            "Get a key at https://bailian.console.aliyun.com/"
        )
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        DASHSCOPE_CHAT_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Qwen HTTP {exc.code}: {detail}") from exc

    try:
        text = (payload["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Qwen unexpected response: {str(payload)[:200]}") from exc
    if not text:
        raise RuntimeError("Qwen returned an empty answer")
    return text


def qwen_answer(
    query: str,
    hits: list[dict],
    scope: str,
    history: list[dict] | None = None,
) -> str:
    model = os.environ.get("CAD_QWEN_MODEL", QWEN_MODEL_DEFAULT)
    return _dashscope_chat(
        [
            {
                "role": "user",
                "content": _llm_user_prompt(query, hits, scope, history=history),
            }
        ],
        model=model,
    )


def deepseek_answer(
    query: str,
    hits: list[dict],
    scope: str,
    history: list[dict] | None = None,
) -> str:
    api_key = _deepseek_key()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    model = os.environ.get("CAD_DEEPSEEK_MODEL", DEEPSEEK_MODEL_DEFAULT)
    url = os.environ.get("CAD_DEEPSEEK_URL", DEEPSEEK_CHAT_URL)
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": _llm_user_prompt(query, hits, scope, history=history),
            }
        ],
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc

    try:
        text = (payload["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"DeepSeek unexpected response: {str(payload)[:200]}"
        ) from exc
    if not text:
        raise RuntimeError("DeepSeek returned an empty answer")
    return text


def synthesize(
    query: str,
    hits: list[dict],
    *,
    scope: str,
    backend: str,
    mode: str | None = None,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """Return {answer, generate_mode, generate_note}."""
    requested = (mode or os.environ.get("CAD_GENERATE", "auto")).strip().lower()
    if requested not in VALID_GENERATE:
        raise ValueError(f"CAD_GENERATE must be one of {sorted(VALID_GENERATE)}")
    resolved = resolve_mode(requested)

    if resolved == "off":
        return {
            "answer": retrieval_only_answer(hits, scope),
            "generate_mode": "off",
            "generate_note": None,
        }

    note = None
    generate_mode = "extractive"
    summary: str

    if resolved in {"deepseek", "qwen"}:
        providers = configured_providers() if requested == "auto" else [resolved]
        provider_labels = {
            "qwen": "通义千问",
            "deepseek": "DeepSeek",
        }
        failures: list[str] = []
        summary = ""
        for provider in providers:
            try:
                llm_started = time.perf_counter()
                if provider == "qwen":
                    summary = qwen_answer(query, hits, scope, history=history)
                elif provider == "deepseek":
                    summary = deepseek_answer(query, hits, scope, history=history)
                generate_mode = provider
                log.info(
                    "llm ok provider=%s hits=%s ms=%s",
                    provider,
                    len(hits),
                    round((time.perf_counter() - llm_started) * 1000),
                )
                if failures:
                    note = "；".join(failures) + (
                        f"；已自动切换至{provider_labels[provider]}。"
                    )
                break
            except Exception as exc:  # noqa: BLE001 — chat must keep working
                failure = _failure_note(provider_labels[provider], exc)
                failures.append(failure)
                log.warning(
                    "llm fallback provider=%s note=%s err=%s",
                    provider,
                    failure,
                    str(exc)[:200],
                )
        if not summary:
            note = "；".join(failures) + "；已切换至本地模式。"
            summary = extractive_answer(query, hits, scope)
            generate_mode = "extractive"
    else:
        summary = extractive_answer(query, hits, scope)
        log.debug("generate mode=%s hits=%s", generate_mode, len(hits))

    if not hits:
        return {
            "answer": summary,
            "generate_mode": generate_mode,
            "generate_note": note,
        }

    answer = (
        summary.rstrip()
        + "\n"
        + evidence_block(hits, scope, backend)
        + "说明: 上方为生成/摘要答案，下方依据可追溯到索引块。"
    )
    return {
        "answer": answer,
        "generate_mode": generate_mode,
        "generate_note": note,
    }
