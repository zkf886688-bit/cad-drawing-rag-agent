"""CAD 智能体（工程知识库）索引上的检索与问答编排。

使用 BM25 而非「词频 ÷ 文档长度」：后者会系统性偏袒最短 chunk，
使简短的 `stats` 块压过真正能回答问题的文本块。
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "datasets" / "index" / "chunks.jsonl"

# bm25 | dense | hybrid —— dense/hybrid 需先跑 scripts/build_embeddings.py
DEFAULT_BACKEND = "bm25"
VALID_BACKENDS = frozenset({"bm25", "dense", "hybrid"})
RRF_K = 60
DENSE_CANDIDATES = 40

WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)

# BM25 参数：k1 控制词频饱和，b 控制长度归一化强度
BM25_K1 = 1.5
BM25_B = 0.75

SUMMARY_BOOST = 1.15

# 「列出全部」与「X 是什么」不是同一类问题。完整列表块天然最长，
# BM25 长度归一会把它排到分批短块后面；问句要求枚举时，必须给完整列表块加权。
LIST_INTENT_TERMS = ("全部", "所有", "列出", "完整", "清单", "一览", "逐个", "都有哪些")
COMPLETE_LIST_KINDS = frozenset({"steel_specs_all"})
COMPLETE_LIST_BOOST = 1.6

# 文件名比摘要里碰巧出现的词更能标识图纸，故名称匹配权重大于正文匹配
NAME_TERM_WEIGHT = 3
CONTENT_TERM_WEIGHT = 1

# 如 L4X4X1/2 这类型号只出现在一张图里，识别力等同文件名；
# 词表从索引自动推导，无需手写。短词与中文片段排除：正文里「唯一」往往是措辞巧合。
UNIQUE_TERM_WEIGHT = 3
MIN_UNIQUE_TERM_LEN = 4
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

# 下列词表示在问图纸的「哪个方面」，而不是「哪张图」。
# 若用它们选图，「门图有哪些图层」会被锁到摘要里碰巧含「图层」的视口样例。
# 它们仍参与排序，只是不参与选定图纸。
ASPECT_TERMS = frozenset(
    {
        # 「规格」与表格图标签「表格」共享「格」字；单字可作别名时，
        # 所有规格类问题都会被误锁到表格样例。列入方面词后，其分词碎片一并排除。
        "规格",
        "型号",
        "单位",
        # 比较词：「哪张图实体最多」曾被锁到别名含「最」的「最简 DXF 样例」
        "最多",
        "最少",
        "最大",
        "最小",
        "图层",
        "块",
        "块引用",
        "引用",
        "定义",
        "尺寸",
        "标注",
        "实体",
        "统计",
        "数量",
        "类型",
        "测试",
        "测试图",
        "示意",
        "构件",
        "样例",
        "layer",
        "layers",
        "block",
        "blocks",
        "entity",
        "entities",
        "dimension",
        "dimensions",
        "text",
        "texts",
        "type",
    }
)

# 把问题指向「整库」而非单张图的短语。
# 会话已锁定时，若仍按锁图检索，「总共有几张图纸」会只答一张。
# 这里只列明确「多张图」语义的短语：单独的「全部/所有」常指单图范围
# （如「角钢图的所有规格」）。
#
# FORCE 类：一律视为全库（除非强名称/唯一型号别名命中）。
# WHICH 类（「哪张」）：仍可用内容别名锁定单图（如「哪张图有被冻结的图层」），
# 仅在没有任何图纸匹配时才强制全库。
FORCE_CORPUS_TERMS = (
    "总共",
    "一共",
    "总数",
    "几张",
    "多少张",
    "最多",
    "最少",
    "最大",
    "最小",
    "哪些图纸",
    "所有图纸",
    "全部图纸",
    "每张",
    "各张",
    "图纸列表",
    "知识库",
    "库里",
    "整个库",
)
WHICH_DRAWING_TERMS = ("哪张", "哪几张")

# 代词/指示性追问应保持会话锁，除非用户强点名另一张图。
# 否则像「的图」（来自「它的图号」）这类正文 bigram 会通过视觉摘要抢锁。
FOLLOWUP_PRONOUNS = ("它", "这张", "该图", "上面这", "其中")

# 标准/规范类问题需要语料级 standard 块与图谱 GOVERNED_BY 边。
# 这些块 drawing=None，若只按会话锁图过滤会被挡掉，
# 导致「C15X50 对应什么标准」看起来无法回答。
STANDARD_INTENT_TERMS = (
    "标准",
    "规范",
    "国标",
    "美标",
    "对照",
    "条文",
    "可引用",
    "AISC",
    "GB/T",
    "GB ",
    "ASTM",
)


def tokenize(text: str) -> list[str]:
    """分词：拉丁词 + 中文单字及其 bigram。

    中文无空格，仅用单字匹配过松；bigram 在无分词器的情况下恢复大部分短语精度。
    """
    parts = WORD_RE.findall(text or "")
    tokens: list[str] = []
    chinese: list[str] = []

    def flush() -> None:
        if not chinese:
            return
        tokens.extend(chinese)
        for i in range(len(chinese) - 1):
            tokens.append(chinese[i] + chinese[i + 1])
        chinese.clear()

    for part in parts:
        if "\u4e00" <= part[0] <= "\u9fff":
            chinese.append(part)
        else:
            flush()
            tokens.append(part.lower())
    flush()
    return tokens


@dataclass
class SessionState:
    drawing: str | None = None
    history: list[dict] = field(default_factory=list)


class CadRAG:
    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        backend: str = DEFAULT_BACKEND,
        generate: str | None = None,
    ) -> None:
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
        self.backend = backend
        # None 表示在 answer 时再尊重 CAD_GENERATE / auto
        self.generate = generate
        self.chunks: list[dict] = []
        self.term_freqs: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_length = 0.0
        self.name_terms: dict[str, set[str]] = {}
        self.content_terms: dict[str, set[str]] = {}
        self.unique_terms: dict[str, set[str]] = {}
        self._embed = None
        self._id_to_index: dict[str, int] = {}
        self._graph = None
        self._load(index_path)
        if backend in {"dense", "hybrid"}:
            self._load_embeddings()
        self._load_graph()

    def _load_graph(self) -> None:
        try:
            from app.graph import KnowledgeGraph

            self._graph = KnowledgeGraph()
        except Exception:
            self._graph = None

    def reload(self) -> None:
        """从磁盘重建。高并发下请优先用 RagService.reload 做原子替换。"""
        fresh = CadRAG(backend=self.backend, generate=self.generate)
        self.chunks = fresh.chunks
        self.term_freqs = fresh.term_freqs
        self.lengths = fresh.lengths
        self.doc_freq = fresh.doc_freq
        self.avg_length = fresh.avg_length
        self.name_terms = fresh.name_terms
        self.content_terms = fresh.content_terms
        self.unique_terms = fresh.unique_terms
        self._id_to_index = fresh._id_to_index
        self._embed = fresh._embed
        self._graph = fresh._graph

    def _load_embeddings(self) -> None:
        from app.embeddings import EmbeddingIndex

        self._embed = EmbeddingIndex()
        # 按 id 把向量行与 chunk 列表对齐；build_embeddings 读同一份 jsonl，
        # 理应 1:1，这里仍显式映射以便发现过期索引。
        missing = [chunk["id"] for chunk in self.chunks if chunk["id"] not in self._embed._id_to_row]
        if missing:
            raise RuntimeError(
                f"Embedding index out of date ({len(missing)} missing ids). "
                "Re-run: py scripts/build_embeddings.py"
            )

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Index not found: {path}. Run: py scripts/build_index.py"
            )

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                tokens = tokenize(chunk.get("text", ""))
                # 词频在此一次性算好，避免每次查询重复计算
                self._id_to_index[chunk["id"]] = len(self.chunks)
                self.chunks.append(chunk)
                self.term_freqs.append(Counter(tokens))
                self.lengths.append(len(tokens))
                for term in set(tokens):
                    self.doc_freq[term] += 1

        if self.lengths:
            self.avg_length = sum(self.lengths) / len(self.lengths)
        self._build_drawing_terms()

    def _build_drawing_terms(self) -> None:
        """从索引自举每张图的识别词，避免手写别名表。

        硬编码别名会随图集变更失效；像 layer 这类通用词也会误匹配多张图。
        """
        names: dict[str, set[str]] = {}
        contents: dict[str, set[str]] = {}

        for chunk in self.chunks:
            drawing = chunk.get("drawing")
            if not drawing:
                continue
            # 按分隔符拆开，使 wipeout_door 也能匹配 door
            terms = names.setdefault(drawing, set())
            terms.update(tokenize(re.sub(r"[_\-]+", " ", drawing)))
            if chunk.get("kind") == "summary":
                # drawing_type 是刻意标注的中文类型，当作名称别名
                terms.update(tokenize(chunk.get("drawing_type", "")))
                contents.setdefault(drawing, set()).update(
                    tokenize(chunk.get("text", ""))
                )

        # 中文 bigram 会跨词产生碎片，如「块引用」→「块引」。
        # 用同一套分词扩展方面词表，可自动排除这些碎片。
        excluded = set(ASPECT_TERMS)
        for term in ASPECT_TERMS:
            excluded.update(tokenize(term))

        def discriminating(
            groups: dict[str, set[str]], min_length: int
        ) -> dict[str, set[str]]:
            counts: Counter[str] = Counter()
            for terms in groups.values():
                counts.update(terms)
            return {
                drawing: {
                    term
                    for term in terms
                    if counts[term] == 1
                    and len(term) >= min_length
                    and term not in excluded
                }
                for drawing, terms in groups.items()
            }

        # 名称与类型是刻意标签，唯一单字如「门」可作别名；
        # 自由正文不行：偶然的「有」不能成为某张图的标识。
        self.name_terms = discriminating(names, min_length=1)
        self.content_terms = discriminating(contents, min_length=2)
        self.unique_terms = self._build_unique_terms(excluded)

    def _build_unique_terms(self, excluded: set[str]) -> dict[str, set[str]]:
        """只出现在一张图的 chunk 中、别处没有的词（多为型号）。

        跳过语料级块：它们提到所有图，计入后几乎没有词能保持唯一。
        """
        owners: dict[str, set[str]] = {}
        for chunk, freqs in zip(self.chunks, self.term_freqs):
            drawing = chunk.get("drawing")
            if not drawing:
                continue
            for term in freqs:
                if len(term) < MIN_UNIQUE_TERM_LEN or term in excluded:
                    continue
                if CHINESE_RE.search(term):
                    continue
                owners.setdefault(term, set()).add(drawing)

        unique: dict[str, set[str]] = {drawing: set() for drawing in self.name_terms}
        for term, drawings in owners.items():
            if len(drawings) == 1:
                unique.setdefault(next(iter(drawings)), set()).add(term)
        return unique

    def _score(
        self, index: int, query_terms: list[str], list_intent: bool = False
    ) -> float:
        term_freq = self.term_freqs[index]
        length = self.lengths[index] or 1
        total = len(self.chunks)
        score = 0.0

        for term in query_terms:
            freq = term_freq.get(term)
            if not freq:
                continue
            idf = math.log(
                1 + (total - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5)
            )
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / (self.avg_length or 1))
            score += idf * (freq * (BM25_K1 + 1)) / (freq + norm)

        if not score:
            return score

        kind = self.chunks[index].get("kind")
        if kind == "summary":
            score *= SUMMARY_BOOST
        if list_intent and kind in COMPLETE_LIST_KINDS:
            score *= COMPLETE_LIST_BOOST
        return score

    def search(
        self,
        query: str,
        top_k: int = 4,
        drawing: str | None = None,
    ) -> list[dict]:
        if self.backend == "bm25":
            return self._search_bm25(query, top_k=top_k, drawing=drawing)
        if self.backend == "dense":
            return self._search_dense(query, top_k=top_k, drawing=drawing)
        return self._search_hybrid(query, top_k=top_k, drawing=drawing)

    def _search_bm25(
        self,
        query: str,
        top_k: int = 4,
        drawing: str | None = None,
    ) -> list[dict]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        list_intent = any(term in query for term in LIST_INTENT_TERMS)
        scored: list[tuple[float, int]] = []
        for index, chunk in enumerate(self.chunks):
            if drawing and chunk.get("drawing") != drawing:
                continue
            score = self._score(index, query_terms, list_intent)
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            dict(self.chunks[index], score=round(score, 6), backend="bm25")
            for score, index in scored[:top_k]
        ]

    def _search_dense(
        self,
        query: str,
        top_k: int = 4,
        drawing: str | None = None,
    ) -> list[dict]:
        assert self._embed is not None
        # 先对全集打分，再按图纸过滤，保证会话锁图时仍用稠密排序
        ranked = self._embed.top_indices(query, top_k=max(top_k * 5, DENSE_CANDIDATES))
        hits: list[dict] = []
        for row, score in ranked:
            chunk_id = self._embed.ids[row]
            index = self._id_to_index.get(chunk_id)
            if index is None:
                continue
            chunk = self.chunks[index]
            if drawing and chunk.get("drawing") != drawing:
                continue
            hits.append(dict(chunk, score=round(score, 6), backend="dense"))
            if len(hits) >= top_k:
                break
        return hits

    def _search_hybrid(
        self,
        query: str,
        top_k: int = 4,
        drawing: str | None = None,
    ) -> list[dict]:
        """BM25 与稠密排序的 Reciprocal Rank Fusion（倒数排名融合）。

        精确型号更偏 BM25，改写问法更偏 dense；RRF 无需学习融合权重。
        """
        bm25_hits = self._search_bm25(query, top_k=DENSE_CANDIDATES, drawing=drawing)
        dense_hits = self._search_dense(query, top_k=DENSE_CANDIDATES, drawing=drawing)

        fused: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        for rank, hit in enumerate(bm25_hits, start=1):
            fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (RRF_K + rank)
            payloads[hit["id"]] = hit
        for rank, hit in enumerate(dense_hits, start=1):
            fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (RRF_K + rank)
            payloads.setdefault(hit["id"], hit)

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [
            dict(payloads[chunk_id], score=round(score, 6), backend="hybrid")
            for chunk_id, score in ordered[:top_k]
        ]

    def _best_drawing(
        self,
        query_terms: set[str],
        *,
        use_content: bool,
    ) -> str | None:
        """按别名得分选出最高的图纸；平局或零分则返回 None。"""
        if not query_terms:
            return None

        best: str | None = None
        best_score = 0
        tied = False
        for drawing in self.name_terms:
            score = (
                NAME_TERM_WEIGHT
                * len(query_terms & self.name_terms.get(drawing, set()))
                + UNIQUE_TERM_WEIGHT
                * len(query_terms & self.unique_terms.get(drawing, set()))
            )
            if use_content:
                score += CONTENT_TERM_WEIGHT * len(
                    query_terms & self.content_terms.get(drawing, set())
                )
            if score > best_score:
                best, best_score, tied = drawing, score, False
            elif score == best_score and score > 0:
                tied = True

        if best_score == 0 or tied:
            return None
        return best

    def match_drawing(self, query: str) -> str | None:
        """仅根据问句措辞判断指向哪张图（忽略会话上下文）。

        对每张图累计命中的区分性词数，结果不依赖字典遍历顺序。
        无匹配或两图平局时返回 None。
        """
        return self._best_drawing(set(tokenize(query)), use_content=True)

    def match_drawing_strong(self, query: str) -> str | None:
        """仅用名称/唯一型号别名，忽略自由正文 content_terms。

        视觉摘要会引入偶然 bigram（如「的实」来自「的实体」），
        否则会干扰「哪张图的实体最多」这类全库问题。
        """
        return self._best_drawing(set(tokenize(query)), use_content=False)

    def infer_drawing(self, query: str, state: SessionState) -> str | None:
        """解析本轮主题图纸，必要时回退到会话中已锁定的图纸。

        该回退使「那它有几个图层」这类追问成立。
        问句明显是代词追问且会话已锁时，只有强名称/唯一型号才能换图——
        正文别名在 VLM 摘要进索引后噪声太大。
        """
        if state.drawing and any(term in query for term in FOLLOWUP_PRONOUNS):
            return self.match_drawing_strong(query) or state.drawing
        return self.match_drawing(query) or state.drawing

    def is_standard_query(self, query: str) -> bool:
        return any(term in query for term in STANDARD_INTENT_TERMS)

    def is_corpus_query(self, query: str) -> bool:
        """问句是否在问整库，而非单张图？

        问句中点名的图纸优先：故「角钢图总共有多少规格」仍是单图问题。
        只有强别名（文件名/类型/唯一型号）能取消 FORCE 全库短语——
        摘要正文在阶段 3 VLM 文本入库后噪声过大。

        单独的「哪张」不同：「哪张图有被冻结的图层」应锁到内容唯一提到冻结的图；
        「哪张图的实体最多」因含「最多」被强制全库，避免摘要 bigram 抢锁。
        """
        if self.match_drawing_strong(query):
            return False
        if any(term in query for term in FORCE_CORPUS_TERMS):
            return True
        if any(term in query for term in WHICH_DRAWING_TERMS):
            return self.match_drawing(query) is None
        return False

    def answer(
        self,
        query: str,
        state: SessionState | None = None,
        generate: str | None = None,
    ) -> dict:
        state = state or SessionState()
        corpus_scope = self.is_corpus_query(query)
        standard_intent = self.is_standard_query(query)
        list_intent = any(term in query for term in LIST_INTENT_TERMS)
        is_followup = bool(
            state.drawing and any(term in query for term in FOLLOWUP_PRONOUNS)
        )
        # 全库问题故意忽略会话锁：上一轮主题仍留在 state 供后续追问，
        # 但本轮检索全库，才能命中语料级 chunk。
        identified = None if corpus_scope else self.infer_drawing(query, state)

        from app.intent import classify_intent

        intent = classify_intent(
            query,
            corpus_scope=corpus_scope,
            standard_intent=standard_intent,
            identified=identified,
            session_drawing=state.drawing,
            is_followup=is_followup,
            list_intent=list_intent,
        )

        # 产品/上传 FAQ：勿走图纸检索，否则 LLM 会据样例图编造「不支持上传」
        if intent.kind == "help":
            from app.intent import HELP_ANSWER, HELP_EVIDENCE

            answer = HELP_ANSWER
            state.history.append({"role": "user", "content": query})
            state.history.append({"role": "assistant", "content": answer})
            return {
                "answer": answer,
                "drawing": None,
                "sources": [
                    {
                        "id": hit.get("id"),
                        "drawing": hit.get("drawing"),
                        "file_name": hit.get("file_name"),
                        "kind": hit.get("kind"),
                        "score": hit.get("score"),
                        "text": (hit.get("text") or "")[:320],
                        "title": None,
                        "tool": None,
                        "parent_id": None,
                    }
                    for hit in HELP_EVIDENCE
                ],
                "intent": intent.to_dict(),
                "tools_used": [],
                "generate_mode": "help",
                "generate_note": None,
                "history_turns": len(state.history) // 2,
            }

        # 先召回较宽候选池，再精排缩小供生成使用
        candidate_k = 12
        hits = self.search(query, top_k=candidate_k, drawing=identified)

        if not hits:
            hits = self.search(query, top_k=candidate_k, drawing=None)

        # 已识别图纸但措辞未命中：回退该图概览块，仍能给出该图事实
        if not hits and identified:
            overview = {"summary", "layers", "blocks", "stats"}
            hits = [
                dict(chunk, score=0.0)
                for chunk in self.chunks
                if chunk.get("drawing") == identified and chunk.get("kind") in overview
            ][:4]

        # 只读工具（Pydantic 校验）增强证据，便于答辩展示 Tool 闭环
        tools_used: list[str] = []
        use_tools = os.environ.get("CAD_TOOLS", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if use_tools:
            try:
                from app.tools import ToolRuntime, tools_for_intent

                runtime = ToolRuntime(self)
                for name, arguments in tools_for_intent(intent, query):
                    tools_used.append(name)
                    tool_hits = runtime.call(name, arguments)
                    merged = list(hits)
                    seen = {h.get("id") for h in merged}
                    for th in tool_hits:
                        if th.get("id") in seen:
                            continue
                        # 针对性意图下，把已校验的工具事实插到前面
                        if intent.kind in {
                            "layers",
                            "dimensions",
                            "stats",
                            "standard",
                        } and th.get("kind") == "tool":
                            merged.insert(0, th)
                        else:
                            merged.append(th)
                        seen.add(th.get("id"))
                    hits = merged[:16]
            except Exception:
                tools_used = []

        # 标准问句务必拉取语料级 standard/clause 块。
        # 它们 drawing=None，若仅按型号锁图会被过滤掉。
        if standard_intent or intent.clause_code:
            standard_hits = [
                hit
                for hit in self.search(query, top_k=8, drawing=None)
                if hit.get("kind") in {"standard", "clause"}
            ]
            if intent.clause_code:
                code = intent.clause_code.lower()
                exact = [
                    dict(chunk, score=1.2, backend="clause")
                    for chunk in self.chunks
                    if chunk.get("kind") == "clause"
                    and (chunk.get("clause_code") or "").lower() == code
                ]
                standard_hits = exact + standard_hits
            if not standard_hits:
                # 回退：任意提到查询 token 的 standard/clause 块
                tokens = {t.lower() for t in tokenize(query) if len(t) >= 2}
                for chunk in self.chunks:
                    if chunk.get("kind") not in {"standard", "clause"}:
                        continue
                    text = (chunk.get("text") or "").lower()
                    if any(token in text for token in tokens):
                        standard_hits.append(
                            dict(chunk, score=0.5, backend="standard")
                        )
                    if len(standard_hits) >= 3:
                        break
            ordered = list(standard_hits) + list(hits)
            hits = []
            seen: set = set()
            for hit in ordered:
                if hit.get("id") in seen:
                    continue
                hits.append(hit)
                seen.add(hit.get("id"))
                if len(hits) >= 10:
                    break

        # GraphRAG-lite：融合多跳知识图谱事实（默认 3 跳）
        graph_hits: list[dict] = []
        if self._graph is not None:
            try:
                graph_hits = self._graph.facts_for_query(query, limit=6)
            except Exception:
                graph_hits = []
        if graph_hits:
            if not standard_intent and not intent.clause_code:
                # 标准/规范问句保留条文引用；否则教学条文边会淹没普通图层/尺寸题
                graph_hits = [
                    item
                    for item in graph_hits
                    if not any(
                        token in (item.get("text") or "")
                        for token in ("可引用条文", "CLAUSE_OF", "HAS_CLAUSE", "CITED_BY")
                    )
                ]
            if standard_intent or intent.clause_code:
                # 优先 GOVERNED_BY / 可引用条文类边
                graph_hits = sorted(
                    graph_hits,
                    key=lambda item: (
                        0
                        if any(
                            token in (item.get("text") or "")
                            for token in (
                                "GOVERNED_BY",
                                "CLAUSE_OF",
                                "可引用条文",
                                "CITED_BY",
                            )
                        )
                        else 1,
                        0
                        if identified and item.get("drawing") == identified
                        else 1,
                    ),
                )
            else:
                graph_hits = sorted(
                    graph_hits,
                    key=lambda item: (
                        0
                        if identified and item.get("drawing") == identified
                        else 1
                    ),
                )
            merged_hits = list(hits)
            seen_ids = {h.get("id") for h in merged_hits}
            # 标准问句：图谱事实插到靠前位置
            insert_at = 1 if (standard_intent or intent.clause_code) else len(merged_hits)
            for gh in graph_hits:
                if gh.get("id") in seen_ids:
                    continue
                merged_hits.insert(insert_at, gh)
                insert_at += 1
                seen_ids.add(gh.get("id"))
            hits = merged_hits[:12]

        use_rerank = os.environ.get("CAD_RERANK", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if use_rerank and hits:
            from app.rerank import rerank_hits

            final_k = 8 if (standard_intent or intent.clause_code) else 6
            hits = rerank_hits(
                query,
                hits,
                top_k=final_k,
                clause_code=intent.clause_code,
            )

        hit_drawings = {hit.get("drawing") for hit in hits if hit.get("drawing")}

        # 仅在图纸真正定下来时锁定会话：问句点名了，或全部命中块同属一张图。
        # 「总共有几张图纸」会拉到多张图的块，若锁到排名第一的那张，
        # 会静默污染后续每一轮追问。
        drawing = (
            None
            if corpus_scope
            else identified
            or (next(iter(hit_drawings)) if len(hit_drawings) == 1 else None)
        )
        if drawing:
            state.drawing = drawing

        if corpus_scope:
            scope = "全库范围提问，不限定单张图纸"
        elif (standard_intent or intent.clause_code) and drawing:
            scope = f"当前图纸: {drawing}（含工程标准/图谱关联）"
        elif drawing:
            scope = f"当前图纸: {drawing}"
        else:
            scope = f"未锁定单张图纸，证据来自 {len(hit_drawings)} 张图"

        from app.generate import synthesize

        # 把近期对话交给 LLM，使追问能用对话上下文，而不只靠锁定的 drawing id。
        # 本轮问答在合成后再追加进 history。
        generated = synthesize(
            query,
            hits,
            scope=scope,
            backend=self.backend,
            mode=generate if generate is not None else self.generate,
            history=state.history,
        )
        answer = generated["answer"]

        state.history.append({"role": "user", "content": query})
        state.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            # 本轮主题图纸；证据跨多张图时为 None。
            # state.drawing 仍保留上次锁定的图，供后续追问使用。
            "drawing": drawing,
            "sources": [
                {
                    "id": hit.get("id"),
                    "drawing": hit.get("drawing"),
                    "file_name": hit.get("file_name"),
                    "kind": hit.get("kind"),
                    "score": hit.get("rerank_score", hit.get("score")),
                    "text": (hit.get("text") or "")[:320],
                    "title": hit.get("title")
                    or hit.get("jurisdiction")
                    or None,
                    "tool": hit.get("tool"),
                    "parent_id": hit.get("parent_id"),
                }
                for hit in hits
            ],
            "intent": intent.to_dict(),
            "tools_used": tools_used,
            "generate_mode": generated["generate_mode"],
            "generate_note": generated["generate_note"],
            "history_turns": len(state.history) // 2,
        }


class RagService:
    """进程级 RAG 句柄；reload 原子替换，进行中的问答仍使用旧索引。"""

    def __init__(
        self,
        *,
        backend: str = DEFAULT_BACKEND,
        generate: str | None = None,
    ) -> None:
        self.backend = backend
        self.generate = generate
        self._lock = threading.Lock()
        self._impl = CadRAG(backend=backend, generate=generate)

    def _snapshot(self) -> CadRAG:
        with self._lock:
            return self._impl

    def reload(self) -> None:
        fresh = CadRAG(backend=self.backend, generate=self.generate)
        with self._lock:
            self._impl = fresh

    @property
    def chunks(self) -> list[dict]:
        return self._snapshot().chunks

    @property
    def name_terms(self) -> dict[str, set[str]]:
        return self._snapshot().name_terms

    def answer(
        self,
        query: str,
        state: SessionState | None = None,
        generate: str | None = None,
    ) -> dict:
        return self._snapshot().answer(query, state, generate=generate)
