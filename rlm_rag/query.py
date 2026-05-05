"""Single-shot RAG query: retrieve → triage in parallel → synthesize.

Sub-agents see chunks grouped by file by default — a single sub-call covers
every retrieved chunk from one file, which gives the model cross-chunk
context within the file (e.g. it can see that `authenticate` calls
`hash_password` defined two functions up). Pass group="chunk" to revert to
the legacy one-chunk-per-sub-call mode (more parallel, less local context,
more sub-calls).

For the iterative root-loop variant where the model can re-query, navigate
the call graph, and decide for itself when to finalize, see iterative.py.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass

from . import _rlm_helper as rlm_helper
from .embedder import Embedder
from .models import ModelConfig, ModelTier, default_config
from .retrieval import Reranker, retrieve
from .review import ReviewResult, with_review
from .store import ChunkStore, SearchHit


# Back-compat with earlier env vars; ModelConfig overrides take precedence.
DEFAULT_ROOT_MODEL = os.environ.get("RLM_RAG_ROOT_MODEL", "claude-sonnet-4-6")
DEFAULT_SUB_MODEL = os.environ.get("RLM_RAG_SUB_MODEL", "claude-haiku-4-5")

TRIAGE_SYSTEM_FILE = (
    "You are a code analyst. Given a user question and the chunks retrieved "
    "from a single source file, decide which (if any) are relevant and write "
    "one concise summary (3-7 sentences) of what this file tells us about the "
    "question. Cross-reference chunks from this file when they relate. If "
    "nothing in the file is relevant, reply with exactly the token "
    "NOT_RELEVANT and nothing else."
)

TRIAGE_SYSTEM_CHUNK = (
    "You are a code analyst. Given a code chunk and a user question, decide "
    "whether the chunk is relevant. If yes, summarize concisely (3-5 sentences) "
    "what the chunk tells us about the question. If no, reply with exactly the "
    "token NOT_RELEVANT and nothing else."
)


@dataclass
class RelevantGroup:
    """A set of chunks the sub-agent assessed together, plus its summary."""
    hits: list[SearchHit]
    summary: str


@dataclass
class QueryResult:
    answer: str
    candidates: list[SearchHit]
    relevant_groups: list[RelevantGroup]
    tokens_in: int
    tokens_out: int
    reviews: list[ReviewResult] | None = None  # populated when --review is on


def _group_by_file(candidates: list[SearchHit]) -> list[list[SearchHit]]:
    """Group hits by file_path, preserving first-seen file order. Within
    each group, sort by start_line so the sub-agent reads in source order.
    """
    order: list[str] = []
    buckets: dict[str, list[SearchHit]] = {}
    for h in candidates:
        path = h.chunk.file_path
        if path not in buckets:
            buckets[path] = []
            order.append(path)
        buckets[path].append(h)
    return [sorted(buckets[p], key=lambda h: h.chunk.start_line) for p in order]


def _per_file_prompt(question: str, hits: list[SearchHit]) -> str:
    first = hits[0].chunk
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        blocks.append(textwrap.dedent(f"""\
            ### chunk {i}: {c.symbol_kind} `{c.symbol_name}` (lines {c.start_line}-{c.end_line}, score={h.score:.3f})

            ```
            {c.text}
            ```"""))
    return textwrap.dedent(f"""\
        Question:
        {question}

        File: `{first.file_path}` ({first.language})
        Retrieved chunks (in source order):

        {chr(10).join(blocks)}

        Apply the rules in the system message.""")


def _per_chunk_prompt(question: str, hit: SearchHit) -> str:
    c = hit.chunk
    return textwrap.dedent(f"""\
        Question:
        {question}

        Code chunk: {c.symbol_kind} `{c.symbol_name}` from `{c.file_path}` (lines {c.start_line}-{c.end_line}, {c.language})

        ```
        {c.text}
        ```

        Apply the rules in the system message.""")


def _aggregate_prompt(question: str, groups: list[RelevantGroup]) -> str:
    blocks = []
    for g in groups:
        path = g.hits[0].chunk.file_path
        symbols = ", ".join(f"`{h.chunk.symbol_name}`" for h in g.hits)
        top = max(h.score for h in g.hits)
        blocks.append(f"### `{path}` — {symbols} (top score={top:.3f})\n{g.summary}")
    return textwrap.dedent(f"""\
        Question:
        {question}

        Relevant code summaries (one block per file, ordered by retrieval score):

        {chr(10).join(blocks)}

        Synthesize a clear, accurate answer. Cite specific files and symbols.
        If the evidence is incomplete or contradictory, say so explicitly.""")


def query(
    question: str,
    store: ChunkStore,
    embedder: Embedder,
    *,
    top_k: int = 20,
    sub_model: str | None = None,
    root_model: str | None = None,
    concurrency: int = 10,
    retrieval_mode: str = "hybrid",     # "cosine" | "bm25" | "hybrid"
    rerank: bool = False,
    rerank_model: str | None = None,
    model_config: ModelConfig | None = None,
    review: bool = False,
    reviewer_model: str | None = None,
    review_rounds: int = 1,
    group: str = "file",                # "file" | "chunk"
) -> QueryResult:
    """Run a single-shot RAG query.

    Model selection: if `model_config` is provided, per-chunk triage uses
    the FAST tier and synthesis uses the SMART tier. `sub_model` and
    `root_model` are explicit string overrides for the same purpose
    (back-compat with earlier callers).

    `group` controls how retrieved chunks are batched into sub-calls:
      - "file"  (default): one sub-call per file, covering all retrieved
                chunks from that file. Better local context, fewer calls.
      - "chunk": one sub-call per chunk. Maximally parallel, no
                cross-chunk reasoning.
    """
    cfg = model_config or default_config()
    triage_model = sub_model or cfg.model_for(ModelTier.FAST)
    triage_thinking = cfg.thinking_for(ModelTier.FAST)
    synth_model = root_model or cfg.model_for(ModelTier.SMART)
    synth_thinking = cfg.thinking_for(ModelTier.SMART)

    reranker = Reranker(model_name=rerank_model) if rerank else None
    candidates = retrieve(
        question, store, embedder,
        top_k=top_k, mode=retrieval_mode,
        rerank=rerank, reranker=reranker,
    )

    if not candidates:
        return QueryResult(
            answer="(no chunks indexed; run `rlm-rag index --root <path>` first)",
            candidates=[],
            relevant_groups=[],
            tokens_in=0,
            tokens_out=0,
        )

    if group == "chunk":
        groups_in: list[list[SearchHit]] = [[h] for h in candidates]
        prompts = [_per_chunk_prompt(question, g[0]) for g in groups_in]
        triage_system = TRIAGE_SYSTEM_CHUNK
    else:
        groups_in = _group_by_file(candidates)
        prompts = [_per_file_prompt(question, g) for g in groups_in]
        triage_system = TRIAGE_SYSTEM_FILE

    summaries_raw = rlm_helper.llm_query_batch(
        prompts,
        system=triage_system,
        model=triage_model,
        concurrency=concurrency,
        thinking_budget=triage_thinking,
    )

    relevant: list[RelevantGroup] = []
    for hits, summary in zip(groups_in, summaries_raw):
        text = summary.strip()
        toks = text.split()
        if toks and toks[0] == "NOT_RELEVANT":
            continue
        relevant.append(RelevantGroup(hits=hits, summary=text))

    if not relevant:
        return QueryResult(
            answer=(
                "Top-k retrieval returned candidates, but none were judged "
                "relevant. Try rephrasing, raising --top-k, or switching "
                "--retrieval-mode."
            ),
            candidates=candidates,
            relevant_groups=[],
            tokens_in=0,
            tokens_out=0,
        )

    aggregate_prompt_text = _aggregate_prompt(question, relevant)
    final = rlm_helper.llm_query(
        aggregate_prompt_text,
        system="You are a precise code explainer. Be specific and grounded in the provided summaries.",
        model=synth_model,
        max_tokens=4096,
        thinking_budget=synth_thinking,
    )

    reviews_chain: list[ReviewResult] | None = None
    if review:
        # Evidence shown to the reviewer is the relevant summaries (what
        # the synthesizer actually saw), not the whole retrieved pool.
        evidence = aggregate_prompt_text
        reviewed = with_review(
            question, final, evidence,
            rounds=review_rounds,
            reviewer_model=reviewer_model,
            config=cfg,
        )
        final = reviewed.answer
        reviews_chain = reviewed.reviews

    tokens_in = tokens_out = 0
    try:
        bp = os.environ.get(rlm_helper.BUDGET_PATH_ENV)
        if bp and os.path.exists(bp):
            state = json.loads(open(bp).read())
            tokens_in = state.get("tokens_in", 0)
            tokens_out = state.get("tokens_out", 0)
    except Exception:
        pass

    return QueryResult(
        answer=final,
        candidates=candidates,
        relevant_groups=relevant,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        reviews=reviews_chain,
    )
