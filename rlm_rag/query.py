"""Single-shot RAG query: retrieve → triage in parallel → synthesize.

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

PER_CHUNK_SYSTEM = (
    "You are a code analyst. Given a code chunk and a user question, decide "
    "whether the chunk is relevant. If yes, summarize concisely (3-5 sentences) "
    "what the chunk tells us about the question. If no, reply with exactly the "
    "token NOT_RELEVANT and nothing else."
)


@dataclass
class QueryResult:
    answer: str
    candidates: list[SearchHit]
    relevant_summaries: list[tuple[SearchHit, str]]
    tokens_in: int
    tokens_out: int
    reviews: list[ReviewResult] | None = None  # populated when --review is on


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


def _aggregate_prompt(question: str, summaries: list[tuple[SearchHit, str]]) -> str:
    blocks = [
        f"### `{h.chunk.file_path}` — `{h.chunk.symbol_name}` (score={h.score:.3f})\n{s}"
        for h, s in summaries
    ]
    return textwrap.dedent(f"""\
        Question:
        {question}

        Relevant code summaries (ordered by retrieval score):

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
) -> QueryResult:
    """Run a single-shot RAG query.

    Model selection: if `model_config` is provided, per-chunk triage uses
    the FAST tier and synthesis uses the SMART tier. `sub_model` and
    `root_model` are explicit string overrides for the same purpose
    (back-compat with earlier callers).
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
            relevant_summaries=[],
            tokens_in=0,
            tokens_out=0,
        )

    prompts = [_per_chunk_prompt(question, h) for h in candidates]
    summaries_raw = rlm_helper.llm_query_batch(
        prompts,
        system=PER_CHUNK_SYSTEM,
        model=triage_model,
        concurrency=concurrency,
        thinking_budget=triage_thinking,
    )

    relevant: list[tuple[SearchHit, str]] = []
    for hit, summary in zip(candidates, summaries_raw):
        toks = summary.strip().split()
        if toks and toks[0] == "NOT_RELEVANT":
            continue
        relevant.append((hit, summary.strip()))

    if not relevant:
        return QueryResult(
            answer=(
                "Top-k retrieval returned candidates, but none were judged "
                "relevant. Try rephrasing, raising --top-k, or switching "
                "--retrieval-mode."
            ),
            candidates=candidates,
            relevant_summaries=[],
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
        relevant_summaries=relevant,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        reviews=reviews_chain,
    )
