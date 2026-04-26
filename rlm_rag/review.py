"""Adversarial review pass.

Given a question, an answer, and the evidence used, run a separate critique
model that looks for specific failure modes:
  - factual errors not supported by the evidence
  - missing evidence the answer should have addressed
  - unjustified leaps or hand-waves
  - over-confidence on ambiguous evidence

The critique returns a structured `ReviewResult` with severity and a list
of issues. `regenerate_with_review` feeds the critique back to the original
synthesizer so it can produce an improved answer; the loop runs up to
`review_rounds` times or stops early on `severity == ok`.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass

from . import _rlm_helper as rlm_helper
from .models import ModelConfig, ModelTier, default_config


REVIEW_SYSTEM = textwrap.dedent("""\
    You are a strict adversarial reviewer of code-research answers. You have
    seen the user's question, the answer drafted by another model, and the
    evidence cited.

    Output a single JSON object inside a fenced ```json block, with fields:

      severity:   "ok" | "minor" | "major"
      issues:     list of strings, each a specific concrete problem
      missing:    list of strings, each a piece of evidence the answer should
                  have addressed but didn't

    Severity rubric:
      ok:    the answer is well-grounded; issues and missing may be empty
      minor: phrasing/clarity gaps, but no factual problems
      major: at least one concrete factual error, missing key evidence, or
             an unjustified claim

    Be specific. If a claim is unsupported, name the claim. If a piece of
    evidence is missing, name the evidence. Do not paraphrase the answer.
""")


@dataclass
class ReviewResult:
    severity: str          # "ok" | "minor" | "major"
    issues: list[str]
    missing: list[str]
    raw: str               # full response, in case parsing falls back


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_review(text: str) -> ReviewResult:
    """Parse a review response. On parse failure, classify severity by
    keyword heuristics so a malformed response doesn't silently say 'ok'.
    """
    m = _FENCED_JSON.search(text)
    payload = None
    if m:
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    if payload is None:
        try:
            payload = json.loads(text.strip())
        except Exception:
            payload = None

    if isinstance(payload, dict):
        sev = str(payload.get("severity", "")).lower().strip()
        if sev not in ("ok", "minor", "major"):
            sev = "minor"  # defensive: parsed but invalid severity → assume minor
        return ReviewResult(
            severity=sev,
            issues=list(payload.get("issues", []) or []),
            missing=list(payload.get("missing", []) or []),
            raw=text,
        )

    # Fallback: keyword heuristic.
    lower = text.lower()
    if any(t in lower for t in ("major", "incorrect", "wrong", "factually")):
        sev = "major"
    elif any(t in lower for t in ("minor", "could", "should clarify")):
        sev = "minor"
    else:
        sev = "minor"  # never trust an unparseable response as "ok"
    return ReviewResult(severity=sev, issues=[], missing=[], raw=text)


def review_answer(
    question: str,
    answer: str,
    evidence: str,
    *,
    reviewer_model: str | None = None,
    thinking_budget: int | None = None,
    config: ModelConfig | None = None,
) -> ReviewResult:
    cfg = config or default_config()
    model = reviewer_model or cfg.model_for(ModelTier.SMART)
    if thinking_budget is None:
        thinking_budget = cfg.thinking_for(ModelTier.SMART)

    prompt = textwrap.dedent(f"""\
        Question:
        {question}

        Drafted answer:
        ---
        {answer}
        ---

        Evidence cited:
        ---
        {evidence}
        ---

        Apply the rubric in the system message and produce the JSON object.""")

    raw = rlm_helper.llm_query(
        prompt,
        system=REVIEW_SYSTEM,
        model=model,
        max_tokens=2048,
        thinking_budget=thinking_budget,
    )
    return _parse_review(raw)


REGENERATE_SYSTEM = (
    "You are a precise code explainer. Your previous answer was reviewed and "
    "found to have issues. Produce a corrected, more grounded answer. Do not "
    "argue with the review — incorporate the corrections directly."
)


def regenerate_with_review(
    question: str,
    prior_answer: str,
    evidence: str,
    review: ReviewResult,
    *,
    model: str | None = None,
    thinking_budget: int | None = None,
    config: ModelConfig | None = None,
) -> str:
    cfg = config or default_config()
    model = model or cfg.model_for(ModelTier.SMART)
    if thinking_budget is None:
        thinking_budget = cfg.thinking_for(ModelTier.SMART)

    issues_block = "\n".join(f"- {i}" for i in review.issues) or "  (no specific issues listed)"
    missing_block = "\n".join(f"- {m}" for m in review.missing) or "  (no missing items listed)"
    prompt = textwrap.dedent(f"""\
        Question:
        {question}

        Prior answer (rejected):
        ---
        {prior_answer}
        ---

        Reviewer's issues:
        {issues_block}

        Reviewer's missing evidence:
        {missing_block}

        Evidence available:
        ---
        {evidence}
        ---

        Produce a corrected answer.""")

    return rlm_helper.llm_query(
        prompt,
        system=REGENERATE_SYSTEM,
        model=model,
        max_tokens=4096,
        thinking_budget=thinking_budget,
    )


@dataclass
class ReviewedAnswer:
    answer: str
    reviews: list[ReviewResult]
    rounds_run: int


def with_review(
    question: str,
    initial_answer: str,
    evidence: str,
    *,
    rounds: int = 1,
    reviewer_model: str | None = None,
    config: ModelConfig | None = None,
) -> ReviewedAnswer:
    """Run up to `rounds` review-and-regenerate cycles. Stops early on
    severity=='ok'. Returns the final answer plus every round's review
    so callers can show the audit trail.
    """
    rounds = max(1, min(int(rounds), 5))
    answer = initial_answer
    reviews: list[ReviewResult] = []
    for r in range(rounds):
        review = review_answer(
            question, answer, evidence,
            reviewer_model=reviewer_model, config=config,
        )
        reviews.append(review)
        if review.severity == "ok":
            return ReviewedAnswer(answer=answer, reviews=reviews, rounds_run=r + 1)
        answer = regenerate_with_review(
            question, answer, evidence, review, config=config,
        )
    return ReviewedAnswer(answer=answer, reviews=reviews, rounds_run=rounds)
