"""rlm-rag command-line interface.

  rlm-rag index --root <path>
  rlm-rag query "<q>" --root <path> [--top-k 20] [--mode hybrid|cosine|bm25] [--rerank]
  rlm-rag iterate "<q>" --root <path> [--max-iterations 5]
  rlm-rag stats --root <path>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .diff_mode import analyze_diff, diff_from_git
from .embedder import Embedder
from .graph_export import export_graph
from .indexer import index_directory
from .iterative import iterative_query
from .models import ModelConfig
from .query import query as run_query
from .store import ChunkStore


def _models_from_args(args: argparse.Namespace) -> ModelConfig:
    """Build a ModelConfig: config file first, then CLI flags override."""
    cfg = load_config()
    m = cfg.models
    if getattr(args, "fast_model", None):
        m.fast_model = args.fast_model
    if getattr(args, "balanced_model", None):
        m.balanced_model = args.balanced_model
    if getattr(args, "smart_model", None):
        m.smart_model = args.smart_model
    if getattr(args, "fast_thinking_budget", None) is not None:
        m.fast_thinking_budget = args.fast_thinking_budget
    if getattr(args, "balanced_thinking_budget", None) is not None:
        m.balanced_thinking_budget = args.balanced_thinking_budget
    if getattr(args, "smart_thinking_budget", None) is not None:
        m.smart_thinking_budget = args.smart_thinking_budget
    return m


def _add_model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fast-model", default=None,
                   help="Override fast tier (per-chunk triage, etc.). Default: claude-haiku-4-5")
    p.add_argument("--balanced-model", default=None,
                   help="Override balanced tier (iterative root). Default: claude-sonnet-4-6")
    p.add_argument("--smart-model", default=None,
                   help="Override smart tier (synthesis, review). Default: claude-opus-4-7")
    p.add_argument("--fast-thinking-budget", type=int, default=None,
                   help="Extended thinking tokens for fast tier (0 = off)")
    p.add_argument("--balanced-thinking-budget", type=int, default=None,
                   help="Extended thinking tokens for balanced tier (0 = off)")
    p.add_argument("--smart-thinking-budget", type=int, default=None,
                   help="Extended thinking tokens for smart tier (0 = off)")


def _add_review_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--review", action="store_true",
                   help="Run an adversarial reviewer pass on the answer")
    p.add_argument("--reviewer-model", default=None,
                   help="Reviewer model (default: smart tier)")
    p.add_argument("--review-rounds", type=int, default=None,
                   help="Max review-and-regenerate cycles (default: 1, max: 5)")


def _store_in_project(root: Path) -> Path:
    return root / ".rlm-rag" / "index.db"


def _store_in_user(root: Path) -> Path:
    # Hash the absolute path so two repos sharing a basename don't collide
    # and a moved repo gets a fresh slot rather than reusing a stale index.
    h = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    return Path.home() / ".rlm-rag" / "projects" / f"{root.name}-{h}" / "index.db"


def _find_existing_store(root: Path) -> Path | None:
    p = _store_in_project(root)
    if p.exists():
        return p
    p = _store_in_user(root)
    if p.exists():
        return p
    return None


def _resolve_index_path(root: Path, location: str | None) -> Path:
    """Pick where the index lives. If one already exists, use it. Otherwise
    honor `location` (project|user), or prompt when running interactively,
    or default to in-project for non-interactive invocations.
    """
    existing = _find_existing_store(root)
    if existing is not None:
        return existing
    if location == "project":
        return _store_in_project(root)
    if location == "user":
        return _store_in_user(root)
    if not sys.stdin.isatty():
        return _store_in_project(root)
    in_p = _store_in_project(root)
    in_u = _store_in_user(root)
    print(f"\nNo rlm-rag index found for {root}.")
    print("Where should it live?")
    print(f"  [1] In the project: {in_p.parent}")
    print(f"  [2] In your home:   {in_u.parent}")
    while True:
        choice = (input("Choose [1/2] (default 1): ").strip() or "1").lower()
        if choice in ("1", "p", "project"):
            return in_p
        if choice in ("2", "u", "user", "home"):
            return in_u
        print("please answer 1 or 2")


def _require_store(root: Path) -> Path | None:
    """For read-only commands: locate an existing index or print a hint."""
    p = _find_existing_store(root)
    if p is None:
        print(
            f"error: no rlm-rag index found for {root}\n"
            f"hint:  rlm-rag index --root {root}",
            file=sys.stderr,
        )
        return None
    return p


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: --root must be a directory: {root}", file=sys.stderr)
        return 2

    db_path = _resolve_index_path(root, args.index_location)
    store = ChunkStore(db_path)
    embedder = None if args.graph_only else Embedder()

    def progress(rel: str, action: str) -> None:
        if args.verbose or action in ("rebuilt", "removed"):
            print(f"  [{action:>9}] {rel}")

    mode = " (graph-only, skipping embeddings)" if args.graph_only else ""
    print(f"indexing {root}{mode}...")
    result = index_directory(root, store, embedder, progress=progress, graph_only=args.graph_only)
    print()
    print(f"  rebuilt:   {len(result.rebuilt)}")
    print(f"  unchanged: {len(result.unchanged)}")
    print(f"  removed:   {len(result.removed)}")
    print(f"  skipped:   {len(result.skipped)}")
    print(f"  chunks:    {result.total_chunks}")
    print(f"  index at:  {db_path}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    db = _require_store(root)
    if db is None:
        return 1

    store = ChunkStore(db)
    embedder = Embedder()
    file_cfg = load_config()
    review_enabled = args.review or file_cfg.review.enabled
    review_rounds = args.review_rounds if args.review_rounds is not None else file_cfg.review.rounds
    reviewer_model = args.reviewer_model or file_cfg.review.reviewer_model or None
    result = run_query(
        args.question,
        store=store,
        embedder=embedder,
        top_k=args.top_k,
        concurrency=args.concurrency,
        retrieval_mode=args.mode,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        model_config=_models_from_args(args),
        review=review_enabled,
        reviewer_model=reviewer_model,
        review_rounds=review_rounds,
    )

    if args.show_candidates:
        print("=== retrieved candidates ===")
        for h in result.candidates:
            c = h.chunk
            print(f"  {h.score:+.3f}  {c.file_path}:{c.start_line:>4}  "
                  f"{c.symbol_kind} {c.symbol_name}")
        print("\n=== judged relevant ===")
        for h, _ in result.relevant_summaries:
            c = h.chunk
            print(f"  {h.score:+.3f}  {c.file_path}:{c.start_line:>4}  "
                  f"{c.symbol_kind} {c.symbol_name}")
        print()

    print("=== answer ===")
    print(result.answer)
    if result.reviews:
        print()
        print("=== review trail ===")
        for i, rv in enumerate(result.reviews, 1):
            print(f"\nround {i} — severity: {rv.severity}")
            for issue in rv.issues:
                print(f"  issue:   {issue}")
            for missing in rv.missing:
                print(f"  missing: {missing}")
    if result.tokens_in or result.tokens_out:
        print(f"\n(tokens: in={result.tokens_in} out={result.tokens_out})")
    return 0


def cmd_iterate(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    db = _require_store(root)
    if db is None:
        return 1

    store = ChunkStore(db)
    embedder = Embedder()
    result = iterative_query(
        args.question,
        store=store,
        embedder=embedder,
        max_iterations=args.max_iterations,
        retrieval_mode=args.mode,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        initial_top_k=args.top_k,
        model_config=_models_from_args(args),
    )

    if args.show_iterations:
        print("=== iterations ===")
        for log in result.iterations:
            print(f"\n--- iter {log.iteration}: {log.action} ---")
            print(f"payload: {log.payload}")
            print(f"summary:\n{log.summary}")
        print()

    print("=== answer ===")
    print(result.answer)
    print(f"\n(stopped: {result.stopped_reason}, iterations: {len(result.iterations)})")
    return 0


def cmd_pr(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    db = _require_store(root)
    if db is None:
        return 1

    # Source the diff: explicit file, stdin, or `git diff <rev>`.
    if args.diff_file:
        diff_text = Path(args.diff_file).read_text()
    elif args.git:
        diff_text = diff_from_git(root, args.git)
    else:
        diff_text = sys.stdin.read()

    if not diff_text.strip():
        print("error: empty diff (use --diff-file PATH, --git REV, or pipe via stdin)",
              file=sys.stderr)
        return 2

    store = ChunkStore(db)
    impact = analyze_diff(diff_text, store)

    print("=== changed symbols ===")
    for s in impact.changed_symbols:
        callers = impact.callers_per_symbol.get(s.name, [])
        print(f"  {s.file_path}::{s.name}  ({s.kind}, {len(callers)} caller(s))")
    print()
    print("=== impact ===")
    print(impact.summary)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    db = _require_store(root)
    if db is None:
        return 1
    store = ChunkStore(db)
    print(f"chunks: {store.count()}")
    print(f"files:  {len(store.known_files())}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    db = _require_store(root)
    if db is None:
        return 1

    out = Path(args.output).resolve()
    store = ChunkStore(db)
    stats = export_graph(
        store,
        out_path=out,
        granularity=args.granularity,
        package_depth=args.package_depth,
        include_external=args.include_external,
        render_format=None if args.no_render else args.format,
        title=args.title or f"rlm-rag dependency graph: {root.name}",
    )

    print(f"nodes:   {stats.nodes}")
    print(f"edges:   {stats.edges}")
    print(f"dot:     {stats.dot_path}")
    if stats.rendered_path:
        print(f"render:  {stats.rendered_path}")
    elif not args.no_render:
        print("render:  (skipped — `dot` not on PATH; `brew install graphviz`)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rlm-rag",
        description="Code-aware RAG using RLM-style sub-calls for generation.",
    )
    p.add_argument("--version", action="version", version=f"rlm-rag {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="Build or refresh the index for a directory")
    pi.add_argument("--root", required=True)
    pi.add_argument("-v", "--verbose", action="store_true")
    pi.add_argument("--graph-only", action="store_true",
                    help="Skip embeddings; build only the symbol/import/call graph "
                         "(use this when you only want `rlm-rag graph` later, not `query`)")
    pi.add_argument("--index-location", choices=["project", "user"], default=None,
                    help="Where to put the index on first run. 'project' = "
                         "<root>/.rlm-rag/, 'user' = ~/.rlm-rag/projects/<name>-<hash>/. "
                         "If omitted and stdin is a tty you'll be prompted; "
                         "non-interactive runs default to 'project'.")
    pi.set_defaults(func=cmd_index)

    pq = sub.add_parser("query", help="Single-shot retrieve → triage → synthesize")
    pq.add_argument("question")
    pq.add_argument("--root", required=True)
    pq.add_argument("--top-k", type=int, default=20)
    pq.add_argument("--concurrency", type=int, default=10)
    pq.add_argument("--mode", choices=["cosine", "bm25", "hybrid"], default="hybrid")
    pq.add_argument("--rerank", action="store_true",
                    help="Rerank candidates with a cross-encoder")
    pq.add_argument("--rerank-model", default=None,
                    help="Reranker model alias or HF path (e.g. 'mxbai-rerank-base')")
    pq.add_argument("--show-candidates", action="store_true")
    _add_model_flags(pq)
    _add_review_flags(pq)
    pq.set_defaults(func=cmd_query)

    pit = sub.add_parser("iterate",
                         help="RLM-style iterative loop with graph navigation")
    pit.add_argument("question")
    pit.add_argument("--root", required=True)
    pit.add_argument("--max-iterations", type=int, default=5)
    pit.add_argument("--top-k", type=int, default=10,
                     help="initial retrieval depth before the loop starts")
    pit.add_argument("--mode", choices=["cosine", "bm25", "hybrid"], default="hybrid")
    pit.add_argument("--rerank", action="store_true")
    pit.add_argument("--rerank-model", default=None,
                     help="Reranker model alias or HF path (e.g. 'mxbai-rerank-base')")
    pit.add_argument("--show-iterations", action="store_true")
    _add_model_flags(pit)
    pit.set_defaults(func=cmd_iterate)

    ppr = sub.add_parser("pr", help="Analyze a diff or git rev range against the indexed call graph")
    ppr.add_argument("--root", required=True)
    grp = ppr.add_mutually_exclusive_group()
    grp.add_argument("--diff-file", help="Path to a unified-diff file")
    grp.add_argument("--git", help="git rev range, e.g. 'HEAD~5..HEAD' or 'main...HEAD'")
    ppr.set_defaults(func=cmd_pr)

    ps = sub.add_parser("stats", help="Show index size")
    ps.add_argument("--root", required=True)
    ps.set_defaults(func=cmd_stats)

    pg = sub.add_parser("graph", help="Export the dependency graph as Graphviz DOT")
    pg.add_argument("--root", required=True)
    pg.add_argument("--output", default="dependency-graph.dot",
                    help="Path for the .dot file (rendered output uses the same stem)")
    pg.add_argument("--granularity", choices=["file", "package"], default="package",
                    help="Aggregate per file or per top-level package directory")
    pg.add_argument("--package-depth", type=int, default=1,
                    help="When granularity=package, how many path components to keep")
    pg.add_argument("--include-external", action="store_true",
                    help="Show external imports (numpy, os, etc.) bucketed by top-level name")
    pg.add_argument("--format", default="svg",
                    help="Render format passed to `dot` (svg / png / pdf). Default: svg")
    pg.add_argument("--no-render", action="store_true",
                    help="Skip the `dot` render step; only emit the .dot file")
    pg.add_argument("--title", default=None)
    pg.set_defaults(func=cmd_graph)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
