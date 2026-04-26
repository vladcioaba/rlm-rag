# rlm-rag

> ⚠ **Alpha — depth varies by language.** Python is deeply supported via stdlib `ast`. JS / TS / Go / Rust / C# / C++ use regex extractors that are correct on common patterns but shallow on edge cases. See "Limits" below.

Code-aware RAG that uses [RLM](https://arxiv.org/abs/2512.24601)-style sub-calls for generation. Designed for asking real questions about real codebases:

- *"How does authentication work here?"* — the iterative root loop searches, navigates the call graph, and reads specific symbols before answering.
- *"What changes if I rename `authenticate`?"* — `callers_of` walks the call graph; the loop fetches each caller and reasons about impact.
- *"Where do we validate user input?"* — hybrid BM25+cosine retrieval handles both keyword-exact and semantic queries.

## Pipeline

| Stage | What it does |
|---|---|
| **Chunker** | Per-symbol chunks. Python via `ast`; JS/TS/Go/Rust/C#/C++ via regex. Captures function/class/method bodies, plus a module-level remainder for imports/constants. Also extracts each chunk's import list and call sites for the graph layer. |
| **Embedder** | Local embeddings via `sentence-transformers/all-MiniLM-L6-v2` (~80 MB, CPU-friendly). Vectors are unit-normalized so cosine == dot product. |
| **Store** | Single-file SQLite. Tables: `chunks` (text + float32 embedding BLOB), `symbols`, `imports`, `calls`. |
| **Vector backend** | Default `numpy` (in-memory matrix). Optional `faiss` (auto-activated if installed and chunk count > 50K). |
| **Retrieval** | Cosine, BM25 (in-process Okapi), or hybrid via reciprocal rank fusion. Optional cross-encoder reranker on top. |
| **Query (one-shot)** | retrieve → parallel per-chunk relevance/summary (Haiku, with cached system prompt) → single Sonnet aggregation. |
| **Query (iterative)** | RLM-style root loop: model emits one action per turn (`search`, `find`, `grep`, `callers`, `imports`, `files_importing`, `fetch`, `final`); bounded by `max_iterations` and an LLM-call budget. |

## Install

```bash
git clone https://github.com/vladcioaba/rlm-rag.git
cd rlm-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                    # core
pip install -e ".[treesitter]"      # optional — proper C++/C# parsing (recommended for game-dev)
pip install -e ".[faiss]"           # optional — faster vector search for >50K chunks
```

Then configure an LLM provider (one of the below). First run downloads the embedding model (~80MB).

### LLM provider

`rlm-rag` ships with two backends — Anthropic native, and OpenAI-compatible (which covers OpenAI, Azure OpenAI, OpenRouter, Ollama, LM Studio, GitHub Models, and any service that speaks the OpenAI Chat Completions protocol).

The provider is auto-detected from whichever API key is set. Override with `LLM_PROVIDER`.

```bash
# Anthropic (default if ANTHROPIC_API_KEY is set)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI native
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai          # only needed if you also have an Anthropic key

# Azure OpenAI
export OPENAI_API_KEY=<azure-key>
export LLM_BASE_URL=https://<resource>.openai.azure.com/openai/v1
export LLM_PROVIDER=openai

# OpenRouter (proxies Claude / GPT / Gemini / Llama / Mistral under one API)
export OPENAI_API_KEY=sk-or-...
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_PROVIDER=openai

# GitHub Models (separate product from Copilot — has a free tier)
export OPENAI_API_KEY=<your-github-pat>
export LLM_BASE_URL=https://models.github.ai/inference
export LLM_PROVIDER=openai

# Local model via Ollama
ollama serve &  # in another terminal
export OPENAI_API_KEY=ignored          # Ollama doesn't check it but the SDK requires the var
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_PROVIDER=openai
export RLM_RAG_OPENAI_FAST=llama3.1
export RLM_RAG_OPENAI_BALANCED=llama3.1:70b
export RLM_RAG_OPENAI_SMART=llama3.1:70b

# Local model via LM Studio (start the server in the LM Studio app first)
export OPENAI_API_KEY=ignored
export LLM_BASE_URL=http://localhost:1234/v1
export LLM_PROVIDER=openai
```

Per-tier model picks default to claude-haiku/sonnet/opus on Anthropic, and gpt-4o-mini/gpt-4o/gpt-4o on OpenAI. Override per tier via `RLM_RAG_FAST_MODEL` / `RLM_RAG_BALANCED_MODEL` / `RLM_RAG_SMART_MODEL`, or specifically for OpenAI defaults via `RLM_RAG_OPENAI_FAST` / `_BALANCED` / `_SMART`.

#### A note on GitHub Copilot specifically

Copilot is an editor product, not a programmable backend — there's no general-purpose chat completions API exposed to third-party apps. If you want a "Copilot-like" experience with `rlm-rag`, the closest options are:

1. **GitHub Models** (different product, link above) — OpenAI-compatible, free tier, lets you call GPT-4o, GPT-5, Llama, Phi, etc. from `rlm-rag`.
2. **OpenRouter** — gives you Claude/GPT/Gemini under one API.
3. **A native key** — Anthropic, OpenAI, etc. directly.

## Use

```bash
# 1. Index your codebase. Stores at <root>/.rlm-rag/index.db
rlm-rag index --root /path/to/repo

# (Faster: graph-only mode skips embeddings — useful when you only want
# `rlm-rag graph` and not `query`.)
rlm-rag index --root /path/to/repo --graph-only

# 2a. Single-shot query (fast, simple).
rlm-rag query "how does authentication work?" --root /path/to/repo
rlm-rag query "what calls hash_password?" --root /path/to/repo --mode bm25
rlm-rag query "..." --root /path/to/repo --rerank --rerank-model mxbai-rerank-base

# 2b. Iterative root loop (RLM-shaped — searches, navigates the graph, reasons).
rlm-rag iterate "trace what would break if I renamed authenticate" --root /path/to/repo
rlm-rag iterate "..." --root /path/to/repo --max-iterations 10 --show-iterations

# 3. PR / diff impact analysis. Three input modes:
rlm-rag pr --root /path/to/repo --diff-file /tmp/my.diff
rlm-rag pr --root /path/to/repo --git "main...HEAD"
git diff main | rlm-rag pr --root /path/to/repo

# 4. Visualize the import/call graph. .dot output is rendered to SVG
#    automatically when `dot` (graphviz) is installed.
rlm-rag graph --root /path/to/repo --output deps.dot                 # package level
rlm-rag graph --root /path/to/repo --output deps.dot --granularity file
rlm-rag graph --root /path/to/repo --output deps.dot --include-external
```

See [`examples/`](examples/) for a real-world demo against OpenCocosStudio.

Re-running `index` only re-embeds files whose `sha1` changed. Deleted files are pruned from chunks AND graph tables. Indexing is parallelized across files (default 8 workers) and embeddings are batched in one big call.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` *or* `OPENAI_API_KEY` | *(one required)* | Picks the provider; can also be set via `LLM_API_KEY` |
| `LLM_PROVIDER` | auto-detect | `anthropic` \| `openai`; explicit override |
| `LLM_BASE_URL` | OpenAI public | Endpoint for the OpenAI-compatible provider (Azure / OpenRouter / Ollama / LM Studio / GitHub Models) |
| `LLM_API_KEY` | *(unset)* | Generic key fallback used by either provider |
| `RLM_RAG_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RLM_RAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for `--rerank` |
| `RLM_RAG_FAST_MODEL` | provider default | Fast tier (per-chunk triage) |
| `RLM_RAG_BALANCED_MODEL` | provider default | Balanced tier (iterative root) |
| `RLM_RAG_SMART_MODEL` | provider default | Smart tier (synthesis, review) |
| `RLM_RAG_OPENAI_FAST` / `_BALANCED` / `_SMART` | gpt-4o-mini / gpt-4o / gpt-4o | OpenAI tier defaults; overrideable for Ollama / OpenRouter etc. |
| `RLM_RAG_FAST_THINKING_BUDGET` | `0` | Extended thinking tokens for fast tier (translates to `reasoning_effort` on OpenAI o-series) |
| `RLM_RAG_BALANCED_THINKING_BUDGET` | `0` | Same, balanced tier |
| `RLM_RAG_SMART_THINKING_BUDGET` | `0` | Same, smart tier |
| `RLM_RAG_CONFIG` | `~/.rlm-rag/config.toml` | Persistent config file path |

### Persistent config (`~/.rlm-rag/config.toml`)

```toml
[tiers]
fast      = "claude-haiku-4-5"
balanced  = "claude-sonnet-4-6"
smart     = "claude-opus-4-7"

[thinking]
fast      = 0
balanced  = 0
smart     = 8000        # extended thinking on the synthesizer

[review]
enabled        = false
reviewer_model = "claude-opus-4-7"
rounds         = 1

[retrieval]
mode         = "hybrid"
rerank       = false
rerank_model = "ms-marco-mini"

[budget]
concurrency = 10
```

Resolution order: CLI flags > config file > env vars > package defaults.

### Power-user invocation

```bash
rlm-rag iterate "trace what would break if I renamed authenticate" \
  --root /path/to/repo \
  --fast-model claude-haiku-4-5 \
  --balanced-model claude-sonnet-4-6 \
  --smart-model claude-opus-4-7 \
  --smart-thinking-budget 8000 \
  --rerank --rerank-model mxbai-rerank-base
```

```bash
rlm-rag query "audit auth flow for race conditions" \
  --root /path/to/repo \
  --review --reviewer-model claude-opus-4-7 --review-rounds 2 \
  --smart-thinking-budget 12000
```

## Tests

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

103 unit tests — chunkers (Python + JS/TS/Go/Rust/C#/C++ regex + tree-sitter), store, indexer, BM25, RRF, retrieval pipeline, graph queries, diff parser, model tiering, review parser, config loader, thinking-budget plumbing, both providers (Anthropic + OpenAI-compatible), graph export. All offline; provider tests mock the HTTP layer so no API keys needed; tree-sitter tests skip cleanly when grammars aren't installed.

## What's in this version

- **Multi-language chunking** (Python, JS, TS, Go, Rust, C#, C++). Python is deep; the others are correct on common patterns and degrade gracefully (fall back to a whole-file chunk on edge cases). Tree-sitter is on the roadmap for proper non-Python depth.

  | Language | Extensions | Symbol kinds extracted |
  |---|---|---|
  | Python | `.py` | function, async_function, class, method, module |
  | JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | function, class, type, module |
  | TypeScript | `.ts`, `.tsx` | function, class, type (interface/type alias), module |
  | Go | `.go` | function, type (struct/interface), module |
  | Rust | `.rs` | function, type (struct/enum/trait), impl, module |
  | C# | `.cs` | namespace, class, interface, struct, record, enum, method, constructor |
  | C++ | `.cpp` `.cc` `.cxx` `.h` `.hpp` `.hh` `.hxx` `.inl` | namespace, class, struct, enum, function (incl. `Class::method`) |

- **Symbol/import/call graph** persisted alongside chunks. Query helpers: `find_symbol`, `grep_symbol`, `callers_of`, `imports_of`, `files_importing`, `get_chunk`.
- **Hybrid retrieval** — BM25 (in-process Okapi) + cosine, fused via reciprocal rank fusion (k=60). BM25 alone, cosine alone, and hybrid are all selectable per query.
- **Cross-encoder reranker** (optional, `--rerank`). Defaults to `ms-marco-MiniLM-L-6-v2`. Pass `--rerank-model` to swap in something stronger; supported aliases: `ms-marco-mini`, `ms-marco-base`, `bge-reranker-base`, `bge-reranker-large`, `mxbai-rerank-base`. Or pass any HuggingFace cross-encoder path. Gracefully no-ops if the model can't load.

- **Tree-sitter parsing** (optional, `pip install -e ".[treesitter]"`). When installed, C++ and C# use tree-sitter for proper symbol extraction (multi-line signatures, templates, partial classes, nested namespaces). Without it, both languages fall back to the regex extractors. Other languages keep regex.

- **PR/diff impact analysis** (`rlm-rag pr`). Parses a unified diff, finds which indexed symbols overlap the changes, looks up callers via the call graph, and runs a sub-LLM call per impacted symbol asking whether the change is safe. Three input modes: `--diff-file <path>`, `--git <rev_spec>`, or piped on stdin.

- **Concurrent indexing.** Files are read, hashed, and chunked in parallel via a thread pool (default 8 workers). All chunks across all changed files are embedded in one batch call to amortize sentence-transformers warmup.

- **Model tiering** (`--fast-model / --balanced-model / --smart-model`). Per-task-type model routing: per-chunk triage uses the fast tier (default Haiku), iterative root uses the balanced tier (default Sonnet), synthesis and review use the smart tier (default Opus). Override per-tier via CLI, env vars, or the persistent config file.

- **Extended thinking** (`--fast-thinking-budget / --balanced-thinking-budget / --smart-thinking-budget`). Sonnet and Opus 4.x can be configured to think before responding. Setting `--smart-thinking-budget 8000` makes the synthesizer reason for 8K tokens before producing the answer. Off by default (cost trade-off).

- **Adversarial reviewer** (`--review`). After the synthesizer produces an answer, a separate model (default: smart tier) critiques it against the same evidence and emits a structured `{severity, issues, missing}` JSON response. On non-`ok` severity, the answer is regenerated with the critique as feedback, up to `--review-rounds N` cycles (default 1, max 5). Severity defaults to `minor` rather than `ok` if the critique can't be parsed — defensive against silent failures.

- **Persistent config** (`~/.rlm-rag/config.toml`, override via `$RLM_RAG_CONFIG`). All tier/thinking/review/retrieval defaults live here. Resolution order: CLI flags > config file > env vars > package defaults.

- **Multi-provider** (Anthropic native + OpenAI-compatible). The OpenAI backend works against OpenAI public, Azure OpenAI, OpenRouter, Ollama, LM Studio, and GitHub Models — anywhere that speaks OpenAI Chat Completions. Provider auto-detects from which API key is set; override via `LLM_PROVIDER`. Pluggable through `rlm_rag.providers` if you want to add a new backend.
- **faiss backend** (optional, via `pip install -e ".[faiss]"`). Auto-activates when chunk count crosses 50K. Falls back to numpy otherwise.
- **Iterative RLM root loop** (`rlm-rag iterate`). Bounded by `--max-iterations` and the underlying LLM call budget.
- **One-shot query** (`rlm-rag query`) — retrieve → triage → synthesize. Faster and cheaper when you don't need iteration.

## Limits

- **Multi-language depth.** Regex extractors miss:
  - Multi-line declarations where the opening `{` is on the next line (some C++ / C# styles).
  - C++ templates with newlines inside `<...>`, function pointers, multiple definitions per line.
  - C# expression-bodied methods (`public int X => 1;`) and properties (those are intentionally not chunked as methods).
  - Method definitions outside the immediately matched class — for C++, `Class::method` is captured but not linked back to its class.
  - Pure-declaration-only files (header-only with no inline bodies). The closing-brace counter requires at least one `{...}` block to anchor; pure forward declarations may surface as a single `<file>` chunk.
  - Nested function declarations in any language.
  
  Python via `ast` doesn't have these issues.
- **Call graph is name-only.** "Callers of `authenticate`" matches any call to a function literally named `authenticate`, regardless of which module it's defined in. Python's dynamic dispatch (decorators, `getattr`, etc.) is invisible to AST parsing.
- **In-memory cosine** caps practical scale around ~50K chunks. Above that, install the `faiss` extra.
- **No reranker training.** Uses an off-the-shelf MS MARCO cross-encoder, which is general-purpose web-text. A code-tuned reranker would be better; that's not in this version.
- **No incremental embedding within a file.** A 1-line change re-embeds every chunk in that file.
- **Single root model in the iterative loop.** No depth>1 recursion.

If any of these is the blocker for your use case, open an issue.

## Roadmap

- [x] Tree-sitter integration for C++ and C# (opt-in via `[treesitter]` extra)
- [ ] Tree-sitter for the remaining languages (JS/TS/Go/Rust)
- [x] Reranker model selection (5 curated alternatives + arbitrary HF paths)
- [ ] Code-tuned reranker (no good off-the-shelf option exists yet)
- [ ] Streaming sub-call output for long answers
- [x] Concurrent index builds
- [x] Model tiering (fast/balanced/smart) per task type
- [x] Extended thinking budget per tier
- [x] Adversarial reviewer with iterative regenerate
- [x] Persistent config file
- [ ] Richer graph: scope-aware name resolution (genuinely months of work — not promised)
- [x] PR / git-diff input mode
- [x] Multi-provider LLM backends (Anthropic + OpenAI-compatible)
- [ ] Native Google Gemini provider (currently usable via OpenRouter)
- [ ] Streaming sub-call output

## License

MIT — see [`LICENSE`](LICENSE).
