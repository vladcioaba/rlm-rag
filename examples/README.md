# Example: dependency graph of OpenCocosStudio

Real-world demonstration of `rlm-rag graph` against [OpenCocosStudio](https://github.com/) — a mixed Python + C++ codebase (41 source files / 504 chunks after build-dir filtering).

## Reproduce

```bash
# Build the graph (no embeddings — fast, doesn't require ANTHROPIC_API_KEY)
rlm-rag index --root /path/to/OpenCocosStudio --graph-only

# Three views
rlm-rag graph --root /path/to/OpenCocosStudio \
  --output opencocosstudio-packages.dot \
  --granularity package --package-depth 2

rlm-rag graph --root /path/to/OpenCocosStudio \
  --output opencocosstudio-files.dot \
  --granularity file

rlm-rag graph --root /path/to/OpenCocosStudio \
  --output opencocosstudio-externals.dot \
  --granularity package --include-external
```

Each command writes `<output>.dot` and (if `dot` is on PATH) renders an SVG.

## What you see

**`opencocosstudio-packages.png`** — top-level structure aggregated to two-level packages:

- The Python entry chain: `packaging/launcher.py → opencocosstudio/__main__.py → opencocosstudio/ui → opencocosstudio/{model, exporters}` with edge weights showing how many distinct imports cross each boundary.
- The C++ side: `app/AppDelegate.cpp` is the hub, pulling in 4 imports from `extensions/UILayoutEditor` plus several local `app/*.h` headers.
- `extensions/UILayoutEditor` is the biggest single subtree (20 files, 283 chunks — darker blue indicating density).
- Tests (`test_csd_roundtrip.py`, `test_edit_roundtrip.py`) reach directly into `opencocosstudio/model`.

**`opencocosstudio-files.png`** — same data, one node per file. Shows the internal structure of UILayoutEditor (panels importing each other, `Editor.h` central) and the per-file Python imports.

**`opencocosstudio-externals.png`** — adds external imports bucketed by top-level package name (`<ext>os`, `<ext>PySide6`, `<ext>ImGui`, etc.). Useful for spotting which third-party deps are most heavily used.

## How the graph is built

1. `rlm-rag index --graph-only` walks the source tree, dispatches each file to its language's chunker, and writes per-file rows into the SQLite tables `chunks`, `symbols`, `imports`, `calls`. Embeddings are skipped.
2. `rlm-rag graph` reads `imports` and resolves each import to a node:
   - Python relative imports (`from .sibling import X` inside `opencocosstudio/ui/main.py`) are resolved to absolute paths (`opencocosstudio.ui.sibling.X`) using the file's package context (chain of `__init__.py` ancestors).
   - C++ `#include "AppDelegate.h"` is resolved by basename match against indexed `.h` files.
   - Imports that don't match any indexed file are treated as external.
3. Node fill color and pen width scale with chunk count; edge labels show the number of distinct imports between the two nodes.

## Why some nodes are isolated

- `opencocosstudio/__init__.py` has no incoming edges because nothing in the codebase imports the package init by name.
- C++ system headers (`<vector>`, `<algorithm>`, `<filesystem>`) match nothing in the index and only show up with `--include-external`.
- Files with only external dependencies (e.g. a small standalone script) appear as isolated nodes.
