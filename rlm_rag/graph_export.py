"""Export the indexed call/import graph as Graphviz DOT.

Two granularities:
  - file:    one node per indexed source file; edges = imports (file → module).
             Best for small codebases (<100 files).
  - package: nodes aggregated to top-level directory (e.g. `opencocosstudio/ui`
             becomes the `ui` package). Best for medium-to-large codebases.

Edge weights = number of imports between the two nodes (after aggregation).
Node sizes = chunk count for that file or aggregated package.

Render with `dot -Tsvg out.dot > out.svg` (or PNG / PDF).
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .store import ChunkStore


@dataclass
class GraphStats:
    nodes: int
    edges: int
    granularity: str          # "file" | "package"
    dot_path: Path
    rendered_path: Path | None  # only set if `dot` was found and ran


# ---------- aggregation -------------------------------------------------

def _package_for(rel_path: str, depth: int = 1) -> str:
    """Return the top-N path components as the package id, e.g.
    'opencocosstudio/ui/widgets/foo.py' with depth=2 → 'opencocosstudio/ui'.
    Files at the root get 'root'.
    """
    parts = Path(rel_path).parts
    if len(parts) <= 1:
        return "root"
    return "/".join(parts[:depth])


def _module_to_node(module: str, indexed_files: set[str], depth: int) -> str | None:
    """Try to map an imported module name to a node in our graph.

    Three resolution strategies, tried in order:

      1. Python-style fully-qualified path: `foo.bar.baz` → `foo/bar/baz.py`.
         Falls back to progressively shorter prefixes if the full path
         doesn't match (handles symbol-imported-from-module).

      2. C++/C#-style relative path: imports like `app/Foo.h` or `Models/User.cs`
         match the file directly when present in the index.

      3. C++ basename match: `AppDelegate.h` matches any indexed file whose
         basename is `AppDelegate.h`. Coarse — picks the first match if there
         are duplicates — but useful for typical project layouts.

    Returns None when nothing matches (treat as external dep).
    """
    # Strategy 2: literal path lookup.
    if module in indexed_files:
        return _package_for(module, depth) if depth else module

    # Strategy 1: dotted module → progressive prefix.
    parts = module.split(".")
    while parts:
        candidate = "/".join(parts)
        for f in indexed_files:
            if f == candidate + ".py" or f == candidate + "/__init__.py":
                return _package_for(f, depth) if depth else f
            if f.startswith(candidate + "/"):
                return _package_for(f, depth) if depth else f
        parts.pop()

    # Strategy 3: basename match (mainly for C/C++ #include "foo.h" style).
    if "/" in module or module.endswith((".h", ".hpp", ".hh", ".hxx")):
        leaf = module.rsplit("/", 1)[-1]
        for f in indexed_files:
            if f.endswith("/" + leaf) or f == leaf:
                return _package_for(f, depth) if depth else f

    return None


# ---------- DOT writers --------------------------------------------------

def _safe_id(s: str) -> str:
    """Make a Graphviz-safe node id."""
    return '"' + s.replace('"', '\\"') + '"'


def _write_dot(
    nodes: dict[str, dict],     # node_id → {chunks: int, files: int}
    edges: Counter,             # (src, dst) → weight
    out_path: Path,
    title: str,
    granularity: str,
) -> None:
    lines = [
        "digraph rlm_rag_dependencies {",
        f'  label="{title}";',
        '  labelloc="t";',
        '  fontname="Helvetica";',
        '  rankdir=LR;',
        '  node [shape=box, fontname="Helvetica", fontsize=10, style=filled, fillcolor="#f0f0f0"];',
        '  edge [fontname="Helvetica", fontsize=8, color="#888888"];',
        "",
    ]

    if not nodes:
        lines.append('  empty [label="(no indexed files)", shape=plaintext];')
    else:
        # Scale node fillcolor / penwidth by chunk count.
        max_chunks = max((n["chunks"] for n in nodes.values()), default=1)
        for node_id, info in sorted(nodes.items()):
            label = node_id
            if granularity == "package":
                label = f'{node_id}\\n{info["files"]}f / {info["chunks"]}c'
            else:
                label = f'{node_id}\\n{info["chunks"]}c'
            # Heatmap fill: bigger packages = darker.
            intensity = min(0.85, 0.2 + 0.65 * (info["chunks"] / max_chunks))
            grey = int(255 - 255 * intensity * 0.6)
            color = f"#{grey:02x}{grey:02x}f0"
            penwidth = 1 + 2 * (info["chunks"] / max_chunks)
            lines.append(
                f'  {_safe_id(node_id)} [label="{label}", fillcolor="{color}", penwidth={penwidth:.2f}];'
            )
        lines.append("")
        for (src, dst), weight in sorted(edges.items()):
            penwidth = 1 + min(4, weight / 2)
            lines.append(
                f'  {_safe_id(src)} -> {_safe_id(dst)} '
                f'[label="{weight if weight > 1 else ""}", penwidth={penwidth:.2f}];'
            )

    lines.append("}")
    out_path.write_text("\n".join(lines) + "\n")


# ---------- public entry point -------------------------------------------

def export_graph(
    store: ChunkStore,
    out_path: Path,
    granularity: str = "package",  # "file" | "package"
    package_depth: int = 1,
    include_external: bool = False,
    render_format: str | None = "svg",  # set None to skip rendering
    title: str = "rlm-rag dependency graph",
) -> GraphStats:
    """Build a dependency graph from the indexed import edges and write a
    Graphviz .dot file. If the `dot` executable is on PATH, also render it
    to <out_path>.<render_format>.
    """
    indexed_files = store.known_files()

    file_chunks: Counter = Counter()
    for r in store.db.execute("SELECT file_path, COUNT(*) FROM chunks GROUP BY file_path"):
        file_chunks[r[0]] = r[1]

    nodes: dict[str, dict] = {}
    files_per_node: dict[str, set[str]] = {}

    def node_id_for(rel: str) -> str:
        return _package_for(rel, package_depth) if granularity == "package" else rel

    for f in indexed_files:
        nid = node_id_for(f)
        nodes.setdefault(nid, {"chunks": 0, "files": 0})
        nodes[nid]["chunks"] += file_chunks.get(f, 0)
        files_per_node.setdefault(nid, set()).add(f)
    for nid, files in files_per_node.items():
        nodes[nid]["files"] = len(files)

    edges: Counter = Counter()
    rows = store.db.execute("SELECT file_path, module FROM imports").fetchall()
    for src_file, module in rows:
        src_node = node_id_for(src_file)
        dst_node = _module_to_node(module, indexed_files, package_depth if granularity == "package" else 0)
        if dst_node is None:
            if include_external:
                # Bucket external imports into one synthetic node per top-level package.
                dst_node = f"<ext>{module.split('.')[0]}"
                nodes.setdefault(dst_node, {"chunks": 0, "files": 0})
            else:
                continue
        if src_node == dst_node:
            continue
        edges[(src_node, dst_node)] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_dot(nodes, edges, out_path, title=title, granularity=granularity)

    rendered: Path | None = None
    if render_format and shutil.which("dot"):
        rendered = out_path.with_suffix("." + render_format)
        try:
            subprocess.run(
                ["dot", f"-T{render_format}", str(out_path), "-o", str(rendered)],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            rendered = None

    return GraphStats(
        nodes=len(nodes),
        edges=sum(edges.values()),
        granularity=granularity,
        dot_path=out_path,
        rendered_path=rendered,
    )
