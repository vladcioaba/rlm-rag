"""Tests for the Graphviz graph exporter (no rendering, no API)."""

from __future__ import annotations

from rlm_rag.graph_export import _module_to_node, _package_for, export_graph
from rlm_rag.indexer import index_directory
from rlm_rag.store import ChunkStore


# ---------- helpers -----------------------------------------------------

def test_package_for_simple():
    assert _package_for("myapp/ui/widgets/foo.py", depth=1) == "myapp"
    assert _package_for("myapp/ui/widgets/foo.py", depth=2) == "myapp/ui"
    assert _package_for("standalone.py", depth=1) == "root"


def test_module_to_node_resolves_internal_to_package():
    indexed = {
        "myapp/ui/widgets/canvas.py",
        "myapp/model/scene.py",
    }
    # Internal: dotted module → maps onto the right indexed file → package id.
    node = _module_to_node("myapp.model.scene", indexed, depth=1)
    assert node == "myapp"


def test_module_to_node_returns_none_for_external():
    indexed = {"a/b.py"}
    assert _module_to_node("numpy", indexed, depth=1) is None
    assert _module_to_node("os.path", indexed, depth=1) is None


# ---------- end-to-end via the indexer + exporter -----------------------

def test_export_graph_writes_dot_and_counts_nodes_edges(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g.db")
    index_directory(tmp_repo, store, fake_embedder)

    out = tmp_path / "deps.dot"
    stats = export_graph(store, out_path=out, granularity="file", render_format=None)

    assert out.exists()
    body = out.read_text()
    assert body.startswith("digraph ")
    # tmp_repo has src/auth.py and src/models.py.
    assert "src/auth.py" in body or "src" in body
    assert stats.nodes >= 2


def test_export_graph_package_granularity_aggregates(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g2.db")
    index_directory(tmp_repo, store, fake_embedder)

    out = tmp_path / "pkg.dot"
    stats = export_graph(store, out_path=out, granularity="package", package_depth=1, render_format=None)
    body = out.read_text()
    # Top-level dirs in tmp_repo are 'src' and 'docs'.
    assert "src" in body
    assert stats.granularity == "package"


def test_export_graph_includes_external_when_asked(tmp_repo, fake_embedder, tmp_path):
    store = ChunkStore(tmp_path / "g3.db")
    index_directory(tmp_repo, store, fake_embedder)

    out = tmp_path / "ext.dot"
    export_graph(store, out_path=out, granularity="package",
                 include_external=True, render_format=None)
    body = out.read_text()
    # tmp_repo's auth.py imports `hashlib` (stdlib, external relative to repo).
    assert "<ext>hashlib" in body or "hashlib" in body


def test_export_graph_empty_store_does_not_crash(tmp_path):
    store = ChunkStore(tmp_path / "empty.db")
    out = tmp_path / "empty.dot"
    stats = export_graph(store, out_path=out, render_format=None)
    assert stats.nodes == 0
    assert stats.edges == 0
    assert "(no indexed files)" in out.read_text()
