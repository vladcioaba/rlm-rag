"""Unit tests for the Python AST chunker."""

from __future__ import annotations

from rlm_rag.chunker import chunk_python_text, chunk_python_file


SIMPLE = '''\
"""module docstring"""
import os

CONST = 42

def add(x, y):
    """Add two numbers."""
    return x + y

async def fetch(url):
    return url

class User:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"hi {self.name}"
'''


def test_chunks_top_level_functions_and_classes():
    chunks = chunk_python_text(SIMPLE, file_path="x.py")
    kinds = {(c.symbol_kind, c.symbol_name) for c in chunks}
    assert ("function", "add") in kinds
    assert ("async_function", "fetch") in kinds
    assert ("class", "User") in kinds


def test_module_chunk_captures_imports_and_constants():
    chunks = chunk_python_text(SIMPLE, file_path="x.py")
    mod = next((c for c in chunks if c.symbol_name == "<module>"), None)
    assert mod is not None
    assert "import os" in mod.text
    assert "CONST = 42" in mod.text
    # Function and class bodies should NOT be in the module remainder.
    assert "def add" not in mod.text
    assert "class User" not in mod.text


def test_chunks_carry_correct_line_numbers():
    chunks = chunk_python_text(SIMPLE, file_path="x.py")
    add = next(c for c in chunks if c.symbol_name == "add")
    assert add.start_line < add.end_line
    # Body must contain the def line.
    assert "def add(x, y)" in add.text


def test_syntax_error_falls_back_to_whole_file():
    bad = "def broken(:\n    pass\n"
    chunks = chunk_python_text(bad, file_path="bad.py")
    assert len(chunks) == 1
    assert chunks[0].symbol_kind == "module"
    assert "broken" in chunks[0].text


def test_empty_text_returns_empty_chunk_list_via_file():
    # chunk_python_text on a non-empty whitespace-only string still yields a
    # chunk; chunk_python_file on a strict-empty file returns [].
    chunks = chunk_python_text("   \n", file_path="ws.py")
    # Whitespace-only: AST parses fine; no top-level symbols; remainder empty
    # after .strip(); _whole_file_chunk fallback fires.
    assert len(chunks) == 1


def test_chunk_python_file_returns_empty_for_empty_file(tmp_path):
    p = tmp_path / "empty.py"
    p.write_text("")
    assert chunk_python_file(p) == []


def test_relative_imports_resolved_to_absolute_paths(tmp_path):
    """from .sibling import X inside myapp/ui/mainwindow.py should
    resolve to myapp.ui.sibling.X — i.e. the package context is
    detected from __init__.py ancestors.
    """
    pkg_root = tmp_path / "myapp"
    ui = pkg_root / "ui"
    ui.mkdir(parents=True)
    (pkg_root / "__init__.py").write_text("")
    (ui / "__init__.py").write_text("")
    (ui / "sibling.py").write_text("def x(): pass\n")
    f = ui / "mainwindow.py"
    f.write_text(
        "from .sibling import x\n"
        "from ..model.csd import load_csd\n"
        "import os\n"
    )
    chunks = chunk_python_file(f)
    imports = chunks[0].imports
    assert "myapp.ui.sibling.x" in imports
    assert "myapp.model.csd.load_csd" in imports
    assert "os" in imports  # absolute imports unchanged
