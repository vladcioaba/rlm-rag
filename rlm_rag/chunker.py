"""Per-symbol chunking, dispatched by file extension.

Python uses stdlib `ast` (accurate). JS/TS/Go/Rust use regex extractors
(shallow but no dependencies). The contract is the same Chunk dataclass —
swap in tree-sitter later without changing callers.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    file_path: str
    symbol_name: str         # e.g. "authenticate", "User", "<module>"
    symbol_kind: str         # "function" | "async_function" | "class" | "method" | "module"
    start_line: int          # 1-based, inclusive
    end_line: int            # 1-based, inclusive
    text: str                # raw source of the chunk
    language: str = "python"
    # Symbols this chunk references by name (for the call graph).
    calls: list[str] = field(default_factory=list)
    # Modules this file imports (file-level; same on every chunk from this file).
    imports: list[str] = field(default_factory=list)


# ---------- public dispatch ----------------------------------------------

EXTENSION_LANGUAGE = {
    ".py":  "python",
    ".js":  "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts":  "typescript",
    ".tsx": "typescript",
    ".go":  "go",
    ".rs":  "rust",
    ".cs":  "csharp",
    ".cpp": "cpp",
    ".cc":  "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".h":   "cpp",
    ".hpp": "cpp",
    ".hh":  "cpp",
    ".hxx": "cpp",
    ".h++": "cpp",
    ".inl": "cpp",
}


def language_of(path: Path | str) -> str | None:
    return EXTENSION_LANGUAGE.get(Path(path).suffix.lower())


def chunk_file(path: Path) -> list[Chunk]:
    """Read the file and dispatch to the right chunker. Empty files → []."""
    text = path.read_text(errors="replace")
    if not text.strip():
        return []
    lang = language_of(path)
    if lang is None:
        return []
    return chunk_text(text, file_path=str(path), language=lang)


def chunk_text(text: str, file_path: str, language: str) -> list[Chunk]:
    if language == "python":
        return _chunk_python(text, file_path)
    if language in ("javascript", "typescript"):
        return _chunk_jsts(text, file_path, language)
    if language == "go":
        return _chunk_go(text, file_path)
    if language == "rust":
        return _chunk_rust(text, file_path)
    if language == "csharp":
        return _chunk_csharp(text, file_path)
    if language == "cpp":
        return _chunk_cpp(text, file_path)
    return []


# ---------- Python (ast, deep) -------------------------------------------

def _chunk_python(text: str, file_path: str) -> list[Chunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [_whole_file_chunk(text, file_path, "python")]

    lines = text.split("\n")
    imports = _python_imports(tree, file_path=file_path)
    chunks: list[Chunk] = []
    covered: set[int] = set()

    for node in tree.body:
        kind = _python_kind(node)
        if kind is None:
            continue
        start = node.lineno
        end = node.end_lineno or start
        body = "\n".join(lines[start - 1:end])
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=node.name,
            symbol_kind=kind,
            start_line=start,
            end_line=end,
            text=body,
            language="python",
            calls=_python_calls(node),
            imports=imports,
        ))
        covered.update(range(start, end + 1))

        # For classes, also yield each method as its own chunk.
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                sk = _python_kind(sub)
                if sk in ("function", "async_function"):
                    s = sub.lineno
                    e = sub.end_lineno or s
                    chunks.append(Chunk(
                        file_path=file_path,
                        symbol_name=f"{node.name}.{sub.name}",
                        symbol_kind="method",
                        start_line=s,
                        end_line=e,
                        text="\n".join(lines[s - 1:e]),
                        language="python",
                        calls=_python_calls(sub),
                        imports=imports,
                    ))

    remainder_lines = [ln for i, ln in enumerate(lines, start=1) if i not in covered]
    remainder = "\n".join(remainder_lines).strip()
    if remainder:
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name="<module>",
            symbol_kind="module",
            start_line=1,
            end_line=len(lines),
            text=remainder,
            language="python",
            calls=[],
            imports=imports,
        ))

    if not chunks:
        chunks.append(_whole_file_chunk(text, file_path, "python", imports=imports))
    return chunks


def _python_kind(node: ast.AST) -> str | None:
    if isinstance(node, ast.FunctionDef):
        return "function"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_function"
    if isinstance(node, ast.ClassDef):
        return "class"
    return None


def _python_imports(tree: ast.AST, file_path: str = "") -> list[str]:
    """Extract imports from a Python module, resolving relative imports
    (`from .foo import X`, `from ..bar import Y`) to absolute paths based
    on the file's package context (chain of __init__.py-bearing dirs).
    """
    pkg_parts = _python_package_parts(file_path) if file_path else ()
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            level = getattr(node, "level", 0) or 0
            if level > 0:
                # Relative: keep len(pkg_parts) - (level - 1) leading parts.
                keep = len(pkg_parts) - (level - 1)
                if keep < 0:
                    # Can't resolve; fall back to the literal module name.
                    full_mod = mod
                else:
                    base = list(pkg_parts[:keep])
                    full_mod = ".".join(base + (mod.split(".") if mod else []))
            else:
                full_mod = mod
            for a in node.names:
                out.append(f"{full_mod}.{a.name}" if full_mod else a.name)
    return sorted(set(out))


def _python_package_parts(file_path: str) -> tuple[str, ...]:
    """Walk up from file_path collecting directories that contain an
    __init__.py. Returns ('opencocosstudio', 'ui') for
    'opencocosstudio/ui/mainwindow.py'. Returns () if file has no
    __init__.py ancestor.
    """
    p = Path(file_path)
    if not p.is_absolute():
        # Caller passed a relative path (e.g. from the indexer post-rewrite).
        # We can't probe the filesystem reliably, so derive from path parts:
        # treat every parent as a package part — accurate when the indexer
        # passes paths relative to a package root.
        return tuple(p.parts[:-1])
    parts: list[str] = []
    parent = p.parent
    while True:
        try:
            has_init = (parent / "__init__.py").exists()
        except OSError:
            break
        if has_init:
            parts.append(parent.name)
            parent = parent.parent
        else:
            break
    parts.reverse()
    return tuple(parts)


def _python_calls(node: ast.AST) -> list[str]:
    out: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = _call_name(sub.func)
            if name:
                out.append(name)
    return sorted(set(out))


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# ---------- JS / TS (regex, shallow) -------------------------------------

# Captures: function foo(...) { ... }, async function foo, class Foo, const foo = (..) =>,
# export {default ?,async ?} function|class, type/interface (TS).
_JS_PATTERNS = [
    (re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("), "function"),
    (re.compile(r"^\s*(?:export\s+(?:default\s+)?)?class\s+([A-Za-z_$][\w$]*)"), "class"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?\("), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:type|interface)\s+([A-Za-z_$][\w$]*)"), "type"),
]
# Multiple import styles: ES6 `from`, side-effect `import 'foo'`, and CommonJS `require(...)`.
_JS_IMPORTS = [
    re.compile(r"\bimport\s+.*?\bfrom\s+['\"]([^'\"]+)['\"]"),
    re.compile(r"\bimport\s+['\"]([^'\"]+)['\"]"),
    re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
]
_JS_CALL = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")


def _chunk_jsts(text: str, file_path: str, language: str) -> list[Chunk]:
    return _regex_chunk(
        text, file_path, language,
        symbol_patterns=_JS_PATTERNS,
        import_patterns=_JS_IMPORTS,
        call_pattern=_JS_CALL,
        block_open="{", block_close="}",
    )


# ---------- Go (regex, shallow) ------------------------------------------

_GO_PATTERNS = [
    (re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z_][\w]*)\s*\("), "function"),
    (re.compile(r"^\s*type\s+([A-Za-z_][\w]*)\s+(?:struct|interface)\b"), "type"),
]
_GO_IMPORT = re.compile(r"^\s*\"([^\"]+)\"", re.MULTILINE)  # inside an import block; coarse
_GO_CALL = re.compile(r"\b([A-Za-z_][\w]*)\s*\(")


def _chunk_go(text: str, file_path: str) -> list[Chunk]:
    return _regex_chunk(
        text, file_path, "go",
        symbol_patterns=_GO_PATTERNS,
        import_patterns=[_GO_IMPORT],
        call_pattern=_GO_CALL,
        block_open="{", block_close="}",
    )


# ---------- Rust (regex, shallow) ----------------------------------------

_RS_PATTERNS = [
    (re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][\w]*)\s*[<(]"), "function"),
    (re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_][\w]*)\b"), "type"),
    (re.compile(r"^\s*impl\s+(?:<[^>]*>\s+)?([A-Za-z_][\w]*)\b"), "impl"),
]
_RS_IMPORT = re.compile(r"^\s*use\s+([A-Za-z_][\w:]*)", re.MULTILINE)
_RS_CALL = re.compile(r"\b([A-Za-z_][\w]*)\s*\(")


def _chunk_rust(text: str, file_path: str) -> list[Chunk]:
    return _regex_chunk(
        text, file_path, "rust",
        symbol_patterns=_RS_PATTERNS,
        import_patterns=[_RS_IMPORT],
        call_pattern=_RS_CALL,
        block_open="{", block_close="}",
    )


# ---------- C# (regex, shallow) ------------------------------------------

# Order matters: more-specific patterns first so methods don't match before classes.
_CS_PATTERNS = [
    (re.compile(
        r"^\s*(?:(?:public|private|protected|internal|abstract|sealed|static|partial)\s+)*"
        r"class\s+([A-Za-z_]\w*)"
    ), "class"),
    (re.compile(
        r"^\s*(?:(?:public|private|protected|internal)\s+)*interface\s+([A-Za-z_]\w*)"
    ), "interface"),
    (re.compile(
        r"^\s*(?:(?:public|private|protected|internal|readonly)\s+)*struct\s+([A-Za-z_]\w*)"
    ), "struct"),
    (re.compile(
        r"^\s*(?:(?:public|private|protected|internal)\s+)*record\s+(?:class\s+|struct\s+)?([A-Za-z_]\w*)"
    ), "record"),
    (re.compile(
        r"^\s*(?:(?:public|private|protected|internal)\s+)*enum\s+([A-Za-z_]\w*)"
    ), "enum"),
    (re.compile(r"^\s*namespace\s+([A-Za-z_][\w\.]*)"), "namespace"),
    # Method: needs at least one access/lifecycle modifier + return type + name + ()
    (re.compile(
        r"^\s*(?:(?:public|private|protected|internal|static|virtual|override|"
        r"abstract|async|sealed|new|partial|extern)\s+)+"
        r"[\w<>,\?\[\]\.\s]+?\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*(?:where\s[^{]+)?\s*\{"
    ), "method"),
    # Constructor: modifier + Name(...) { (no return type before name)
    (re.compile(
        r"^\s*(?:public|private|protected|internal)\s+"
        r"([A-Za-z_]\w*)\s*\([^)]*\)\s*(?::\s*(?:base|this)\s*\([^)]*\))?\s*\{"
    ), "constructor"),
]
_CS_IMPORTS = [
    re.compile(r"^\s*using\s+(?:static\s+)?([A-Za-z_][\w\.]*)\s*;", re.MULTILINE),
    re.compile(r"^\s*using\s+[A-Za-z_]\w*\s*=\s*([A-Za-z_][\w\.]*)\s*;", re.MULTILINE),
]
_CS_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _chunk_csharp(text: str, file_path: str) -> list[Chunk]:
    # Prefer tree-sitter if available — it handles modifiers, multi-line
    # signatures, generics, and partial classes correctly.
    try:
        from .treesitter_extractors import extract_csharp
        ts = extract_csharp(text, file_path)
        if ts is not None:
            return ts
    except ImportError:
        pass
    return _regex_chunk(
        text, file_path, "csharp",
        symbol_patterns=_CS_PATTERNS,
        import_patterns=_CS_IMPORTS,
        call_pattern=_CS_CALL,
        block_open="{", block_close="}",
    )


# ---------- C++ (regex, shallow) -----------------------------------------

# C++ is famously hard to parse with regex. These patterns cover common
# idiomatic forms and intentionally accept some false negatives on edge
# cases (multi-line declarations, complex templates, function pointers).
_CPP_PATTERNS = [
    # class/struct definition (with body — declarations end with ; not {)
    (re.compile(
        r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+([A-Za-z_]\w*)\s*"
        r"(?:final\s+)?(?::[^{]+)?\s*\{"
    ), "class"),
    (re.compile(r"^\s*namespace\s+([A-Za-z_][\w:]*)\s*\{"), "namespace"),
    (re.compile(r"^\s*enum(?:\s+class|\s+struct)?\s+([A-Za-z_]\w*)"), "enum"),
    # Function definition with body. Captures `name` or `Class::name` or `operator+`.
    (re.compile(
        r"^\s*(?:(?:inline|static|virtual|explicit|constexpr|extern|friend|"
        r"[A-Za-z_]\w*::|template\s*<[^>]*>)\s+)*"
        r"[\w:&\*<>,\s]+?\s+"
        r"((?:[A-Za-z_]\w*::)*(?:[A-Za-z_]\w*|operator\s*\S+))\s*"
        r"\([^;]*?\)\s*"
        r"(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?"
        r"(?:=\s*\w+\s*)?\{"
    ), "function"),
]
_CPP_IMPORTS = [
    re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE),
]
_CPP_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _chunk_cpp(text: str, file_path: str) -> list[Chunk]:
    try:
        from .treesitter_extractors import extract_cpp
        ts = extract_cpp(text, file_path)
        if ts is not None:
            return ts
    except ImportError:
        pass
    return _regex_chunk(
        text, file_path, "cpp",
        symbol_patterns=_CPP_PATTERNS,
        import_patterns=_CPP_IMPORTS,
        call_pattern=_CPP_CALL,
        block_open="{", block_close="}",
    )


# ---------- Generic regex chunker ----------------------------------------

def _regex_chunk(
    text: str,
    file_path: str,
    language: str,
    symbol_patterns,
    import_patterns,
    call_pattern,
    block_open: str,
    block_close: str,
) -> list[Chunk]:
    lines = text.split("\n")
    imports_set: set[str] = set()
    for pat in import_patterns:
        # findall over the whole text catches require()/import on any line position.
        imports_set.update(pat.findall(text))
    imports = sorted(imports_set)

    starts: list[tuple[int, str, str]] = []  # (line_index, name, kind)
    for i, line in enumerate(lines):
        for pat, kind in symbol_patterns:
            m = pat.match(line)
            if m:
                starts.append((i, m.group(1), kind))
                break

    chunks: list[Chunk] = []
    covered: set[int] = set()

    for idx, (line_i, name, kind) in enumerate(starts):
        end = _scan_block_end(lines, line_i, block_open, block_close)
        body = "\n".join(lines[line_i:end + 1])
        calls = sorted(set(call_pattern.findall(body))) if call_pattern else []
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=name,
            symbol_kind=kind,
            start_line=line_i + 1,
            end_line=end + 1,
            text=body,
            language=language,
            calls=calls,
            imports=imports,
        ))
        covered.update(range(line_i, end + 1))

    remainder_lines = [ln for i, ln in enumerate(lines) if i not in covered]
    remainder = "\n".join(remainder_lines).strip()
    if remainder:
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name="<module>",
            symbol_kind="module",
            start_line=1,
            end_line=len(lines),
            text=remainder,
            language=language,
            calls=[],
            imports=imports,
        ))

    if not chunks:
        chunks.append(_whole_file_chunk(text, file_path, language, imports=imports))
    return chunks


def _scan_block_end(lines: list[str], start: int, open_c: str, close_c: str) -> int:
    """Find the closing brace of the block that opens on or after `start`.

    Counts open/close characters (best effort — does not respect strings or
    comments). On unmatched braces, returns the last line index.
    """
    depth = 0
    saw_open = False
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == open_c:
                depth += 1
                saw_open = True
            elif ch == close_c and depth > 0:
                depth -= 1
                if saw_open and depth == 0:
                    return i
    return len(lines) - 1


def _whole_file_chunk(text: str, file_path: str, language: str, imports=None) -> Chunk:
    return Chunk(
        file_path=file_path,
        symbol_name="<file>",
        symbol_kind="module",
        start_line=1,
        end_line=text.count("\n") + 1,
        text=text,
        language=language,
        imports=imports or [],
    )


# ---------- back-compat shims (don't break existing callers/tests) -------

def chunk_python_text(text: str, file_path: str) -> list[Chunk]:
    return _chunk_python(text, file_path)


def chunk_python_file(path: Path) -> list[Chunk]:
    return chunk_file(path) if path.suffix == ".py" else []
