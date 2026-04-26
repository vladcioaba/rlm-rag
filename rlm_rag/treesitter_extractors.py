"""Optional tree-sitter extractors for C++ and C#.

These are used when `tree-sitter` and the per-language grammar packages are
installed (the `[treesitter]` extra in pyproject.toml). When unavailable,
the chunker falls back to the regex extractors in chunker.py.

Both extractors return Chunk objects with the same shape as the regex
extractors so the rest of the pipeline (graph, retrieval, iterative loop)
doesn't notice the difference.
"""

from __future__ import annotations

from typing import Optional

# Lazy import of the tree-sitter machinery so this module can be imported
# even when the grammars aren't installed.
_TS_AVAILABLE: dict[str, bool] = {}
_LANGUAGES: dict[str, object] = {}


def _try_load(language_name: str) -> Optional[object]:
    """Lazily load a tree-sitter language. Returns None if unavailable."""
    if language_name in _LANGUAGES:
        return _LANGUAGES[language_name]
    if _TS_AVAILABLE.get(language_name) is False:
        return None
    try:
        from tree_sitter import Language  # noqa: PLC0415
        if language_name == "cpp":
            import tree_sitter_cpp  # noqa: PLC0415
            lang = Language(tree_sitter_cpp.language())
        elif language_name == "csharp":
            import tree_sitter_c_sharp  # noqa: PLC0415
            lang = Language(tree_sitter_c_sharp.language())
        else:
            _TS_AVAILABLE[language_name] = False
            return None
        _LANGUAGES[language_name] = lang
        _TS_AVAILABLE[language_name] = True
        return lang
    except Exception:
        _TS_AVAILABLE[language_name] = False
        return None


def is_available(language: str) -> bool:
    return _try_load(language) is not None


# ---------- shared helpers ---------------------------------------------

def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _walk_descendants(node, predicate):
    """Iterate descendants matching predicate(node) -> bool."""
    cursor = node.walk()
    visited_children = False
    while True:
        if not visited_children:
            if predicate(cursor.node):
                yield cursor.node
            if cursor.goto_first_child():
                continue
            visited_children = True
        if cursor.goto_next_sibling():
            visited_children = False
            continue
        if not cursor.goto_parent():
            break


# ---------- C++ ---------------------------------------------------------

def extract_cpp(text: str, file_path: str):
    """Returns list[Chunk] or None if tree-sitter isn't available for C++."""
    lang = _try_load("cpp")
    if lang is None:
        return None

    from tree_sitter import Parser  # noqa: PLC0415
    from .chunker import Chunk  # noqa: PLC0415

    parser = Parser(lang)
    source = text.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node

    # Imports.
    imports: list[str] = []
    for inc in _walk_descendants(root, lambda n: n.type == "preproc_include"):
        for c in inc.children:
            if c.type in ("system_lib_string", "string_literal"):
                s = _node_text(c, source).strip("<>\"")
                if s:
                    imports.append(s)
                break
    imports = sorted(set(imports))

    chunks: list[Chunk] = []
    covered: set[int] = set()

    # Free functions (top-level function_definition).
    for fn in _walk_descendants(root, lambda n: n.type == "function_definition"):
        # Skip functions inside class bodies — those are handled when we walk the class.
        if _is_inside_class(fn):
            continue
        name = _cpp_function_name(fn, source)
        if not name:
            continue
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=name,
            symbol_kind="function",
            language="cpp",
            start_line=fn.start_point[0] + 1,
            end_line=fn.end_point[0] + 1,
            text=_node_text(fn, source),
            calls=sorted(set(_cpp_calls(fn, source))),
            imports=imports,
        ))
        for ln in range(fn.start_point[0] + 1, fn.end_point[0] + 2):
            covered.add(ln)

    # Classes / structs.
    for cs in _walk_descendants(root, lambda n: n.type in ("class_specifier", "struct_specifier")):
        name_node = cs.child_by_field_name("name")
        if name_node is None:
            continue
        name = _node_text(name_node, source)
        kind = "class" if cs.type == "class_specifier" else "struct"
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=name,
            symbol_kind=kind,
            language="cpp",
            start_line=cs.start_point[0] + 1,
            end_line=cs.end_point[0] + 1,
            text=_node_text(cs, source),
            calls=sorted(set(_cpp_calls(cs, source))),
            imports=imports,
        ))
        for ln in range(cs.start_point[0] + 1, cs.end_point[0] + 2):
            covered.add(ln)

        # Methods inline inside the class body.
        for m in _walk_descendants(cs, lambda n: n.type == "function_definition"):
            mname = _cpp_function_name(m, source)
            if not mname:
                continue
            chunks.append(Chunk(
                file_path=file_path,
                symbol_name=f"{name}::{mname}" if "::" not in mname else mname,
                symbol_kind="method",
                language="cpp",
                start_line=m.start_point[0] + 1,
                end_line=m.end_point[0] + 1,
                text=_node_text(m, source),
                calls=sorted(set(_cpp_calls(m, source))),
                imports=imports,
            ))

    # Namespaces.
    for ns in _walk_descendants(root, lambda n: n.type == "namespace_definition"):
        name_node = ns.child_by_field_name("name")
        name = _node_text(name_node, source) if name_node else "<anonymous>"
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=name,
            symbol_kind="namespace",
            language="cpp",
            start_line=ns.start_point[0] + 1,
            end_line=ns.end_point[0] + 1,
            text=_node_text(ns, source),
            calls=[],
            imports=imports,
        ))

    # Enums.
    for en in _walk_descendants(root, lambda n: n.type == "enum_specifier"):
        name_node = en.child_by_field_name("name")
        if name_node is None:
            continue
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=_node_text(name_node, source),
            symbol_kind="enum",
            language="cpp",
            start_line=en.start_point[0] + 1,
            end_line=en.end_point[0] + 1,
            text=_node_text(en, source),
            calls=[],
            imports=imports,
        ))

    if not chunks:
        chunks.append(Chunk(
            file_path=file_path, symbol_name="<file>", symbol_kind="module",
            language="cpp", start_line=1, end_line=text.count("\n") + 1,
            text=text, imports=imports,
        ))
    return chunks


def _is_inside_class(node) -> bool:
    p = node.parent
    while p is not None:
        if p.type in ("class_specifier", "struct_specifier"):
            return True
        p = p.parent
    return False


def _cpp_function_name(fn_node, source: bytes) -> Optional[str]:
    """Pull `name` or `Class::name` out of a function_definition."""
    fd = fn_node.child_by_field_name("declarator")
    while fd is not None and fd.type in (
        "pointer_declarator", "reference_declarator",
    ):
        fd = fd.child_by_field_name("declarator")
    if fd is None:
        return None
    if fd.type != "function_declarator":
        # Try walking inside.
        for c in fd.children:
            if c.type == "function_declarator":
                fd = c
                break
        else:
            return None
    inner = fd.child_by_field_name("declarator")
    if inner is None:
        return None
    return _node_text(inner, source).strip()


def _cpp_calls(node, source: bytes) -> list[str]:
    out: list[str] = []
    for call in _walk_descendants(node, lambda n: n.type == "call_expression"):
        fn = call.child_by_field_name("function")
        if fn is None:
            continue
        if fn.type == "identifier":
            out.append(_node_text(fn, source))
        elif fn.type == "field_expression":
            field = fn.child_by_field_name("field")
            if field is not None:
                out.append(_node_text(field, source))
        elif fn.type == "qualified_identifier":
            # Take the rightmost name.
            txt = _node_text(fn, source)
            out.append(txt.split("::")[-1])
    return out


# ---------- C# ----------------------------------------------------------

def extract_csharp(text: str, file_path: str):
    """Returns list[Chunk] or None if tree-sitter isn't available for C#."""
    lang = _try_load("csharp")
    if lang is None:
        return None

    from tree_sitter import Parser  # noqa: PLC0415
    from .chunker import Chunk  # noqa: PLC0415

    parser = Parser(lang)
    source = text.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node

    # Imports: using_directive.
    imports: list[str] = []
    for ud in _walk_descendants(root, lambda n: n.type == "using_directive"):
        # The interesting child is the qualified_name or identifier (or the
        # right-hand side of a name_equals).
        target = None
        for c in ud.children:
            if c.type == "name_equals":
                # find next sibling that's the target
                continue
            if c.type in ("qualified_name", "identifier"):
                target = c
        if target is not None:
            imports.append(_node_text(target, source))
    imports = sorted(set(imports))

    chunks: list[Chunk] = []

    declaration_kinds = {
        "class_declaration":      "class",
        "interface_declaration":  "interface",
        "struct_declaration":     "struct",
        "record_declaration":     "record",
        "enum_declaration":       "enum",
        "namespace_declaration":  "namespace",
        "file_scoped_namespace_declaration": "namespace",
    }

    for node in _walk_descendants(root, lambda n: n.type in declaration_kinds):
        kind = declaration_kinds[node.type]
        name_node = node.child_by_field_name("name")
        name = _node_text(name_node, source) if name_node else "<anonymous>"
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=name,
            symbol_kind=kind,
            language="csharp",
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            text=_node_text(node, source),
            calls=sorted(set(_csharp_calls(node, source))),
            imports=imports,
        ))

    # Methods + constructors live inside class/struct bodies.
    for m in _walk_descendants(root, lambda n: n.type == "method_declaration"):
        name_node = m.child_by_field_name("name")
        if name_node is None:
            continue
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=_node_text(name_node, source),
            symbol_kind="method",
            language="csharp",
            start_line=m.start_point[0] + 1,
            end_line=m.end_point[0] + 1,
            text=_node_text(m, source),
            calls=sorted(set(_csharp_calls(m, source))),
            imports=imports,
        ))

    for c in _walk_descendants(root, lambda n: n.type == "constructor_declaration"):
        name_node = c.child_by_field_name("name")
        if name_node is None:
            continue
        chunks.append(Chunk(
            file_path=file_path,
            symbol_name=_node_text(name_node, source),
            symbol_kind="constructor",
            language="csharp",
            start_line=c.start_point[0] + 1,
            end_line=c.end_point[0] + 1,
            text=_node_text(c, source),
            calls=sorted(set(_csharp_calls(c, source))),
            imports=imports,
        ))

    if not chunks:
        chunks.append(Chunk(
            file_path=file_path, symbol_name="<file>", symbol_kind="module",
            language="csharp", start_line=1, end_line=text.count("\n") + 1,
            text=text, imports=imports,
        ))
    return chunks


def _csharp_calls(node, source: bytes) -> list[str]:
    out: list[str] = []
    for call in _walk_descendants(node, lambda n: n.type == "invocation_expression"):
        fn = call.child_by_field_name("function")
        if fn is None:
            continue
        if fn.type == "identifier":
            out.append(_node_text(fn, source))
        elif fn.type == "member_access_expression":
            name = fn.child_by_field_name("name")
            if name is not None:
                out.append(_node_text(name, source))
        elif fn.type == "generic_name":
            ident = next((c for c in fn.children if c.type == "identifier"), None)
            if ident is not None:
                out.append(_node_text(ident, source))
    return out
