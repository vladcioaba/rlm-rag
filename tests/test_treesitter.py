"""Tests that exercise the tree-sitter extractors specifically — cases
where the regex extractor would miss but tree-sitter should catch.

Skipped when tree-sitter isn't installed.
"""

from __future__ import annotations

import pytest

from rlm_rag import treesitter_extractors as tse


pytestmark = pytest.mark.skipif(
    not (tse.is_available("cpp") and tse.is_available("csharp")),
    reason="tree-sitter grammars not installed",
)


# ---------- C++ ---------------------------------------------------------

CPP_HARD = """\
#include <vector>
#include <memory>

namespace outer {
namespace inner {

template <typename T, typename U>
class Container final : public Base<T> {
public:
    Container() = default;

    void
    Add(const T& item)
    {
        items_.push_back(item);
    }

    template <typename V>
    auto
    Transform(V func) const -> Container<U, V>
    {
        return Container<U, V>{};
    }

private:
    std::vector<T> items_;
};

template <typename T>
T identity(T x) {
    return x;
}

} // namespace inner
} // namespace outer
"""


def test_cpp_treesitter_handles_multi_line_signatures():
    chunks = tse.extract_cpp(CPP_HARD, "x.cpp")
    fn_names = {c.symbol_name for c in chunks if c.symbol_kind in ("function", "method")}
    # Both methods (with multi-line signatures) and the free template function
    # should be picked up — regex would miss these.
    assert "identity" in fn_names
    assert any("Add" in n for n in fn_names)
    assert any("Transform" in n for n in fn_names)


def test_cpp_treesitter_extracts_nested_namespaces():
    chunks = tse.extract_cpp(CPP_HARD, "x.cpp")
    namespaces = {c.symbol_name for c in chunks if c.symbol_kind == "namespace"}
    assert "outer" in namespaces
    assert "inner" in namespaces


def test_cpp_treesitter_methods_qualified_with_class_name():
    chunks = tse.extract_cpp(CPP_HARD, "x.cpp")
    methods = [c for c in chunks if c.symbol_kind == "method"]
    assert any(c.symbol_name.startswith("Container::") for c in methods), \
        f"expected at least one Container::* method, got: {[c.symbol_name for c in methods]}"


def test_cpp_treesitter_extracts_class_with_template():
    chunks = tse.extract_cpp(CPP_HARD, "x.cpp")
    classes = {c.symbol_name for c in chunks if c.symbol_kind == "class"}
    assert "Container" in classes


# ---------- C# ----------------------------------------------------------

CS_HARD = """\
using System;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace MyApp {

    public partial class UserService<T> where T : class {

        private readonly IRepo<T> _repo;

        public UserService(IRepo<T> repo) {
            _repo = repo;
        }

        public async Task<T?> GetAsync(
            int id,
            CancellationToken ct = default
        ) {
            return await _repo.FindAsync(id, ct);
        }

        public IEnumerable<T> All() => _repo.GetAll();
    }

    public partial class UserService<T> {
        public void Touch() { /* partial in another file in real code */ }
    }
}
"""


def test_csharp_treesitter_handles_multi_line_method_signatures():
    chunks = tse.extract_csharp(CS_HARD, "x.cs")
    method_names = {c.symbol_name for c in chunks if c.symbol_kind == "method"}
    assert "GetAsync" in method_names  # multi-line signature, regex would miss


def test_csharp_treesitter_handles_partial_classes():
    chunks = tse.extract_csharp(CS_HARD, "x.cs")
    classes = [c for c in chunks if c.symbol_kind == "class"]
    # Both `partial class` declarations of UserService should appear.
    user_service_classes = [c for c in classes if c.symbol_name == "UserService"]
    assert len(user_service_classes) >= 2


def test_csharp_treesitter_extracts_constructor_with_generic_class():
    chunks = tse.extract_csharp(CS_HARD, "x.cs")
    ctors = {c.symbol_name for c in chunks if c.symbol_kind == "constructor"}
    assert "UserService" in ctors


def test_csharp_treesitter_extracts_calls_through_member_access():
    chunks = tse.extract_csharp(CS_HARD, "x.cs")
    get_async = next(c for c in chunks if c.symbol_name == "GetAsync")
    # `_repo.FindAsync(...)` should surface FindAsync as a call.
    assert "FindAsync" in get_async.calls
