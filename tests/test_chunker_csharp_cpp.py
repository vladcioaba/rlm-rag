"""Tests for C# and C++ chunkers."""

from __future__ import annotations

from rlm_rag.chunker import chunk_text, language_of


# ---------- dispatch ----------------------------------------------------

def test_dispatch_csharp():
    assert language_of("foo.cs") == "csharp"


def test_dispatch_cpp_extensions():
    for ext in (".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inl"):
        assert language_of(f"foo{ext}") == "cpp", ext


# ---------- C# ----------------------------------------------------------

CS = """\
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Alias = MyApp.Internal.Helper;

namespace MyApp.Services {

    public interface IUserRepo {
        User Get(int id);
    }

    public class UserService : IUserRepo {
        private readonly Database _db;

        public UserService(Database db) {
            _db = db;
        }

        public async Task<User> GetUser(int id) {
            return await _db.GetUserAsync(id);
        }

        private void log(string msg) {
            Console.WriteLine(msg);
        }
    }

    public struct Point {
        public int X;
        public int Y;
    }

    public record User(string Name, int Age);

    public enum Status { Active, Inactive }
}
"""


def test_csharp_extracts_namespace():
    chunks = chunk_text(CS, "x.cs", "csharp")
    names = {(c.symbol_kind, c.symbol_name) for c in chunks}
    assert ("namespace", "MyApp.Services") in names


def test_csharp_extracts_class_interface_struct_record_enum():
    chunks = chunk_text(CS, "x.cs", "csharp")
    names = {(c.symbol_kind, c.symbol_name) for c in chunks}
    assert ("class", "UserService") in names
    assert ("interface", "IUserRepo") in names
    assert ("struct", "Point") in names
    assert ("record", "User") in names
    assert ("enum", "Status") in names


def test_csharp_extracts_methods_and_constructor():
    chunks = chunk_text(CS, "x.cs", "csharp")
    method_names = {c.symbol_name for c in chunks if c.symbol_kind == "method"}
    ctor_names = {c.symbol_name for c in chunks if c.symbol_kind == "constructor"}
    assert "GetUser" in method_names
    assert "log" in method_names
    assert "UserService" in ctor_names


def test_csharp_extracts_using_imports_including_aliases():
    chunks = chunk_text(CS, "x.cs", "csharp")
    imports = chunks[0].imports
    assert "System" in imports
    assert "System.Collections.Generic" in imports
    # The aliased form `using Alias = MyApp.Internal.Helper;` resolves to the target.
    assert "MyApp.Internal.Helper" in imports


def test_csharp_extracts_calls():
    chunks = chunk_text(CS, "x.cs", "csharp")
    get_user = next(c for c in chunks if c.symbol_name == "GetUser")
    assert "GetUserAsync" in get_user.calls


# ---------- C++ ---------------------------------------------------------

CPP = """\
#include <vector>
#include <string>
#include "player.h"

namespace mygame {

    class Player {
    public:
        Player(int hp);
        void TakeDamage(int dmg);
        int Health() const;

    private:
        int hp_;
    };

    Player::Player(int hp) : hp_(hp) {
    }

    void Player::TakeDamage(int dmg) {
        hp_ -= dmg;
        Log("damage taken");
    }

    int Player::Health() const {
        return hp_;
    }

    int compute_score(const std::vector<int>& runs) {
        int total = 0;
        for (auto r : runs) total += r;
        return total;
    }

    enum class Difficulty { Easy, Normal, Hard };

}  // namespace mygame
"""


def test_cpp_extracts_includes_both_styles():
    chunks = chunk_text(CPP, "x.cpp", "cpp")
    imports = chunks[0].imports
    assert "vector" in imports
    assert "string" in imports
    assert "player.h" in imports


def test_cpp_extracts_namespace_and_class_and_enum():
    chunks = chunk_text(CPP, "x.cpp", "cpp")
    names = {(c.symbol_kind, c.symbol_name) for c in chunks}
    assert ("namespace", "mygame") in names
    assert ("class", "Player") in names
    assert ("enum", "Difficulty") in names


def test_cpp_extracts_qualified_method_definitions():
    chunks = chunk_text(CPP, "x.cpp", "cpp")
    fn_names = {c.symbol_name for c in chunks if c.symbol_kind == "function"}
    # Out-of-class method definitions should appear as `Player::Foo`.
    assert "Player::TakeDamage" in fn_names
    assert "Player::Health" in fn_names


def test_cpp_extracts_free_function():
    chunks = chunk_text(CPP, "x.cpp", "cpp")
    fn_names = {c.symbol_name for c in chunks if c.symbol_kind == "function"}
    assert "compute_score" in fn_names


def test_cpp_extracts_calls():
    chunks = chunk_text(CPP, "x.cpp", "cpp")
    take_damage = next(
        (c for c in chunks if c.symbol_name == "Player::TakeDamage"), None,
    )
    assert take_damage is not None
    assert "Log" in take_damage.calls
