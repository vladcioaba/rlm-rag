"""Tests for multi-language chunkers (JS/TS/Go/Rust) and the dispatch."""

from __future__ import annotations

from pathlib import Path

from rlm_rag.chunker import chunk_text, chunk_file, language_of


def test_dispatch_by_extension(tmp_path):
    assert language_of("foo.py") == "python"
    assert language_of("foo.js") == "javascript"
    assert language_of("foo.tsx") == "typescript"
    assert language_of("foo.go") == "go"
    assert language_of("foo.rs") == "rust"
    assert language_of("foo.unknown") is None


# ---------- JavaScript ---------------------------------------------------

JS = """\
import express from 'express';
const auth = require('./auth');

export function login(user, password) {
  if (auth.check(user, password)) {
    return makeToken(user);
  }
  return null;
}

class UserStore {
  constructor() { this.users = {}; }
  add(u) { this.users[u.name] = u; }
}

export const greet = (name) => {
  return `hi ${name}`;
};
"""


def test_js_extracts_function_class_and_arrow():
    chunks = chunk_text(JS, "foo.js", "javascript")
    names = {(c.symbol_kind, c.symbol_name) for c in chunks}
    assert ("function", "login") in names
    assert ("class", "UserStore") in names
    assert ("function", "greet") in names


def test_js_extracts_imports():
    chunks = chunk_text(JS, "foo.js", "javascript")
    imports = chunks[0].imports
    assert "express" in imports
    assert "./auth" in imports


def test_js_extracts_calls():
    chunks = chunk_text(JS, "foo.js", "javascript")
    login = next(c for c in chunks if c.symbol_name == "login")
    assert "check" in login.calls
    assert "makeToken" in login.calls


# ---------- TypeScript ---------------------------------------------------

TS = """\
import { User } from './models';

interface Session {
  id: string;
}

type Token = string;

export async function authenticate(user: User, pw: string): Promise<Token | null> {
  return verify(user, pw);
}
"""


def test_ts_extracts_interface_type_function():
    chunks = chunk_text(TS, "foo.ts", "typescript")
    names = {(c.symbol_kind, c.symbol_name) for c in chunks}
    assert ("type", "Session") in names
    assert ("type", "Token") in names
    assert ("function", "authenticate") in names


# ---------- Go -----------------------------------------------------------

GO = """\
package auth

import (
    "crypto/sha256"
    "fmt"
)

type User struct {
    Name string
}

func HashPassword(pw string) string {
    return fmt.Sprintf("%x", sha256.Sum256([]byte(pw)))
}

func (u *User) Greet() string {
    return "hi " + u.Name
}
"""


def test_go_extracts_functions_and_types():
    chunks = chunk_text(GO, "foo.go", "go")
    names = {(c.symbol_kind, c.symbol_name) for c in chunks}
    assert ("function", "HashPassword") in names
    assert ("function", "Greet") in names
    assert ("type", "User") in names


# ---------- Rust ---------------------------------------------------------

RS = """\
use std::collections::HashMap;
use crate::auth;

pub struct User {
    name: String,
}

pub fn hash_password(pw: &str) -> String {
    String::from("hashed")
}

impl User {
    pub fn greet(&self) -> String {
        format!("hi {}", self.name)
    }
}
"""


def test_rust_extracts_fn_struct_impl():
    chunks = chunk_text(RS, "foo.rs", "rust")
    names = {(c.symbol_kind, c.symbol_name) for c in chunks}
    assert ("function", "hash_password") in names
    assert ("type", "User") in names
    assert ("impl", "User") in names


# ---------- chunk_file dispatch -----------------------------------------

def test_chunk_file_dispatches_correctly(tmp_path):
    py = tmp_path / "x.py"
    py.write_text("def foo(): pass\n")
    js = tmp_path / "x.js"
    js.write_text("function bar() { return 1; }\n")
    rs = tmp_path / "x.rs"
    rs.write_text("pub fn baz() {}\n")

    assert any(c.symbol_name == "foo" for c in chunk_file(py))
    assert any(c.symbol_name == "bar" for c in chunk_file(js))
    assert any(c.symbol_name == "baz" for c in chunk_file(rs))
