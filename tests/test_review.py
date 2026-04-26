"""Tests for the review parser. The actual LLM critique call isn't exercised
here (no API key); we test the JSON parsing and severity heuristics.
"""

from __future__ import annotations

from rlm_rag.review import _parse_review


def test_parse_review_clean_json():
    raw = '''Some preamble.
```json
{
  "severity": "major",
  "issues": ["claim about X is unsupported", "missed file foo.py"],
  "missing": ["the BAR class"]
}
```
trailing chatter
'''
    r = _parse_review(raw)
    assert r.severity == "major"
    assert "claim about X is unsupported" in r.issues
    assert "the BAR class" in r.missing


def test_parse_review_bare_json():
    raw = '{"severity": "ok", "issues": [], "missing": []}'
    r = _parse_review(raw)
    assert r.severity == "ok"
    assert r.issues == []


def test_parse_review_invalid_severity_defaults_to_minor():
    raw = '```json\n{"severity": "lgtm", "issues": [], "missing": []}\n```'
    r = _parse_review(raw)
    assert r.severity == "minor"


def test_parse_review_unparseable_falls_back_to_keyword_heuristic():
    # No JSON at all; severity must NOT default to "ok" silently.
    raw = "The drafted answer has a major factual error in the second paragraph."
    r = _parse_review(raw)
    assert r.severity == "major"


def test_parse_review_unparseable_clean_text_defaults_minor_not_ok():
    raw = "Looks fine to me, no concerns."
    r = _parse_review(raw)
    # Crucial: an unparseable response must never silently say "ok".
    assert r.severity != "ok"


def test_parse_review_missing_fields_handled():
    raw = '```json\n{"severity": "minor"}\n```'
    r = _parse_review(raw)
    assert r.severity == "minor"
    assert r.issues == []
    assert r.missing == []
