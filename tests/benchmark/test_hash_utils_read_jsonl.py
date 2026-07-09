"""Fail-closed regression tests for ``hash_utils.read_jsonl`` (issue #4926).

The canonical JSONL reader is consumed by ~85 consolidated call sites that all
expect ``list[dict]`` records. These tests pin the fail-closed contract that the
mechanical consolidation (PR #4929) originally dropped: the pre-consolidation
``run_multi_amv_smoke._read_jsonl`` rejected non-object rows with
``ValueError("<path>:<line> is not a JSON object")``; the canonical owner must
preserve that behavior rather than silently return junk records that only
explode later as ``AttributeError`` downstream.
"""

from __future__ import annotations

import json

import pytest

from robot_sf.benchmark.identity.hash_utils import read_jsonl


def test_read_jsonl_parses_object_lines_and_skips_blanks(tmp_path):
    """Object lines are parsed in order; blank/whitespace lines are skipped."""
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n\n  \n{"b": 2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_rejects_non_object_line_with_context(tmp_path):
    """Non-object rows fail closed with the path:line 'not a JSON object' message."""
    path = tmp_path / "list_line.jsonl"
    path.write_text('{"a": 1}\n[1, 2, 3]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object") as excinfo:
        read_jsonl(path)

    message = str(excinfo.value)
    assert f"{path}:2" in message


def test_read_jsonl_propagates_malformed_json(tmp_path):
    """Malformed JSON keeps the standard json decode error (a ValueError subclass)."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n{"b": oops}\n', encoding="utf-8")

    # Malformed JSON keeps the standard json decode error (a ValueError subclass),
    # matching the pre-consolidation behavior.
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)
