"""Tests for the shared RFC6901 JSON-pointer helper (issue #3386)."""

from __future__ import annotations

import pytest

from robot_sf.common.json_pointer import json_pointer


def test_root_path_renders_empty_string():
    """An empty path is the whole document -> RFC6901 root pointer ``""``."""
    assert json_pointer([]) == ""


def test_single_key():
    """A single string key renders as ``/key``."""
    assert json_pointer(["scenarios"]) == "/scenarios"


def test_nested_keys_and_indices():
    """Mixed string keys and integer indices join into a slash-separated pointer."""
    assert json_pointer(["scenarios", 0, "id"]) == "/scenarios/0/id"


def test_integer_only_path():
    """Integer indices render via ``str`` (no escaping needed)."""
    assert json_pointer([2]) == "/2"


@pytest.mark.parametrize(
    ("element", "expected"),
    [
        ("a/b", "/a~1b"),  # forward slash -> ~1
        ("m~n", "/m~0n"),  # tilde -> ~0
        ("~/", "/~0~1"),  # both, escaped in order (~ before /)
    ],
)
def test_rfc6901_escaping(element: str, expected: str):
    """Tilde and slash are escaped per RFC6901 (``~`` -> ``~0``, ``/`` -> ``~1``)."""
    assert json_pointer([element]) == expected


def test_accepts_any_iterable():
    """A generator (like jsonschema's ``absolute_path``) works, not just lists."""
    assert json_pointer(iter(["a", "b"])) == "/a/b"
