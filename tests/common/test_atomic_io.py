"""Tests for the shared atomic JSON-write helper (issue #3386)."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from robot_sf.common.atomic_io import atomic_write_json

if TYPE_CHECKING:
    from pathlib import Path


def test_writes_pretty_sorted_json(tmp_path: Path):
    """Payload is written as indented, key-sorted JSON readable back as a dict."""
    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"b": 1, "a": {"d": 2, "c": 3}})

    text = target.read_text(encoding="utf-8")
    # Keys are sorted at every level and the output is indented.
    assert text.index('"a"') < text.index('"b"')
    assert text.index('"c"') < text.index('"d"')
    assert "\n" in text  # indent=2 produces multi-line output
    assert json.loads(text) == {"a": {"c": 3, "d": 2}, "b": 1}


def test_creates_missing_parent_directories(tmp_path: Path):
    """The destination's parent directory is created when absent."""
    target = tmp_path / "nested" / "deeper" / "out.json"
    atomic_write_json(target, {"x": 1})

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1}


def test_overwrites_existing_file_atomically(tmp_path: Path):
    """A second write replaces the prior content and leaves no temp files behind."""
    target = tmp_path / "data.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}
    # Only the final file remains in the directory (temp file was cleaned up).
    assert [p.name for p in tmp_path.iterdir()] == ["data.json"]


def test_closes_temp_fd_when_fdopen_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The raw mkstemp fd is closed if os.fdopen fails before owning it."""
    target = tmp_path / "data.json"
    captured_fd: int | None = None

    def fail_fdopen(fd: int, *args: object, **kwargs: object):
        nonlocal captured_fd
        captured_fd = fd
        raise OSError("fdopen failed")

    monkeypatch.setattr(os, "fdopen", fail_fdopen)

    with pytest.raises(OSError, match="fdopen failed"):
        atomic_write_json(target, {"v": 1})

    assert captured_fd is not None
    with pytest.raises(OSError):
        os.fstat(captured_fd)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_closes_temp_fd_when_fdopen_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A ValueError from os.fdopen is also caught and the raw fd is closed.

    ``os.fdopen`` documents ``OSError`` (bad fd) and ``ValueError`` (incompatible
    mode/arg combination) as its realistic failure types; the narrowed handler
    (issue #6459) must close the raw mkstemp fd and re-raise for both.
    """
    target = tmp_path / "data.json"
    captured_fd: int | None = None

    def fail_fdopen(fd: int, *args: object, **kwargs: object):
        nonlocal captured_fd
        captured_fd = fd
        raise ValueError("bad mode")

    monkeypatch.setattr(os, "fdopen", fail_fdopen)

    with pytest.raises(ValueError, match="bad mode"):
        atomic_write_json(target, {"v": 1})

    assert captured_fd is not None
    with pytest.raises(OSError):
        os.fstat(captured_fd)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
