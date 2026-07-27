"""Contract tests for ``robot_sf/benchmark/full_classic/io_utils.py``.

Locks the persistence contracts owned by this module:
- ``append_episode_record``: compact-separator JSONL append that is line-oriented,
  preserves multi-record insertion order, and creates missing parent directories.
- ``_serialize_obj``: recursive object serialization across the primitive, mapping,
  sequence/set, instance-dict, attribute-only, and string-fallback branches.
- ``write_manifest``: required manifest-key rejection and delegation to
  ``atomic_write_json`` with the fully serialized payload and destination path.

All filesystem effects are confined to the pytest ``tmp_path`` fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from robot_sf.benchmark.full_classic import io_utils
from robot_sf.benchmark.full_classic.io_utils import (
    _serialize_obj,
    append_episode_record,
    write_manifest,
)

REQUIRED_KEYS = ("git_hash", "scenario_matrix_hash", "config")


# ---------------------------------------------------------------------------
# append_episode_record
# ---------------------------------------------------------------------------


def test_append_episode_record_creates_missing_parent_directory(tmp_path):
    """First write must create the full missing parent chain before appending."""
    target = tmp_path / "nested" / "deeper" / "episodes.jsonl"
    assert not target.parent.exists()

    append_episode_record(target, {"episode": 1})

    assert target.parent.is_dir()
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == {"episode": 1}


def test_append_episode_record_uses_compact_separators(tmp_path):
    """Records must be serialized with compact (``(",", ":")``) separators."""
    target = tmp_path / "episodes.jsonl"

    append_episode_record(target, {"a": 1, "b": 2})

    # Compact: no whitespace after ',' or ':', exactly one trailing newline.
    assert target.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'


def test_append_episode_record_multiple_records_remain_ordered_and_parseable(tmp_path):
    """Multiple appends stay line-oriented, independently parseable, in order."""
    target = tmp_path / "episodes.jsonl"
    records = [{"episode": 0, "ok": True}, {"episode": 1, "ok": False}, {"episode": 2}]

    for record in records:
        append_episode_record(target, record)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    # Each line is independently parseable and insertion order is preserved.
    parsed = [json.loads(line) for line in lines]
    assert parsed == records
    # Line-oriented contract: the file ends with a trailing newline.
    assert target.read_text(encoding="utf-8").endswith("\n")


# ---------------------------------------------------------------------------
# _serialize_obj branch coverage
# ---------------------------------------------------------------------------


def test_serialize_obj_primitives_pass_through():
    """Primitive and None inputs are returned unchanged (identity where relevant)."""
    assert _serialize_obj(42) == 42
    assert _serialize_obj(3.5) == 3.5
    assert _serialize_obj("text") == "text"
    assert _serialize_obj(True) is True
    assert _serialize_obj(False) is False
    assert _serialize_obj(None) is None


def test_serialize_obj_mapping_recurses_and_keeps_keys():
    """Dicts recurse into values and, unlike instance dicts, keep underscore keys."""
    data = {"a": 1, "_kept": 2, "nested": {"c": (1, 2)}}

    assert _serialize_obj(data) == {"a": 1, "_kept": 2, "nested": {"c": [1, 2]}}


def test_serialize_obj_sequence_and_set_become_list():
    """Lists recurse, tuples and sets are normalized to lists."""
    assert _serialize_obj([1, "x", None]) == [1, "x", None]
    assert _serialize_obj((1, 2, 3)) == [1, 2, 3]
    # Sets normalize to a list (order is unspecified, so compare as a set).
    set_out = _serialize_obj({1, 2, 3})
    assert isinstance(set_out, list)
    assert sorted(set_out) == [1, 2, 3]
    # Recursion through nested sequences.
    assert _serialize_obj([(1, 2), [3, 4]]) == [[1, 2], [3, 4]]


def test_serialize_obj_instance_dict_filters_underscore_keys():
    """Non-empty instance ``__dict__`` is serialized, dropping underscore keys."""

    class WithDict:
        def __init__(self) -> None:
            self.visible = 1
            self.nested = {"k": (1, 2)}
            self._hidden = 99

    out = _serialize_obj(WithDict())

    assert out == {"visible": 1, "nested": {"k": [1, 2]}}
    assert "_hidden" not in out


def test_serialize_obj_attribute_only_gathers_class_attributes():
    """Empty instance ``__dict__`` gathers non-underscore, non-callable class attrs."""

    class AttrOnly:
        x = 1
        label = "hi"
        _secret = 2

        def method(self) -> None: ...

    out = _serialize_obj(AttrOnly())

    assert out == {"x": 1, "label": "hi"}
    assert "_secret" not in out
    assert "method" not in out


def test_serialize_obj_string_fallback_for_objects_without_dict():
    """Objects lacking ``__dict__`` and matching no branch fall back to ``str()``."""

    class NoDict:
        __slots__ = ()

        def __str__(self) -> str:
            return "NoDictRepr"

    # bytes has no __dict__ and is not primitive/mapping/sequence-handled.
    assert _serialize_obj(b"abc") == str(b"abc")
    assert _serialize_obj(NoDict()) == "NoDictRepr"


# ---------------------------------------------------------------------------
# write_manifest: required-key rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", REQUIRED_KEYS)
def test_write_manifest_rejects_missing_required_key(tmp_path, missing_key):
    """A missing required key raises ValueError before any write is delegated."""
    manifest = {key: f"value-{key}" for key in REQUIRED_KEYS}
    del manifest[missing_key]

    with (
        patch.object(io_utils, "atomic_write_json") as mock_write,
        pytest.raises(ValueError, match=rf"required key: {missing_key}"),
    ):
        write_manifest(manifest, tmp_path / "manifest.json")

    # Rejection must happen before delegation to the atomic writer.
    mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# write_manifest: parent creation and atomic-writer delegation
# ---------------------------------------------------------------------------


def test_write_manifest_creates_missing_parent_directory(tmp_path):
    """``write_manifest`` creates the missing parent chain before delegating."""
    target = tmp_path / "nested" / "manifest.json"
    manifest = {key: f"value-{key}" for key in REQUIRED_KEYS}

    with patch.object(io_utils, "atomic_write_json") as mock_write:
        write_manifest(manifest, target)

    # atomic_write_json is mocked, so only write_manifest's _ensure_parent ran.
    assert target.parent.is_dir()
    mock_write.assert_called_once()


def test_write_manifest_delegates_atomic_write_with_fully_serialized_data_and_path(tmp_path):
    """The fully serialized payload and destination path are passed to atomic_write_json."""

    class Config:
        def __init__(self) -> None:
            self.horizon = (1, 2, 3)  # tuple -> list under serialization
            self._internal = "skip"  # underscore instance key is dropped

    target = tmp_path / "out" / "manifest.json"
    manifest = {
        "git_hash": "abc123",
        "scenario_matrix_hash": "def456",
        "config": Config(),
    }
    expected_serialized = {
        "git_hash": "abc123",
        "scenario_matrix_hash": "def456",
        "config": {"horizon": [1, 2, 3]},
    }

    with patch.object(io_utils, "atomic_write_json") as mock_write:
        write_manifest(manifest, target)

    mock_write.assert_called_once()
    called_path, called_data = mock_write.call_args.args
    # Destination path is the resolved target path.
    assert called_path == Path(target)
    # Payload passed onward is the fully serialized form (object -> dict, tuple -> list).
    assert called_data == expected_serialized
    assert isinstance(called_data["config"]["horizon"], list)
