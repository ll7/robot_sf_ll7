"""Unit coverage for :mod:`robot_sf.data_analysis.pickle_utils`.

Locks the behavior of the shared helpers extracted for issue #6455 and the
re-export contract that keeps every legacy public import path working.
"""

from pathlib import Path

from robot_sf.data_analysis import extract_json_from_pickle, extract_obj_from_pickle
from robot_sf.data_analysis.pickle_utils import ensure_dir_exists, extract_timestamp


def test_ensure_dir_exists_creates_missing_directory(tmp_path: Path) -> None:
    """A missing (nested) directory is created."""
    target = tmp_path / "plots" / "nested"

    ensure_dir_exists(str(target))

    assert target.is_dir()


def test_ensure_dir_exists_is_idempotent_for_existing_directory(tmp_path: Path) -> None:
    """An existing directory is left untouched without error."""
    ensure_dir_exists(str(tmp_path))

    assert tmp_path.is_dir()


def test_extract_timestamp_returns_matched_timestamp() -> None:
    """A recording-style timestamp is extracted verbatim."""
    assert extract_timestamp("rec_2026-01-02_03-04-05.json") == "2026-01-02_03-04-05"


def test_extract_timestamp_falls_back_to_unknown() -> None:
    """Filenames without a timestamp fall back to 'unknown'."""
    assert extract_timestamp("no_timestamp.pkl") == "unknown"


def test_legacy_modules_reexport_shared_helpers() -> None:
    """Every legacy import path keeps resolving to the same objects."""
    assert extract_obj_from_pickle.ensure_dir_exists is ensure_dir_exists
    assert extract_json_from_pickle.ensure_dir_exists is ensure_dir_exists
    assert extract_json_from_pickle.extract_timestamp is extract_timestamp
    assert extract_obj_from_pickle.extract_timestamp is extract_timestamp
