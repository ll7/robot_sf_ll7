"""Skip-if-absent shape-contract tests for ATC pedestrian tracking external data.

These tests never require the license-gated ATC bytes. Exactly one test path
depends on locally staged real data and skips when it is absent; every other test
builds a tiny synthetic ATC-shaped layout under ``tmp_path``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.data.external import atc

if TYPE_CHECKING:
    from pathlib import Path


# A minimal well-formed ATC daily row: eight comma-separated numeric columns
# (time, person_id, x, y, z, velocity, motion_angle, facing_angle), no header.
_ATC_ROWS = (
    "1350100000.0,1,1000.0,2000.0,1200.0,500.0,0.10,0.20\n"
    "1350100000.033,1,1010.0,2005.0,1200.0,505.0,0.11,0.21\n"
)


def _write(path: Path, text: str) -> None:
    """Create parent directories and write a small text fixture file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _stage_minimal_dataset(root: Path, *, csv_name: str = "atc-20121024.csv") -> None:
    """Create a tiny documented ATC layout: one daily CSV plus a terms note."""

    _write(root / csv_name, _ATC_ROWS)
    _write(root / "README.md", "# Local copy of the ATC research-use terms\n")


def test_atc_absent_data_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """External clones without staged ATC bytes skip instead of failing."""

    monkeypatch.setenv(atc.EXTERNAL_DATA_ROOT_ENV, str(tmp_path))
    if not atc.is_available():
        pytest.skip("external dataset not staged")
    pytest.fail("temporary empty external-data root unexpectedly satisfied ATC contract")


def test_atc_shape_contract_with_synthetic_layout(tmp_path: Path) -> None:
    """A complete synthetic layout resolves and produces the shape contract."""

    root = tmp_path / "atc_pedestrian"
    _stage_minimal_dataset(root)

    assert atc.is_available(root)
    dataset = atc.require_available(root)
    assert [path.name for path in dataset.csv_files] == ["atc-20121024.csv"]

    contract = atc.load_shape_contract(root)
    assert contract["asset_id"] == "atc-pedestrian"
    assert contract["docs_path"] == "docs/datasets/atc.md"
    assert contract["delimiter"] == "comma"
    assert contract["column_count"] == 8
    assert contract["column_names"][:2] == ["time", "person_id"]
    assert contract["file_count"] == 1

    entry = contract["files"][0]
    assert entry["path"] == "atc-20121024.csv"
    assert entry["scanned_rows"] == 2
    assert entry["scan_truncated"] is False


def test_atc_accepts_generic_csv_name(tmp_path: Path) -> None:
    """A CSV staged without the ``atc-`` prefix still resolves via the fallback glob."""

    root = tmp_path / "atc_pedestrian"
    _write(root / "daily_csv" / "day1.csv", _ATC_ROWS)
    _write(root / "TERMS.txt", "ATC research-use terms\n")

    assert atc.is_available(root)
    contract = atc.load_shape_contract(root)
    assert contract["files"][0]["path"] == "daily_csv/day1.csv"


def test_atc_missing_readme_is_unavailable(tmp_path: Path) -> None:
    """A trajectory CSV without a local terms/README note is unavailable."""

    root = tmp_path / "atc_pedestrian"
    _write(root / "atc-20121024.csv", _ATC_ROWS)

    assert not atc.is_available(root)
    with pytest.raises(atc.AtcDataError, match="docs/datasets/atc.md"):
        atc.require_available(root)


def test_atc_missing_csv_is_unavailable(tmp_path: Path) -> None:
    """A terms note without any trajectory CSV is unavailable."""

    root = tmp_path / "atc_pedestrian"
    _write(root / "README.md", "terms only\n")

    assert not atc.is_available(root)
    with pytest.raises(atc.AtcDataError, match="not staged or is incomplete"):
        atc.load_shape_contract(root)


def test_atc_wrong_column_count_fails_closed(tmp_path: Path) -> None:
    """A CSV row with the wrong column width fails closed with a docs pointer."""

    root = tmp_path / "atc_pedestrian"
    _stage_minimal_dataset(root)
    _write(root / "atc-20121024.csv", "1350100000.0,1,1000.0,2000.0\n")

    with pytest.raises(atc.AtcDataError, match="columns"):
        atc.load_shape_contract(root)


def test_atc_non_numeric_value_fails_closed(tmp_path: Path) -> None:
    """A non-numeric trajectory value fails closed with an actionable error."""

    root = tmp_path / "atc_pedestrian"
    _stage_minimal_dataset(root)
    _write(root / "atc-20121024.csv", "1350100000.0,1,x,2000.0,1200.0,500.0,0.1,0.2\n")

    with pytest.raises(atc.AtcDataError, match="non-numeric"):
        atc.load_shape_contract(root)


def test_atc_non_finite_value_fails_closed(tmp_path: Path) -> None:
    """A non-finite trajectory value fails closed with an actionable error."""

    root = tmp_path / "atc_pedestrian"
    _stage_minimal_dataset(root)
    _write(root / "atc-20121024.csv", "1350100000.0,1,inf,2000.0,1200.0,500.0,0.1,0.2\n")

    with pytest.raises(atc.AtcDataError, match="non-finite"):
        atc.load_shape_contract(root)


def test_atc_empty_csv_fails_closed(tmp_path: Path) -> None:
    """An empty staged CSV fails closed with an actionable error."""

    root = tmp_path / "atc_pedestrian"
    _stage_minimal_dataset(root)
    _write(root / "atc-20121024.csv", "\n# comment only\n")

    with pytest.raises(atc.AtcDataError, match="no numeric data rows"):
        atc.load_shape_contract(root)


def test_atc_bounded_scan_reports_truncation(tmp_path: Path) -> None:
    """A bounded ``max_rows`` scan stops early and honestly reports truncation."""

    root = tmp_path / "atc_pedestrian"
    _stage_minimal_dataset(root)
    _write(root / "atc-20121024.csv", _ATC_ROWS * 50)

    contract = atc.load_shape_contract(root, max_rows=10)
    entry = contract["files"][0]
    assert entry["scanned_rows"] == 10
    assert entry["scan_truncated"] is True
    assert contract["max_rows"] == 10

    # A full scan (max_rows=None) reads every row and reports no truncation.
    full = atc.load_shape_contract(root, max_rows=None)
    assert full["files"][0]["scanned_rows"] == 100
    assert full["files"][0]["scan_truncated"] is False
