"""Freeze-safe golden regression coverage for the remaining benchmark metric-output surfaces (#5014).

PR #4998 froze the aggregate-dictionary surface only. This module extends the
committed deterministic golden harness to the three remaining metric-output
surfaces named in issue #5014, without changing metric semantics:

1. **Summary surface** -- ``robot_sf.benchmark.summary.aggregate_training_metrics_with_bootstrap``
   over the frozen episode fixture, with a fixed bootstrap seed so the output is
   byte-stable and reviewable.
2. **Canonical-table export surface** -- ``robot_sf.benchmark.canonical_table_export.export_canonical_table``
   producing the ``planner_outcome_summary`` table in CSV / Markdown / LaTeX, plus
   a provenance-scrubbed metadata comparison (volatile timestamp/git fields removed).
3. **Parquet export surface** -- ``robot_sf.benchmark.parquet_export.export_episodes_jsonl_to_parquet``
   over the frozen episode fixture, read back as canonical JSON per table, plus the
   deterministic DuckDB examples SQL and a provenance-scrubbed metadata comparison.

This is implementation-integrity coverage only: it protects the end-to-end output
contract of those surfaces against accidental drift. It makes no
benchmark-performance, paper, or evidence-strength claim.

To intentionally update the reviewed output goldens, run:

``ROBOT_SF_BLESS_GOLDEN=1 uv run pytest tests/benchmark/test_metric_output_golden_surfaces.py -v -s``
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.canonical_table_export import (
    TABLE_SPECS,
    export_canonical_table,
    load_rows_json,
)
from robot_sf.benchmark.parquet_export import export_episodes_jsonl_to_parquet
from robot_sf.benchmark.summary import aggregate_training_metrics_with_bootstrap

_BLESS_ENV = "ROBOT_SF_BLESS_GOLDEN"
_GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "benchmark" / "golden"
_EPISODES_PATH = _GOLDEN_DIR / "aggregate_episodes.jsonl"

# Fixed summary inputs. The seed makes the bootstrap confidence intervals
# deterministic and therefore freezable; the metric paths exercise every numeric
# field present in the frozen episode fixture.
_SUMMARY_METRIC_KEYS: tuple[str, ...] = (
    "metrics.success",
    "metrics.collisions",
    "metrics.avg_speed",
    "metrics.near_misses",
    "metrics.min_clearance",
    "metrics.time_to_goal",
    "metrics.path_length",
)
_SUMMARY_GOLDEN_PATH = _GOLDEN_DIR / "summary_training_metrics.json"

# Canonical-table inputs/goldens.
_CANONICAL_TABLE_ID = "planner_outcome_summary"
_CANONICAL_TABLE_ROWS_PATH = _GOLDEN_DIR / "canonical_table_planner_outcome_summary_rows.json"
_CANONICAL_TABLE_GOLDEN_PATHS = {
    fmt: _GOLDEN_DIR / f"canonical_table_{_CANONICAL_TABLE_ID}.{fmt}"
    for fmt in ("csv", "md", "tex")
}
_CANONICAL_TABLE_METADATA_GOLDEN_PATH = (
    _GOLDEN_DIR / f"canonical_table_{_CANONICAL_TABLE_ID}.metadata.scrubbed.json"
)

# Parquet inputs/goldens. The frozen episode JSONL is the shared input; goldens
# are the read-back per-table canonical JSON, the DuckDB examples SQL, and a
# provenance-scrubbed metadata document.
_PARQUET_TABLE_NAMES: tuple[str, ...] = (
    "episodes",
    "metrics",
    "scenario_params",
    "algorithm_metadata",
)
_PARQUET_TABLE_GOLDEN_PATHS = {
    name: _GOLDEN_DIR / f"parquet_episodes_{name}.json" for name in _PARQUET_TABLE_NAMES
}
_PARQUET_DUCKDB_GOLDEN_PATH = _GOLDEN_DIR / "parquet_duckdb_examples.sql"
_PARQUET_METADATA_GOLDEN_PATH = _GOLDEN_DIR / "parquet_metadata.scrubbed.json"

# Metadata fields that are intentionally volatile and therefore excluded from the
# golden comparison so the freeze remains stable across commits and clock reads.
_VOLATILE_METADATA_FIELDS = ("generated_at_utc", "git_commit", "created_at_utc")
# Provenance fields that carry useful reviewable structure (format/file mapping,
# checksums) but whose values include non-portable absolute paths; these are
# normalized to basenames so the golden is stable across checkouts and temp dirs.
_PATH_LIKE_PROVENANCE_FIELDS = ("outputs",)


def _canonical_json(payload: Any) -> str:
    """Return the reviewable, platform-independent JSON representation."""
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _is_bless() -> bool:
    """Return True only when the opt-in, reviewed bless path is requested."""
    return os.environ.get(_BLESS_ENV) == "1"


def _assert_or_bless_text(*, actual: str, golden_path: Path) -> None:
    """Compare text output to a golden, or deliberately rewrite it with a visible diff."""
    if not _is_bless() and not golden_path.is_file():
        raise FileNotFoundError(
            f"Golden file not found at {golden_path}. "
            f"Run with {_BLESS_ENV}=1 only for a reviewed intentional update."
        )
    expected = golden_path.read_text(encoding="utf-8") if golden_path.is_file() else ""
    if _is_bless():
        if actual != expected:
            print(
                "".join(
                    difflib.unified_diff(
                        expected.splitlines(keepends=True),
                        actual.splitlines(keepends=True),
                        fromfile=str(golden_path),
                        tofile=str(golden_path),
                    )
                ),
                end="",
            )
        golden_path.write_text(actual, encoding="utf-8")
        return

    assert actual == expected, (
        f"Metric output drifted from {golden_path}. "
        f"Review the change, then intentionally bless it with "
        f"{_BLESS_ENV}=1 uv run pytest tests/benchmark/test_metric_output_golden_surfaces.py -v -s."
    )


def _scrub_volatile(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a provenance mapping with volatile and non-portable fields normalized.

    Volatile timestamp/git fields are dropped, absolute output paths are reduced to
    reviewable basenames, and source-file checksums are kept while their absolute
    paths are reduced to basenames. The result is stable across commits, clock
    reads, checkouts, and temporary directories.
    """
    scrubbed: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _VOLATILE_METADATA_FIELDS:
            continue
        if key in _PATH_LIKE_PROVENANCE_FIELDS and isinstance(value, dict):
            scrubbed[key] = {fmt: Path(path).name for fmt, path in value.items()}
            continue
        if key == "source_files" and isinstance(value, list):
            scrubbed[key] = [
                {**entry, "path": Path(entry["path"]).name} if isinstance(entry, dict) else entry
                for entry in value
            ]
            continue
        scrubbed[key] = value
    return scrubbed


def _assert_or_bless_payload(*, actual: dict[str, Any], golden_path: Path) -> None:
    """Compare a JSON payload to a golden after scrubbing volatile metadata fields."""
    _assert_or_bless_text(actual=_canonical_json(actual), golden_path=golden_path)


# --------------------------------------------------------------------------- #
# Summary surface
# --------------------------------------------------------------------------- #


def _summary_fixture() -> str:
    """Run the summary surface over the frozen episodes with a fixed bootstrap seed."""
    from robot_sf.benchmark.aggregate import read_jsonl

    records = read_jsonl(_EPISODES_PATH)
    payload = aggregate_training_metrics_with_bootstrap(
        records,
        _SUMMARY_METRIC_KEYS,
        confidence=0.95,
        n_samples=500,
        seed=42,
    )
    return _canonical_json(payload)


def test_summary_surface_matches_canonical_golden() -> None:
    """Frozen summary (bootstrap-CI) output must exactly match its reviewed JSON golden."""
    first_run = _summary_fixture()
    second_run = _summary_fixture()

    assert first_run == second_run
    _assert_or_bless_text(actual=first_run, golden_path=_SUMMARY_GOLDEN_PATH)


# --------------------------------------------------------------------------- #
# Canonical-table export surface
# --------------------------------------------------------------------------- #


def _canonical_table_fixture(
    *,
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Export the frozen canonical-table rows fixture and return text outputs + scrubbed metadata."""
    rows = load_rows_json(_CANONICAL_TABLE_ROWS_PATH)
    result = export_canonical_table(
        rows,
        table_id=_CANONICAL_TABLE_ID,
        output_dir=out_dir,
        formats=tuple(_CANONICAL_TABLE_GOLDEN_PATHS),
        precision=4,
        source_paths=(),
        command="benchmark-golden-harness#5014",
    )
    text_outputs = {
        fmt: path.read_text(encoding="utf-8") for fmt, path in result.output_paths.items()
    }
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    return text_outputs, _scrub_volatile(metadata)


def test_canonical_table_surface_matches_canonical_golden(
    tmp_path: Path,
) -> None:
    """Frozen canonical-table export (CSV/MD/TEX + scrubbed metadata) must match its goldens."""
    first_texts, first_meta = _canonical_table_fixture(out_dir=tmp_path / "first")
    second_texts, second_meta = _canonical_table_fixture(out_dir=tmp_path / "second")

    assert first_texts == second_texts
    assert first_meta == second_meta
    assert set(first_texts) == set(_CANONICAL_TABLE_GOLDEN_PATHS)
    for fmt, text in first_texts.items():
        _assert_or_bless_text(actual=text, golden_path=_CANONICAL_TABLE_GOLDEN_PATHS[fmt])
    _assert_or_bless_payload(actual=first_meta, golden_path=_CANONICAL_TABLE_METADATA_GOLDEN_PATH)


def test_canonical_table_golden_columns_match_spec_contract() -> None:
    """The frozen canonical-table golden must match the committed spec column contract."""
    spec = TABLE_SPECS[_CANONICAL_TABLE_ID]
    for fmt in _CANONICAL_TABLE_GOLDEN_PATHS:
        text = _CANONICAL_TABLE_GOLDEN_PATHS[fmt].read_text(encoding="utf-8")
        if fmt == "csv":
            header = text.splitlines()[0]
            assert header == ",".join(spec.columns)
        elif fmt == "md":
            header = text.splitlines()[0]
            assert all(column in header for column in spec.columns)


# --------------------------------------------------------------------------- #
# Parquet export surface
# --------------------------------------------------------------------------- #


def _parquet_fixture(
    *,
    out_dir: Path,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    """Export the frozen episodes to Parquet and return canonical table/metadata/SQL text."""
    result = export_episodes_jsonl_to_parquet(_EPISODES_PATH, out_dir, overwrite=True)

    import pyarrow.parquet as pq

    table_texts: dict[str, str] = {}
    for name in _PARQUET_TABLE_NAMES:
        rows = pq.read_table(result.table_paths[name]).to_pylist()
        table_texts[name] = _canonical_json(rows)
    duckdb_sql = result.duckdb_examples_path.read_text(encoding="utf-8")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    return table_texts, duckdb_sql, _scrub_volatile(metadata)


def test_parquet_surface_matches_canonical_golden(tmp_path: Path) -> None:
    """Frozen Parquet export (per-table JSON + DuckDB SQL + scrubbed metadata) must match goldens."""
    # Parquet export is backed by the optional analytics extra; skip cleanly when it is absent
    # rather than fail collection with an unrelated dependency error.
    pytest.importorskip("pyarrow")
    first_tables, first_sql, first_meta = _parquet_fixture(out_dir=tmp_path / "first")
    second_tables, second_sql, second_meta = _parquet_fixture(out_dir=tmp_path / "second")

    assert first_tables == second_tables
    assert first_sql == second_sql
    assert first_meta == second_meta
    assert set(first_tables) == set(_PARQUET_TABLE_GOLDEN_PATHS)
    for name, text in first_tables.items():
        _assert_or_bless_text(actual=text, golden_path=_PARQUET_TABLE_GOLDEN_PATHS[name])
    _assert_or_bless_text(actual=first_sql, golden_path=_PARQUET_DUCKDB_GOLDEN_PATH)
    _assert_or_bless_payload(actual=first_meta, golden_path=_PARQUET_METADATA_GOLDEN_PATH)


def test_parquet_goldens_cover_all_exported_tables() -> None:
    """Every exported Parquet table must have a committed canonical golden."""
    for name in _PARQUET_TABLE_NAMES:
        assert _PARQUET_TABLE_GOLDEN_PATHS[name].is_file(), (
            f"missing canonical golden for Parquet table {name!r}"
        )


# --------------------------------------------------------------------------- #
# Bless-path characterization (explicit, isolated, review-visible)
# --------------------------------------------------------------------------- #


def test_bless_path_rewrites_a_golden_and_prints_its_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The opt-in update path is explicit, isolated, and review-visible across surfaces."""
    golden_path = tmp_path / "summary_training_metrics.json"
    golden_path.write_text('{\n  "before": true\n}\n', encoding="utf-8")
    actual = '{\n  "after": 1.0\n}\n'
    monkeypatch.setenv(_BLESS_ENV, "1")

    _assert_or_bless_text(actual=actual, golden_path=golden_path)

    assert golden_path.read_text(encoding="utf-8") == actual
    assert '-  "before": true' in capsys.readouterr().out


@pytest.mark.parametrize("path_name", ["missing.json", "directory"])
def test_non_bless_path_must_be_a_regular_golden_file(
    tmp_path: Path, path_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing and directory golden paths fail loudly before any comparison can pass."""
    monkeypatch.delenv(_BLESS_ENV, raising=False)
    golden_path = tmp_path / path_name
    if path_name == "directory":
        golden_path.mkdir()

    with pytest.raises(FileNotFoundError, match="Golden file not found"):
        _assert_or_bless_text(actual="{}\n", golden_path=golden_path)


def test_scrub_volatile_removes_intentionally_non_deterministic_fields() -> None:
    """Provenance-scrubbing removes exactly the volatile timestamp/git fields and nothing else."""
    payload = {
        "generated_at_utc": "2026-07-10T00:00:00+00:00",
        "git_commit": "abc123",
        "created_at_utc": "2026-07-10T00:00:00+00:00",
        "schema_version": "v1",
        "row_count": 4,
    }
    assert _scrub_volatile(payload) == {"schema_version": "v1", "row_count": 4}
