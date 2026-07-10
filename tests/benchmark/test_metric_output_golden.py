"""Freeze-safe golden regression coverage for benchmark aggregate outputs (#4985).

The fixture is deliberately small, committed, and static: four episodes across two
fixed scenarios and seeds.  This test protects the end-to-end aggregate output
contract, not a benchmark-performance claim.

To intentionally update the reviewed output golden, run:

``ROBOT_SF_BLESS_GOLDEN=1 uv run pytest tests/benchmark/test_metric_output_golden.py -v -s``
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.aggregate import compute_aggregates, read_jsonl

_BLESS_ENV = "ROBOT_SF_BLESS_GOLDEN"
_GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "benchmark" / "golden"
_EPISODES_PATH = _GOLDEN_DIR / "aggregate_episodes.jsonl"
_GOLDEN_PATH = _GOLDEN_DIR / "aggregate_metrics.json"


def _canonical_json(payload: dict[str, Any]) -> str:
    """Return the reviewable, platform-independent JSON representation."""
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _aggregate_fixture() -> str:
    """Aggregate the frozen episodes fixture using the production default surface."""
    return _canonical_json(compute_aggregates(read_jsonl(_EPISODES_PATH)))


def _assert_or_bless_golden(*, actual: str, golden_path: Path) -> None:
    """Compare output to the golden, or deliberately rewrite it with a visible diff."""
    is_bless = os.environ.get(_BLESS_ENV) == "1"
    if not is_bless and not golden_path.is_file():
        raise FileNotFoundError(
            f"Golden file not found at {golden_path}. "
            f"Run with {_BLESS_ENV}=1 only for a reviewed intentional update."
        )
    expected = golden_path.read_text(encoding="utf-8") if golden_path.is_file() else ""
    if is_bless:
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
        f"Aggregate metric output drifted from {golden_path}. "
        f"Review the change, then intentionally bless it with "
        f"{_BLESS_ENV}=1 uv run pytest tests/benchmark/test_metric_output_golden.py -v -s."
    )


def test_aggregate_metric_output_matches_canonical_golden() -> None:
    """Frozen aggregate metric output must exactly match its reviewed JSON golden."""
    first_run = _aggregate_fixture()
    second_run = _aggregate_fixture()

    assert first_run == second_run
    _assert_or_bless_golden(actual=first_run, golden_path=_GOLDEN_PATH)


def test_bless_path_rewrites_a_golden_and_prints_its_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The opt-in update path is explicit, isolated, and review-visible."""
    golden_path = tmp_path / "aggregate_metrics.json"
    golden_path.write_text('{\n  "before": true\n}\n', encoding="utf-8")
    actual = '{\n  "after": 1.0\n}\n'
    monkeypatch.setenv(_BLESS_ENV, "1")

    _assert_or_bless_golden(actual=actual, golden_path=golden_path)

    assert golden_path.read_text(encoding="utf-8") == actual
    assert '-  "before": true' in capsys.readouterr().out


@pytest.mark.parametrize("path_name", ["missing.json", "directory"])
def test_non_bless_path_must_be_a_regular_golden_file(tmp_path: Path, path_name: str) -> None:
    """Missing and directory golden paths fail loudly before any comparison can pass."""

    golden_path = tmp_path / path_name
    if path_name == "directory":
        golden_path.mkdir()

    with pytest.raises(FileNotFoundError, match="Golden file not found"):
        _assert_or_bless_golden(actual="{}\n", golden_path=golden_path)
