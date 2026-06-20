"""Tests for the external-prior divergence de-risk tool (issue #3192).

These prove the honesty contract (uncited reference statistics are never treated as agreement)
and the canonical per-dataset verdict logic, using small in-memory fixtures.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_PATH = _REPO_ROOT / "scripts" / "tools" / "external_prior_divergence.py"

_spec = importlib.util.spec_from_file_location("external_prior_divergence", _TOOL_PATH)
assert _spec and _spec.loader
epd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(epd)


def _reference(value, citation):
    return {
        "datasets": [
            {
                "key": "sdd",
                "name": "Stanford Drone Dataset",
                "statistics": [
                    {
                        "key": "pedestrian_speed_mean_mps",
                        "value": value,
                        "unit": "m/s",
                        "source_citation": citation,
                    }
                ],
            }
        ]
    }


def test_close_priors_are_sufficient():
    """Authored value within the sufficient threshold yields priors-sufficient-for-diagnostic."""
    ref = _reference(1.30, "Robicquet et al. 2016, Table 2")
    authored = {"datasets": {"sdd": {"pedestrian_speed_mean_mps": 1.34}}}
    report = epd.compute_report(ref, authored)
    assert report["datasets"][0]["verdict"] == epd.VERDICT_SUFFICIENT


def test_far_priors_change_scope():
    """A divergence beyond the material threshold yields raw-data-materially-changes-scope."""
    ref = _reference(1.30, "Robicquet et al. 2016, Table 2")
    authored = {"datasets": {"sdd": {"pedestrian_speed_mean_mps": 2.60}}}
    report = epd.compute_report(ref, authored)
    assert report["datasets"][0]["verdict"] == epd.VERDICT_MATERIAL


def test_uncited_reference_is_not_comparable():
    """An uncited reference value must be not-comparable and verdict inconclusive (fail-closed)."""
    ref = _reference(None, "NEEDS_CITATION")
    authored = {"datasets": {"sdd": {"pedestrian_speed_mean_mps": 1.34}}}
    report = epd.compute_report(ref, authored)
    dataset = report["datasets"][0]
    assert dataset["verdict"] == epd.VERDICT_INCONCLUSIVE
    assert dataset["comparable_count"] == 0
    assert dataset["comparisons"][0]["status"] == "not-comparable"


def test_missing_authored_value_is_not_comparable():
    """A cited reference with no authored counterpart is not-comparable, not agreement."""
    ref = _reference(1.30, "Robicquet et al. 2016, Table 2")
    authored = {"datasets": {"sdd": {}}}
    report = epd.compute_report(ref, authored)
    assert report["datasets"][0]["comparisons"][0]["status"] == "not-comparable"
    assert report["datasets"][0]["verdict"] == epd.VERDICT_INCONCLUSIVE


def test_non_finite_authored_value_fails_closed():
    """Comparable statistics must reject non-finite authored values before metric output."""
    ref = _reference(1.30, "Robicquet et al. 2016, Table 2")
    authored = {"datasets": {"sdd": {"pedestrian_speed_mean_mps": float("nan")}}}

    with pytest.raises(ValueError, match="authored value.*must be finite"):
        epd.compute_report(ref, authored)


def test_non_finite_cited_reference_value_fails_closed():
    """Comparable statistics must reject non-finite cited references before metric output."""
    ref = _reference(float("inf"), "Robicquet et al. 2016, Table 2")
    authored = {"datasets": {"sdd": {"pedestrian_speed_mean_mps": 1.34}}}

    with pytest.raises(ValueError, match="reference value.*must be finite"):
        epd.compute_report(ref, authored)


def test_cli_rejects_non_finite_values_without_writing_report(tmp_path: Path, capsys):
    """The CLI must exit fail-closed instead of writing NaN/Inf metrics."""
    reference = tmp_path / "reference.yaml"
    authored = tmp_path / "authored.yaml"
    out = tmp_path / "report.json"
    reference.write_text(
        """
datasets:
  - key: sdd
    statistics:
      - key: pedestrian_speed_mean_mps
        value: .inf
        source_citation: Robicquet et al. 2016, Table 2
"""
    )
    authored.write_text(
        """
datasets:
  sdd:
    pedestrian_speed_mean_mps: 1.34
"""
    )

    with pytest.raises(SystemExit) as exc_info:
        epd.main(["--reference", str(reference), "--authored", str(authored), "--out", str(out)])

    assert exc_info.value.code == 2
    assert "reference value" in capsys.readouterr().err
    assert not out.exists()


def test_scaffold_configs_yield_inconclusive(tmp_path: Path):
    """The shipped uncited scaffolds must currently produce inconclusive-need-pilot for all."""
    reference = epd._load_yaml(
        _REPO_ROOT / "configs" / "research" / "external_prior_reference_stats.yaml"
    )
    authored = epd._load_yaml(
        _REPO_ROOT / "configs" / "research" / "authored_prior_summary_stats.yaml"
    )
    report = epd.compute_report(reference, authored)
    assert report["datasets"]
    assert all(d["verdict"] == epd.VERDICT_INCONCLUSIVE for d in report["datasets"])
