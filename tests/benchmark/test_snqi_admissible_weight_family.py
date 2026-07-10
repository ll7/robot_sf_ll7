"""Contract tests for issue #5142's ex-ante SNQI weight-family policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    ADMISSIBLE_WEIGHT_FAMILY_SCHEMA,
    build_scalarization_sensitivity_report,
    classify_admissible_weight_vector,
    load_admissible_weight_family_config,
)
from scripts.tools.analyze_snqi_scalarization_sensitivity import main as scalarization_cli_main

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs/analysis/issue_5142_snqi_admissible_weight_family.yaml"

_WEIGHTS = {
    "w_success": 0.19,
    "w_time": 0.09,
    "w_collisions": 0.11,
    "w_near": 0.31,
    "w_comfort": 0.18,
    "w_force_exceed": 0.07,
    "w_jerk": 0.05,
}
_BASELINE = {
    "collisions": {"med": 0.0, "p95": 1.0},
    "near_misses": {"med": 0.0, "p95": 1.0},
    "force_exceed_events": {"med": 0.0, "p95": 1.0},
    "jerk_mean": {"med": 0.0, "p95": 1.0},
}


def _records() -> list[dict[str, object]]:
    return [
        {
            "planner_key": "fast_risky",
            "scenario_id": "crossing",
            "horizon": "h500",
            "metrics": {
                "success": 1.0,
                "time_to_goal_norm": 0.1,
                "collisions": 1.0,
                "near_misses": 0.0,
                "comfort_exposure": 0.05,
                "force_exceed_events": 0.0,
                "jerk_mean": 0.05,
            },
        },
        {
            "planner_key": "safe_slow",
            "scenario_id": "crossing",
            "horizon": "h500",
            "metrics": {
                "success": 1.0,
                "time_to_goal_norm": 0.8,
                "collisions": 0.0,
                "near_misses": 0.0,
                "comfort_exposure": 0.2,
                "force_exceed_events": 0.0,
                "jerk_mean": 0.2,
            },
        },
    ]


def test_committed_family_classifies_simplex_vector_and_zero_probe() -> None:
    """The tracked policy admits its balanced vector but labels a zero term as stress."""
    family = load_admissible_weight_family_config(_CONFIG)

    assert family["id"] == "constraint_completion_comfort_simplex_v1"
    assert classify_admissible_weight_vector(_WEIGHTS, family)["classification"] == "admissible"

    zeroed = dict(_WEIGHTS)
    zeroed["w_jerk"] = 0.0
    stress_probe = classify_admissible_weight_vector(zeroed, family)
    assert stress_probe["classification"] == "stress_probe"
    assert stress_probe["stress_probe_label"] == "out_of_family_stress_probe"
    assert "below_component_minimum:w_jerk" in stress_probe["violations"]


def test_report_separates_admissible_and_full_sweep_inversions() -> None:
    """Headline totals exclude out-of-family probes while retaining their full-sweep totals."""
    family = load_admissible_weight_family_config(_CONFIG)
    report = build_scalarization_sensitivity_report(
        _records(),
        weights=_WEIGHTS,
        baseline=_BASELINE,
        admissible_weight_family=family,
    )

    sensitivity = report["admissible_weight_family"]
    full_sweep = sensitivity["full_sweep"]
    admissible = sensitivity["admissible_family"]
    stress_probes = sensitivity["stress_probes"]
    assert sensitivity["headline_scope"] == "admissible_family_only"
    assert full_sweep["vector_count"] == (
        admissible["vector_count"] + stress_probes["vector_count"]
    )
    assert stress_probes["vector_count"] > 0
    assert report["summary"]["headline_scope"] == "admissible_family_only"
    assert (
        report["summary"]["admissible_family_pairwise_reversal_count_vs_base"]
        == admissible["pairwise_reversal_count_vs_base"]
    )


def test_loader_rejects_an_incomplete_weight_partition(tmp_path: Path) -> None:
    """A policy cannot silently omit a SNQI component from classification."""
    malformed = tmp_path / "malformed_family.yaml"
    malformed.write_text(
        "\n".join(
            [
                f"schema_version: {ADMISSIBLE_WEIGHT_FAMILY_SCHEMA}",
                "admissible_family:",
                "  id: incomplete",
                "  normalization: simplex_l1",
                "  component_bounds: {minimum: 0.0, maximum: 1.0}",
                "  groups: {only: [w_success]}",
                "  ordered_group_masses: [only, missing]",
                "  stress_probe_label: stress",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="omit SNQI weights"):
        load_admissible_weight_family_config(malformed)


def test_cli_writes_weight_family_provenance_and_markdown_summary(tmp_path: Path) -> None:
    """The public CLI wires the tracked policy into JSON and human-readable output."""
    episodes = tmp_path / "episodes.jsonl"
    weights = tmp_path / "weights.json"
    baseline = tmp_path / "baseline.json"
    output_dir = tmp_path / "output"
    episodes.write_text("\n".join(json.dumps(row) for row in _records()), encoding="utf-8")
    weights.write_text(json.dumps(_WEIGHTS), encoding="utf-8")
    baseline.write_text(json.dumps(_BASELINE), encoding="utf-8")

    assert (
        scalarization_cli_main(
            [
                "--episodes",
                str(episodes),
                "--weights",
                str(weights),
                "--baseline",
                str(baseline),
                "--weight-family-config",
                str(_CONFIG),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    report = json.loads((output_dir / "snqi_scalarization_sensitivity.json").read_text())
    markdown = (output_dir / "snqi_scalarization_sensitivity.md").read_text()
    assert report["inputs"]["provenance"]["weight_family_config"]["sha256"]
    assert report["admissible_weight_family"]["headline_scope"] == "admissible_family_only"
    assert "## Ex-Ante Admissible Weight Family" in markdown
