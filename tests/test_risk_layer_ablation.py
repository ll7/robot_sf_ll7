"""Tests for the risk-layer ablation harness (issue #5832).

The tests assert the acceptance criteria from the issue:
- layer configs load (from the canonical YAML and from the module constants);
- metrics differ between layers on a fixture where they must (the L0-vs-L1
  dynamics inversion);
- the report artifact carries per-planner rank deltas and bootstrap CIs;
- L2 degrades gracefully when no zone metadata is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark import risk_layer_ablation as rla

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/benchmark/risk_layer_ablation.yaml"

BASELINE_STATS = {
    "collisions": {"med": 0.0, "p95": 2.0},
    "force_exceed_events": {"med": 0.0, "p95": 2.0},
    "near_misses": {"med": 0.0, "p95": 2.0},
    "comfort_exposure": {"med": 0.0, "p95": 2.0},
    "time_to_goal_norm": {"med": 0.5, "p95": 1.0},
    "semantic_risk_exposure": {"med": 0.0, "p95": 2.0},
}


def _dynamics_flip_records(n: int = 4) -> list[dict]:
    """Two planners designed so L1 (dynamics) inverts the L0 ranking.

    Planner A has clean geometry but high comfort exposure (bad dynamics).
    Planner B has one collision (worse geometry) but clean dynamics.
    """
    records: list[dict] = []

    def add(planner: str, metrics: dict) -> None:
        for i in range(n):
            records.append(
                {
                    "episode_id": f"{planner}-{i}",
                    "scenario_id": "fam-1",
                    "seed": i,
                    "scenario_params": {"algo": planner},
                    "algo": planner,
                    "metrics": metrics,
                }
            )

    add(
        "A",
        {
            "success": 1.0,
            "collisions": 0.0,
            "near_misses": 0.0,
            "comfort_exposure": 2.0,
            "force_exceed_events": 0.0,
            "jerk_mean": 0.0,
            "time_to_goal_norm": 0.5,
            "semantic_risk_exposure": 0.0,
        },
    )
    add(
        "B",
        {
            "success": 1.0,
            "collisions": 1.0,
            "near_misses": 0.0,
            "comfort_exposure": 0.0,
            "force_exceed_events": 0.0,
            "jerk_mean": 0.0,
            "time_to_goal_norm": 0.5,
            "semantic_risk_exposure": 0.0,
        },
    )
    return records


def test_risk_layer_config_loads() -> None:
    """The canonical ablation config must load with three progressive layers."""
    config = rla.load_risk_layer_config(CONFIG_PATH)

    assert config["schema_version"] == rla.RISK_LAYER_ABLATION_SCHEMA
    layer_names = config["layer_names"]
    assert layer_names == [
        "L0_geometry_only",
        "L1_plus_dynamics",
        "L2_plus_semantic_risk",
    ]
    # Each higher layer must enable strictly more weight terms than the prior one.
    active_counts = [
        sum(1 for weight in config["weights_by_layer"][name].values() if weight > 0.0)
        for name in layer_names
    ]
    assert active_counts == sorted(active_counts)
    assert active_counts[-1] > active_counts[0]
    assert config["weights_by_layer"] == rla.RISK_LAYER_WEIGHTS


def test_risk_layer_constants_consistent() -> None:
    """Module constants expose the documented layer set and weight schema."""
    assert [layer.name for layer in rla.RISK_LAYERS] == [
        "L0_geometry_only",
        "L1_plus_dynamics",
        "L2_plus_semantic_risk",
    ]
    for name in ("L0_geometry_only", "L1_plus_dynamics", "L2_plus_semantic_risk"):
        assert name in rla.RISK_LAYER_WEIGHTS
        assert set(rla.RISK_LAYER_WEIGHTS[name]) == set(rla.RISK_WEIGHT_NAMES)


def test_metrics_differ_between_layers_on_fixture() -> None:
    """The dynamics fixture must invert the ranking between L0 and L1."""
    records = _dynamics_flip_records()
    rows = rla.compute_risk_layer_ablation(
        records,
        baseline_stats=BASELINE_STATS,
        group_by="scenario_params.algo",
    )
    by_planner = {row.planner: row for row in rows}

    # A is best at L0 (geometry-only) but drops at L1; B is the inverse.
    assert by_planner["A"].per_layer_rank["L0_geometry_only"] == 1
    assert by_planner["A"].per_layer_rank["L1_plus_dynamics"] == 2
    assert by_planner["A"].rank_changed is True

    assert by_planner["B"].per_layer_rank["L0_geometry_only"] == 2
    assert by_planner["B"].per_layer_rank["L1_plus_dynamics"] == 1

    # Per-planner rank delta from L0 must capture the inversion.
    assert by_planner["A"].rank_delta_from_l0["L1_plus_dynamics"] == 1.0
    assert by_planner["B"].rank_delta_from_l0["L1_plus_dynamics"] == -1.0


def test_report_artifact_carries_bootstrap_cis() -> None:
    """The report artifact must include per-planner rank CIs and stability."""
    records = _dynamics_flip_records()
    report = rla.build_risk_layer_report(
        records,
        baseline_stats=BASELINE_STATS,
        bootstrap={"seed": 7, "samples": 200},
    )
    assert report["schema_version"] == rla.RISK_LAYER_ABLATION_SCHEMA
    assert len(report["rows"]) == 2

    boot = report["bootstrap"]["L0_geometry_only"]
    assert boot["status"] == "ok"
    assert 0.0 <= boot["stability"] <= 1.0
    assert "rank_cis" in boot
    for planner in ("A", "B"):
        ci = boot["rank_cis"][planner]
        assert ci["ci95_low"] <= ci["mean_rank"] <= ci["ci95_high"]


def test_l2_degrades_gracefully_without_zone_metadata() -> None:
    """Without semantic_risk_exposure, L2 scores remaining active terms only."""
    records = _dynamics_flip_records()
    # Remove the semantic metric entirely from every record.
    for record in records:
        record["metrics"].pop("semantic_risk_exposure", None)

    report = rla.build_risk_layer_report(
        records,
        baseline_stats=BASELINE_STATS,
        bootstrap={"seed": 1, "samples": 50},
    )
    assert report["semantic_risk_available"] is False
    assert report["semantic_risk_coverage"] == {
        "present": 0,
        "total": len(records),
        "status": "unavailable",
    }
    # L2 should still produce finite ranks for both planners.
    by_planner = {row["planner"]: row for row in report["rows"]}
    assert by_planner["A"]["per_layer_rank"]["L2_plus_semantic_risk"] in (1, 2)
    assert by_planner["B"]["per_layer_rank"]["L2_plus_semantic_risk"] in (1, 2)
    assert all(
        row["per_layer_score"]["L2_plus_semantic_risk"]
        == row["per_layer_score"]["L1_plus_dynamics"]
        for row in report["rows"]
    )


def test_semantic_risk_changes_l2_ranking() -> None:
    """L2 must apply semantic exposure instead of a hidden time penalty."""
    records = _dynamics_flip_records()
    for record in records:
        record["metrics"]["comfort_exposure"] = 0.0
        record["metrics"]["semantic_risk_exposure"] = (
            2.0 if record["scenario_params"]["algo"] == "A" else 0.0
        )

    report = rla.build_risk_layer_report(records, baseline_stats=BASELINE_STATS)
    rows = {row["planner"]: row for row in report["rows"]}

    assert report["semantic_risk_available"] is True
    assert rows["A"]["per_layer_rank"]["L0_geometry_only"] == 1
    assert rows["A"]["per_layer_rank"]["L2_plus_semantic_risk"] == 2


def test_partial_semantic_coverage_fails_closed() -> None:
    """A partial semantic metric cannot be silently imputed as neutral."""
    records = _dynamics_flip_records()
    records[0]["metrics"].pop("semantic_risk_exposure")

    with pytest.raises(ValueError, match="partial semantic-risk coverage"):
        rla.build_risk_layer_report(records, baseline_stats=BASELINE_STATS)


def test_mismatched_planner_seed_coverage_fails_closed() -> None:
    """Planner rankings require the same scenario/seed multiset."""
    records = _dynamics_flip_records()
    records.pop()

    with pytest.raises(ValueError, match="scenario/seed coverage differs"):
        rla.build_risk_layer_report(records, baseline_stats=BASELINE_STATS)


def test_config_drives_report_runtime() -> None:
    """The checked-in YAML is consumed by the public config-first entry point."""
    report = rla.build_risk_layer_report_from_config(
        _dynamics_flip_records(),
        baseline_stats=BASELINE_STATS,
        config_path=CONFIG_PATH,
    )

    assert report["bootstrap"]["L0_geometry_only"]["samples"] == 200


def test_markdown_report_renders_delta_table() -> None:
    """The Markdown formatter must render a per-planner delta table."""
    records = _dynamics_flip_records()
    report = rla.build_risk_layer_report(records, baseline_stats=BASELINE_STATS)
    md = rla.format_risk_layer_markdown(report)
    assert "| Planner | L0 rank |" in md
    assert "Δrank L1_plus_dynamics" in md


def test_unknown_layer_rejected() -> None:
    """Requesting an unknown risk layer must fail closed."""
    with pytest.raises(ValueError):
        rla.compute_risk_layer_ablation(
            _dynamics_flip_records(),
            layer_names=["L0_geometry_only", "nope"],
        )
