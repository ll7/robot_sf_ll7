"""Mean-matched heterogeneous-population ablation harness tests for issue #3574."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.heterogeneous_population_ablation import (
    HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
    MEAN_MATCHED_HETEROGENEITY_HARNESS_SCHEMA,
    ArchetypePopulationSpec,
    audit_smoke_mean_match,
    build_mean_matched_harness_manifest,
    build_mean_matched_population_pair,
    build_per_archetype_ablation_report,
)
from robot_sf.benchmark.pedestrian_control_trace import PEDESTRIAN_CONTROL_TRACE_LABELS_KEY


def _archetypes() -> dict[str, ArchetypePopulationSpec]:
    return {
        "cautious": ArchetypePopulationSpec(desired_speed_factor=0.7, radius_m=0.35),
        "standard": ArchetypePopulationSpec(desired_speed_factor=1.0, radius_m=0.30),
        "hurried": ArchetypePopulationSpec(desired_speed_factor=1.4, radius_m=0.25),
    }


def _control_trace() -> dict[str, object]:
    return {
        "schema_version": "pedestrian-control-trace.v1",
        "near_field_clearance_threshold_m": 1.0,
        "pedestrians": [
            {
                "id": "ped_cautious",
                "archetype": "cautious",
                "steps": [
                    {"step": 0, "clearance_m": 1.0, "near_field_exposure_s": 0.0},
                    {"step": 1, "clearance_m": 0.8, "near_field_exposure_s": 0.1},
                ],
            },
            {
                "id": "ped_hurried",
                "archetype": "hurried",
                "steps": [
                    {"step": 0, "clearance_m": 0.5, "near_field_exposure_s": 0.1},
                    {"step": 1, "clearance_m": 0.3, "near_field_exposure_s": 0.1},
                ],
            },
        ],
    }


def test_mean_matched_population_pair_preserves_weighted_speed_and_radius() -> None:
    """The homogeneous arm uses the heterogeneous mixture's exact parameter means."""

    report = build_mean_matched_population_pair(
        population_size=12,
        composition={"cautious": 0.25, "standard": 0.5, "hurried": 0.25},
        archetypes=_archetypes(),
        seed=3574,
    )

    assert report["schema_version"] == HETEROGENEOUS_POPULATION_ABLATION_SCHEMA
    assert report["status"] == "analysis_harness_only"
    assert report["mean_matched_parameters"]["desired_speed_factor"] == pytest.approx(1.025)
    assert report["mean_matched_parameters"]["radius_m"] == pytest.approx(0.3)
    assert report["arms"]["heterogeneous"]["counts"] == {
        "cautious": 3,
        "hurried": 3,
        "standard": 6,
    }
    homogeneous_records = report["arms"]["mean_matched_homogeneous"]["records"]
    assert {record["archetype"] for record in homogeneous_records} == {"mean_matched_homogeneous"}
    assert {record["desired_speed_factor"] for record in homogeneous_records} == {1.025}
    assert {record["radius_m"] for record in homogeneous_records} == {0.3}
    heterogeneous_labels = report["arms"]["heterogeneous"][PEDESTRIAN_CONTROL_TRACE_LABELS_KEY]
    assert len(heterogeneous_labels) == 12
    assert heterogeneous_labels[0]["simulator_index"] == 0
    assert heterogeneous_labels[0]["source"] == "mean_matched_harness.heterogeneous_population"
    assert {label["desired_speed_factor"] for label in heterogeneous_labels} == {0.7, 1.0, 1.4}
    homogeneous_labels = report["arms"]["mean_matched_homogeneous"][
        PEDESTRIAN_CONTROL_TRACE_LABELS_KEY
    ]
    assert {label["archetype"] for label in homogeneous_labels} == {"mean_matched_homogeneous"}
    assert {label["desired_speed_factor"] for label in homogeneous_labels} == {1.025}


def test_mean_matched_population_pair_rejects_unknown_archetype() -> None:
    """Unknown mixture entries fail before producing misleading arm records."""

    with pytest.raises(ValueError, match="unknown archetypes"):
        build_mean_matched_population_pair(
            population_size=4,
            composition={"unknown": 1.0},
            archetypes=_archetypes(),
        )


def test_mean_matched_population_pair_reports_missing_archetype_spec_key() -> None:
    """Mapping specs fail with a descriptive missing-key error."""

    with pytest.raises(ValueError, match="missing key: radius_m"):
        build_mean_matched_population_pair(
            population_size=4,
            composition={"cautious": 1.0},
            archetypes={"cautious": {"desired_speed_factor": 0.7}},
        )


def test_mean_matched_population_pair_normalizes_tolerance_slop() -> None:
    """Tiny composition roundoff is normalized before mean calculations."""

    report = build_mean_matched_population_pair(
        population_size=10,
        composition={"cautious": 0.1, "standard": 0.2, "hurried": 0.7000001},
        archetypes=_archetypes(),
    )

    expected = (0.1 * 0.7 + 0.2 * 1.0 + 0.7000001 * 1.4) / 1.0000001
    assert report["mean_matched_parameters"]["desired_speed_factor"] == pytest.approx(expected)


def test_per_archetype_ablation_report_blocks_missing_control_trace() -> None:
    """Missing traces stay blocked diagnostics, not claim-supporting metrics."""

    report = build_per_archetype_ablation_report(
        control_traces_by_arm={"heterogeneous": None},
        metric_key="clearance_m",
    )

    assert report["arms"]["heterogeneous"] == {
        "status": "blocked",
        "ready": False,
        "blockers": ["pedestrian_control_trace missing"],
    }


def test_per_archetype_ablation_report_uses_trace_metrics_when_ready() -> None:
    """Ready traces feed the existing per-archetype metric harness per arm."""

    report = build_per_archetype_ablation_report(
        control_traces_by_arm={"heterogeneous": _control_trace()},
        metric_key="clearance_m",
        higher_is_safer=True,
        cvar_alpha=0.5,
    )

    metrics = report["arms"]["heterogeneous"]["metrics"]
    assert metrics["source"] == "pedestrian_control_trace"
    assert metrics["worst_archetype_by_mean"] == "hurried"
    assert metrics["per_archetype"]["cautious"]["mean"] == pytest.approx(0.9)
    assert metrics["per_archetype"]["hurried"]["mean"] == pytest.approx(0.4)


def test_smoke_mean_match_fallback_ignores_metadata_mappings() -> None:
    """Fallback smoke parsing should not confuse metadata maps for condition arms."""

    audit = audit_smoke_mean_match(
        {
            "schema_version": "smoke.v1",
            "metadata": {"run": "ignored"},
            "config": {"also": "ignored"},
            "homogeneous_standard": {"mean_min_clearance": {"mean": 1.0}},
            "heterogeneous_mixed": {"mean_min_clearance": {"mean": 1.0}},
        },
        metric_key="mean_min_clearance",
    )

    assert audit["status"] == "ready"


def test_issue_3206_three_seed_smoke_audits_as_mean_matched_but_not_per_archetype_ready() -> None:
    """Existing three-seed smoke artifact proves aggregate mean matching only."""

    aggregate_report = json.loads(
        Path(
            "docs/context/evidence/issue_3206_heterogeneous_pedestrian_smoke_2026-06-20/"
            "aggregate_by_condition.json"
        ).read_text(encoding="utf-8")
    )
    detailed_report = json.loads(
        Path(
            "docs/context/evidence/issue_3206_heterogeneous_pedestrian_smoke_2026-06-20/"
            "smoke_report.json"
        ).read_text(encoding="utf-8")
    )

    audit = audit_smoke_mean_match(aggregate_report)

    assert audit["schema_version"] == HETEROGENEOUS_POPULATION_ABLATION_SCHEMA
    assert audit["status"] == "ready"
    assert audit["mean_matched"] is True
    assert audit["absolute_delta"] == pytest.approx(0.0)
    assert audit["arm_means"] == {
        "homogeneous_standard": pytest.approx(0.9795916847173137),
        "mixed_balanced": pytest.approx(0.9795916847173137),
    }
    assert (
        detailed_report["per_archetype_distributional_status"]
        == "not_computable_from_current_smoke"
    )


def _manifest_config() -> dict[str, object]:
    return {
        "trace_metric_keys": ["clearance_m", "near_field_exposure_s"],
        "planners": [{"key": "goal", "algo": "goal"}, {"key": "social_force"}],
        "seeds": [101, 102],
        "scenarios": [
            {
                "id": "classic_density_002",
                "density": 0.02,
                "population_size": 4,
                "archetype_seed": 3574,
                "composition": {"cautious": 0.25, "standard": 0.5, "hurried": 0.25},
                "archetypes": {
                    "cautious": {"desired_speed_factor": 0.7, "radius_m": 0.35},
                    "standard": {"desired_speed_factor": 1.0, "radius_m": 0.3},
                    "hurried": {"desired_speed_factor": 1.4, "radius_m": 0.25},
                },
            }
        ],
    }


def test_mean_matched_harness_manifest_attributes_rows_by_pairing_keys() -> None:
    """Dry-run rows are attributable by scenario, seed, planner, density, and arm."""

    manifest = build_mean_matched_harness_manifest(_manifest_config(), config_path="smoke.yaml")

    assert manifest["schema_version"] == MEAN_MATCHED_HETEROGENEITY_HARNESS_SCHEMA
    assert manifest["issue"] == 3574
    assert manifest["claim_boundary"] == "harness_only_no_ablation_result"
    assert manifest["paired_arms"] == ["heterogeneous", "mean_matched_homogeneous"]
    assert manifest["row_count"] == 8
    rows = manifest["manifest_rows"]
    row_keys = {
        (row["scenario_id"], row["planner"], row["seed"], row["density"], row["population_arm"])
        for row in rows
    }
    assert ("classic_density_002", "goal", 101, 0.02, "heterogeneous") in row_keys
    assert ("classic_density_002", "goal", 101, 0.02, "mean_matched_homogeneous") in row_keys
    assert {row["population_composition_hash"] for row in rows} == {
        manifest["scenario_rows"][0]["population_composition_hash"]
    }
    assert {
        row["arm_population"]["counts"]["mean_matched_homogeneous"]
        for row in rows
        if row["population_arm"] == "mean_matched_homogeneous"
    } == {4}
    assert manifest["scenario_rows"][0]["mean_matched_parameters"][
        "desired_speed_factor"
    ] == pytest.approx(1.025)


def test_mean_matched_harness_manifest_fails_closed_on_missing_control_trace_inputs() -> None:
    """Pre-run manifests name the exact trace fields future metrics require."""

    manifest = build_mean_matched_harness_manifest(_manifest_config())

    assert manifest["status"] == "blocked_pending_control_trace"
    assert any(
        "metadata.pedestrian_control_trace missing" in blocker for blocker in manifest["blockers"]
    )
    assert any("steps[].clearance_m missing" in blocker for blocker in manifest["blockers"])
    assert any(
        "steps[].near_field_exposure_s missing" in blocker for blocker in manifest["blockers"]
    )
    first_row = manifest["manifest_rows"][0]
    assert first_row["trace_readiness"]["ready"] is False
    assert (
        f"scenario.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY}"
        in first_row["expected_episode_output_keys"]
    )
    assert (
        "metadata.pedestrian_control_trace.pedestrians[].steps[].clearance_m"
        in first_row["expected_episode_output_keys"]
    )
    assert (
        "metadata.pedestrian_control_trace.pedestrians[].steps[].near_field_exposure_s"
        in first_row["expected_episode_output_keys"]
    )
    assert (
        "metadata.pedestrian_control_trace.near_field_clearance_threshold_m"
        in first_row["expected_episode_output_keys"]
    )
    assert PEDESTRIAN_CONTROL_TRACE_LABELS_KEY in first_row["arm_population"]
    assert len(first_row["arm_population"][PEDESTRIAN_CONTROL_TRACE_LABELS_KEY]) == 4


def test_mean_matched_harness_manifest_accepts_fixture_control_traces() -> None:
    """Fixture-only dry-run proves bridge to existing trace-readiness primitive."""

    config = _manifest_config()
    scenario = config["scenarios"][0]
    assert isinstance(scenario, dict)
    scenario["control_traces"] = {
        "heterogeneous": _control_trace(),
        "mean_matched_homogeneous": _control_trace(),
    }

    manifest = build_mean_matched_harness_manifest(config)

    assert manifest["status"] == "ready"
    assert manifest["blockers"] == []
    readiness = manifest["scenario_rows"][0]["trace_readiness_by_arm"]["heterogeneous"]
    assert readiness["metrics"]["clearance_m"]["status"] == "ready"
    assert readiness["metrics"]["near_field_exposure_s"]["status"] == "ready"
    assert readiness["metrics"]["clearance_m"]["archetype_counts"] == {
        "cautious": 1,
        "hurried": 1,
    }


def test_mean_matched_harness_manifest_blocks_exposure_without_threshold_metadata() -> None:
    """Near-field exposure cannot be interpreted when its clearance threshold is absent."""

    config = _manifest_config()
    scenario = config["scenarios"][0]
    assert isinstance(scenario, dict)
    trace = _control_trace()
    trace.pop("near_field_clearance_threshold_m")
    scenario["control_traces"] = {
        "heterogeneous": trace,
        "mean_matched_homogeneous": _control_trace(),
    }

    manifest = build_mean_matched_harness_manifest(config)

    assert manifest["status"] == "blocked_pending_control_trace"
    assert any(
        "control_trace.near_field_clearance_threshold_m missing" in blocker
        for blocker in manifest["blockers"]
    )


@pytest.mark.parametrize("threshold", [True, "1.0", [], math.nan, -0.1])
def test_mean_matched_harness_manifest_blocks_invalid_exposure_threshold(
    threshold: object,
) -> None:
    """Near-field exposure provenance must be a finite non-negative numeric threshold."""

    config = _manifest_config()
    scenario = config["scenarios"][0]
    assert isinstance(scenario, dict)
    trace = _control_trace()
    trace["near_field_clearance_threshold_m"] = threshold
    scenario["control_traces"] = {
        "heterogeneous": trace,
        "mean_matched_homogeneous": _control_trace(),
    }

    manifest = build_mean_matched_harness_manifest(config)

    assert manifest["status"] == "blocked_pending_control_trace"
    assert any(
        "control_trace.near_field_clearance_threshold_m must be finite non-negative number"
        in blocker
        for blocker in manifest["blockers"]
    )


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"trace_metric_keys": "clearance_m"}, "trace_metric_keys must be sequence"),
        ({"seeds": [True]}, "seeds"),
        ({"planners": [object()]}, "planners\\[0\\] must be string or mapping"),
    ],
)
def test_mean_matched_harness_manifest_rejects_malformed_config(
    patch: dict[str, object],
    match: str,
) -> None:
    """Malformed dry-run configs fail before producing ambiguous row contracts."""

    config = _manifest_config()
    config.update(patch)

    with pytest.raises(ValueError, match=match):
        build_mean_matched_harness_manifest(config)


def test_mean_matched_harness_manifest_reports_bad_fixture_trace_metric() -> None:
    """Trace fixtures with missing metric fields stay blocked with field-level detail."""

    config = _manifest_config()
    scenario = config["scenarios"][0]
    assert isinstance(scenario, dict)
    scenario["control_traces"] = {
        "heterogeneous": {
            "pedestrians": [{"id": "ped_cautious", "archetype": "cautious", "steps": [{}]}]
        },
        "mean_matched_homogeneous": _control_trace(),
    }

    manifest = build_mean_matched_harness_manifest(config)

    assert manifest["status"] == "blocked_pending_control_trace"
    assert any(
        "control_trace.pedestrians[0].steps[0] missing 'clearance_m'" in blocker
        for blocker in manifest["blockers"]
    )


_REPO_HARNESS_CONFIG_PATH = (
    Path(__file__).parents[2] / "configs/benchmarks/issue_3574_mean_matched_harness_smoke.yaml"
)


def _load_repo_harness_config() -> dict[str, object]:
    return yaml.safe_load(_REPO_HARNESS_CONFIG_PATH.read_text(encoding="utf-8"))


def test_repo_harness_config_loads_and_has_at_least_three_planners() -> None:
    """Repo config meets the ≥3-planner DoD for rank-order sensitivity analysis."""

    config = _load_repo_harness_config()
    planners = config.get("planners", [])
    assert isinstance(planners, list), "planners must be a list"
    assert len(planners) >= 3, f"DoD requires ≥3 planners; got {len(planners)}: {planners}"


def test_repo_harness_config_builds_valid_manifest_and_fails_closed() -> None:
    """Repo config builds a valid manifest and blocks on missing episode traces."""

    config = _load_repo_harness_config()
    manifest = build_mean_matched_harness_manifest(
        config, config_path=str(_REPO_HARNESS_CONFIG_PATH)
    )

    assert manifest["schema_version"] == MEAN_MATCHED_HETEROGENEITY_HARNESS_SCHEMA
    assert manifest["issue"] == 3574
    # Without episode runs the manifest is blocked — it fails closed, not open.
    assert manifest["status"] == "blocked_pending_control_trace"
    assert manifest["blockers"], "manifest must name the missing trace fields"
    # Paired arms must appear in every row.
    arms_seen = {row["population_arm"] for row in manifest["manifest_rows"]}
    assert arms_seen == {"heterogeneous", "mean_matched_homogeneous"}


def test_repo_harness_config_manifest_includes_near_field_exposure_metric() -> None:
    """Repo config declares near_field_exposure_s for per-archetype tail risk."""

    config = _load_repo_harness_config()
    metric_keys = config.get("trace_metric_keys", [])
    assert "near_field_exposure_s" in metric_keys, (
        "near_field_exposure_s must be declared for per-archetype tail-risk analysis"
    )
    manifest = build_mean_matched_harness_manifest(config)
    # near_field_clearance_threshold_m must be surfaced in the row-level output contract.
    first_row = manifest["manifest_rows"][0]
    assert any(
        "near_field_clearance_threshold_m" in key
        for key in first_row["expected_episode_output_keys"]
    )
