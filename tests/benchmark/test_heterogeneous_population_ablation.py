"""Mean-matched heterogeneous-population ablation harness tests for issue #3574."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.heterogeneous_population_ablation import (
    HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
    ArchetypePopulationSpec,
    audit_smoke_mean_match,
    build_mean_matched_population_pair,
    build_per_archetype_ablation_report,
)


def _archetypes() -> dict[str, ArchetypePopulationSpec]:
    return {
        "cautious": ArchetypePopulationSpec(desired_speed_factor=0.7, radius_m=0.35),
        "standard": ArchetypePopulationSpec(desired_speed_factor=1.0, radius_m=0.30),
        "hurried": ArchetypePopulationSpec(desired_speed_factor=1.4, radius_m=0.25),
    }


def _control_trace() -> dict[str, object]:
    return {
        "schema_version": "pedestrian-control-trace.v1",
        "pedestrians": [
            {
                "id": "ped_cautious",
                "archetype": "cautious",
                "steps": [
                    {"step": 0, "clearance_m": 1.0},
                    {"step": 1, "clearance_m": 0.8},
                ],
            },
            {
                "id": "ped_hurried",
                "archetype": "hurried",
                "steps": [
                    {"step": 0, "clearance_m": 0.5},
                    {"step": 1, "clearance_m": 0.3},
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
