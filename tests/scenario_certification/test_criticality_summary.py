"""Tests for criticality_summary.v1 writer/validator contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from robot_sf.scenario_certification.criticality_summary import (
    CRITICALITY_SUMMARY_SCHEMA_VERSION,
    CriticalitySummaryV1,
    _aggregate_subset,
    _classify_episode_row_status,
    _zero_row_status_counts,
    build_criticality_summary_from_compact_evidence,
    build_criticality_summary_from_pilot,
    build_pair_summary_with_statuses,
    criticality_summary_to_dict,
    validate_criticality_summary,
)


def _sample_manifest_metadata() -> tuple[str, str]:
    """Return a tiny (manifest, manifest_id) pair."""
    return (
        "configs/scenarios/perturbations/example_pilot_v1.yaml",
        "example_pilot_v1",
    )


def _sample_materialization() -> dict[str, object]:
    """Return a tiny materialization payload."""
    return {
        "schema_version": "scenario_perturbation_pilot_matrix.v1",
        "manifest_id": "example_pilot_v1",
        "included_variants": ["demo_noop", "demo_offset"],
        "excluded_variants": [],
        "variant_count": 2,
        "local_artifact_boundary": (
            "materialized scenario matrix, route overrides, and raw episode JSONL "
            "remain ignored local outputs reproducible from the tracked manifest and command"
        ),
    }


def _sample_planner_runs() -> dict[str, dict[str, object]]:
    """Return a tiny planner-runs payload."""
    return {
        "goal": {
            "algo": "goal",
            "algo_config_path": None,
            "source": "raw_algo",
            "episodes": 3,
        },
    }


def _sample_scenario_metadata() -> dict[str, dict[str, object]]:
    """Return tiny scenario metadata with one noop and one offset."""
    return {
        "demo_noop": {
            "source_scenario_id": "demo",
            "variant_id": "demo_noop",
            "family": "noop",
        },
        "demo_offset": {
            "source_scenario_id": "demo",
            "variant_id": "demo_offset",
            "family": "robot_route_offset",
        },
    }


def _sample_records() -> dict[str, list[dict[str, object]]]:
    """Return minimal episode records for one planner and two variants."""
    return {
        "goal": [
            {
                "scenario_id": "demo_noop",
                "seed": 111,
                "termination_reason": "success",
                "metrics": {"min_distance": 2.0},
            },
            {
                "scenario_id": "demo_offset",
                "seed": 111,
                "termination_reason": "max_steps",
                "metrics": {"min_distance": 1.5},
            },
            {
                "scenario_id": "demo_noop",
                "seed": 222,
                "termination_reason": "collision",
                "metrics": {"min_distance": 1.2},
            },
            {
                "scenario_id": "demo_offset",
                "seed": 222,
                "termination_reason": "collision",
                "metrics": {"min_distance": 1.1},
            },
        ],
    }


def test_schema_version_is_stable() -> None:
    """The constant must match the schema file's version."""
    assert CRITICALITY_SUMMARY_SCHEMA_VERSION == "criticality_summary.v1"


def test_zero_row_status_counts_has_all_required_statuses() -> None:
    """Zeroed row status counts must include all six required status keys."""
    zeroed = _zero_row_status_counts()
    assert set(zeroed.keys()) == {
        "completed",
        "invalid",
        "fallback",
        "degraded",
        "missing",
        "failed",
    }
    assert all(value == 0 for value in zeroed.values())


def test_aggregate_subset_tracks_explicit_row_statuses() -> None:
    """Aggregation must split status counts from completed-pair means."""
    pair_rows = [
        {
            "planner": "goal",
            "source_scenario_id": "demo",
            "noop_variant_id": "demo_noop",
            "perturbed_variant_id": "demo_offset",
            "perturbed_family": "robot_route_offset",
            "seed": 111,
            "pair_status": "completed",
            "noop_status": "completed",
            "perturbed_status": "completed",
            "success_delta": 1.0,
            "collision_delta": 0.0,
        },
        {
            "planner": "goal",
            "source_scenario_id": "demo",
            "noop_variant_id": "demo_noop",
            "perturbed_variant_id": "demo_offset",
            "perturbed_family": "robot_route_offset",
            "seed": 222,
            "pair_status": "excluded",
            "noop_status": "completed",
            "perturbed_status": "fallback",
            "success_delta": None,
            "collision_delta": None,
        },
        {
            "planner": "goal",
            "source_scenario_id": "demo",
            "noop_variant_id": "demo_noop",
            "perturbed_variant_id": "demo_other",
            "perturbed_family": "robot_route_offset",
            "seed": 333,
            "pair_status": "excluded",
            "noop_status": "completed",
            "perturbed_status": "degraded",
            "success_delta": None,
            "collision_delta": None,
        },
    ]
    result = _aggregate_subset(pair_rows)
    assert result["pairs"] == 3
    assert result["row_status_counts"]["completed"] == 1
    assert result["row_status_counts"]["fallback"] == 1
    assert result["row_status_counts"]["degraded"] == 1
    assert result["row_status_counts"]["invalid"] == 0
    assert result["row_status_counts"]["missing"] == 0
    assert result["row_status_counts"]["failed"] == 0
    assert result["mean_deltas_completed_pairs"]["success_delta"] == pytest.approx(1.0)
    assert "success_delta" not in result.get("by_fallback", {})


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (None, "missing"),
        ({"scenario_exclusion": {"reason": "bound_exceeded"}}, "invalid"),
        ({"algorithm_metadata": {"mode": "planner fallback"}}, "fallback"),
        ({"algorithm_metadata": {"nested": ["degraded runtime"]}}, "degraded"),
        ({"termination_reason": "error"}, "failed"),
        ({"termination_reason": "success"}, "completed"),
    ],
)
def test_classify_episode_row_statuses(row: dict[str, object] | None, expected: str) -> None:
    """Episode row classification must preserve fail-closed diagnostic statuses."""
    assert _classify_episode_row_status(row) == expected


def test_build_criticality_summary_from_pilot_builds_valid_payload() -> None:
    """Building from pilot records must produce a validatable v1 summary."""
    manifest, manifest_id = _sample_manifest_metadata()
    summary = build_criticality_summary_from_pilot(
        records_by_planner=_sample_records(),
        scenario_metadata=_sample_scenario_metadata(),
        manifest=manifest,
        manifest_id=manifest_id,
        planners=["goal"],
        horizon=80,
        dt=0.1,
        seed_limit=1,
        materialization=_sample_materialization(),
        planner_runs=_sample_planner_runs(),
    )
    assert summary.schema_version == "criticality_summary.v1"
    assert summary.manifest == manifest
    assert summary.manifest_id == manifest_id
    assert summary.planners == ["goal"]
    assert summary.horizon == 80
    assert summary.dt == 0.1
    assert summary.seed_limit == 1
    row_status_counts = summary.pair_summary["row_status_counts"]
    assert isinstance(row_status_counts, dict)
    assert set(row_status_counts.keys()) == {
        "completed",
        "invalid",
        "fallback",
        "degraded",
        "missing",
        "failed",
    }
    assert summary.pair_summary["pairs"] == 2
    assert len(summary.pair_rows) == 2


def test_build_criticality_summary_records_missing_noop() -> None:
    """A perturbed row without a noop baseline must remain excluded and counted missing."""
    manifest, manifest_id = _sample_manifest_metadata()
    summary = build_criticality_summary_from_pilot(
        records_by_planner={"goal": []},
        scenario_metadata={
            "demo_offset": {
                "source_scenario_id": "demo",
                "variant_id": "demo_offset",
                "family": "robot_route_offset",
            },
        },
        manifest=manifest,
        manifest_id=manifest_id,
        planners=["goal"],
        horizon=80,
        dt=0.1,
        seed_limit=1,
        materialization=_sample_materialization(),
        planner_runs=_sample_planner_runs(),
        description="missing noop example",
    )
    payload = criticality_summary_to_dict(summary)
    assert payload["description"] == "missing noop example"
    assert summary.pair_rows == [
        {
            "planner": "goal",
            "source_scenario_id": "demo",
            "noop_variant_id": None,
            "perturbed_variant_id": "demo_offset",
            "perturbed_family": "robot_route_offset",
            "seed": None,
            "pair_status": "missing_noop",
            "noop_status": "missing",
            "perturbed_status": "missing",
        }
    ]
    assert summary.pair_summary["row_status_counts"]["missing"] == 1


def test_build_criticality_summary_includes_grouped_breakdowns() -> None:
    """The v1 summary must include by-planner, by-scenario, and by-family groupings."""
    manifest, manifest_id = _sample_manifest_metadata()
    summary = build_criticality_summary_from_pilot(
        records_by_planner=_sample_records(),
        scenario_metadata=_sample_scenario_metadata(),
        manifest=manifest,
        manifest_id=manifest_id,
        planners=["goal"],
        horizon=80,
        dt=0.1,
        seed_limit=1,
        materialization=_sample_materialization(),
        planner_runs=_sample_planner_runs(),
    )
    assert "by_planner" in summary.pair_summary
    assert "by_source_scenario" in summary.pair_summary
    assert "by_perturbation_family" in summary.pair_summary
    by_planner = summary.pair_summary["by_planner"]["goal"]
    assert "row_status_counts" in by_planner
    assert "mean_deltas_completed_pairs" in by_planner


def test_criticality_summary_to_dict_is_json_safe() -> None:
    """Serialization should produce JSON-safe primitives."""
    manifest, manifest_id = _sample_manifest_metadata()
    summary = build_criticality_summary_from_pilot(
        records_by_planner=_sample_records(),
        scenario_metadata=_sample_scenario_metadata(),
        manifest=manifest,
        manifest_id=manifest_id,
        planners=["goal"],
        horizon=80,
        dt=0.1,
        seed_limit=1,
        materialization=_sample_materialization(),
        planner_runs=_sample_planner_runs(),
    )
    payload = criticality_summary_to_dict(summary)
    assert isinstance(payload, dict)
    json.dumps(payload)


def test_criticality_summary_is_immutable() -> None:
    """CriticalitySummaryV1 must be frozen."""
    summary = CriticalitySummaryV1(
        schema_version=CRITICALITY_SUMMARY_SCHEMA_VERSION,
        manifest="test.yaml",
        manifest_id="test",
        planners=["goal"],
        horizon=80,
        dt=0.1,
        seed_limit=1,
        materialization={},
        planner_runs={},
        pair_summary={
            "pairs": 0,
            "row_status_counts": _zero_row_status_counts(),
            "mean_deltas_completed_pairs": {},
        },
    )
    with pytest.raises(FrozenInstanceError):
        summary.horizon = 100  # type: ignore[misc]


def test_validate_criticality_summary_accepts_minimal_payload() -> None:
    """Schema validation must accept a correctly structured minimal summary."""
    payload = {
        "schema_version": "criticality_summary.v1",
        "manifest": "configs/scenarios/perturbations/test.yaml",
        "manifest_id": "test",
        "planners": ["goal"],
        "horizon": 80,
        "dt": 0.1,
        "seed_limit": 1,
        "materialization": {
            "schema_version": "scenario_perturbation_pilot_matrix.v1",
            "manifest_id": "test",
            "included_variants": ["demo_noop"],
            "excluded_variants": [],
            "variant_count": 1,
            "local_artifact_boundary": "test boundary",
        },
        "planner_runs": {
            "goal": {
                "algo": "goal",
                "source": "raw_algo",
                "episodes": 1,
            },
        },
        "pair_summary": {
            "pairs": 0,
            "row_status_counts": {
                "completed": 0,
                "invalid": 0,
                "fallback": 0,
                "degraded": 0,
                "missing": 0,
                "failed": 0,
            },
            "mean_deltas_completed_pairs": {},
        },
        "claim_boundary": "diagnostic local pilot only; not benchmark-strength or paper-facing evidence",
    }
    validate_criticality_summary(payload)


def test_validate_criticality_summary_rejects_missing_row_statuses() -> None:
    """Schema must reject a payload missing required row_status_counts fields."""
    payload = {
        "schema_version": "criticality_summary.v1",
        "manifest": "test.yaml",
        "manifest_id": "test",
        "planners": ["goal"],
        "horizon": 80,
        "dt": 0.1,
        "seed_limit": 1,
        "materialization": {
            "schema_version": "scenario_perturbation_pilot_matrix.v1",
            "manifest_id": "test",
            "included_variants": [],
            "excluded_variants": [],
            "variant_count": 0,
            "local_artifact_boundary": "test boundary",
        },
        "planner_runs": {
            "goal": {"algo": "goal", "source": "raw_algo", "episodes": 0},
        },
        "pair_summary": {
            "pairs": 0,
            "row_status_counts": {
                "completed": 0,
            },
            "mean_deltas_completed_pairs": {},
        },
        "claim_boundary": "diagnostic",
    }
    with pytest.raises(ValueError, match="row_status_counts"):
        validate_criticality_summary(payload)


def test_validate_criticality_summary_rejects_wrong_schema_version() -> None:
    """Schema must reject wrong schema_version."""
    payload = {
        "schema_version": "criticality_summary.v2",
        "manifest": "test.yaml",
        "manifest_id": "test",
        "planners": ["goal"],
        "horizon": 80,
        "dt": 0.1,
        "seed_limit": 1,
        "materialization": {
            "schema_version": "x",
            "manifest_id": "test",
            "included_variants": [],
            "excluded_variants": [],
            "variant_count": 0,
            "local_artifact_boundary": "test",
        },
        "planner_runs": {
            "goal": {"algo": "goal", "source": "raw_algo", "episodes": 0},
        },
        "pair_summary": {
            "pairs": 0,
            "row_status_counts": _zero_row_status_counts(),
            "mean_deltas_completed_pairs": {},
        },
        "claim_boundary": "diagnostic",
    }
    with pytest.raises(ValueError, match="schema_version"):
        validate_criticality_summary(payload)


def test_build_from_compact_evidence_represents_existing_summary(tmp_path: Path) -> None:
    """An existing #1610 compact evidence summary must be representable."""
    evidence_path = Path(
        "docs/context/evidence/issue_1937_ped_route_offset_2026-05-31/summary.json"
    )
    if not evidence_path.exists():
        pytest.skip("Existing compact evidence not available")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    summary = build_criticality_summary_from_compact_evidence(evidence)
    payload = criticality_summary_to_dict(summary)
    validate_criticality_summary(payload)
    assert payload["schema_version"] == "criticality_summary.v1"
    row_status_counts = payload["pair_summary"]["row_status_counts"]
    assert isinstance(row_status_counts, dict)
    assert set(row_status_counts.keys()) == {
        "completed",
        "invalid",
        "fallback",
        "degraded",
        "missing",
        "failed",
    }
    assert "mean_deltas_completed_pairs" in payload["pair_summary"]


def test_build_from_compact_evidence_includes_grouped_row_statuses(
    tmp_path: Path,
) -> None:
    """Grouped summaries must carry explicit row_status_counts."""
    evidence_path = Path(
        "docs/context/evidence/issue_1937_ped_route_offset_2026-05-31/summary.json"
    )
    if not evidence_path.exists():
        pytest.skip("Existing compact evidence not available")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    summary = build_criticality_summary_from_compact_evidence(evidence)
    payload = criticality_summary_to_dict(summary)
    by_planner = payload["pair_summary"].get("by_planner", {})
    for group in by_planner.values():
        assert "row_status_counts" in group
        assert "mean_deltas_completed_pairs" in group


def test_build_from_compact_evidence_does_not_require_raw_output_paths(
    tmp_path: Path,
) -> None:
    """Compact evidence must not require output/ paths to be representable."""
    evidence_path = Path(
        "docs/context/evidence/issue_1937_ped_route_offset_2026-05-31/summary.json"
    )
    if not evidence_path.exists():
        pytest.skip("Existing compact evidence not available")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    summary = build_criticality_summary_from_compact_evidence(evidence)
    payload = criticality_summary_to_dict(summary)
    local_artifact_boundary = payload["materialization"].get("local_artifact_boundary")
    assert local_artifact_boundary is not None
    assert "output/" not in str(payload.get("materialization", {}).get("scenario_matrix_path", ""))


def test_build_from_compact_evidence_sets_claim_boundary() -> None:
    """Compact evidence must carry the diagnostic-vs-benchmark boundary."""
    evidence_path = Path(
        "docs/context/evidence/issue_1937_ped_route_offset_2026-05-31/summary.json"
    )
    if not evidence_path.exists():
        pytest.skip("Existing compact evidence not available")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    summary = build_criticality_summary_from_compact_evidence(evidence)
    payload = criticality_summary_to_dict(summary)
    assert "claim_boundary" in payload
    assert "diagnostic" in payload["claim_boundary"].lower()
    assert "not benchmark" in payload["claim_boundary"].lower()


def test_pair_summary_excludes_fallback_from_means() -> None:
    """Non-completed rows must not influence completed-pair means."""
    pair_rows = [
        {
            "planner": "goal",
            "source_scenario_id": "demo",
            "noop_variant_id": "demo_noop",
            "perturbed_variant_id": "demo_fallback",
            "perturbed_family": "robot_route_offset",
            "seed": 111,
            "pair_status": "excluded",
            "noop_status": "completed",
            "perturbed_status": "fallback",
            "success_delta": 0.5,
        },
        {
            "planner": "goal",
            "source_scenario_id": "demo",
            "noop_variant_id": "demo_noop",
            "perturbed_variant_id": "demo_offset",
            "perturbed_family": "robot_route_offset",
            "seed": 111,
            "pair_status": "completed",
            "noop_status": "completed",
            "perturbed_status": "completed",
            "success_delta": 1.0,
        },
    ]
    result = _aggregate_subset(pair_rows)
    assert result["row_status_counts"]["completed"] == 1
    assert result["row_status_counts"]["fallback"] == 1
    assert result["mean_deltas_completed_pairs"]["success_delta"] == pytest.approx(1.0)


def test_build_pair_summary_derives_row_statuses_when_not_supplied() -> None:
    """The public pair summary helper should derive row status counts when omitted."""
    result = build_pair_summary_with_statuses(
        [
            {
                "pair_status": "excluded",
                "perturbed_status": "failed",
                "success_delta": 1.0,
            },
            {
                "pair_status": "completed",
                "perturbed_status": "completed",
                "success_delta": 0.25,
            },
        ]
    )
    assert result["row_status_counts"]["failed"] == 1
    assert result["row_status_counts"]["completed"] == 1
    assert result["mean_deltas_completed_pairs"]["success_delta"] == pytest.approx(0.25)
