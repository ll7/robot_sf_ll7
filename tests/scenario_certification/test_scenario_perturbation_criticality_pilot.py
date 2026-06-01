"""Tests for scenario perturbation criticality pilot v1 evidence summary.

Row-status classification and pair-table construction now route through
``robot_sf.scenario_certification.criticality_summary`` v1 helpers.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from robot_sf.scenario_certification.criticality_summary import (
    _classify_episode_row_status,
)
from scripts.validation import run_scenario_perturbation_criticality_pilot as pilot


def test_resolve_planner_run_spec_uses_policy_search_candidate_registry(tmp_path) -> None:
    """Registry-backed candidate keys should preserve labels while running their base algo."""
    candidate_config = tmp_path / "candidate.yaml"
    candidate_config.write_text("algo: hybrid_rule_local_planner\n", encoding="utf-8")
    registry = tmp_path / "candidate_registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "candidates:",
                "  scenario_adaptive_demo:",
                "    candidate_config_path: candidate.yaml",
                "",
            ]
        ),
        encoding="utf-8",
    )

    spec = pilot.resolve_planner_run_spec(
        "scenario_adaptive_demo", candidate_registry_path=registry
    )

    assert spec.label == "scenario_adaptive_demo"
    assert spec.algo == "hybrid_rule_local_planner"
    assert spec.algo_config_path == candidate_config
    assert spec.source == "policy_search_candidate"


def test_resolve_planner_run_spec_keeps_unknown_planner_as_raw_algo(tmp_path) -> None:
    """Plain planner algos should remain usable without registry entries."""
    registry = tmp_path / "candidate_registry.yaml"
    registry.write_text("candidates: {}\n", encoding="utf-8")

    spec = pilot.resolve_planner_run_spec("goal", candidate_registry_path=registry)

    assert spec.label == "goal"
    assert spec.algo == "goal"
    assert spec.algo_config_path is None
    assert spec.source == "raw_algo"


def test_classify_row_status_uses_v1_helper() -> None:
    """Row-status classification must come from criticality_summary v1."""
    assert _classify_episode_row_status(None) == "missing"
    assert _classify_episode_row_status({"scenario_exclusion": {"status": "invalid"}}) == "invalid"
    assert (
        _classify_episode_row_status(
            {"algorithm_metadata": {"planner": {"execution_mode": "fallback"}}}
        )
        == "fallback"
    )
    assert (
        _classify_episode_row_status(
            {"algorithm_metadata": {"planner": {"execution_mode": "degraded"}}}
        )
        == "degraded"
    )
    assert _classify_episode_row_status({"termination_reason": "error"}) == "failed"
    assert _classify_episode_row_status({"termination_reason": "success"}) == "completed"


def test_cli_builds_valid_v1_summary_with_row_status_counts() -> None:
    """The pilot CLI builder must produce a schema-valid criticality_summary.v1 payload."""
    metadata = {
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
    records = {
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
        ]
    }

    payload, pair_rows = pilot.build_validated_criticality_summary_payload(
        records_by_planner=records,
        scenario_metadata=metadata,
        manifest="configs/scenarios/perturbations/test.yaml",
        manifest_id="test",
        planners=["goal"],
        horizon=80,
        dt=0.1,
        seed_limit=1,
        materialization={
            "schema_version": "scenario_perturbation_pilot_matrix.v1",
            "manifest_id": "test",
            "included_variants": ["demo_noop", "demo_offset"],
            "excluded_variants": [],
            "variant_count": 2,
            "local_artifact_boundary": "test boundary",
        },
        planner_runs={
            "goal": {
                "algo": "goal",
                "algo_config_path": None,
                "source": "raw_algo",
                "episodes": 2,
            },
        },
    )

    assert payload["schema_version"] == "criticality_summary.v1"
    assert payload["manifest_id"] == "test"
    assert payload["planners"] == ["goal"]

    row_status_counts = payload["pair_summary"]["row_status_counts"]
    assert set(row_status_counts.keys()) == {
        "completed",
        "invalid",
        "fallback",
        "degraded",
        "missing",
        "failed",
    }
    assert row_status_counts["completed"] == 1
    for status in ("invalid", "fallback", "degraded", "missing", "failed"):
        assert row_status_counts[status] == 0

    mean_deltas = payload["pair_summary"]["mean_deltas_completed_pairs"]
    assert mean_deltas["success_delta"] == pytest.approx(-1.0)
    assert mean_deltas["collision_delta"] == pytest.approx(0.0)
    assert mean_deltas["timeout_delta"] == pytest.approx(1.0)
    assert mean_deltas["min_distance_delta"] == pytest.approx(-0.5)

    assert pair_rows == payload["pair_rows"]
    assert len(pair_rows) == 1
    assert pair_rows[0]["pair_status"] == "completed"
    assert pair_rows[0]["perturbed_status"] == "completed"

    assert "by_planner" in payload["pair_summary"]
    assert "by_source_scenario" in payload["pair_summary"]
    assert "by_perturbation_family" in payload["pair_summary"]

    grouped = payload["pair_summary"]["by_perturbation_family"]["robot_route_offset"]
    assert grouped["row_status_counts"]["completed"] == 1
    assert grouped["mean_deltas_completed_pairs"]["success_delta"] == pytest.approx(-1.0)


def test_cli_v1_summary_excludes_fallback_rows_from_means() -> None:
    """Fallback perturbed rows must be counted under ``row_status_counts``
    and excluded from ``mean_deltas_completed_pairs``."""
    metadata = {
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
    records = {
        "orca": [
            {"scenario_id": "demo_noop", "seed": 111, "termination_reason": "success"},
            {
                "scenario_id": "demo_offset",
                "seed": 111,
                "termination_reason": "success",
                "algorithm_metadata": {"mode": "fallback"},
            },
        ]
    }

    payload, pair_rows = pilot.build_validated_criticality_summary_payload(
        records_by_planner=records,
        scenario_metadata=metadata,
        manifest="configs/scenarios/perturbations/test.yaml",
        manifest_id="test_fallback",
        planners=["orca"],
        horizon=80,
        dt=0.1,
        seed_limit=1,
        materialization={
            "schema_version": "scenario_perturbation_pilot_matrix.v1",
            "manifest_id": "test_fallback",
            "included_variants": ["demo_noop", "demo_offset"],
            "excluded_variants": [],
            "variant_count": 2,
            "local_artifact_boundary": "test boundary",
        },
        planner_runs={
            "orca": {
                "algo": "orca",
                "algo_config_path": None,
                "source": "raw_algo",
                "episodes": 2,
            },
        },
    )

    assert pair_rows == payload["pair_rows"]
    assert pair_rows[0]["pair_status"] == "excluded"
    assert pair_rows[0]["perturbed_status"] == "fallback"

    row_status_counts = payload["pair_summary"]["row_status_counts"]
    assert row_status_counts["completed"] == 0
    assert row_status_counts["fallback"] == 1

    assert payload["pair_summary"]["mean_deltas_completed_pairs"] == {}

    assert json.dumps(payload["claim_boundary"]).lower().count("diagnostic") >= 1
    assert json.dumps(payload["claim_boundary"]).lower().count("not benchmark") >= 1


def test_cli_v1_counts_perturbed_rows_not_completed_pairs() -> None:
    """Completed row counts are per perturbed row even when the noop side fails."""
    metadata = {
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
    records = {
        "goal": [
            {"scenario_id": "demo_noop", "seed": 111, "termination_reason": "error"},
            {"scenario_id": "demo_offset", "seed": 111, "termination_reason": "success"},
        ]
    }

    payload, pair_rows = pilot.build_validated_criticality_summary_payload(
        records_by_planner=records,
        scenario_metadata=metadata,
        manifest="configs/scenarios/perturbations/test.yaml",
        manifest_id="test_noop_failed",
        planners=["goal"],
        horizon=80,
        dt=0.1,
        seed_limit=1,
        materialization={
            "schema_version": "scenario_perturbation_pilot_matrix.v1",
            "manifest_id": "test_noop_failed",
            "included_variants": ["demo_noop", "demo_offset"],
            "excluded_variants": [],
            "variant_count": 2,
            "local_artifact_boundary": "test boundary",
        },
        planner_runs={
            "goal": {
                "algo": "goal",
                "algo_config_path": None,
                "source": "raw_algo",
                "episodes": 2,
            },
        },
    )

    assert pair_rows[0]["pair_status"] == "excluded"
    assert pair_rows[0]["noop_status"] == "failed"
    assert pair_rows[0]["perturbed_status"] == "completed"
    assert payload["pair_summary"]["row_status_counts"]["completed"] == 1
    assert payload["pair_summary"]["mean_deltas_completed_pairs"] == {}


def test_main_writes_v1_evidence_summary_and_legacy_local_summary(
    monkeypatch,
    tmp_path,
) -> None:
    """--evidence-summary should write validated v1 JSON while local summary stays legacy."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("manifest_id: test\n", encoding="utf-8")
    materialized_dir = tmp_path / "materialized"
    pilot_dir = tmp_path / "pilot"
    evidence_summary = tmp_path / "evidence" / "summary.json"
    matrix_path = materialized_dir / "matrix.yaml"
    materialized = SimpleNamespace(
        schema_version="scenario_perturbation_pilot_matrix.v1",
        manifest_id="test",
        scenario_matrix_path=matrix_path.as_posix(),
        summary_path=(materialized_dir / "summary.json").as_posix(),
        included_variants=["demo_noop", "demo_offset"],
        excluded_variants=[],
    )
    scenarios = [
        {
            "scenario_id": "demo_noop",
            "metadata": {
                "scenario_perturbation": {
                    "source_scenario_id": "demo",
                    "variant_id": "demo_noop",
                    "family": "noop",
                }
            },
        },
        {
            "scenario_id": "demo_offset",
            "metadata": {
                "scenario_perturbation": {
                    "source_scenario_id": "demo",
                    "variant_id": "demo_offset",
                    "family": "robot_route_offset",
                }
            },
        },
    ]

    def fake_run_map_batch(_matrix_path, jsonl_path, **_kwargs):
        rows = [
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
        ]
        jsonl_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return {"episodes": len(rows)}

    monkeypatch.setattr(
        pilot,
        "materialize_perturbation_pilot_matrix",
        lambda *_args, **_kwargs: materialized,
    )
    monkeypatch.setattr(pilot, "load_scenarios", lambda _path: scenarios)
    monkeypatch.setattr(pilot, "run_map_batch", fake_run_map_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scenario_perturbation_criticality_pilot.py",
            manifest.as_posix(),
            "--materialized-output-dir",
            materialized_dir.as_posix(),
            "--pilot-output-dir",
            pilot_dir.as_posix(),
            "--planner",
            "goal",
            "--evidence-summary",
            evidence_summary.as_posix(),
        ],
    )

    assert pilot.main() == 0

    evidence_payload = json.loads(evidence_summary.read_text(encoding="utf-8"))
    assert evidence_payload["schema_version"] == "criticality_summary.v1"
    assert evidence_payload["pair_summary"]["row_status_counts"]["completed"] == 1
    assert "jsonl_path" not in evidence_payload["planner_runs"]["goal"]

    local_payload = json.loads((pilot_dir / "summary.json").read_text(encoding="utf-8"))
    assert local_payload["schema_version"] == pilot.SCHEMA_VERSION
    assert local_payload["pair_summary"]["status_counts"] == {"completed": 1}
    assert local_payload["planner_runs"]["goal"]["jsonl_path"].endswith("goal.episodes.jsonl")
