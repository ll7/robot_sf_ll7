"""Focused contract tests for the issue #8849 trace admission gate."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

from scripts.validation.check_risk_calibration_preflight import _inspect_record, build_report

TARGET = {
    "target_distribution_id": "target-v1",
    "collision_predicate_id": "predicate-v1",
    "required_execution_mode": "native",
    "horizon_steps": 20,
    "dt_s": 0.1,
}


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "fixture.v1",
        "scenario_id": "s1",
        "planner_id": "planner1",
        "seed": 7,
        "episode_id": "episode-7",
        "git_commit": "a" * 40,
        "horizon": 20,
        "dt_s": 0.1,
        "execution_mode": "native",
        "fallback_count": 0,
        "degraded": False,
        "history": [{"state": [0, 0]}],
        "frames": [{"planner": {"selected_action": {"v": 1.0, "omega": 0.0}}}],
        "candidate_action_trajectories": [{"action": [1.0, 0.0], "trajectory": [[0, 0]]}],
        "forecast_inputs": {"pedestrian_state": [[1, 1]]},
        "outcome": {"collision_event": False, "action_conditioned": True},
        "collision_predicate_id": "predicate-v1",
        "prediction_model": "cv-v1",
        "target_distribution_id": "target-v1",
        "sampling_provenance": {"kind": "iid", "source": "target-v1"},
        "proposal_weight": 1.0,
        "leakage_check": True,
        "footprints": {"robot_radius_m": 0.3, "pedestrian_radius_m": 0.3},
    }
    record.update(overrides)
    return record


def _fixture_tree(tmp_path: Path) -> Path:
    (tmp_path / "matrix.yaml").write_text(
        "scenarios:\n- name: s1\n  metadata: {archetype: crossing, density: low}\n",
        encoding="utf-8",
    )
    (tmp_path / "records.json").write_text(json.dumps(_record()), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        """schema_version: risk_calibration_preflight_config.v1
target:
  target_distribution_id: target-v1
  collision_predicate_id: predicate-v1
  required_execution_mode: native
  scenario_matrix: matrix.yaml
  horizon_steps: 20
  dt_s: 0.1
  minimum_eligible_samples: 1
packet:
  estimators: [{id: baseline}]
  calibration_split: fixed
  weighting: unit
  primary_metrics: [ece]
  runtime_cap_ms: 100
  compute_estimate: bounded
  stop_rule: stop_on_missing_contract
sources:
  - {id: fixture, kind: trace_series, glob: records.json, expected_schema: fixture.v1}
""",
        encoding="utf-8",
    )
    return tmp_path / "config.yaml"


def test_retained_packet_is_insufficient_and_deterministically_inventoried() -> None:
    report = build_report()
    assert report["status"] == "ok"
    assert report["decision"] == "insufficient_eligible_traces"
    assert report["record_count"] == 25
    assert report["eligible_count"] == 0
    assert {row["disposition"] for row in report["source_dispositions"]} == {"stale", "ineligible"}
    assert len(report["staging_bundle"]["identity_digest"]) == 64
    assert report["reason_counts"]["missing_action_conditioned_label"] == 25


def test_complete_native_record_is_admitted(tmp_path: Path) -> None:
    report = build_report(tmp_path, _fixture_tree(tmp_path))
    assert report["decision"] == "admitted_for_private_execution"
    assert report["eligible_count"] == 1
    assert report["source_dispositions"][0]["disposition"] == "eligible"
    assert report["rows"][0]["reasons"] == []


def test_adapter_fallback_and_leakage_are_explicit_failures() -> None:
    record = _record(execution_mode="adapter", fallback_count=2, leakage_check=False)
    result = _inspect_record(
        record,
        {"expected_schema": "fixture.v1"},
        {"s1": {"scenario_family": "crossing", "density": "low"}},
        TARGET,
    )
    assert result["status"] == "ineligible"
    assert {
        "non_native_execution",
        "fallback_or_degraded_execution",
        "future_leakage_detected",
    } <= set(result["reasons"])
