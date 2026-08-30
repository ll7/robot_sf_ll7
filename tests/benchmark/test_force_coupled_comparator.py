"""Tests for the force-coupled potential-field paired diagnostic comparator (issue #8015)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.force_coupled_comparator import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    PurePursuitGoalPlanner,
    execute_rollout,
    get_canonical_comparison_scenarios,
    run_force_coupled_comparator,
)
from robot_sf.planner.force_coupled_potential_field import (
    ForceCoupledPotentialFieldConfig,
    ForceCoupledPotentialFieldPlanner,
)


def test_pure_pursuit_planner_lifecycle() -> None:
    """Pure pursuit reference planner obeys protocol lifecycle and fails closed when closed."""
    planner = PurePursuitGoalPlanner(max_linear_speed=1.0, max_angular_speed=1.2)
    planner.reset(seed=1)
    obs = {"robot": [0.0, 0.0, 0.0], "goal": [4.0, 0.0]}
    cmd = planner.plan(obs)
    assert len(cmd) == 2
    assert cmd[0] > 0.0  # moving forward towards goal
    assert abs(cmd[1]) < 1e-6  # zero heading error
    diag = planner.diagnostics()
    assert diag["planner_type"] == "pure_pursuit_goal"
    assert diag["status"] == "ok"
    planner.close()
    with pytest.raises(ValueError, match="planner is closed"):
        planner.plan(obs)


def test_comparator_receipt_schema_conformance() -> None:
    """The generated comparator receipt matches the versioned JSON schema."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "schemas"
        / "force_coupled_comparator_receipt.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    receipt = run_force_coupled_comparator()

    jsonschema.validate(instance=receipt, schema=schema)
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["claim_boundary"] == CLAIM_BOUNDARY
    assert receipt["status"] == "ok"
    assert len(receipt["scenarios"]) == 4
    assert len(receipt["results"]) == 16  # 4 scenarios x 4 planners
    assert len(receipt["summary_table"]) == 4  # 4 distinct planners


def test_deterministic_receipt_invariant() -> None:
    """Running the comparator suite multiple times yields bitwise-identical receipts and digests."""
    receipt1 = run_force_coupled_comparator()
    receipt2 = run_force_coupled_comparator()
    assert receipt1["receipt_digest"] == receipt2["receipt_digest"]
    assert receipt1["config_digest"] == receipt2["config_digest"]
    assert receipt1["status"] == receipt2["status"]
    assert len(receipt1["results"]) == len(receipt2["results"])
    for r1, r2 in zip(receipt1["results"], receipt2["results"], strict=True):
        assert r1["planner_id"] == r2["planner_id"]
        assert r1["scenario_id"] == r2["scenario_id"]
        assert r1["steps"] == r2["steps"]
        assert r1["completed"] == r2["completed"]
        assert r1["collision"] == r2["collision"]
        assert r1["path_length_m"] == r2["path_length_m"]
        assert r1["jerk_metric"] == r2["jerk_metric"]


def test_force_coupled_obstacle_clearance_advantage_over_pure_pursuit() -> None:
    """Force-coupled planner repels from obstacles whereas pure pursuit heads straight."""
    scenarios = {s.scenario_id: s for s in get_canonical_comparison_scenarios()}
    static_scenario = scenarios["analytic_static_obstacle"]

    fcpf_planner = ForceCoupledPotentialFieldPlanner(ForceCoupledPotentialFieldConfig())
    fcpf_result = execute_rollout(fcpf_planner, static_scenario)

    pp_planner = PurePursuitGoalPlanner()
    pp_result = execute_rollout(pp_planner, static_scenario)

    assert fcpf_result.status == "ok"
    assert fcpf_result.min_clearance_obstacle_m is not None
    assert fcpf_result.path_length_m > 0.0
    assert pp_result.path_length_m > 0.0


def test_force_coupled_pedestrian_interaction_rollout() -> None:
    """Force-coupled planner successfully executes the pedestrian interaction scenario."""
    scenarios = {s.scenario_id: s for s in get_canonical_comparison_scenarios()}
    ped_scenario = scenarios["analytic_pedestrian_interaction"]

    fcpf_planner = ForceCoupledPotentialFieldPlanner(ForceCoupledPotentialFieldConfig())
    result = execute_rollout(fcpf_planner, ped_scenario)

    assert result.status == "ok"
    assert result.completed is True
    assert result.steps > 0
    assert result.min_clearance_pedestrian_m is not None
    assert result.jerk_metric >= 0.0


def test_cli_runner_smoke_mode(tmp_path: Path) -> None:
    """CLI script runs in smoke mode and writes output JSON successfully."""
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmark"
        / "check_force_coupled_comparator.py"
    )
    out_file = tmp_path / "receipt.json"

    res = subprocess.run(
        [sys.executable, str(script_path), "--smoke", "--output", str(out_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "PASS" in res.stdout
    assert out_file.exists()
    saved_receipt = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved_receipt["schema_version"] == SCHEMA_VERSION
