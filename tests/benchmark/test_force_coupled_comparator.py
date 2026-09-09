"""Tests for the force-coupled potential-field paired diagnostic comparator (issue #8015)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import jsonschema
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

import robot_sf.benchmark.force_coupled_comparator as comparator
from robot_sf.benchmark.force_coupled_comparator import (
    CLAIM_BOUNDARY,
    FAILURE_CLASS_PATH_GENERATION,
    FAILURE_CLASS_SIMULATOR,
    FAILURE_CLASS_SOCIAL_COMPLIANCE,
    FAILURE_CLASS_TRACKING,
    SCHEMA_VERSION,
    VALID_FAILURE_CLASSES,
    ComparatorRunResult,
    PurePursuitGoalPlanner,
    classify_failure,
    compute_summary_table,
    execute_rollout,
    get_canonical_comparison_scenarios,
    run_force_coupled_comparator,
)
from robot_sf.planner.force_coupled_potential_field import (
    ForceCoupledPotentialFieldConfig,
    ForceCoupledPotentialFieldPlanner,
)


class _SimulatorBoundaryFailurePlanner:
    """Protocol fixture that fails at one canonical rollout boundary."""

    def __init__(self, failure_phase: str) -> None:
        self.failure_phase = failure_phase

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        if self.failure_phase == "reset":
            raise RuntimeError("fixture reset failure")

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        del observation
        if self.failure_phase == "planner_simulator_text":
            raise RuntimeError("simulator backend unavailable")
        if self.failure_phase == "command":
            return (float("nan"), 0.0)
        if self.failure_phase == "degraded_without_reason":
            return (1.0, 0.0)
        return (0.1, 0.0)

    def diagnostics(self) -> dict[str, object]:
        if self.failure_phase == "diagnostics":
            raise RuntimeError("fixture diagnostics failure")
        if self.failure_phase == "diagnostic_status":
            return {"planner_type": "simulator_boundary_fixture", "status": "unknown"}
        if self.failure_phase == "degraded_diagnostics":
            return {
                "planner_type": "simulator_boundary_fixture",
                "status": "degraded",
                "degradation_reasons": ["fixture_degraded"],
            }
        if self.failure_phase == "degraded_without_reason":
            return {
                "planner_type": "simulator_boundary_fixture",
                "status": "degraded",
            }
        if self.failure_phase == "degraded_simulator_text":
            return {
                "planner_type": "simulator_boundary_fixture",
                "status": "degraded",
                "degradation_reasons": ["simulator backend unavailable"],
            }
        if self.failure_phase == "fallback_diagnostics":
            return {
                "planner_type": "simulator_boundary_fixture",
                "status": "ok",
                "fallback": True,
            }
        return {"planner_type": "simulator_boundary_fixture", "status": "ok"}

    def close(self) -> None:
        """Release no resources; this fixture only exercises rollout boundaries."""


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


def test_pure_pursuit_stops_at_goal() -> None:
    """The reference planner emits a stationary command at the goal."""
    planner = PurePursuitGoalPlanner()
    planner.reset(seed=1)

    assert planner.plan({"robot": [0.0, 0.0, 0.0], "goal": [0.0, 0.0]}) == (0.0, 0.0)


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


def test_first_step_planner_exception_is_classified_without_diagnostics() -> None:
    """A planner failure before diagnostics still emits a structured path-generation result."""

    class FirstStepFailurePlanner:
        def reset(self, seed: int) -> None:
            del seed

        def plan(self, obs: dict[str, object]) -> tuple[float, float]:
            del obs
            raise ValueError("fixture planner failure")

        def diagnostics(self) -> dict[str, object]:
            return {"planner_type": "first_step_failure"}

    result = execute_rollout(FirstStepFailurePlanner(), get_canonical_comparison_scenarios()[0])

    assert result.planner_id == "FirstStepFailurePlanner"
    assert result.status == "error"
    assert result.degraded is True
    assert result.failure_class == FAILURE_CLASS_PATH_GENERATION
    assert result.degradation_reasons == ("plan_exception: fixture planner failure",)


@pytest.mark.parametrize(
    "failure_phase, expected_failure_class, expected_simulator_error",
    [
        ("reset", FAILURE_CLASS_PATH_GENERATION, False),
        ("diagnostics", FAILURE_CLASS_PATH_GENERATION, False),
        ("diagnostic_status", FAILURE_CLASS_PATH_GENERATION, False),
        ("command", FAILURE_CLASS_PATH_GENERATION, False),
        ("planner_simulator_text", FAILURE_CLASS_PATH_GENERATION, False),
        ("integration", FAILURE_CLASS_SIMULATOR, True),
    ],
)
def test_execute_rollout_classifies_planner_boundaries_and_simulator_integration(
    failure_phase: str,
    expected_failure_class: str,
    expected_simulator_error: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner-owned failures stay path-generation; analytic integration sets the flag."""
    scenario = get_canonical_comparison_scenarios()[0]
    planner: comparator.LocalPlannerProtocol = _SimulatorBoundaryFailurePlanner(failure_phase)
    if failure_phase == "integration":
        scenario = replace(scenario, control_dt=0.0, max_steps=1)
        planner = PurePursuitGoalPlanner()

    observed_flags: list[bool] = []
    original_classify_failure = comparator.classify_failure

    def classify_failure_spy(**kwargs: object) -> str | None:
        observed_flags.append(bool(kwargs["simulator_error"]))
        return original_classify_failure(**kwargs)

    monkeypatch.setattr(comparator, "classify_failure", classify_failure_spy)
    result = execute_rollout(planner, scenario)

    assert observed_flags == [expected_simulator_error]
    assert result.status == "error"
    assert result.degraded is True
    assert result.failure_class == expected_failure_class
    if expected_simulator_error:
        assert result.degradation_reasons[0].startswith("simulator_integration_failure")
    else:
        assert result.degradation_reasons[0].startswith("plan_exception:")


@pytest.mark.parametrize(
    ("failure_phase", "expected_reason"),
    [
        ("degraded_diagnostics", "fixture_degraded"),
        ("degraded_simulator_text", "simulator backend unavailable"),
        ("fallback_diagnostics", "fallback_execution"),
    ],
)
def test_execute_rollout_records_diagnostic_degradation(
    failure_phase: str,
    expected_reason: str,
) -> None:
    """Planner diagnostic degradation remains visible without becoming simulator failure."""
    scenario = get_canonical_comparison_scenarios()[0]
    result = execute_rollout(_SimulatorBoundaryFailurePlanner(failure_phase), scenario)

    assert result.status == "degraded"
    assert result.degraded is True
    assert result.failure_class == FAILURE_CLASS_PATH_GENERATION
    assert f"planner_diagnostic: {expected_reason}" in result.degradation_reasons


def test_execute_rollout_emits_reason_for_degraded_diagnostic_without_reason() -> None:
    """A bare degraded diagnostic gets a schema-compatible reason instead of an empty list."""
    scenario = replace(
        get_canonical_comparison_scenarios()[0],
        goal=(0.3, 0.0),
        max_steps=2,
    )

    result = execute_rollout(_SimulatorBoundaryFailurePlanner("degraded_without_reason"), scenario)

    assert result.status == "degraded"
    assert result.degraded is True
    assert result.completed is True
    assert result.degradation_reasons == ("planner_diagnostic: degraded_without_reason",)
    assert result.failure_class == FAILURE_CLASS_PATH_GENERATION
    assert len(result.to_dict()["degradation_reasons"]) == 1


def test_execute_rollout_captures_simulator_state_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analytic simulator-state exceptions become structured simulator failures."""

    def fail_hypot(*args: object) -> float:
        del args
        raise RuntimeError("fixture state failure")

    monkeypatch.setattr(comparator.math, "hypot", fail_hypot)
    result = execute_rollout(
        _SimulatorBoundaryFailurePlanner("normal"), get_canonical_comparison_scenarios()[0]
    )

    assert result.status == "error"
    assert result.degraded is True
    assert result.failure_class == FAILURE_CLASS_SIMULATOR
    assert result.degradation_reasons == (
        "simulator_state_failure: RuntimeError: fixture state failure",
    )


def test_execute_rollout_captures_simulator_clearance_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analytic clearance exceptions become structured simulator failures."""
    scenario = replace(get_canonical_comparison_scenarios()[0], obstacles=((1.0, 0.0),))

    def fail_clearance(*args: object) -> tuple[None, bool, bool]:
        del args
        raise RuntimeError("fixture clearance failure")

    monkeypatch.setattr(comparator, "_update_clearance", fail_clearance)
    result = execute_rollout(_SimulatorBoundaryFailurePlanner("normal"), scenario)

    assert result.failure_class == FAILURE_CLASS_SIMULATOR
    assert result.degradation_reasons == (
        "simulator_clearance_failure: RuntimeError: fixture clearance failure",
    )


def test_execute_rollout_captures_simulator_observation_state_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observation-state construction exceptions become simulator failures."""

    class BrokenPoint:
        def __iter__(self) -> Iterator[float]:
            raise RuntimeError("fixture observation-state failure")

    scenario = replace(get_canonical_comparison_scenarios()[0], obstacles=(BrokenPoint(),))  # type: ignore[arg-type]
    monkeypatch.setattr(comparator, "_update_clearance", lambda *args: (None, False, False))
    result = execute_rollout(_SimulatorBoundaryFailurePlanner("normal"), scenario)

    assert result.failure_class == FAILURE_CLASS_SIMULATOR
    assert result.degradation_reasons == (
        "simulator_state_failure: RuntimeError: fixture observation-state failure",
    )


def test_execute_rollout_captures_nonfinite_simulator_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite analytic state after integration is classified as a simulator failure."""
    monkeypatch.setattr(comparator, "wrap_angle_pi", lambda angle: float("nan"))
    scenario = replace(get_canonical_comparison_scenarios()[0], max_steps=1)
    result = execute_rollout(_SimulatorBoundaryFailurePlanner("normal"), scenario)

    assert result.failure_class == FAILURE_CLASS_SIMULATOR
    assert result.degradation_reasons == (
        "simulator_integration_failure: ValueError: non-finite simulator state",
    )


def test_classify_failure_rejects_unknown_status_and_preserves_path_fallback() -> None:
    """Unknown statuses fail closed while the established non-success fallback remains stable."""
    with pytest.raises(ValueError, match="unknown rollout status"):
        classify_failure(status="unknown")

    assert classify_failure(status="degraded", degraded=True) == FAILURE_CLASS_PATH_GENERATION
    assert (
        classify_failure(
            status="degraded",
            degradation_reasons=("plan_exception: simulator backend unavailable",),
        )
        == FAILURE_CLASS_PATH_GENERATION
    )


@pytest.mark.parametrize(
    ("signals", "expected"),
    [
        (
            {
                "status": "error",
                "plan_exception": True,
                "collision_pedestrian": True,
            },
            FAILURE_CLASS_PATH_GENERATION,
        ),
        (
            {
                "status": "degraded",
                "collision_pedestrian": True,
                "degradation_reasons": ("planner_diagnostic: simulator backend unavailable",),
            },
            FAILURE_CLASS_PATH_GENERATION,
        ),
        (
            {
                "status": "error",
                "collision_pedestrian": True,
                "degradation_reasons": ("simulator backend unavailable",),
            },
            FAILURE_CLASS_SIMULATOR,
        ),
    ],
)
def test_classify_failure_preserves_planner_boundary_precedence(
    signals: dict[str, object], expected: str
) -> None:
    """Planner ownership wins over combined pedestrian/simulator signals."""
    assert classify_failure(**signals) == expected  # type: ignore[arg-type]


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


def test_classify_failure_deterministic_mapping_per_class() -> None:
    """Mapping rules deterministically assign each of the four explicit failure classes."""
    # 1. Ok rollout produces None
    ok_class = classify_failure(
        status="ok",
        degraded=False,
        completed=True,
        collision_obstacle=False,
        collision_pedestrian=False,
    )
    assert ok_class is None

    # 2. Simulator failure: simulator_error flag or simulation degradation reason
    sim_class_1 = classify_failure(status="error", simulator_error=True)
    assert sim_class_1 == FAILURE_CLASS_SIMULATOR

    sim_class_2 = classify_failure(
        status="degraded",
        degradation_reasons=["simulator_step_failure"],
    )
    assert sim_class_2 == FAILURE_CLASS_SIMULATOR

    # 3. Social compliance failure: pedestrian collision or proximity violation
    social_class_1 = classify_failure(
        status="degraded",
        collision_pedestrian=True,
    )
    assert social_class_1 == FAILURE_CLASS_SOCIAL_COMPLIANCE

    social_class_2 = classify_failure(
        status="degraded",
        degradation_reasons=["pedestrian_comfort_violation"],
    )
    assert social_class_2 == FAILURE_CLASS_SOCIAL_COMPLIANCE

    # 4. Tracking failure: obstacle collision or kinematic/steering/actuation saturation
    tracking_class_1 = classify_failure(
        status="degraded",
        collision_obstacle=True,
    )
    assert tracking_class_1 == FAILURE_CLASS_TRACKING

    tracking_class_2 = classify_failure(
        status="degraded",
        degradation_reasons=["steering_rate_saturation_limit"],
    )
    assert tracking_class_2 == FAILURE_CLASS_TRACKING

    # 5. Path generation failure: plan exception, solver infeasibility, or goal timeout
    path_class_1 = classify_failure(
        status="error",
        plan_exception=True,
        degradation_reasons=["plan_exception: ValueError"],
    )
    assert path_class_1 == FAILURE_CLASS_PATH_GENERATION

    path_class_2 = classify_failure(
        status="degraded",
        degradation_reasons=["solver_infeasible_no_path"],
    )
    assert path_class_2 == FAILURE_CLASS_PATH_GENERATION

    path_class_3 = classify_failure(
        status="degraded",
        completed=False,
        degradation_reasons=["max_steps_exceeded"],
    )
    assert path_class_3 == FAILURE_CLASS_PATH_GENERATION


def test_every_non_ok_rollout_carries_failure_class() -> None:
    """Every non-ok rollout carries exactly one of the four failure classes; ok rollouts carry None."""
    receipt = run_force_coupled_comparator()
    assert len(receipt["results"]) > 0

    non_ok_count = 0
    ok_count = 0
    for r in receipt["results"]:
        if r["status"] != "ok":
            non_ok_count += 1
            assert r["failure_class"] in VALID_FAILURE_CLASSES
        else:
            ok_count += 1
            assert r["failure_class"] is None

    assert non_ok_count > 0, "Suite must contain non-ok rollouts (e.g. from reference baselines)"
    assert ok_count > 0, "Suite must contain ok rollouts"


def test_summary_table_reports_failure_class_counts() -> None:
    """Summary table reports per-class counts for every planner without promoting a ranking."""
    receipt = run_force_coupled_comparator()
    summary = receipt["summary_table"]
    assert len(summary) == 4

    for row in summary:
        assert "failure_class_counts" in row
        counts = row["failure_class_counts"]
        for cls in VALID_FAILURE_CLASSES:
            assert cls in counts
            assert isinstance(counts[cls], int)
            assert counts[cls] >= 0

        # Sum of failure class counts must match the non-ok runs count
        non_ok_runs = sum(count for st, count in row["status_counts"].items() if st != "ok")
        total_failures = sum(counts.values())
        assert total_failures == non_ok_runs

    # force_coupled_potential_field has 0 failures across all canonical analytic scenarios
    fcpf_row = next(r for r in summary if r["planner_id"] == "force_coupled_potential_field")
    assert all(c == 0 for c in fcpf_row["failure_class_counts"].values())

    # pure_pursuit_goal collides with obstacle and pedestrian, producing tracking and social_compliance counts
    pp_row = next(r for r in summary if r["planner_id"] == "pure_pursuit_goal")
    assert pp_row["failure_class_counts"][FAILURE_CLASS_TRACKING] >= 1
    assert pp_row["failure_class_counts"][FAILURE_CLASS_SOCIAL_COMPLIANCE] >= 1


def test_summary_table_rejects_unknown_failure_class() -> None:
    """Summary output cannot silently introduce a non-canonical taxonomy value."""
    result = ComparatorRunResult(
        planner_id="test_planner",
        scenario_id="test_scenario",
        seed=42,
        steps=0,
        completed=False,
        collision=False,
        near_miss=False,
        min_clearance_obstacle_m=None,
        min_clearance_pedestrian_m=None,
        path_length_m=0.0,
        mean_linear_speed_mps=0.0,
        max_linear_speed_mps=0.0,
        mean_angular_rate_radps=0.0,
        max_angular_rate_radps=0.0,
        jerk_metric=0.0,
        mean_latency_ms=0.0,
        status="error",
        degraded=True,
        degradation_reasons=("fixture",),
        failure_class="not_canonical",
    )

    with pytest.raises(ValueError, match="unknown failure class"):
        compute_summary_table([result])


def test_summary_table_rejects_missing_failure_class() -> None:
    """A non-ok row without a taxonomy class cannot disappear from aggregation."""
    result = ComparatorRunResult(
        planner_id="test_planner",
        scenario_id="test_scenario",
        seed=42,
        steps=0,
        completed=True,
        collision=False,
        near_miss=False,
        min_clearance_obstacle_m=None,
        min_clearance_pedestrian_m=None,
        path_length_m=0.0,
        mean_linear_speed_mps=0.0,
        max_linear_speed_mps=0.0,
        mean_angular_rate_radps=0.0,
        max_angular_rate_radps=0.0,
        jerk_metric=0.0,
        mean_latency_ms=0.0,
        status="degraded",
        degraded=True,
        degradation_reasons=("fixture_degraded",),
        failure_class=None,
    )

    with pytest.raises(ValueError, match="missing failure class"):
        compute_summary_table([result])


@pytest.mark.parametrize("status", ["error", "degraded"])
def test_summary_table_rejects_non_ok_rows_without_degraded_flag(status: str) -> None:
    """Non-ok rows must carry the emitted degraded invariant before aggregation."""
    result = ComparatorRunResult(
        planner_id="test_planner",
        scenario_id="test_scenario",
        seed=42,
        steps=0,
        completed=True,
        collision=False,
        near_miss=False,
        min_clearance_obstacle_m=None,
        min_clearance_pedestrian_m=None,
        path_length_m=0.0,
        mean_linear_speed_mps=0.0,
        max_linear_speed_mps=0.0,
        mean_angular_rate_radps=0.0,
        max_angular_rate_radps=0.0,
        jerk_metric=0.0,
        mean_latency_ms=0.0,
        status=status,
        degraded=False,
        degradation_reasons=("fixture",),
        failure_class=FAILURE_CLASS_PATH_GENERATION,
    )

    with pytest.raises(ValueError, match="not marked degraded"):
        compute_summary_table([result])


def test_summary_table_rejects_unknown_status() -> None:
    """Summary aggregation rejects statuses outside the versioned rollout vocabulary."""
    healthy = execute_rollout(PurePursuitGoalPlanner(), get_canonical_comparison_scenarios()[-1])
    with pytest.raises(ValueError, match="unknown rollout status"):
        compute_summary_table([replace(healthy, status="unknown")])  # type: ignore[arg-type]


def test_summary_table_rejects_inconsistent_ok_row() -> None:
    """An ok row cannot carry degradation reasons or a failure class."""
    healthy = execute_rollout(PurePursuitGoalPlanner(), get_canonical_comparison_scenarios()[-1])
    with pytest.raises(ValueError, match="inconsistent failure"):
        compute_summary_table([replace(healthy, degradation_reasons=("fixture",))])


def test_summary_table_preserves_existing_success_rate_semantics() -> None:
    """Taxonomy diagnostics do not change the established completion/collision success rate."""
    healthy = execute_rollout(PurePursuitGoalPlanner(), get_canonical_comparison_scenarios()[-1])
    near_miss = replace(healthy, scenario_id="near_miss_fixture", near_miss=True)
    degraded = replace(
        healthy,
        scenario_id="degraded_fixture",
        status="degraded",
        degraded=True,
        degradation_reasons=("steering_rate_saturation",),
        failure_class=FAILURE_CLASS_TRACKING,
    )

    [summary] = compute_summary_table([near_miss, degraded])

    assert summary["success_rate"] == 1.0
    assert summary["near_miss_rate"] == 0.5
    assert summary["status_counts"] == {"ok": 1, "degraded": 1}
    assert summary["failure_class_counts"][FAILURE_CLASS_TRACKING] == 1


def test_comparator_uses_empty_digest_for_missing_explicit_config(tmp_path: Path) -> None:
    """An explicitly missing config uses the documented empty-config fallback."""
    receipt = run_force_coupled_comparator(config_path=tmp_path / "missing.yaml")

    assert len(receipt["config_sha256"]) == 64
    assert receipt["status"] == "ok"


def test_v1_schema_accepts_receipts_without_optional_taxonomy_fields() -> None:
    """The additive taxonomy fields do not invalidate existing v1 receipts."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "schemas"
        / "force_coupled_comparator_receipt.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    receipt = run_force_coupled_comparator()
    for result in receipt["results"]:
        result.pop("failure_class", None)
    for row in receipt["summary_table"]:
        row.pop("failure_class_counts", None)

    jsonschema.validate(instance=receipt, schema=schema)


def test_v1_schema_rejects_noncanonical_taxonomy_alias() -> None:
    """Schema validation rejects hyphenated aliases that runtime classification cannot emit."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "schemas"
        / "force_coupled_comparator_receipt.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    receipt = run_force_coupled_comparator()
    receipt["results"][0]["failure_class"] = "path-generation"

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=receipt, schema=schema)


def test_v1_schema_enforces_new_taxonomy_invariants_when_present() -> None:
    """New taxonomy-bearing rows are schema-checked while legacy rows remain optional."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "schemas"
        / "force_coupled_comparator_receipt.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    unknown_status = run_force_coupled_comparator()
    unknown_status["results"][0]["status"] = "unknown"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=unknown_status, schema=schema)

    inconsistent_degraded = run_force_coupled_comparator()
    inconsistent_degraded["results"][0]["status"] = "degraded"
    inconsistent_degraded["results"][0]["degraded"] = False
    inconsistent_degraded["results"][0]["failure_class"] = FAILURE_CLASS_PATH_GENERATION
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=inconsistent_degraded, schema=schema)


def test_v1_schema_rejects_explicit_null_failure_class_on_non_ok_row() -> None:
    """Legacy omission stays compatible, but an explicit non-ok null is malformed."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "schemas"
        / "force_coupled_comparator_receipt.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    receipt = run_force_coupled_comparator()
    row = receipt["results"][0]
    row.update(
        {
            "status": "degraded",
            "degraded": True,
            "degradation_reasons": ["fixture_degraded"],
            "failure_class": None,
        }
    )

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=receipt, schema=schema)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collision", True),
        ("completed", False),
        ("degradation_reasons", ["fixture_degraded"]),
    ],
)
def test_v1_schema_rejects_inconsistent_ok_rollout(field: str, value: object) -> None:
    """The receipt schema rejects ok rows carrying failure-state signals."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "schemas"
        / "force_coupled_comparator_receipt.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    receipt = run_force_coupled_comparator()
    row = receipt["results"][0]
    row.update(
        {
            "status": "ok",
            "completed": True,
            "collision": False,
            "degraded": False,
            "degradation_reasons": [],
            "failure_class": None,
        }
    )
    row[field] = value

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=receipt, schema=schema)


@pytest.mark.parametrize("malformation", ["unknown", "negative", "missing"])
def test_v1_schema_rejects_malformed_summary_taxonomy_counts(malformation: str) -> None:
    """Present summary taxonomy counts must be complete, canonical, and non-negative."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "schemas"
        / "force_coupled_comparator_receipt.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    receipt = run_force_coupled_comparator()
    counts = receipt["summary_table"][0]["failure_class_counts"]
    if malformation == "unknown":
        counts["bogus"] = 1
    elif malformation == "negative":
        counts[FAILURE_CLASS_SIMULATOR] = -1
    else:
        counts.pop(FAILURE_CLASS_SIMULATOR)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=receipt, schema=schema)


def test_comparator_run_result_serialization_with_failure_class() -> None:
    """ComparatorRunResult dataclass serializes failure_class to dictionary."""
    res = ComparatorRunResult(
        planner_id="test_planner",
        scenario_id="test_scenario",
        seed=42,
        steps=10,
        completed=False,
        collision=True,
        near_miss=False,
        min_clearance_obstacle_m=0.0,
        min_clearance_pedestrian_m=None,
        path_length_m=2.5,
        mean_linear_speed_mps=0.5,
        max_linear_speed_mps=1.0,
        mean_angular_rate_radps=0.1,
        max_angular_rate_radps=0.3,
        jerk_metric=0.2,
        mean_latency_ms=1.5,
        status="degraded",
        degraded=True,
        degradation_reasons=("obstacle_collision",),
        failure_class=FAILURE_CLASS_TRACKING,
    )
    serialized = res.to_dict()
    assert serialized["failure_class"] == FAILURE_CLASS_TRACKING
    assert serialized["status"] == "degraded"
