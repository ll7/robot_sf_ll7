"""Tests for the issue #6970 paired-effect retained-row contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from robot_sf.benchmark import runner
from robot_sf.benchmark.paired_effect_metric_contract import (
    CLARIFICATION_METRIC_NAMES,
    REQUIRED_METRIC_NAMES,
    PairedEffectMetricContractError,
    evaluate_paired_effect_metric_fields,
    load_paired_effect_metric_clarification,
    load_paired_effect_metric_contract,
    validate_paired_effect_metric_contract,
    validate_paired_effect_metric_record,
    validate_paired_effect_metric_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/benchmarks/paired_effect_metric_contract_v1.yaml"
CLARIFICATION_PATH = (
    REPO_ROOT / "configs/benchmarks/paired_effect_metric_contract_v1_clarification.yaml"
)
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
CHECK_SCRIPT = REPO_ROOT / "scripts/benchmark/check_paired_effect_metric_contract.py"


def _contract() -> dict[str, object]:
    return load_paired_effect_metric_contract(CONTRACT_PATH)


def _valid_metric_values() -> dict[str, float]:
    return {
        "exact_collision_probability": 0.0,
        "near_miss_probability": 1.0,
        "min_predicted_separation_m": 0.42,
        "completion_probability": 1.0,
        "progress_at_timeout": 0.75,
        "false_positive_stop_rate": 0.0,
        "stop_yield_latency_s": 0.8,
        "wrapper_intervention_rate": 0.25,
    }


def _map_scenario() -> dict[str, object]:
    return {
        "name": "issue-6970-contract-smoke",
        "map_file": "maps/svg_maps/francis2023/francis2023_blind_corner.svg",
        "simulation_config": {"max_episode_steps": 1},
        "robot_config": {"kinematics": "differential_drive"},
    }


def _native_trace(
    arm: str,
    *,
    length: int = 25,
    stop_steps: tuple[int, ...] = (),
    recovery_step: int | None = None,
    invalid_command_steps: tuple[int, ...] = (),
    unavailable_outcome_steps: tuple[int, ...] = (),
    collision_steps: tuple[int, ...] = (),
    near_miss_steps: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    """Build a fully declared native wrapper trace for contract tests."""

    stops = set(stop_steps)
    invalid_commands = set(invalid_command_steps)
    collision_steps_set = set(collision_steps)
    near_miss_steps_set = set(near_miss_steps)
    trace: list[dict[str, Any]] = []
    for step in range(length):
        if step in stops:
            command_status = "not_forward_progress"
            linear_velocity: float = 0.0
        elif stops and recovery_step is None and step > max(stops):
            command_status = "not_forward_progress"
            linear_velocity = 0.0
        elif stops and recovery_step is not None and step < recovery_step and step > max(stops):
            command_status = "not_forward_progress"
            linear_velocity = 0.0
        else:
            command_status = "valid"
            linear_velocity = 1.0
        if step in invalid_commands:
            command_status = "valid"
            linear_velocity = float("nan")
        if step in unavailable_outcome_steps:
            post_step_outcome: dict[str, Any] = {
                "status": "unavailable",
                "reason": "native_outcome_missing",
            }
        else:
            post_step_outcome = {
                "status": "available",
                "collision": step in collision_steps_set,
                "near_miss": step in near_miss_steps_set,
            }
        trace.append(
            {
                "schema_version": "safety_wrapper_runtime_step.v1",
                "step": step,
                "time_s": (step + 1) * 0.1,
                "arm_key": arm,
                "enabled": arm == "wrapper_on",
                "eligible_for_wrapper": True,
                "intervention": "hard_stop" if step in stops else "none",
                "intervened": step in stops,
                "forward_progress_command": {
                    "status": command_status,
                    "linear_velocity_m_s": linear_velocity,
                    "source": "fixture_native_final_command",
                },
                "post_step_outcome": post_step_outcome,
            }
        )
    return trace


def _native_record(
    arm: str,
    *,
    length: int = 25,
    stop_steps: tuple[int, ...] = (),
    recovery_step: int | None = None,
    timeout: bool = True,
    initial_distance: float = 10.0,
    final_x: float | None = None,
    nonfinite_final: bool = False,
    **trace_kwargs: Any,
) -> dict[str, Any]:
    """Build one arm record with explicit pair and simulation provenance."""

    trace = _native_trace(
        arm,
        length=length,
        stop_steps=stop_steps,
        recovery_step=recovery_step,
        **trace_kwargs,
    )
    final_position = [
        float(final_x if final_x is not None else max(0.0, (length - 1) * 0.2)),
        0.0,
    ]
    if nonfinite_final:
        final_position[0] = float("nan")
    simulation_steps = [
        {
            "step": step,
            "time_s": (step + 1) * 0.1,
            "robot": {
                "position": (final_position if step == length - 1 else [float(step) * 0.2, 0.0])
            },
        }
        for step in range(length)
    ]
    termination_reason = "max_steps" if timeout else "success"
    arm_independent_params = {
        "id": "fixture_scenario",
        "algo": "social_force",
        "run_dt": 0.1,
        "safety_wrapper": {"enabled": arm == "wrapper_on", "arm_key": arm},
    }
    arm_independent_params.pop("safety_wrapper")
    pair_config_hash = hashlib.sha256(
        json.dumps(arm_independent_params, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {
        "algo": "social_force",
        "scenario_id": "fixture_scenario",
        "seed": 111,
        "scenario_params": {
            "id": "fixture_scenario",
            "algo": "social_force",
            "run_dt": 0.1,
            "safety_wrapper": {"enabled": arm == "wrapper_on", "arm_key": arm},
        },
        "config_hash": "arm-specific-config-hash",
        "git_hash": "fixture-source-commit",
        "provenance": {"git_hash": "fixture-source-commit"},
        "horizon": length,
        "termination_reason": termination_reason,
        "outcome": {"timeout_event": timeout},
        "algorithm_metadata": {
            "safety_wrapper": {
                "schema_version": "safety_wrapper_episode_summary.v1",
                "arm_key": arm,
                "time_per_step_s": 0.1,
                "step_trace": trace,
            },
            "paired_effect_native_trace": {
                "schema_version": "paired_effect_native_trace.v1",
                "arm_key": arm,
                "dt_s": 0.1,
                "horizon_steps": length,
                "initial_goal_distance_m": initial_distance,
                "goal_position": [10.0, 0.0],
                "declared_timeout": timeout,
                "termination_reason": termination_reason,
                "source_commit": "fixture-source-commit",
                "pair_config_hash": pair_config_hash,
            },
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "dt": 0.1,
                "steps": simulation_steps,
            },
        },
    }


def test_contract_declares_exact_report_builder_roster() -> None:
    """The versioned contract is aligned with the #4598 report-builder outcome roster."""
    contract = _contract()
    assert contract["schema_version"] == "paired_effect_metric_contract.v1"
    assert contract["report_builder_issue"] == 4598
    assert tuple(contract["required_metric_names"]) == REQUIRED_METRIC_NAMES  # type: ignore[arg-type]
    assert [field["path"] for field in contract["fields"]] == [  # type: ignore[index]
        f"metric_values.{name}" for name in REQUIRED_METRIC_NAMES
    ]


def test_clarification_companion_declares_only_the_three_deferred_fields() -> None:
    """The #8567 companion is versioned separately from the frozen v1 manifest."""

    clarification = load_paired_effect_metric_clarification(CLARIFICATION_PATH)

    assert clarification["schema_version"] == "paired_effect_metric_clarification.v1"
    assert clarification["issue"] == 8567
    assert clarification["parent_contract_schema_version"] == "paired_effect_metric_contract.v1"
    assert clarification["no_campaign_compute"] is True
    assert tuple(clarification["required_metric_names"]) == CLARIFICATION_METRIC_NAMES
    assert [field["name"] for field in clarification["fields"]] == list(CLARIFICATION_METRIC_NAMES)


def test_native_paired_fields_are_trace_derived_and_pair_aware() -> None:
    """A complete native pair produces all three clarified fields with provenance."""

    wrapper_off = _native_record("wrapper_off")
    wrapper_on = _native_record(
        "wrapper_on",
        stop_steps=(1, 2),
        recovery_step=3,
    )

    result = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
    )

    assert result["status"] == "available"
    assert result["source"]["clarification_schema_version"] == (
        "paired_effect_metric_clarification.v1"
    )
    assert result["metric_values"]["false_positive_stop_rate"] == 1.0
    assert result["metric_values"]["stop_yield_latency_s"] == pytest.approx(0.2)
    assert result["metric_values"]["progress_at_timeout"] == pytest.approx(0.48)


def test_native_producer_adds_clarified_fields_without_changing_five_existing_fields() -> None:
    """The finalizer appends valid native fields while retaining the original five sources."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    wrapper_off = _native_record("wrapper_off")
    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    algo_meta = deepcopy(wrapper_on["algorithm_metadata"])
    native_record = dict(wrapper_on)
    native_record["algorithm_metadata"] = algo_meta

    metrics = map_runner_episode._finalize_episode_metrics(
        {"success": 0.0, "collisions": 0.0, "near_misses": 0.0, "min_distance": 0.5},
        algo_meta=algo_meta,
        actuation_controller=None,
        actuation_summary={},
        tracking_precision_summary={
            "min_separation_corrupted_m": 0.5,
            "contract_honored": True,
            "contract_honored_rate": 1.0,
        },
        tracking_precision_spec={"target_motp_m": 0.3},
        safety_wrapper_summary=None,
        cbf_filter_summary=None,
        snqi_weights=None,
        snqi_baseline=None,
        native_record=native_record,
        paired_wrapper_off_record=wrapper_off,
    )

    values = metrics["metric_values"]
    assert values["exact_collision_probability"] == 0.0
    assert values["near_miss_probability"] == 0.0
    assert values["min_predicted_separation_m"] == 0.5
    assert values["completion_probability"] == 0.0
    assert values["false_positive_stop_rate"] == 1.0
    assert values["stop_yield_latency_s"] == pytest.approx(0.1)
    assert values["progress_at_timeout"] == pytest.approx(0.48)
    assert algo_meta["paired_effect_metric_producer"]["status"] == "available"


@pytest.mark.parametrize("mismatch", ["identity", "config", "source", "arm"])
def test_native_pair_mismatch_fails_closed(mismatch: str) -> None:
    """Pair identity, config, source, and arm drift cannot produce a causal rate."""

    wrapper_off = _native_record("wrapper_off")
    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    if mismatch == "identity":
        wrapper_off["seed"] = 112
    elif mismatch == "config":
        wrapper_off["algorithm_metadata"]["paired_effect_native_trace"]["pair_config_hash"] = (
            "different"
        )
    elif mismatch == "source":
        wrapper_off["algorithm_metadata"]["paired_effect_native_trace"]["source_commit"] = (
            "different"
        )
    else:
        wrapper_off["algorithm_metadata"]["safety_wrapper"]["arm_key"] = "wrapper_on"

    result = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
    )

    assert result["fields"]["false_positive_stop_rate"]["status"] in {
        "unavailable",
        "invalid",
    }
    assert "false_positive_stop_rate" not in result["metric_values"]


def test_native_pair_requires_declared_native_provenance() -> None:
    """Generic row provenance cannot substitute for native pair identity."""

    wrapper_off = _native_record("wrapper_off")
    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    wrapper_on["algorithm_metadata"]["paired_effect_native_trace"].pop("source_commit")
    wrapper_on["algorithm_metadata"]["paired_effect_native_trace"].pop("pair_config_hash")

    result = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
    )

    assert result["fields"]["false_positive_stop_rate"] == {
        "status": "unavailable",
        "reason": "pair_config_identity_unavailable",
        "details": {"side": "wrapper_on"},
    }


def test_native_pair_does_not_fallback_to_generic_source_provenance() -> None:
    """A generic row git hash cannot substitute for native source identity."""

    wrapper_off = _native_record("wrapper_off")
    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    wrapper_on["algorithm_metadata"]["paired_effect_native_trace"].pop("source_commit")

    result = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
    )

    assert result["fields"]["false_positive_stop_rate"] == {
        "status": "unavailable",
        "reason": "pair_source_identity_unavailable",
        "details": {"side": "wrapper_on"},
    }


def test_native_pair_recomputes_config_identity_before_pairing() -> None:
    """A copied native hash cannot hide drift in the arm-independent parameters."""

    wrapper_off = _native_record("wrapper_off")
    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    wrapper_off["scenario_params"]["planner_override"] = "drifted"

    result = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
    )

    assert result["fields"]["false_positive_stop_rate"]["status"] == "invalid"
    assert result["fields"]["false_positive_stop_rate"]["reason"] == (
        "pair_config_identity_mismatch"
    )
    assert "false_positive_stop_rate" not in result["metric_values"]


def test_native_false_positive_requires_complete_inclusive_two_second_window() -> None:
    """The endpoint at onset+20 is required at dt=0.1; a missing endpoint is unavailable."""

    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    complete_off = _native_record("wrapper_off", length=22)
    complete = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=complete_off,
    )
    assert complete["fields"]["false_positive_stop_rate"]["status"] == "available"
    assert (
        complete["fields"]["false_positive_stop_rate"]["details"]["windows"][0]["window_end_step"]
        == 21
    )

    truncated_off = _native_record("wrapper_off", length=21)
    truncated = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=truncated_off,
    )
    assert truncated["fields"]["false_positive_stop_rate"] == {
        "status": "unavailable",
        "reason": "counterfactual_window_truncated",
        "details": {"event_onset_step": 1, "window_end_step": 21},
    }


def test_native_false_positive_missing_counterfactual_step_is_unavailable() -> None:
    """An interior missing wrapper-off step cannot be silently repaired."""

    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    wrapper_off = _native_record("wrapper_off")
    wrapper_off["algorithm_metadata"]["safety_wrapper"]["step_trace"].pop(10)

    result = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
    )

    assert result["fields"]["false_positive_stop_rate"]["status"] == "unavailable"
    assert result["fields"]["false_positive_stop_rate"]["reason"] == "missing_trace_step"


def test_native_trace_required_state_and_nonfinite_window_fail_closed() -> None:
    """Missing native state and an invalid counterfactual window stay machine-readable."""

    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    wrapper_off = _native_record("wrapper_off")
    wrapper_on["algorithm_metadata"]["safety_wrapper"]["step_trace"][0].pop("eligible_for_wrapper")

    missing_state = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
    )
    assert missing_state["fields"]["stop_yield_latency_s"] == {
        "status": "unavailable",
        "reason": "missing_eligible_wrapper_state",
    }

    valid_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    invalid_window = evaluate_paired_effect_metric_fields(
        valid_on,
        paired_wrapper_off_record=wrapper_off,
        window_s=float("nan"),
    )
    assert invalid_window["fields"]["false_positive_stop_rate"] == {
        "status": "invalid",
        "reason": "invalid_counterfactual_window",
    }


@pytest.mark.parametrize("window_s", [1.0, 5.0])
def test_native_false_positive_rejects_non_contract_window(window_s: float) -> None:
    """The companion's fixed 2.0-second window cannot be overridden by callers."""

    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    wrapper_off = _native_record("wrapper_off")

    result = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
        window_s=window_s,
    )

    assert result["fields"]["false_positive_stop_rate"] == {
        "status": "invalid",
        "reason": "invalid_counterfactual_window",
    }


def test_native_pair_rejects_wrapper_off_intervention() -> None:
    """The counterfactual arm must remain wrapper-off throughout its native trace."""

    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    wrapper_off = _native_record("wrapper_off")
    off_step = wrapper_off["algorithm_metadata"]["safety_wrapper"]["step_trace"][4]
    off_step["intervention"] = "hard_stop"
    off_step["intervened"] = True

    result = evaluate_paired_effect_metric_fields(
        wrapper_on,
        paired_wrapper_off_record=wrapper_off,
    )

    assert result["fields"]["false_positive_stop_rate"] == {
        "status": "invalid",
        "reason": "wrapper_off_intervention_present",
        "details": {"interventions": ["hard_stop"]},
    }


def test_native_nonfinite_declared_trace_time_step_is_invalid() -> None:
    """A malformed native timestep cannot fall back to a finite summary timestep."""

    record = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    record["algorithm_metadata"]["paired_effect_native_trace"]["dt_s"] = float("nan")

    result = evaluate_paired_effect_metric_fields(record)

    assert result["fields"]["stop_yield_latency_s"] == {
        "status": "invalid",
        "reason": "nonfinite_trace_time_step",
    }


def test_native_no_intervention_or_recovery_is_unavailable() -> None:
    """No opportunity and unrecovered intervention are explicit unavailable states."""

    off = _native_record("wrapper_off")
    no_intervention = _native_record("wrapper_on")
    no_intervention_result = evaluate_paired_effect_metric_fields(
        no_intervention,
        paired_wrapper_off_record=off,
    )
    assert no_intervention_result["fields"]["stop_yield_latency_s"] == {
        "status": "unavailable",
        "reason": "no_stop_yield_opportunity",
    }

    unrecovered = _native_record("wrapper_on", stop_steps=(1, 2))
    unrecovered_result = evaluate_paired_effect_metric_fields(
        unrecovered,
        paired_wrapper_off_record=off,
    )
    assert unrecovered_result["fields"]["stop_yield_latency_s"]["reason"] == (
        "intervention_recovery_missing"
    )
    assert "stop_yield_latency_s" not in unrecovered_result["metric_values"]


def test_native_false_positive_rejects_collision_near_miss_and_invalid_command() -> None:
    """Every required off-arm predicate participates in the false-positive classifier."""

    wrapper_on = _native_record("wrapper_on", stop_steps=(1,), recovery_step=2)
    for kwargs, expected in (
        ({"collision_steps": (5,)}, "available"),
        ({"near_miss_steps": (5,)}, "available"),
        ({"invalid_command_steps": (21,)}, "invalid"),
    ):
        wrapper_off = _native_record("wrapper_off", **kwargs)
        result = evaluate_paired_effect_metric_fields(
            wrapper_on,
            paired_wrapper_off_record=wrapper_off,
        )
        field = result["fields"]["false_positive_stop_rate"]
        assert field["status"] == expected
        if expected == "available":
            assert field["value"] == 0.0
        else:
            assert "false_positive_stop_rate" not in result["metric_values"]


@pytest.mark.parametrize("termination", ["success", "truncated"])
def test_native_timeout_progress_distinguishes_termination(termination: str) -> None:
    """Timeout progress uses the declared timeout and rejects other termination."""

    record = _native_record("wrapper_on", length=3, timeout=termination == "truncated", final_x=2.0)
    record["termination_reason"] = termination
    record["algorithm_metadata"]["paired_effect_native_trace"]["termination_reason"] = termination
    result = evaluate_paired_effect_metric_fields(record)

    field = result["fields"]["progress_at_timeout"]
    if termination == "truncated":
        assert field["status"] == "available"
        assert field["value"] == pytest.approx(0.2)
    else:
        assert field == {"status": "unavailable", "reason": "non_timeout_termination"}


@pytest.mark.parametrize("initial_distance", [0.0, float("nan")])
def test_native_timeout_progress_rejects_invalid_denominator(initial_distance: float) -> None:
    """A zero or non-finite declared start-goal denominator is invalid, not imputed."""

    record = _native_record("wrapper_on", length=3, initial_distance=initial_distance)
    result = evaluate_paired_effect_metric_fields(record)

    field = result["fields"]["progress_at_timeout"]
    assert field["status"] == "invalid"
    assert field["reason"] in {
        "invalid_start_to_goal_denominator",
        "nonfinite_start_to_goal_denominator",
    }
    assert "progress_at_timeout" not in result["metric_values"]


def test_native_timeout_progress_rejects_nonfinite_final_position() -> None:
    """A non-finite final native robot position is an explicit invalid outcome."""

    record = _native_record("wrapper_on", length=3, nonfinite_final=True)
    result = evaluate_paired_effect_metric_fields(record)

    assert result["fields"]["progress_at_timeout"] == {
        "status": "invalid",
        "reason": "nonfinite_timeout_robot_position",
    }


def test_native_timeout_progress_rejects_nonfinite_simulation_time_step() -> None:
    """A non-finite simulation-trace timestep is invalid even with native dt metadata."""

    record = _native_record("wrapper_on", length=3)
    record["algorithm_metadata"]["simulation_step_trace"]["dt"] = float("nan")

    result = evaluate_paired_effect_metric_fields(record)

    assert result["fields"]["progress_at_timeout"] == {
        "status": "invalid",
        "reason": "nonfinite_simulation_trace_time_step",
    }


def test_metric_values_producer_derives_available_sources() -> None:
    """The map-runner producer emits the five fields with available sources."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    values = map_runner_episode._paired_effect_metric_values(
        metrics_raw={
            "success": 1.0,
            "collisions": 1.0,
            "near_misses": 2.0,
            "min_distance": 0.42,
        },
        metrics={"wrapper_intervention_rate": 0.25},
    )
    assert values["exact_collision_probability"] == 1.0
    assert values["near_miss_probability"] == 1.0
    assert values["min_predicted_separation_m"] == 0.42
    assert values["completion_probability"] == 1.0
    assert values["wrapper_intervention_rate"] == 0.25


def test_metric_values_producer_omits_unavailable_sources() -> None:
    """Fields whose sources are missing are omitted, never fabricated as zero."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    values = map_runner_episode._paired_effect_metric_values(
        metrics_raw={},
        metrics={},
    )
    assert values == {}
    assert "exact_collision_probability" not in values


def test_metric_values_producer_finalize_attaches_mapping() -> None:
    """_finalize_episode_metrics attaches the metric_values mapping."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    metrics = map_runner_episode._finalize_episode_metrics(
        {"success": 0.0, "collisions": 1.0, "near_misses": 0.0, "min_distance": 0.5},
        algo_meta={},
        actuation_controller=None,
        actuation_summary={},
        tracking_precision_summary={
            "min_separation_corrupted_m": 0.5,
            "contract_honored": True,
            "contract_honored_rate": 1.0,
        },
        tracking_precision_spec={"target_motp_m": 0.3},
        safety_wrapper_summary=None,
        cbf_filter_summary=None,
        snqi_weights=None,
        snqi_baseline=None,
    )
    assert "metric_values" in metrics
    assert metrics["metric_values"]["exact_collision_probability"] == 1.0
    assert metrics["metric_values"]["completion_probability"] == 0.0


def test_map_episode_record_exposes_retained_mapping_at_contract_path() -> None:
    """The full map episode envelope forwards the retained mapping to the root path."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    metrics = {"metric_values": _valid_metric_values(), "success": 1.0}
    record = map_runner_episode._build_episode_record_dict(
        scenario_id="fixture_scenario",
        seed=111,
        scenario_params={"id": "fixture_scenario"},
        metrics=metrics,
        safety_predicates={},
        public_requirement_events={},
        algo_meta={},
        noise_spec={},
        noise_stats={},
        tracking_precision_spec={},
        algo="social_force",
        active_observation_mode="default",
        active_observation_level="full",
        ts_start="2026-09-08T00:00:00+00:00",
        ts_end="2026-09-08T00:00:01+00:00",
        status="completed",
        steps_taken=1,
        horizon_val=1,
        wall_time=1.0,
        termination_reason="success",
        outcome={"success": True, "collision": False, "timeout_event": False},
        contradictions=[],
        view_integrity=None,
    )

    assert record["metric_values"] == metrics["metric_values"]
    assert record["metrics"]["metric_values"] == metrics["metric_values"]


def test_map_episode_record_uses_empty_mapping_when_metric_values_are_not_a_mapping() -> None:
    """Malformed producer output stays an empty contract mapping at the root."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    record = map_runner_episode._build_episode_record_dict(
        scenario_id="fixture_scenario",
        seed=111,
        scenario_params={"id": "fixture_scenario"},
        metrics={"metric_values": None},
        safety_predicates={},
        public_requirement_events={},
        algo_meta={},
        noise_spec={},
        noise_stats={},
        tracking_precision_spec={},
        algo="social_force",
        active_observation_mode="default",
        active_observation_level="full",
        ts_start="2026-09-08T00:00:00+00:00",
        ts_end="2026-09-08T00:00:01+00:00",
        status="completed",
        steps_taken=1,
        horizon_val=1,
        wall_time=1.0,
        termination_reason="success",
        outcome={"success": True, "collision": False, "timeout_event": False},
        contradictions=[],
        view_integrity=None,
    )

    assert record["metric_values"] == {}


def test_native_timeout_wrapper_trace_truncation_is_unavailable() -> None:
    """A declared timeout cannot yield latency or opportunity from a short trace."""
    record = _native_record("wrapper_on", length=3, stop_steps=(1,), recovery_step=2)
    record["algorithm_metadata"]["paired_effect_native_trace"]["horizon_steps"] = 4

    result = evaluate_paired_effect_metric_fields(record)

    assert result["fields"]["stop_yield_latency_s"] == {
        "status": "unavailable",
        "reason": "timeout_trace_truncated",
    }
    assert result["fields"]["progress_at_timeout"] == {
        "status": "unavailable",
        "reason": "timeout_trace_truncated",
    }


def test_valid_retained_row_passes() -> None:
    """A row with all exact finite fields passes without using legacy aliases."""
    report = validate_paired_effect_metric_record(
        {"metric_values": _valid_metric_values()},
        _contract(),
        row_index=3,
    )
    assert report["status"] == "ok"
    assert report["missing_fields"] == []
    assert report["invalid_fields"] == []


def test_issue_13775_shape_is_blocked_without_alias_substitution() -> None:
    """The historical metrics-only row cannot satisfy the retained paired-effect contract."""
    report = validate_paired_effect_metric_rows(
        [{"metrics": {"clearing_distance_min": 0.42, "success": 1.0}}],
        _contract(),
    )
    assert report["status"] == "blocked"
    assert report["complete"] is False
    assert set(report["missing_field_counts"]) == set(REQUIRED_METRIC_NAMES)


def test_row_validation_uses_bounded_diagnostics_by_default() -> None:
    """Runner-facing reports stay compact while diagnostics retain every row report."""

    rows = [{"metrics": {"clearing_distance_min": float(index)}} for index in range(12)]

    compact = validate_paired_effect_metric_rows(rows, _contract())
    assert "row_reports" not in compact
    assert len(compact["invalid_row_samples"]) == 10
    assert [report["row_index"] for report in compact["invalid_row_samples"]] == list(range(10))

    diagnostic = validate_paired_effect_metric_rows(rows, _contract(), include_row_reports=True)
    assert len(diagnostic["row_reports"]) == len(rows)


@pytest.mark.parametrize(
    "field_name, value, reason",
    [
        ("wrapper_intervention_rate", True, "boolean_is_not_scalar"),
        ("wrapper_intervention_rate", float("nan"), "non_finite"),
        ("wrapper_intervention_rate", 1.1, "out_of_bounds"),
        ("progress_at_timeout", 1.1, "out_of_bounds"),
    ],
)
def test_invalid_retained_values_are_blocked(field_name: str, value: object, reason: str) -> None:
    """Boolean, non-finite, and out-of-range values fail closed."""
    metric_values = _valid_metric_values()
    metric_values[field_name] = value  # type: ignore[assignment]
    report = validate_paired_effect_metric_record(
        {"metric_values": metric_values},
        _contract(),
    )
    assert report["status"] == "blocked"
    assert report["invalid_fields"][0]["reason"] == reason


def test_contract_rejects_wrong_retained_path() -> None:
    """The contract cannot silently move the report-builder fields to a legacy mapping."""
    payload = _contract()
    fields = [dict(field) for field in payload["fields"]]  # type: ignore[index]
    fields[0]["path"] = "metrics.exact_collision_probability"
    payload["fields"] = fields
    with pytest.raises(PairedEffectMetricContractError, match="path"):
        validate_paired_effect_metric_contract(payload)


def test_runner_fails_closed_on_metrics_only_map_output(tmp_path: Path) -> None:
    """The run-batch post-run gate rejects a successful-looking but unidentifiable row."""
    out_path = tmp_path / "episodes.jsonl"

    def fake_run_map_batch(*args: object, **_kwargs: object) -> dict[str, object]:
        Path(args[1]).write_text(
            json.dumps({"metrics": {"clearing_distance_min": 0.42}}) + "\n",
            encoding="utf-8",
        )
        return {"status": "ok", "total_jobs": 1, "written": 1, "failures": []}

    with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
        with pytest.raises(PairedEffectMetricContractError, match="retained-row contract failed"):
            runner.run_batch(
                [_map_scenario()],
                out_path=out_path,
                schema_path=SCHEMA_PATH,
                horizon=1,
                dt=0.1,
                retained_metric_contract_path=CONTRACT_PATH,
            )


def test_runner_contract_error_uses_compact_validation_details(tmp_path: Path) -> None:
    """A failed runner gate does not serialize the full per-row diagnostic collection."""
    out_path = tmp_path / "episodes.jsonl"

    def fake_run_map_batch(*args: object, **_kwargs: object) -> dict[str, object]:
        Path(args[1]).write_text(
            json.dumps({"metrics": {"clearing_distance_min": 0.42}}) + "\n",
            encoding="utf-8",
        )
        return {"status": "ok", "total_jobs": 1, "written": 1, "failures": []}

    with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
        with pytest.raises(
            PairedEffectMetricContractError,
            match="retained-row contract failed",
        ) as exc:
            runner.run_batch(
                [_map_scenario()],
                out_path=out_path,
                schema_path=SCHEMA_PATH,
                horizon=1,
                dt=0.1,
                retained_metric_contract_path=CONTRACT_PATH,
            )
    assert "row_reports" not in str(exc.value)


def test_runner_accepts_complete_retained_metric_row(tmp_path: Path) -> None:
    """The post-run gate returns a structured success report for a complete retained row."""
    out_path = tmp_path / "episodes.jsonl"

    def fake_run_map_batch(*args: object, **_kwargs: object) -> dict[str, object]:
        Path(args[1]).write_text(
            json.dumps({"metric_values": _valid_metric_values()}) + "\n",
            encoding="utf-8",
        )
        return {"status": "ok", "total_jobs": 1, "written": 1, "failures": []}

    with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
        summary = runner.run_batch(
            [_map_scenario()],
            out_path=out_path,
            schema_path=SCHEMA_PATH,
            horizon=1,
            dt=0.1,
            retained_metric_contract_path=CONTRACT_PATH,
        )
    assert summary["retained_metric_contract"]["status"] == "ok"
    assert summary["retained_metric_contract"]["complete"] is True


def test_runner_loads_contract_before_dispatch(tmp_path: Path) -> None:
    """An invalid contract reference blocks dispatch before any arm can run."""
    called = False

    def fake_run_map_batch(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
        with pytest.raises(PairedEffectMetricContractError, match="cannot be read"):
            runner.run_batch(
                [_map_scenario()],
                out_path=tmp_path / "episodes.jsonl",
                schema_path=SCHEMA_PATH,
                retained_metric_contract_path=tmp_path / "missing.yaml",
            )
    assert called is False


def test_contract_audit_cli_reports_contract_complete_exposure_without_running_campaign() -> None:
    """The audit command remains diagnostic-only while reporting complete config coverage."""
    completed = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--contract",
            str(CONTRACT_PATH),
            "--audit-configs",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["exposure"]["status"] == "ok"
    assert payload["exposure"]["config_count"] >= 2
    assert payload["exposure"]["claim_boundary"].startswith("Exposure audit only.")
    assert payload["exposure"]["counts"]["missing_reference"] == 0
    assert payload["exposure"]["counts"]["invalid_reference"] == 0
    assert payload["exposure"]["counts"]["invalid_contract"] == 0


def test_contract_audit_marks_malformed_referenced_contract(tmp_path: Path) -> None:
    """A readable but schema-invalid reference is not reported as covered."""
    from scripts.benchmark.check_paired_effect_metric_contract import _audit_configs

    config_dir = tmp_path / "configs" / "research"
    config_dir.mkdir(parents=True)
    (config_dir / "invalid_reference.yaml").write_text(
        "report_contract:\n  paired_report_builder_issue: 4598\n"
        "retained_metric_contract: invalid_contract.yaml\n",
        encoding="utf-8",
    )
    (config_dir / "invalid_contract.yaml").write_text(
        "schema_version: paired_effect_metric_contract.v1\n",
        encoding="utf-8",
    )

    audit = _audit_configs(tmp_path)

    assert audit["status"] == "findings"
    assert audit["counts"]["invalid_contract"] == 1
    assert audit["configs"][0]["status"] == "invalid_contract"
