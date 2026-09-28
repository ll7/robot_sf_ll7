"""Focused contract tests for the reference-planner oracle report gate."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from robot_sf.benchmark.reference_oracle_report import evaluate_oracles, render_markdown

SCENARIOS = ["open", "probe"]
SEEDS = [111, 112]
SOURCE_SHA = "a" * 40
THRESHOLDS: dict[str, int | float] = {
    "max_goal_failure_count": 0,
    "max_stationary_contact_rate": 0.0,
    "max_dominance_regressions": 0,
}


def _row(
    scenario_id: str,
    seed: int,
    *,
    success: bool = True,
    arm: str = "goal",
    execution_mode: str = "native",
    ped_collision_count: float = 0.0,
    population_mode: str | None = "pedestrian_free_v1",
    source_sha: str = SOURCE_SHA,
) -> dict[str, Any]:
    """Build one typed native episode row for the declared arm role."""
    stationary = arm == "stationary"
    route_complete = success and not stationary
    collision_event = stationary and ped_collision_count > 0
    timeout_event = not route_complete and not collision_event
    termination_reason = (
        "success" if route_complete else "collision" if collision_event else "max_steps"
    )
    params: dict[str, Any] = {
        "simulation_config": {
            "population_size": 5 if stationary else 0,
            "instantiated_population_size": 5 if stationary else 0,
        }
    }
    if population_mode is not None:
        params["reference_population_mode"] = population_mode
    if stationary:
        params["reference_population_capture_version"] = "v1"
    return {
        "version": "v1",
        "episode_id": f"{arm}-{scenario_id}-{seed}",
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": "stand_still" if stationary else "social_force" if arm == "aware" else "goal",
        "termination_reason": termination_reason,
        "status": "success" if route_complete else "collision" if collision_event else "failure",
        "outcome": {
            "route_complete": route_complete,
            "collision_event": collision_event,
            "timeout_event": timeout_event,
        },
        "integrity": {"contradictions": []},
        "execution_mode": execution_mode,
        "algorithm_metadata": (
            {
                "reference_population": {
                    "schema_version": "v1",
                    "instantiated_population_size": 5,
                }
            }
            if stationary
            else {}
        ),
        "git_hash": source_sha,
        "provenance": {"git_hash": source_sha},
        "scenario_params": params,
        "interaction_exposure": {
            "interaction_exposure_status": (
                "computed" if stationary else "not_derivable_no_pedestrians"
            )
        },
        "metrics": {
            "success": int(route_complete),
            "collisions": int(collision_event),
            "ped_collision_count": ped_collision_count,
        },
    }


def _inputs() -> dict[str, Any]:
    goal = [_row(scenario, seed) for scenario in SCENARIOS for seed in SEEDS]
    stationary = [
        _row(
            scenario,
            seed,
            arm="stationary",
            population_mode=None,
            ped_collision_count=1.0 if (scenario, seed) == ("open", 111) else 0.0,
        )
        for scenario in SCENARIOS
        for seed in SEEDS
    ]
    social = [
        _row(scenario, seed, arm="aware", execution_mode="adapter")
        for scenario in SCENARIOS
        for seed in SEEDS
    ]
    return {
        "release_id": "s30-h600-test",
        "scenario_ids": SCENARIOS.copy(),
        "seeds": SEEDS.copy(),
        "rows_by_arm": {"goal": goal, "stationary": stationary, "social": social},
        "goal_key": "goal",
        "stationary_key": "stationary",
        "aware_keys": ["social"],
        "probe_scenario_ids": ["probe"],
        "thresholds": {**THRESHOLDS, "max_stationary_contact_rate": 0.25},
        "source_sha": SOURCE_SHA,
        "expected_algorithms_by_arm": {
            "goal": "goal",
            "stationary": "stand_still",
            "social": "social_force",
        },
    }


def test_complete_native_matrix_passes_and_reports_stationary_contact_rate() -> None:
    report = evaluate_oracles(**_inputs())

    assert report["gate"]["status"] == "pass"
    assert report["gate"]["reasons"] == []
    assert report["stationary_contact"] == {
        "arm": "stationary",
        "contact_episode_count": 1,
        "denominator": 4,
        "expected_denominator": 4,
        "excluded_no_pedestrian_scenario_ids": [],
        "rate": 0.25,
    }
    assert report["coverage"]["social"]["execution_mode_counts"] == {"adapter": 4}
    json.dumps(report, allow_nan=False)


def test_probe_failures_are_listed_but_exempt_only_from_goal_failure_threshold() -> None:
    inputs = _inputs()
    inputs["rows_by_arm"]["goal"] = [
        _row(scenario, seed, success=not (scenario == "probe" and seed == 111))
        for scenario in SCENARIOS
        for seed in SEEDS
    ]

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "pass"
    assert report["counts"]["goal_failure_count"] == 1
    assert report["counts"]["non_probe_goal_failure_count"] == 0
    assert report["goal_failures"][0]["probe_exempt"] is True
    assert report["goal_failures"][0]["counted_for_threshold"] is False
    assert "probe" in render_markdown(report)


def test_non_probe_goal_failure_and_paired_aware_regression_fail_gate() -> None:
    inputs = _inputs()
    inputs["rows_by_arm"]["goal"] = [
        _row(scenario, seed, success=not (scenario == "open" and seed == 111))
        for scenario in SCENARIOS
        for seed in SEEDS
    ]
    inputs["rows_by_arm"]["social"] = [
        _row(
            scenario,
            seed,
            arm="aware",
            execution_mode="adapter",
            success=not (scenario == "probe" and seed == 112),
        )
        for scenario in SCENARIOS
        for seed in SEEDS
    ]

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    assert report["counts"]["non_probe_goal_failure_count"] == 1
    assert report["dominance_regressions"] == [
        {
            "scenario_id": "probe",
            "seed": 112,
            "goal_arm": "goal",
            "aware_arm": "social",
            "goal_status": "success",
            "aware_status": "failure",
            "goal_route_complete": True,
            "aware_route_complete": False,
        }
    ]
    markdown = render_markdown(report)
    assert "## Goal failures" in markdown
    assert "## Dominance regressions" in markdown
    assert "| social | probe | 112 | success | failure |" in markdown


def test_missing_duplicate_and_non_native_rows_block_and_remain_visible() -> None:
    inputs = _inputs()
    rows = inputs["rows_by_arm"]["goal"]
    rows.pop()
    rows.append({**rows[0], "execution_mode": "fallback"})
    rows[0] = {**rows[0], "scenario_id": "outside"}
    inputs["rows_by_arm"]["stationary"].append(inputs["rows_by_arm"]["stationary"][0].copy())

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    assert report["coverage"]["goal"]["missing_cells"]
    assert report["coverage"]["goal"]["unexpected_rows"]
    assert report["coverage"]["goal"]["invalid_cells"]
    assert report["coverage"]["stationary"]["duplicate_cells"]
    assert any(reason.startswith("goal is missing") for reason in report["gate"]["reasons"])


def test_declared_stationary_no_pedestrian_scenario_is_covered_but_excluded_from_rate() -> None:
    inputs = _inputs()
    no_pedestrian_scenario = "classic_bottleneck_low"
    inputs["scenario_ids"].append(no_pedestrian_scenario)
    for arm_key in ("goal", "stationary", "social"):
        role = (
            "stationary" if arm_key == "stationary" else "aware" if arm_key == "social" else "goal"
        )
        for seed in SEEDS:
            row = _row(
                no_pedestrian_scenario,
                seed,
                arm=role,
                population_mode=None if role == "stationary" else "pedestrian_free_v1",
            )
            if role == "stationary":
                row["scenario_params"]["simulation_config"]["population_size"] = 0
                row["scenario_params"]["simulation_config"]["instantiated_population_size"] = 0
                row["algorithm_metadata"]["reference_population"][
                    "instantiated_population_size"
                ] = 0
                row["interaction_exposure"]["interaction_exposure_status"] = (
                    "not_derivable_no_pedestrians"
                )
            inputs["rows_by_arm"][arm_key].append(row)
    inputs["stationary_no_pedestrian_scenario_ids"] = [no_pedestrian_scenario]

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "pass"
    assert report["coverage"]["stationary"]["expected_cell_count"] == 6
    assert report["stationary_contact"]["denominator"] == 4
    assert report["stationary_contact"]["excluded_no_pedestrian_scenario_ids"] == [
        no_pedestrian_scenario
    ]
    assert report["stationary_no_pedestrian_scenarios"]["excluded_cell_count"] == 2


def test_undeclared_stationary_no_pedestrian_status_and_wrong_algo_block() -> None:
    inputs = _inputs()
    inputs["rows_by_arm"]["stationary"][0]["interaction_exposure"][
        "interaction_exposure_status"
    ] = "not_derivable_no_pedestrians"
    inputs["rows_by_arm"]["social"][0]["algo"] = "goal"

    report = evaluate_oracles(**inputs)

    messages = [violation["message"] for violation in report["violations"]]
    assert report["gate"]["status"] == "fail"
    assert any("interaction_exposure_status=computed" in message for message in messages)
    assert any("does not match configured 'social_force'" in message for message in messages)


def test_population_and_source_contracts_block_invalid_rows() -> None:
    inputs = _inputs()
    inputs["rows_by_arm"]["goal"][0]["scenario_params"]["reference_population_mode"] = "original"
    inputs["rows_by_arm"]["goal"][1]["scenario_params"]["simulation_config"][
        "instantiated_population_size"
    ] = 1
    inputs["rows_by_arm"]["goal"][2]["git_hash"] = "b" * 40
    inputs["rows_by_arm"]["goal"][2]["provenance"]["git_hash"] = "b" * 40
    inputs["rows_by_arm"]["stationary"][0]["interaction_exposure"][
        "interaction_exposure_status"
    ] = "not_derivable_no_pedestrians"
    del inputs["rows_by_arm"]["stationary"][1]["metrics"]["ped_collision_count"]

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    messages = [violation["message"] for violation in report["violations"]]
    assert any("reference_population_mode" in message for message in messages)
    assert any("instantiated_population_size" in message for message in messages)
    assert any("does not match source_sha" in message for message in messages)
    assert any("interaction_exposure_status" in message for message in messages)
    assert any("ped_collision_count" in message for message in messages)


def test_malformed_threshold_unknown_probe_and_missing_source_fail_closed() -> None:
    inputs = _inputs()
    inputs["thresholds"]["max_stationary_contact_rate"] = float("nan")
    inputs["probe_scenario_ids"] = ["not-declared"]
    inputs["source_sha"] = "short"

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    assert {violation["code"] for violation in report["violations"]} >= {
        "threshold_invalid",
        "unknown_probe_scenario",
        "source_sha_invalid",
    }
    json.dumps(report, allow_nan=False)


def test_invalid_matrix_and_arm_configuration_cannot_pass() -> None:
    cases: list[tuple[str, Any, str]] = [
        ("release_id", "", "release_id_invalid"),
        ("scenario_ids", [], "scenario_ids_empty"),
        ("scenario_ids", ["open", "open"], "scenario_ids_duplicate"),
        ("seeds", [], "seeds_empty"),
        ("seeds", [111, 111], "seeds_duplicate"),
        ("probe_scenario_ids", ["probe", "probe"], "probe_scenario_ids_duplicate"),
        (
            "stationary_no_pedestrian_scenario_ids",
            ["outside"],
            "unknown_stationary_no_pedestrian_scenario",
        ),
        ("goal_key", "", "goal_key_invalid"),
        ("stationary_key", "", "stationary_key_invalid"),
        ("aware_keys", ["social", "social"], "aware_keys_duplicate"),
        ("rows_by_arm", {"unexpected": []}, "unconfigured_arm_rows"),
        ("expected_algorithms_by_arm", None, "expected_algorithms_missing"),
    ]
    for field, value, expected_code in cases:
        inputs = _inputs()
        inputs[field] = value

        report = evaluate_oracles(**inputs)

        assert report["gate"]["status"] == "fail", field
        assert expected_code in {item["code"] for item in report["violations"]}, field


def test_malformed_and_out_of_matrix_episode_rows_fail_closed() -> None:
    inputs = _inputs()
    rows = inputs["rows_by_arm"]["goal"]
    rows[0] = None
    rows[1] = {"scenario_id": "open", "seed": 112}
    rows.append(_row("open", 999))

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    assert report["coverage"]["goal"]["malformed_rows"]
    assert report["coverage"]["goal"]["invalid_cells"]
    assert report["coverage"]["goal"]["unexpected_rows"]
    assert {item["code"] for item in report["violations"]} >= {
        "row_invalid",
        "missing_cell",
        "unexpected_row",
    }


@pytest.mark.parametrize("missing", ["version", "termination_reason", "integrity"])
def test_saved_row_missing_canonical_episode_field_blocks(missing: str) -> None:
    inputs = _inputs()
    del inputs["rows_by_arm"]["goal"][0][missing]

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    assert any(
        "canonical episode schema" in error
        for cell in report["coverage"]["goal"]["invalid_cells"]
        for error in cell["errors"]
    )


def test_contradictory_terminal_events_and_error_row_block() -> None:
    inputs = _inputs()
    goal_rows = inputs["rows_by_arm"]["goal"]
    goal_rows[0]["outcome"]["collision_event"] = True
    goal_rows[1]["termination_reason"] = "error"
    goal_rows[2]["integrity"]["contradictions"] = ["unresolved collision"]

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    invalid = report["coverage"]["goal"]["invalid_cells"]
    assert len(invalid) == 3
    assert any("exactly one terminal event" in error for error in invalid[0]["errors"])
    assert any("termination_reason must be" in error for error in invalid[1]["errors"])
    assert any("integrity.contradictions must be empty" in error for error in invalid[2]["errors"])


def test_stationary_pedestrian_contact_requires_collision_outcome() -> None:
    inputs = _inputs()
    contact_row = inputs["rows_by_arm"]["stationary"][0]
    contact_row["outcome"] = {
        "route_complete": False,
        "collision_event": False,
        "timeout_event": True,
    }
    contact_row["termination_reason"] = "max_steps"
    contact_row["status"] = "failure"
    contact_row["metrics"]["collisions"] = 0

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    assert report["stationary_contact"]["denominator"] == 3
    assert any(
        "pedestrian contact requires outcome.collision_event=true" in error
        for error in report["coverage"]["stationary"]["invalid_cells"][0]["errors"]
    )


def test_stationary_population_capture_marker_and_runtime_telemetry_are_required() -> None:
    inputs = _inputs()
    marker_row, telemetry_row = inputs["rows_by_arm"]["stationary"][:2]
    del marker_row["scenario_params"]["reference_population_capture_version"]
    del telemetry_row["algorithm_metadata"]["reference_population"]

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    invalid = report["coverage"]["stationary"]["invalid_cells"]
    assert len(invalid) == 2
    assert any("reference_population_capture_version=v1" in error for error in invalid[0]["errors"])
    assert any(
        "algorithm_metadata.reference_population v1" in error for error in invalid[1]["errors"]
    )


@pytest.mark.parametrize("invalid_population", [0, 0.5])
def test_invalid_actual_stationary_population_cannot_enter_contact_denominator(
    invalid_population: float,
) -> None:
    inputs = _inputs()
    row = inputs["rows_by_arm"]["stationary"][0]
    row["scenario_params"] = deepcopy(row["scenario_params"])
    row["algorithm_metadata"]["reference_population"]["instantiated_population_size"] = (
        invalid_population
    )

    report = evaluate_oracles(**inputs)

    assert report["gate"]["status"] == "fail"
    assert report["stationary_contact"]["denominator"] == 3
    assert any(
        "instantiated_population_size must be" in error
        for error in report["coverage"]["stationary"]["invalid_cells"][0]["errors"]
    )
