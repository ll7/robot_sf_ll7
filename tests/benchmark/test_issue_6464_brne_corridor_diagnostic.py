"""Contract tests for the bounded BRNE corridor diagnostic (#6464)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.baselines.brne import BRNE_PINNED_SHA
from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.map_runner.map_runner_observations import obs_to_brne_format
from scripts.benchmark.run_brne_corridor_diagnostic_issue_6464 import (
    _candidate_distribution_summary,
    _mechanism_table_row,
    _validate_candidate_distribution,
    _write_report,
    classify_record,
    summarize_records,
    validate_campaign_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml"


def _config() -> dict[str, Any]:
    """Load and validate the committed diagnostic contract."""
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return validate_campaign_config(payload)


def _record(
    *,
    metadata: dict[str, Any] | None = None,
    pedestrians: list[dict[str, Any]] | None = None,
    positions: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Build a small trace-backed episode fixture."""
    positions = positions or [[0.0, 4.0], [0.0, 5.0]]
    pedestrians = pedestrians or []
    steps = [
        {
            "robot": {"position": position},
            "pedestrians": pedestrians,
            "planner": {
                "selected_action": {"v": 0.0, "omega": 0.0},
                "applied_environment_action": {"v": 0.0, "omega": 0.0},
            },
        }
        for position in positions
    ]
    return {
        "episode_id": "fixture",
        "scenario_id": "classic_head_on_corridor_low",
        "seed": 111,
        "status": "success",
        "metrics": {"success": True},
        "algorithm_metadata": {
            "status": "ok",
            "simulation_step_trace": {"steps": steps},
            **(metadata or {}),
        },
    }


def _candidate_distribution() -> dict[str, Any]:
    """Return bounded candidate/weight summaries for native trace fixtures."""
    stats = {
        "min": 0.0,
        "q25": 0.0,
        "median": 0.0,
        "mean": 0.0,
        "q75": 0.0,
        "max": 0.0,
        "std": 0.0,
    }
    step = {
        "candidate_controls": {
            "v_m_s": stats.copy(),
            "omega_rad_s": stats.copy(),
        },
        "weights": stats.copy(),
        "weighted_mean": {"v_m_s": 0.0, "omega_rad_s": 0.0},
    }
    return {
        "status": "available",
        "schema_version": "brne-candidate-distribution.v1",
        "sample_count": 42,
        "plan_step_count": 25,
        "first": step,
        "second": step,
        "first_to_second": {
            "candidate_mean_delta_v_m_s": 0.0,
            "weighted_mean_delta_v_m_s": 0.0,
            "candidate_mean_delta_omega_rad_s": 0.0,
            "weighted_mean_delta_omega_rad_s": 0.0,
        },
    }


def _native_mechanism_trace() -> dict[str, Any]:
    """Return valid compact BRNE telemetry steps for classifier fixtures."""
    first_step = {
        "step": 0,
        "observation": {
            "robot_position_world_m": [0.0, 4.0],
            "robot_velocity_world_m_s": [0.0, 0.0],
            "declared_heading_rad": 0.0,
            "velocity_derived_heading_rad": None,
            "goal_position_world_m": [0.0, 8.0],
            "goal_bearing_rad": 1.5707963267948966,
            "heading_goal_angular_difference_rad": 1.5707963267948966,
            "heading_reference": "declared_heading",
        },
        "selected_pedestrians": [],
        "pedestrian_selection": {
            "observed_count": 0,
            "within_upstream_activation_radius_count": 0,
            "passed_to_brne_count": 0,
            "upstream_activation_radius_m": 3.5,
            "activation_gate_applied": False,
            "selection_mode": "nearest_up_to_maximum_agents",
        },
        "nominal_command": {
            "v_m_s": 0.4,
            "omega_rad_s": 0.0,
            "construction_mode": "straight_constant",
        },
        "adapter_input_frame": "world",
        "pre_clamp_action": {"v_m_s": 0.0, "omega_rad_s": 0.0},
        "selected_action": {"v_m_s": 0.0, "omega_rad_s": 0.0},
        "action_clipping": {
            "v_clipped": False,
            "omega_clipped": False,
            "any_clipped": False,
        },
        "action_delta": None,
        "ensemble": {
            "requested_num_samples": 49,
            "effective_num_samples": 42,
            "control_ensemble_shape": [25, 42, 2],
            "weight_shape": [8, 42],
            "aggregation_mode": "plan_step_first",
            "aggregation_formula": "mean_plan_step_first_over_samples",
            "candidate_distribution": _candidate_distribution(),
        },
        "runtime": {
            "status": "ok",
            "failure_reason": None,
            "failure_count": 0,
            "failure_reasons": [],
            "elapsed_s": 0.01,
            "budget_s": 0.1,
            "budget_exceeded": False,
        },
    }
    second_step = deepcopy(first_step)
    second_step["step"] = 1
    second_step["observation"]["robot_position_world_m"] = [0.0, 5.0]
    third_step = deepcopy(first_step)
    third_step["step"] = 2
    third_step["observation"]["robot_position_world_m"] = [0.0, 6.0]
    return {
        "schema_version": "brne-mechanism-trace.v1",
        "status": "available",
        "steps": [first_step, second_step, third_step],
    }


def test_issue_6464_config_is_frozen_and_opt_in() -> None:
    """The campaign is exact, bounded, and not silently benchmark-safe."""
    config = _config()

    assert config["scenario_ids"] == ["classic_head_on_corridor_low"]
    assert config["seeds"] == [111, 112, 113]
    assert config["max_pedestrians"] == 7
    assert config["expected_effective_num_samples"] == 42
    assert [planner["key"] for planner in config["planners"]] == [
        "brne",
        "orca",
        "social_force",
    ]

    readiness = get_algorithm_readiness("brne")
    assert readiness is not None
    assert readiness.tier == "experimental"
    assert readiness.requires_explicit_opt_in is True


def test_obs_to_brne_format_reconstructs_world_velocity() -> None:
    """Map observations retain the BRNE robot/agent contract and heading speed."""
    converted = obs_to_brne_format(
        {
            "robot": {
                "position": [1.0, 2.0],
                "heading": [1.5707963267948966],
                "speed": [2.0],
            },
            "goal": {"current": [1.0, 8.0]},
            "pedestrians": {
                "positions": [[4.0, 5.0]],
                "velocities": [[0.0, -1.0]],
                "count": [1],
            },
            "sim": {"timestep": 0.1},
            "obstacles": [{"kind": "corridor_boundary"}],
        }
    )

    assert converted["dt"] == pytest.approx(0.1)
    assert converted["robot"]["velocity"] == pytest.approx([0.0, 2.0])
    assert converted["agents"][0]["velocity"] == pytest.approx([1.0, 0.0])
    assert converted["obstacles"] == [{"kind": "corridor_boundary"}]


def test_obs_to_brne_format_unflattens_grid_enabled_map_observations() -> None:
    """The occupancy-grid map path must not collapse flat state to zero defaults."""
    converted = obs_to_brne_format(
        {
            "robot_position": [3.0, 4.0],
            "robot_velocity_xy": [0.0, 0.5],
            "robot_heading": [1.5707963267948966],
            "robot_speed": [0.5],
            "robot_radius": [1.0],
            "goal_current": [3.0, 20.0],
            "pedestrians_positions": [[3.0, 12.0], [7.0, 10.0]],
            "pedestrians_velocities": [[0.0, -0.3], [0.0, -0.2]],
            "pedestrians_count": [2],
            "pedestrians_radius": [0.35],
            "sim_timestep": [0.1],
        }
    )

    assert converted["robot"]["position"] == [3.0, 4.0]
    assert converted["robot"]["goal"] == [3.0, 20.0]
    assert converted["robot"]["velocity"] == [0.0, 0.5]
    assert len(converted["agents"]) == 2


def test_comparators_do_not_require_brne_dependency_metadata() -> None:
    """Standard comparator adapters remain usable without BRNE-only metadata."""
    classified = classify_record(_record(), _config(), planner_key="social_force")

    assert classified["execution_ok"] is True
    assert classified["native"] is False
    assert classified["status"] == "available_comparator"


def test_brne_requires_native_dependency_status() -> None:
    """BRNE rows without a successful staged-core status are unavailable."""
    config = _config()
    missing = classify_record(_record(), config, planner_key="brne")
    assert missing["execution_ok"] is False
    assert missing["native"] is False
    assert missing["status"] == "unavailable"

    native_metadata = {
        "brne_diagnostic": {
            "status": "native_core_via_adapter",
            "execution_semantics": "native_upstream_core_through_robot_sf_adapter",
        },
        "planner_metadata": {"status": "ok"},
        "planner_kinematics": {
            "execution_mode": "adapter",
            "adapter_active": True,
            "adapter_name": "BRNEPlanner",
            "supports_native_commands": True,
            "supports_adapter_commands": True,
            "planner_command_space": "unicycle_vw",
        },
        "planner_runtime": {
            "planner_metadata": {
                "status": "ok",
                "runtime_status": "ok",
                "failure_count": 0,
                "source_commit": BRNE_PINNED_SHA,
                "source_pin": BRNE_PINNED_SHA,
                "source_integrity": "clean_pinned_worktree",
                "effective_num_samples": 42,
                "step_count": 1,
                "mechanism_trace": _native_mechanism_trace(),
            }
        },
    }
    native = classify_record(
        _record(
            metadata=native_metadata,
            positions=[[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]],
        ),
        config,
        planner_key="brne",
    )
    assert native["native"] is True
    assert native["status"] == "available_native"
    mechanism_row = _mechanism_table_row(
        _record(
            metadata=native_metadata,
            positions=[[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]],
        ),
        native,
        planner_key="brne",
    )
    assert mechanism_row["trace_status"] == "available"
    assert mechanism_row["goal"]["signed_progress_by_phase"]
    assert mechanism_row["aggregation"]["modes"] == ["plan_step_first"]
    assert mechanism_row["aggregation"]["candidate_distribution"]["status"] == "available"
    assert mechanism_row["pre_clamp_action"]["available"] is True
    assert mechanism_row["selected_post_clamp_command"]["available"] is True
    assert mechanism_row["nominal_command"]["construction_modes"] == ["straight_constant"]
    assert mechanism_row["pedestrian_selection"]["first"]["passed_to_brne_count"] == 0
    assert mechanism_row["applied_environment_command"]["available"] is True
    assert mechanism_row["action_clipping"]["clipped_steps"] == 0

    wrong_sample_count = classify_record(
        _record(
            metadata={
                **native_metadata,
                "planner_runtime": {
                    "planner_metadata": {
                        "status": "ok",
                        "runtime_status": "ok",
                        "failure_count": 0,
                        "source_commit": BRNE_PINNED_SHA,
                        "source_pin": BRNE_PINNED_SHA,
                        "source_integrity": "clean_pinned_worktree",
                        "effective_num_samples": 41,
                    }
                },
            }
        ),
        config,
        planner_key="brne",
    )
    assert wrong_sample_count["status"] == "unavailable"


def test_brne_runtime_failure_is_unavailable_even_with_motion() -> None:
    """Fail-closed zero-motion/runtime failures cannot become native evidence."""
    config = _config()
    failed = classify_record(
        _record(
            metadata={
                "brne_diagnostic": {
                    "status": "native_core_via_adapter",
                    "execution_semantics": "native_upstream_core_through_robot_sf_adapter",
                },
                "planner_metadata": {"status": "ok"},
                "planner_kinematics": {
                    "execution_mode": "adapter",
                    "adapter_active": True,
                    "adapter_name": "BRNEPlanner",
                    "supports_native_commands": True,
                    "supports_adapter_commands": True,
                    "planner_command_space": "unicycle_vw",
                },
                "planner_runtime": {
                    "planner_metadata": {
                        "status": "ok",
                        "runtime_status": "failed",
                        "failure_count": 1,
                        "effective_num_samples": 42,
                    }
                },
            }
        ),
        config,
        planner_key="brne",
    )
    assert failed["execution_ok"] is False
    assert failed["status"] == "unavailable"


def test_mechanism_trace_must_be_complete_for_native_rows() -> None:
    """Malformed compact telemetry cannot be promoted to native evidence."""
    config = _config()
    native_metadata = {
        "brne_diagnostic": {
            "status": "native_core_via_adapter",
            "execution_semantics": "native_upstream_core_through_robot_sf_adapter",
        },
        "planner_metadata": {"status": "ok"},
        "planner_kinematics": {
            "execution_mode": "adapter",
            "adapter_active": True,
            "adapter_name": "BRNEPlanner",
            "supports_native_commands": True,
            "supports_adapter_commands": True,
            "planner_command_space": "unicycle_vw",
        },
        "planner_runtime": {
            "planner_metadata": {
                "status": "ok",
                "runtime_status": "ok",
                "failure_count": 0,
                "source_commit": BRNE_PINNED_SHA,
                "source_pin": BRNE_PINNED_SHA,
                "source_integrity": "clean_pinned_worktree",
                "effective_num_samples": 42,
                "mechanism_trace": _native_mechanism_trace(),
            }
        },
    }
    native_metadata["planner_runtime"]["planner_metadata"]["mechanism_trace"]["steps"][0][
        "runtime"
    ]["status"] = "malformed"
    classified = classify_record(
        _record(
            metadata=native_metadata,
            positions=[[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]],
        ),
        config,
        planner_key="brne",
    )
    assert classified["mechanism_trace_valid"] is False
    assert classified["mechanism_trace_status"].startswith("malformed:")
    assert classified["status"] == "unavailable"


def test_report_writer_marks_missing_mechanism_cells_explicitly(tmp_path: Path) -> None:
    """The report must not render unavailable mechanism values as literal nulls."""
    report = {
        "status": "diagnostic_incomplete",
        "config": {
            "scenario_matrix": "configs/scenarios/example.yaml",
            "claim_boundary": "diagnostic only",
        },
        "expected_pairs": 1,
        "arms": [
            {
                "planner": "orca",
                "status": "available",
                "observed_rows": 1,
                "pair_coverage_exact": True,
                "native_rows": 0,
                "diagnostic_eligible_rows": 1,
                "goal_reached_rows": 1,
                "nondegenerate_rows": 1,
                "corridor_violation_rows": 0,
            }
        ],
        "mechanism_table": [
            {
                "planner": "orca",
                "seed": 111,
                "status": "available_comparator",
                "runtime": {"status": "not_applicable", "failure_count": 0},
                "goal": {
                    "initial_distance_m": None,
                    "final_distance_m": None,
                    "signed_progress_by_phase": None,
                },
                "motion": {"displacement_m": 1.0},
                "interaction_zone": {"exposure_share": 0.0},
                "clearance": {"min_clearance_m": 1.0},
                "events": {"collision_step": None, "goal_step": None},
            }
        ],
    }
    _, markdown_path = _write_report(report, tmp_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "goal start -> end (m)" in markdown
    assert "pre-clamp action (v/omega)" in markdown
    assert "not available" in markdown
    assert "| None |" not in markdown


def test_native_mechanism_trace_requires_pre_clamp_action() -> None:
    """Native successful steps must expose the weighted command before safety clipping."""
    config = _config()
    trace = _native_mechanism_trace()
    trace["steps"][0].pop("pre_clamp_action")
    metadata = {
        "planner_runtime": {
            "planner_metadata": {
                "status": "ok",
                "runtime_status": "ok",
                "failure_count": 0,
                "source_commit": BRNE_PINNED_SHA,
                "source_pin": BRNE_PINNED_SHA,
                "source_integrity": "clean_pinned_worktree",
                "effective_num_samples": 42,
                "mechanism_trace": trace,
            }
        },
    }
    classified = classify_record(
        _record(metadata=metadata, positions=[[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]]),
        config,
        planner_key="brne",
    )
    assert classified["mechanism_trace_valid"] is False
    assert "pre-clamp action" in classified["mechanism_trace_status"]


def test_native_mechanism_trace_requires_controller_parity_fields() -> None:
    """Native rows fail closed when nominal or pedestrian-selection telemetry is absent."""
    config = _config()
    trace = _native_mechanism_trace()
    trace["steps"][0].pop("nominal_command")
    metadata = {
        "planner_runtime": {
            "planner_metadata": {
                "status": "ok",
                "runtime_status": "ok",
                "failure_count": 0,
                "source_commit": BRNE_PINNED_SHA,
                "source_pin": BRNE_PINNED_SHA,
                "source_integrity": "clean_pinned_worktree",
                "effective_num_samples": 42,
                "mechanism_trace": trace,
            }
        }
    }
    classified = classify_record(
        _record(metadata=metadata, positions=[[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]]),
        config,
        planner_key="brne",
    )
    assert classified["mechanism_trace_valid"] is False
    assert "nominal command" in classified["mechanism_trace_status"]

    trace = _native_mechanism_trace()
    trace["steps"][0].pop("pedestrian_selection")
    metadata["planner_runtime"]["planner_metadata"]["mechanism_trace"] = trace
    classified = classify_record(
        _record(metadata=metadata, positions=[[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]]),
        config,
        planner_key="brne",
    )
    assert classified["mechanism_trace_valid"] is False
    assert "pedestrian selection" in classified["mechanism_trace_status"]

    metadata["planner_runtime"]["planner_metadata"]["mechanism_trace"] = _native_mechanism_trace()
    record = _record(metadata=metadata, positions=[[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]])
    for simulation_step in record["algorithm_metadata"]["simulation_step_trace"]["steps"]:
        simulation_step["planner"].pop("applied_environment_action")
    classified = classify_record(record, config, planner_key="brne")
    assert classified["mechanism_trace_valid"] is False
    assert "applied environment action" in classified["mechanism_trace_status"]


def test_native_mechanism_trace_requires_candidate_distribution() -> None:
    """Native mechanism rows must expose bounded candidate/weight summaries."""
    config = _config()
    trace = _native_mechanism_trace()
    trace["steps"][0]["ensemble"].pop("candidate_distribution")
    metadata = {
        "planner_runtime": {
            "planner_metadata": {
                "status": "ok",
                "runtime_status": "ok",
                "failure_count": 0,
                "source_commit": BRNE_PINNED_SHA,
                "source_pin": BRNE_PINNED_SHA,
                "source_integrity": "clean_pinned_worktree",
                "effective_num_samples": 42,
                "mechanism_trace": trace,
            }
        }
    }
    classified = classify_record(
        _record(metadata=metadata, positions=[[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]]),
        config,
        planner_key="brne",
    )
    assert classified["mechanism_trace_valid"] is False
    assert "candidate distribution" in classified["mechanism_trace_status"]


def test_available_candidate_distribution_requires_complete_finite_fields() -> None:
    """Available summaries must not pass validation with omitted telemetry."""
    candidate = _candidate_distribution()
    candidate["first"]["weights"].pop("mean")

    with pytest.raises(ValueError, match="first.weights.mean"):
        _validate_candidate_distribution(candidate, step_index=0)


def test_candidate_summary_keeps_plan_step_and_observation_transitions_separate() -> None:
    """The plan-step second summary must come from the first observation."""
    first_distribution = _candidate_distribution()
    second_distribution = deepcopy(_candidate_distribution())
    first_distribution["second"]["weighted_mean"]["v_m_s"] = 1.0
    second_distribution["second"]["weighted_mean"]["v_m_s"] = 2.0

    summary = _candidate_distribution_summary(
        [
            {"candidate_distribution": first_distribution},
            {"candidate_distribution": second_distribution},
        ]
    )

    assert summary["second"]["weighted_mean"]["v_m_s"] == 1.0
    assert summary["observation_step_transition"]["to"] == second_distribution["first"]


def test_summary_requires_unique_exact_pairs_and_excludes_unavailable_goals() -> None:
    """Coverage and outcomes must use exact pairs and eligible rows only."""
    config = _config()
    first = _record()
    first["seed"] = 111
    unavailable_goal = _record(metadata={"fallback_triggered": True})
    unavailable_goal["seed"] = 112
    duplicate = _record()
    duplicate["seed"] = 111

    summary = summarize_records(
        planner_key="social_force",
        records=[first, unavailable_goal, duplicate],
        config=config,
    )

    assert summary["pair_coverage_exact"] is False
    assert summary["duplicate_pairs"] == [["classic_head_on_corridor_low", 111]]
    assert summary["missing_pairs"] == [["classic_head_on_corridor_low", 113]]
    assert summary["goal_reached_rows"] == 2
    assert summary["goal_reached_unavailable_rows"] == 1


def test_corridor_uses_approved_direct_feasible_band() -> None:
    """The approved y-band is not silently shrunk by the robot radius."""
    config = _config()
    classified = classify_record(
        _record(positions=[[0.0, 3.0], [0.0, 37.0]]),
        config,
        planner_key="social_force",
    )

    assert classified["corridor_valid"] is True
    assert classified["corridor_violation_count"] == 0


def test_campaign_rejects_mutated_frozen_inputs() -> None:
    """The diagnostic cannot silently change its predeclared horizon or seed set."""
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["seeds"] = [111, 112, 114]
    with pytest.raises(ValueError, match="seeds are frozen"):
        validate_campaign_config(payload)


def test_fallback_and_over_cap_rows_are_unavailable() -> None:
    """Fallback/degraded execution and unsupported crowds cannot count as evidence."""
    config = _config()
    fallback = classify_record(
        _record(metadata={"fallback_triggered": True}), config, planner_key="social_force"
    )
    assert fallback["status"] == "unavailable"
    assert fallback["fallback_or_degraded"] is True

    crowd = [{"position": [float(idx), 4.0]} for idx in range(8)]
    over_cap = classify_record(_record(pedestrians=crowd), config, planner_key="social_force")
    assert over_cap["crowd_within_budget"] is False
    assert over_cap["status"] == "unavailable"
