"""Regression coverage for the issue #9533 guarded-PPO trace path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from robot_sf.benchmark.map_runner.map_runner import _attach_guard_decision_stats
from robot_sf.benchmark.map_runner.map_runner_episode import _step_build_planner_decision_entry


def _guard_decision(*, label: str, intervened: bool) -> dict[str, object]:
    """Build a small shield-decision payload with safety metrics."""
    return {
        "schema_version": "shield-decision.v1",
        "decision_label": label,
        "proposed_action": [0.8, 0.0],
        "filtered_action": [0.2, 0.1],
        "intervened": intervened,
        "override_applied": intervened,
        "hard_constraint_violation": False,
        "violated_constraints": [],
        "intervention_reason": "fallback_command_satisfied_short_horizon_constraints",
        "fallback_controller_state": {
            "policy": "RiskDWAPlannerAdapter",
            "planner_diagnostics": {
                "schema_version": "risk-dwa-progress-escape.v1",
                "status": "selected",
                "reason": "candidate_score_better",
                "candidate_command": [0.2, 0.1],
                "candidate_score": 1.25,
            },
        },
        "proposed_evaluation": {
            "safe": False,
            "min_ped_clear": 0.42,
            "min_obs_clear": 0.51,
            "min_ttc": 0.63,
        },
        "selected_evaluation": {
            "safe": True,
            "min_ped_clear": 0.81,
            "min_obs_clear": 0.51,
            "min_ttc": 1.04,
        },
    }


def test_guard_policy_stats_expose_latest_decision_and_keep_checkpoint_provenance() -> None:
    decision = _guard_decision(label="fallback_safe", intervened=True)
    policy = SimpleNamespace(
        _planner_stats=lambda: {"checkpoint_provenance": {"load_status": "loaded"}}
    )
    metadata = {"shield_stats": {"last_decision": decision}}

    _attach_guard_decision_stats(policy, metadata)

    stats = policy._planner_stats()
    assert stats["checkpoint_provenance"] == {"load_status": "loaded"}
    assert stats["last_decision"] == decision


def test_guard_trace_retains_evaluations_and_marks_intervention_edges() -> None:
    state = SimpleNamespace(
        goal_vec=np.asarray([10.0, 0.0]),
        initial_goal_distance=10.0,
        planner_decision_trace=[],
    )
    slc = SimpleNamespace(record_planner_decision_trace=True)

    _step_build_planner_decision_entry(
        state,
        slc,
        step_idx=0,
        sim=SimpleNamespace(
            robot_pos=np.asarray([1.0, 0.0]),
            planner_step_decision=_guard_decision(label="fallback_safe", intervened=True),
        ),
    )
    first = state.planner_decision_trace[0]
    assert first["selected_source"] == "fallback_safe"
    assert first["selected_command"] == [0.2, 0.1]
    assert first["proposed_command"] == [0.8, 0.0]
    assert first["route_progress_from_start_m"] == 1.0
    assert first["safety_guard"]["intervention_reason"] == (
        "fallback_command_satisfied_short_horizon_constraints"
    )
    assert first["safety_guard"]["proposed_evaluation"]["min_obs_clear"] == 0.51
    assert first["guard_intervened"] is True
    assert first["guard_intervention_start"] is True
    assert first["guard_intervention_end"] is False
    assert (
        first["safety_guard"]["fallback_controller_state"]["planner_diagnostics"]["status"]
        == "selected"
    )
    assert (
        first["safety_guard"]["fallback_controller_state"]["planner_diagnostics"]["candidate_score"]
        == 1.25
    )

    second_decision = _guard_decision(label="ppo_safe", intervened=False)
    second_decision["intervention_reason"] = "proposed_action_satisfies_shield"
    _step_build_planner_decision_entry(
        state,
        slc,
        step_idx=1,
        sim=SimpleNamespace(
            robot_pos=np.asarray([2.0, 0.0]),
            planner_step_decision=second_decision,
        ),
    )
    second = state.planner_decision_trace[1]
    assert second["guard_intervened"] is False
    assert second["guard_intervention_start"] is False
    assert second["guard_intervention_end"] is True


def _changed_paths(left: object, right: object, prefix: str = "") -> set[str]:
    """Return dotted paths whose config values differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        changed: set[str] = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else str(key)
            changed.update(_changed_paths(left.get(key), right.get(key), child))
        return changed
    return set() if left == right else {prefix}


def test_progress_escape_treatment_is_a_single_knob_delta() -> None:
    config_root = Path(__file__).parents[2] / "configs" / "algos"
    baseline = yaml.safe_load(
        (config_root / "guarded_ppo_camera_ready_cpu.yaml").read_text(encoding="utf-8")
    )
    treatment = yaml.safe_load(
        (config_root / "guarded_ppo_issue_9533_progress_escape.yaml").read_text(encoding="utf-8")
    )

    assert _changed_paths(baseline, treatment) == {"fallback_risk_dwa.progress_escape_enabled"}
    assert baseline["fallback_risk_dwa"]["progress_escape_enabled"] is False
    assert treatment["fallback_risk_dwa"]["progress_escape_enabled"] is True


def test_issue_9533_campaign_configs_freeze_paired_dev_eval_design() -> None:
    """The matched arm, scenario, and trace design must be identical across seed sets."""
    benchmark_root = Path(__file__).parents[2] / "configs" / "benchmarks"
    dev = yaml.safe_load(
        (benchmark_root / "issue_9533_guarded_ppo_progress_escape.yaml").read_text(encoding="utf-8")
    )
    evaluation = yaml.safe_load(
        (benchmark_root / "issue_9533_guarded_ppo_progress_escape_eval.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected_scenarios = [
        "classic_bottleneck_low",
        "classic_bottleneck_medium",
        "classic_bottleneck_high",
        "classic_cross_trap_low",
        "classic_cross_trap_medium",
        "classic_cross_trap_high",
        "classic_doorway_low",
        "classic_doorway_medium",
        "classic_doorway_high",
        "classic_merging_low",
        "classic_merging_medium",
        "classic_overtaking_low",
        "classic_overtaking_medium",
        "classic_t_intersection_low",
        "classic_t_intersection_medium",
        "classic_realworld_double_bottleneck_high",
    ]
    assert dev["seed_policy"]["seed_set"] == "dev"
    assert evaluation["seed_policy"]["seed_set"] == "eval"
    assert dev["scenario_candidates"] == expected_scenarios
    assert evaluation["scenario_candidates"] == expected_scenarios
    assert _changed_paths(dev, evaluation) == {"name", "seed_policy.seed_set"}
    assert dev["workers"] <= 8
    assert evaluation["workers"] <= 8
    assert dev["record_planner_decision_trace"] is True
    assert dev["record_simulation_step_trace"] is True
    expected_planners = [
        (
            "ppo_br06_matched_guard_off",
            "ppo",
            "configs/algos/ppo_v3_issue_9533_cpu.yaml",
        ),
        ("guarded_ppo_baseline", "guarded_ppo", "configs/algos/guarded_ppo_camera_ready_cpu.yaml"),
        (
            "guarded_ppo_progress_escape",
            "guarded_ppo",
            "configs/algos/guarded_ppo_issue_9533_progress_escape.yaml",
        ),
    ]
    actual = [(arm["key"], arm["algo"], arm["algo_config"]) for arm in dev["planners"]]
    assert actual == expected_planners
    model_ids = {
        yaml.safe_load(
            (Path(__file__).parents[2] / arm["algo_config"]).read_text(encoding="utf-8")
        )["model_id"]
        for arm in dev["planners"]
    }
    assert model_ids == {"ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200"}
    canonical_ppo = yaml.safe_load(
        (Path(__file__).parents[2] / "configs/algos/ppo_v3_camera_ready.yaml").read_text(
            encoding="utf-8"
        )
    )
    matched_cpu_ppo = yaml.safe_load(
        (Path(__file__).parents[2] / "configs/algos/ppo_v3_issue_9533_cpu.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert _changed_paths(canonical_ppo, matched_cpu_ppo) == {"device"}
    assert matched_cpu_ppo["device"] == "cpu"
    assert "paper_eval_s30" not in (
        benchmark_root / "issue_9533_guarded_ppo_progress_escape.yaml"
    ).read_text(encoding="utf-8")
