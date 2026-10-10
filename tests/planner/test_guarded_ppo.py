"""Tests for guarded PPO safety and shared planner adapter contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pysocialforce.config import SOCIAL_FORCE_KERNEL_WRAPPED_V2

from robot_sf.benchmark.map_runner.map_runner_native_command import (
    NativeCommandStepError,
    _parse_response,
)
from robot_sf.benchmark.map_runner_policies.map_runner_actions import policy_command_to_env_action
from robot_sf.benchmark.runner import _NativeCommandPolicy
from robot_sf.planner import socnav_sampling_v2 as sampling
from robot_sf.planner.goal_target import select_goal_target
from robot_sf.planner.guarded_ppo import (
    GuardedPPOAdapter,
    GuardedPPOConfig,
    build_guarded_ppo_config,
    build_guarded_ppo_fallback,
    build_guarded_ppo_prior,
)
from robot_sf.planner.socnav_base import SamplingPlannerAdapter, SocNavPlannerConfig
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter
from robot_sf.planner.socnav_sacadrl import SACADRLPlannerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings


def _obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    goal=(2.0, 0.0),
    next_goal=None,
    ped_positions=None,
    ped_velocities=None,
    ped_count=None,
) -> dict[str, object]:
    """Build the minimal observation payload consumed by the guard tests."""
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    next_goal = goal if next_goal is None else next_goal
    ped_count = len(ped_positions) if ped_count is None else ped_count
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([0.2], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(next_goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([ped_count], dtype=float),
        },
    }


def _flat_obs(*, heading: float, pedestrian_velocity: tuple[float, float]) -> dict[str, object]:
    """Build a flat map-runner observation using ego-frame pedestrian velocity."""
    return {
        "robot_position": np.asarray([0.0, 0.0], dtype=float),
        "robot_heading": np.asarray([heading], dtype=float),
        "robot_speed": np.asarray([0.0], dtype=float),
        "goal_current": np.asarray([4.0, 0.0], dtype=float),
        "goal_next": np.asarray([4.0, 0.0], dtype=float),
        "pedestrians_positions": np.asarray([[1.0, 0.5]], dtype=float),
        "pedestrians_velocities": np.asarray([pedestrian_velocity], dtype=float),
        "pedestrians_count": np.asarray([1.0], dtype=float),
    }


class _FallbackAdapter:
    """Planner adapter stub that returns a fixed fallback command."""

    def __init__(self, command: tuple[float, float]) -> None:
        self.command = command
        self.plan_calls = 0

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        """Return the configured command and count the request."""
        del observation
        self.plan_calls += 1
        return self.command


class _PriorAdapter(_FallbackAdapter):
    """Marker subclass used when the guard distinguishes prior adapters."""


class _LifecycleAdapter(_FallbackAdapter):
    """Adapter stub that records lifecycle hook propagation."""

    def __init__(self, command: tuple[float, float]) -> None:
        super().__init__(command)
        self.bound_envs: list[object] = []
        self.reset_seeds: list[int | None] = []
        self.closed = False

    def bind_env(self, env: object) -> None:
        """Record bound environments for propagation assertions."""
        self.bound_envs.append(env)

    def reset(self, *, seed: int | None = None) -> None:
        """Record reset seeds for propagation assertions."""
        self.reset_seeds.append(seed)

    def close(self) -> None:
        """Record that the adapter was closed."""
        self.closed = True


def test_guarded_ppo_keeps_safe_ppo_command() -> None:
    """Safe PPO commands should pass through unchanged."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 2.5}),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )
    command, decision = guard.choose_command(_obs(ped_positions=[(2.0, 1.0)]), (0.4, 0.0))
    assert command == (0.4, 0.0)
    assert decision in {"ppo_clear", "ppo_safe"}


def test_guarded_ppo_uses_fallback_when_ppo_is_unsafe() -> None:
    """Unsafe PPO commands should be replaced by a safe fallback when available."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "guard_near_field_distance": 2.5,
                "guard_hard_ped_clearance": 0.45,
                "guard_first_step_ped_clearance": 0.55,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )
    command, decision = guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 1.0)
    assert decision == "fallback_safe"


def test_guarded_ppo_exposes_structured_shield_decision_for_fallback() -> None:
    """Guard decisions should preserve proposed, filtered, and constraint metadata."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "guard_near_field_distance": 2.5,
                "guard_hard_ped_clearance": 0.45,
                "guard_first_step_ped_clearance": 0.55,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )

    decision = guard.choose_command_decision(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )

    assert decision.decision_label == "fallback_safe"
    assert decision.proposed_action == (0.6, 0.0)
    assert decision.filtered_action == (0.0, 1.0)
    assert decision.intervened is True
    assert "pedestrian_clearance" in decision.violated_constraints
    assert decision.prediction_source == "short_horizon_rollout"
    assert decision.fallback_controller_state["policy"] == "_FallbackAdapter"
    adaptation = decision.fallback_controller_state["action_adaptation"]
    assert adaptation["mode"] == "guard_selected_command"
    assert adaptation["raw_policy_action"] == [0.6, 0.0]
    assert adaptation["adapted_action"] == [0.0, 1.0]
    assert decision.hard_constraint_violation is False


def test_guarded_ppo_blends_safe_orca_prior_in_near_field() -> None:
    """Near-field ORCA-prior blending should apply only when it remains safe."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "guard_near_field_distance": 2.5,
                "prior_blend_weight": 0.5,
                "prior_progress_margin": 0.1,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
        prior_adapter=_PriorAdapter((0.2, 0.4)),
    )
    evaluations = iter(
        [
            {
                "safe": True,
                "progress": 0.4,
                "min_ped_clear": 0.6,
                "first_ped_clear": 0.6,
                "min_obs_clear": float("inf"),
                "min_ttc": 0.8,
            },
            {
                "safe": True,
                "progress": 0.35,
                "min_ped_clear": 0.7,
                "first_ped_clear": 0.7,
                "min_obs_clear": float("inf"),
                "min_ttc": 1.0,
            },
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]

    command, decision = guard.choose_command(
        _obs(ped_positions=[(1.0, 0.4)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )

    assert command == (0.4, 0.2)
    assert decision == "prior_blend_safe"


def test_guarded_ppo_selects_bounded_residual_over_orca_prior() -> None:
    """Residual mode should select ``ORCA + clip(PPO - ORCA)`` when it is safer."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "prior_residual_mode": True,
                "prior_residual_max_linear_delta": 0.2,
                "prior_residual_max_angular_delta": 0.3,
                "prior_near_field_only": False,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
        prior_adapter=_PriorAdapter((0.5, -0.1)),
    )
    evaluations = iter(
        [
            {
                "safe": False,
                "progress": 0.5,
                "min_ped_clear": 0.2,
                "first_ped_clear": 0.2,
                "min_obs_clear": float("inf"),
                "min_ttc": 0.4,
            },
            {
                "safe": True,
                "progress": 0.47,
                "min_ped_clear": 0.65,
                "first_ped_clear": 0.65,
                "min_obs_clear": float("inf"),
                "min_ttc": 0.8,
            },
            {
                "safe": True,
                "progress": 0.48,
                "min_ped_clear": 0.8,
                "first_ped_clear": 0.8,
                "min_obs_clear": float("inf"),
                "min_ttc": 1.2,
            },
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]

    decision = guard.choose_command_decision(_obs(), (0.8, -0.5))

    assert decision.decision_label == "prior_residual_safe"
    assert decision.filtered_action == (0.7, -0.4)
    assert decision.fallback_controller_state["policy"] == "prior_residual"
    metadata = decision.fallback_controller_state["action_adaptation"]
    assert metadata["mode"] == "prior_residual"
    assert metadata["nominal_orca_action"] == [0.5, -0.1]
    assert metadata["raw_policy_action"] == [0.8, -0.5]
    assert metadata["raw_residual_action"] == [0.30000000000000004, -0.4]
    assert metadata["bounded_residual_action"] == [0.2, -0.3]
    assert metadata["adapted_action"] == [0.7, -0.4]
    assert metadata["residual_bounds"] == {"linear": 0.2, "angular": 0.3}
    assert metadata["residual_clipped"] is True
    assert metadata["hard_guard_authoritative"] is True


def test_guarded_ppo_residual_mode_keeps_safe_ppo_passthrough() -> None:
    """Residual mode should not override PPO actions that already satisfy the guard."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {"prior_residual_mode": True, "prior_near_field_only": False}
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
        prior_adapter=_PriorAdapter((0.2, 0.0)),
    )

    decision = guard.choose_command_decision(_obs(), (0.4, 0.0))

    assert decision.decision_label == "ppo_clear"
    assert decision.filtered_action == (0.4, 0.0)
    metadata = decision.fallback_controller_state["action_adaptation"]
    assert metadata["mode"] == "direct_policy_command"
    assert metadata["adapted_action"] == [0.4, 0.0]


def test_guarded_ppo_residual_mode_keeps_safer_orca_prior() -> None:
    """Residual mode should not replace a safe ORCA prior with a less safe residual."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "prior_residual_mode": True,
                "prior_residual_max_linear_delta": 0.2,
                "prior_residual_max_angular_delta": 0.3,
                "prior_near_field_only": False,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
        prior_adapter=_PriorAdapter((0.5, -0.1)),
    )
    evaluations = iter(
        [
            {
                "safe": False,
                "progress": 0.5,
                "min_ped_clear": 0.2,
                "first_ped_clear": 0.2,
                "min_obs_clear": float("inf"),
                "min_ttc": 0.4,
            },
            {
                "safe": True,
                "progress": 0.48,
                "min_ped_clear": 0.9,
                "first_ped_clear": 0.9,
                "min_obs_clear": float("inf"),
                "min_ttc": 1.4,
            },
            {
                "safe": True,
                "progress": 0.48,
                "min_ped_clear": 0.75,
                "first_ped_clear": 0.75,
                "min_obs_clear": float("inf"),
                "min_ttc": 1.0,
            },
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]

    decision = guard.choose_command_decision(_obs(), (0.8, -0.5))

    assert decision.decision_label == "prior_safe"
    assert decision.filtered_action == (0.5, -0.1)
    adaptation = decision.fallback_controller_state["action_adaptation"]
    assert adaptation["mode"] == "guard_selected_command"
    assert adaptation["adapted_action"] == [0.5, -0.1]


def test_guarded_ppo_rejected_residual_does_not_leak_into_prior_safe_metadata() -> None:
    """Rejected residual metadata should not describe a later prior-safe selection."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "prior_residual_mode": True,
                "prior_residual_max_linear_delta": 0.0,
                "prior_residual_max_angular_delta": 0.0,
                "prior_near_field_only": False,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
        prior_adapter=_PriorAdapter((0.5, -0.1)),
    )
    evaluations = iter(
        [
            {
                "safe": False,
                "progress": 0.5,
                "min_ped_clear": 0.2,
                "first_ped_clear": 0.2,
                "min_obs_clear": float("inf"),
                "min_ttc": 0.4,
            },
            {
                "safe": True,
                "progress": 0.48,
                "min_ped_clear": 0.9,
                "first_ped_clear": 0.9,
                "min_obs_clear": float("inf"),
                "min_ttc": 1.4,
            },
            {
                "safe": True,
                "progress": 0.48,
                "min_ped_clear": 0.9,
                "first_ped_clear": 0.9,
                "min_obs_clear": float("inf"),
                "min_ttc": 1.4,
            },
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]

    decision = guard.choose_command_decision(_obs(), (0.8, -0.5))

    assert decision.decision_label == "prior_safe"
    assert decision.filtered_action == (0.5, -0.1)
    adaptation = decision.fallback_controller_state["action_adaptation"]
    assert adaptation["mode"] == "guard_selected_command"
    assert adaptation["adapted_action"] == [0.5, -0.1]


def test_guarded_ppo_residual_mode_falls_through_when_not_safe() -> None:
    """Unsafe residual proposals should fall through to prior or fallback safety handling."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {"prior_residual_mode": True, "prior_near_field_only": False}
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
        prior_adapter=_PriorAdapter((0.2, 0.0)),
    )
    evaluations = iter(
        [
            {
                "safe": False,
                "progress": 0.5,
                "min_ped_clear": 0.2,
                "first_ped_clear": 0.2,
                "min_obs_clear": float("inf"),
                "min_ttc": 0.4,
            },
            {
                "safe": True,
                "progress": 0.2,
                "min_ped_clear": 0.9,
                "first_ped_clear": 0.9,
                "min_obs_clear": float("inf"),
                "min_ttc": 1.0,
            },
            {
                "safe": False,
                "progress": 0.4,
                "min_ped_clear": 0.3,
                "first_ped_clear": 0.3,
                "min_obs_clear": float("inf"),
                "min_ttc": 0.5,
            },
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]

    command, label = guard.choose_command(_obs(), (0.8, 0.0))

    assert command == (0.2, 0.0)
    assert label == "prior_safe"


def test_guarded_ppo_residual_config_defaults_disabled_and_parses_bounds() -> None:
    """Residual mode should be opt-in and parse explicit residual bounds."""
    default_cfg = build_guarded_ppo_config({})
    residual_cfg = build_guarded_ppo_config(
        {
            "prior_residual_mode": True,
            "prior_residual_max_linear_delta": 0.12,
            "prior_residual_max_angular_delta": 0.34,
        }
    )

    assert default_cfg.prior_residual_mode is False
    assert residual_cfg.prior_residual_mode is True
    assert residual_cfg.prior_residual_max_linear_delta == 0.12
    assert residual_cfg.prior_residual_max_angular_delta == 0.34


def test_guarded_ppo_prior_blend_requires_strict_safety_improvement() -> None:
    """Blend selection should avoid equal metrics and handle infinite TTC correctly."""
    guard = GuardedPPOAdapter(config=build_guarded_ppo_config({"guard_near_field_distance": 2.5}))
    base_eval = {
        "safe": True,
        "progress": 0.4,
        "min_ped_clear": 0.6,
        "first_ped_clear": 0.6,
        "min_obs_clear": float("inf"),
        "min_ttc": float("inf"),
    }
    equal_blend_eval = dict(base_eval)
    finite_blend_eval = {**base_eval, "min_ttc": 2.0}
    infinite_improvement_eval = {
        **base_eval,
        "min_ped_clear": 0.5,
        "first_ped_clear": 0.5,
        "min_ttc": float("inf"),
    }
    finite_ppo_eval = {**base_eval, "min_ttc": 1.0}

    assert not guard._blend_is_preferred(base_eval, equal_blend_eval)
    assert not guard._blend_is_preferred(base_eval, finite_blend_eval)
    assert guard._blend_is_preferred(finite_ppo_eval, infinite_improvement_eval)


def test_guarded_ppo_uses_safe_prior_before_fallback_when_ppo_is_unsafe() -> None:
    """Unsafe PPO commands should prefer a safe configured prior over generic fallback."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 2.5}),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
        prior_adapter=_PriorAdapter((0.1, -0.5)),
    )
    evaluations = iter(
        [
            {"safe": False, "min_ped_clear": 0.2, "min_obs_clear": float("inf"), "progress": 0.0},
            {"safe": True, "min_ped_clear": 0.9},
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]

    command, decision = guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )

    assert command == (0.1, -0.5)
    assert decision == "prior_safe"


def test_guarded_ppo_near_field_only_prior_skips_clear_scenes() -> None:
    """Near-field-only priors should not replace fallback behavior in clear scenes."""
    fallback = _FallbackAdapter((0.0, 1.0))
    prior = _PriorAdapter((0.1, -0.5))
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "guard_near_field_distance": 0.5,
                "prior_near_field_only": True,
            }
        ),
        fallback_adapter=fallback,
        prior_adapter=prior,
    )
    evaluations = iter(
        [
            {"safe": False, "min_ped_clear": 0.2, "min_obs_clear": float("inf"), "progress": 0.0},
            {"safe": True, "min_ped_clear": 0.9},
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]

    command, decision = guard.choose_command(
        _obs(ped_positions=[(2.0, 2.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )

    assert command == (0.0, 1.0)
    assert decision == "fallback_safe"
    assert prior.plan_calls == 0


def test_guarded_ppo_propagates_child_adapter_lifecycle_hooks() -> None:
    """Guarded PPO should reset, bind, and close stateful child adapters."""
    fallback = _LifecycleAdapter((0.0, 1.0))
    prior = _LifecycleAdapter((0.1, -0.5))
    guard = GuardedPPOAdapter(fallback_adapter=fallback, prior_adapter=prior)
    env = object()

    guard.bind_env(env)
    guard.reset(seed=7)
    guard.close()

    assert fallback.bound_envs == [env]
    assert prior.bound_envs == [env]
    assert fallback.reset_seeds == [7]
    assert prior.reset_seeds == [7]
    assert fallback.closed
    assert prior.closed


def test_guarded_ppo_falls_back_to_stop_when_no_safe_motion_exists() -> None:
    """Guard should stop when PPO and fallback are both unsafe."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "guard_near_field_distance": 2.5,
                "guard_hard_ped_clearance": 0.45,
                "guard_first_step_ped_clearance": 0.55,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.6, 0.0)),
    )
    command, decision = guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 0.0)
    assert decision == "stop_safe"


def test_guarded_ppo_goal_and_clear_branches() -> None:
    """Guard should short-circuit for reached goals and clear near-field scenes."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"goal_tolerance": 0.3, "guard_near_field_distance": 0.5}),
        fallback_adapter=_FallbackAdapter((0.1, 0.2)),
    )
    command, decision = guard.choose_command(_obs(goal=(0.1, 0.0)), (0.3, 0.1))
    assert command == (0.0, 0.0)
    assert decision == "goal_reached"

    command, decision = guard.choose_command(_obs(ped_positions=[(2.0, 2.0)]), (0.3, 0.1))
    assert command == (0.3, 0.1)
    assert decision == "ppo_clear"


def test_guarded_ppo_tracks_current_goal_before_next_waypoint() -> None:
    """Near next waypoint should not short-circuit while the current goal remains far away."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"goal_tolerance": 0.3, "guard_near_field_distance": 0.5}),
        fallback_adapter=_FallbackAdapter((0.1, 0.2)),
    )

    command, decision = guard.choose_command(
        _obs(robot=(1.0, 1.0), goal=(5.0, 1.0), next_goal=(1.1, 1.0)),
        (0.3, 0.1),
    )

    assert command == (0.3, 0.1)
    assert decision == "ppo_clear"


@pytest.mark.parametrize(
    "profile",
    ["guarded_ppo_camera_ready_cpu_goal_v2.yaml", "guarded_ppo_release_v0_0_8.yaml"],
)
def test_guarded_ppo_outer_guard_looks_ahead_for_one_waypoint_boundary_step(profile: str) -> None:
    """The outer guard keeps its own lookahead while the v2 fallback uses current."""
    release_path = Path(__file__).parents[2] / "configs/algos" / profile
    release_config = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    guard = GuardedPPOAdapter(config=build_guarded_ppo_config(release_config))
    observation = _obs(robot=(8.0, 5.0), goal=(8.0, 5.0), next_goal=(8.0, 8.0))
    _, _, outer_target, _, _ = guard._extract_state(observation)
    np.testing.assert_array_equal(outer_target, [8.0, 8.0])
    fallback_target = select_goal_target(
        observation["robot"]["position"],
        observation["goal"]["current"],
        observation["goal"]["next"],
        version=release_config["fallback_risk_dwa"]["goal_target_version"],
    )
    np.testing.assert_array_equal(fallback_target, [8.0, 5.0])


def test_guarded_ppo_honors_array_pedestrian_count_for_padded_rows() -> None:
    """Padded zero pedestrian rows from SocNav observations should not become real blockers."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 0.5}),
        fallback_adapter=_FallbackAdapter((0.1, 0.2)),
    )

    command, decision = guard.choose_command(
        _obs(
            ped_positions=[(0.0, 0.0), (0.0, 0.0)],
            ped_velocities=[(0.0, 0.0), (0.0, 0.0)],
            ped_count=0,
        ),
        (0.3, 0.1),
    )

    assert command == (0.3, 0.1)
    assert decision == "ppo_clear"


def test_guarded_ppo_best_effort_prefers_fallback_when_clearer() -> None:
    """When nothing is safe, the guard should prefer fallback if it has more clearance."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 2.5}),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )
    evaluations = iter(
        [
            {"safe": False, "min_ped_clear": 0.2, "min_obs_clear": float("inf"), "progress": 0.0},
            {"safe": False, "min_ped_clear": 0.8, "min_obs_clear": float("inf"), "progress": 0.0},
            {"safe": False, "min_ped_clear": 0.5, "min_obs_clear": float("inf"), "progress": 0.0},
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]
    command, decision = guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 1.0)
    assert decision == "fallback_best_effort"


def test_guarded_ppo_handles_malformed_pedestrian_payloads_and_config_builders() -> None:
    """Malformed pedestrian arrays should be sanitized and builder helpers should default cleanly."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(None),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )
    robot_pos, heading, goal, ped_pos, ped_vel = guard._extract_state(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [1.0, 0.0], "next": [1.0, 0.0]},
            "pedestrians": {"positions": [1.0, 2.0, 3.0], "velocities": [0.1]},
        }
    )
    assert robot_pos.tolist() == [0.0, 0.0]
    assert heading == 0.0
    assert goal.tolist() == [1.0, 0.0]
    assert ped_pos.shape == (0, 2)
    assert ped_vel.shape == (0, 2)

    fallback = build_guarded_ppo_fallback(None)
    assert fallback is not None
    assert build_guarded_ppo_prior(None) is None


def test_guarded_ppo_orca_builder_preserves_social_force_kernel_initvar() -> None:
    """The ORCA config bridge retains the shared config's non-field kernel selector."""
    prior = build_guarded_ppo_prior(
        {
            "prior_policy": "orca",
            "prior_orca": {
                "social_force_kernel_version": SOCIAL_FORCE_KERNEL_WRAPPED_V2,
                "unknown_extension": "ignored",
            },
        }
    )

    assert isinstance(prior, ORCAPlannerAdapter)
    assert prior.config.social_force_kernel_version == SOCIAL_FORCE_KERNEL_WRAPPED_V2
    assert "unknown_extension" not in prior.config.to_dict()


def test_guarded_ppo_reshapes_flattened_pedestrian_payloads() -> None:
    """Flattened compatibility payloads should be reshaped using pedestrian count."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(None),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )
    _robot_pos, _heading, _goal, ped_pos, ped_vel = guard._extract_state(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [1.0, 0.0], "next": [1.0, 0.0]},
            "pedestrians": {
                "count": 2,
                "positions": [1.0, 2.0, 3.0, 4.0],
                "velocities": [0.1, 0.2, 0.3, 0.4],
            },
        }
    )
    assert ped_pos.shape == (2, 2)
    assert ped_vel.shape == (2, 2)
    assert ped_pos.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_guarded_ppo_trims_padded_velocities_before_world_conversion() -> None:
    """A count-limited payload retains its matching velocity instead of zeroing it."""
    guard = GuardedPPOAdapter(fallback_adapter=_FallbackAdapter((0.0, 0.0)))
    obs = _obs(
        heading=float(np.pi / 2.0),
        ped_positions=[(1.0, 0.5), (9.0, 9.0)],
        ped_velocities=[(1.25, -0.5), (8.0, 8.0)],
        ped_count=1,
    )

    _robot_pos, _heading, _goal, ped_pos, ped_vel = guard._extract_state(obs)

    assert ped_pos.shape == (1, 2)
    np.testing.assert_allclose(ped_vel, [[0.5, 1.25]], rtol=0.0, atol=1e-12)


def test_guarded_ppo_malformed_flat_velocity_is_zeroed() -> None:
    """Odd-length flattened velocities do not crash the safety rollout."""
    guard = GuardedPPOAdapter(fallback_adapter=_FallbackAdapter((0.0, 0.0)))
    obs = _obs(
        ped_positions=[(1.0, 0.5)],
        ped_velocities=[1.0, 2.0, 3.0],
        ped_count=1,
    )

    _robot_pos, _heading, _goal, ped_pos, ped_vel = guard._extract_state(obs)

    assert ped_pos.shape == (1, 2)
    np.testing.assert_array_equal(ped_vel, [[0.0, 0.0]])


@pytest.mark.parametrize("flat", [False, True], ids=["structured", "flat"])
def test_guarded_ppo_observation_rotates_pedestrian_velocity_to_world(flat: bool) -> None:
    """The world-frame safety rollout converts SOCNAV ego velocities first."""
    guard = GuardedPPOAdapter(fallback_adapter=_FallbackAdapter((0.0, 0.0)))
    heading = float(np.pi / 2.0)
    observation = (
        _flat_obs(heading=heading, pedestrian_velocity=(1.25, -0.5))
        if flat
        else _obs(
            heading=heading,
            ped_positions=[(1.0, 0.5)],
            ped_velocities=[(1.25, -0.5)],
        )
    )

    _robot_pos, _heading, _goal, _ped_pos, ped_vel = guard._extract_state(observation)

    np.testing.assert_allclose(ped_vel, np.asarray([[0.5, 1.25]]), rtol=0.0, atol=1e-12)


def test_guarded_ppo_clear_path_still_checks_obstacle_safety() -> None:
    """Clear-path PPO should not bypass obstacle safety evaluation."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 0.5}),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )
    evaluations = iter(
        [
            {"safe": False, "min_ped_clear": float("inf")},
            {"safe": True, "min_ped_clear": float("inf")},
        ]
    )
    guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]
    command, decision = guard.choose_command(_obs(ped_positions=[(2.0, 2.0)]), (0.3, 0.1))
    assert command == (0.0, 0.0)
    assert decision == "fallback_safe"


def test_guarded_ppo_obstacle_clearance_helper_branches() -> None:
    """Obstacle clearance helper should handle invalid payloads and distance queries."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {"guard_obstacle_threshold": 0.5, "guard_obstacle_search_cells": 2}
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )
    point = np.asarray([0.0, 0.0], dtype=float)

    assert guard._min_obstacle_clearance(point, {}) == float("inf")

    grid = np.zeros((1, 5, 5), dtype=float)
    meta = {"resolution": [0.5]}
    guard._extract_grid_payload = lambda observation: (grid, meta)  # type: ignore[method-assign]

    meta["channel_indices"] = [2]
    with pytest.raises(ValueError, match="static obstacle channel"):
        guard._min_obstacle_clearance(point, {})

    meta["channel_indices"] = [0]
    guard._world_to_grid = lambda point, meta, grid_shape: None  # type: ignore[method-assign]
    assert guard._min_obstacle_clearance(point, {}) == 0.0

    guard._world_to_grid = lambda point, meta, grid_shape: (2, 2)  # type: ignore[method-assign]
    grid[0, 2, 2] = 1.0
    assert guard._min_obstacle_clearance(point, {}) == 0.0

    grid.fill(0.0)
    assert guard._min_obstacle_clearance(point, {}) == float("inf")

    grid[0, 1, 4] = 1.0
    clearance = guard._min_obstacle_clearance(point, {})
    assert 1.0 < clearance < 1.2


def test_guarded_ppo_obstacle_clearance_requires_observation_without_grid_payload(
    monkeypatch,
) -> None:
    """Malformed clearance calls fail explicitly and supplied grids bypass observation."""
    guard = GuardedPPOAdapter(fallback_adapter=_FallbackAdapter((0.0, 0.0)))
    point = np.asarray([0.0, 0.0], dtype=float)

    with pytest.raises(
        ValueError,
        match="Guarded PPO obstacle clearance requires observation when grid_payload is absent",
    ):
        guard._min_obstacle_clearance(point)

    grid = np.zeros((1, 5, 5), dtype=float)
    meta = {"resolution": [0.5], "channel_indices": [0]}
    monkeypatch.setattr(guard, "_world_to_grid", lambda *_args, **_kwargs: None)
    assert guard._min_obstacle_clearance(point, grid_payload=(grid, meta)) == 0.0


def test_guarded_ppo_no_peds_and_stop_best_effort_branch() -> None:
    """No-ped scenes should pass through PPO, and unsafe tie cases should stop."""
    clear_guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 0.5}),
        fallback_adapter=_FallbackAdapter((0.2, 0.3)),
    )
    command, decision = clear_guard.choose_command(
        _obs(ped_positions=[], ped_velocities=[]), (0.4, 0.1)
    )
    assert command == (0.4, 0.1)
    assert decision == "ppo_clear"

    blocked_guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 2.5}),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )
    evaluations = iter(
        [
            {"safe": False, "min_ped_clear": 0.6, "min_obs_clear": float("inf"), "progress": 0.0},
            {"safe": False, "min_ped_clear": 0.5, "min_obs_clear": float("inf"), "progress": 0.0},
            {"safe": False, "min_ped_clear": 0.7, "min_obs_clear": float("inf"), "progress": 0.0},
        ]
    )
    blocked_guard._evaluate_command = lambda observation, command, **kwargs: next(evaluations)  # type: ignore[method-assign]
    command, decision = blocked_guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 0.0)
    assert decision == "stop_best_effort"


def test_guarded_ppo_config_rejects_nan_blend_weight() -> None:
    """A NaN blend weight must fail construction instead of reaching the decision path."""
    with pytest.raises(ValueError, match="prior_blend_weight.*finite"):
        GuardedPPOConfig(prior_blend_weight=float("nan"))


def test_guarded_ppo_config_rejects_nan_goal_tolerance() -> None:
    """A NaN goal tolerance must fail construction instead of reaching the decision path."""
    with pytest.raises(ValueError, match="goal_tolerance.*finite"):
        GuardedPPOConfig(goal_tolerance=float("nan"))


_FIELD_TO_CONFIG_KEY = {
    "rollout_dt": "guard_rollout_dt",
    "rollout_steps": "guard_rollout_steps",
    "goal_tolerance": "goal_tolerance",
    "near_field_distance": "guard_near_field_distance",
    "hard_ped_clearance": "guard_hard_ped_clearance",
    "first_step_ped_clearance": "guard_first_step_ped_clearance",
    "hard_obstacle_clearance": "guard_hard_obstacle_clearance",
    "min_ttc": "guard_min_ttc",
    "obstacle_threshold": "guard_obstacle_threshold",
    "obstacle_search_cells": "guard_obstacle_search_cells",
    "prior_blend_weight": "prior_blend_weight",
    "prior_progress_margin": "prior_progress_margin",
    "prior_residual_max_linear_delta": "prior_residual_max_linear_delta",
    "prior_residual_max_angular_delta": "prior_residual_max_angular_delta",
    "uncertainty_base_radius_m": "uncertainty_base_radius_m",
    "uncertainty_conformal_radius_m": "uncertainty_conformal_radius_m",
    "uncertainty_buffer_intrusion_threshold": "uncertainty_buffer_intrusion_threshold",
    "uncertainty_collision_probability_threshold": "uncertainty_collision_probability_threshold",
    "uncertainty_min_ttc_threshold_s": "uncertainty_min_ttc_threshold_s",
    "uncertainty_slow_down_speed_m_s": "uncertainty_slow_down_speed_m_s",
}

_FINITE_FIELDS = (
    "rollout_dt",
    "goal_tolerance",
    "near_field_distance",
    "hard_ped_clearance",
    "first_step_ped_clearance",
    "hard_obstacle_clearance",
    "min_ttc",
    "obstacle_threshold",
    "prior_blend_weight",
    "prior_progress_margin",
    "prior_residual_max_linear_delta",
    "prior_residual_max_angular_delta",
    "uncertainty_base_radius_m",
    "uncertainty_conformal_radius_m",
    "uncertainty_buffer_intrusion_threshold",
    "uncertainty_collision_probability_threshold",
    "uncertainty_min_ttc_threshold_s",
    "uncertainty_slow_down_speed_m_s",
)

_NON_FINITE_VALUES = (float("nan"), float("inf"), float("-inf"))

_POSITIVE_FIELDS = (
    "rollout_dt",
    "rollout_steps",
    "obstacle_search_cells",
    "uncertainty_base_radius_m",
)

_NON_NEGATIVE_FIELDS = (
    "goal_tolerance",
    "near_field_distance",
    "hard_ped_clearance",
    "first_step_ped_clearance",
    "hard_obstacle_clearance",
    "min_ttc",
    "prior_progress_margin",
    "prior_residual_max_linear_delta",
    "prior_residual_max_angular_delta",
    "uncertainty_conformal_radius_m",
    "uncertainty_slow_down_speed_m_s",
    "uncertainty_min_ttc_threshold_s",
)

_UNIT_INTERVAL_FIELDS = (
    "obstacle_threshold",
    "prior_blend_weight",
    "uncertainty_buffer_intrusion_threshold",
    "uncertainty_collision_probability_threshold",
)


@pytest.mark.parametrize(
    "field,value",
    [(field, value) for field in _FINITE_FIELDS for value in _NON_FINITE_VALUES],
)
def test_guarded_ppo_config_rejects_non_finite(field: str, value: float) -> None:
    """Non-finite guard parameters must fail direct construction."""
    with pytest.raises(ValueError, match=f"{field}.*finite"):
        GuardedPPOConfig(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [(field, value) for field in _FINITE_FIELDS for value in _NON_FINITE_VALUES],
)
def test_guarded_ppo_build_config_rejects_non_finite(field: str, value: float) -> None:
    """Non-finite guard parameters must fail mapping-based parsing naming the field."""
    with pytest.raises(ValueError, match=f"{field}.*finite"):
        build_guarded_ppo_config({_FIELD_TO_CONFIG_KEY[field]: value})


@pytest.mark.parametrize(
    "field,value",
    [(field, value) for field in _POSITIVE_FIELDS for value in (0, -1)],
)
def test_guarded_ppo_config_rejects_non_positive_counts(field: str, value: int) -> None:
    """Zero or negative timestep/count fields must fail direct construction."""
    with pytest.raises(ValueError, match=f"{field}.*positive"):
        GuardedPPOConfig(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [(field, value) for field in _POSITIVE_FIELDS for value in (0, -1)],
)
def test_guarded_ppo_build_config_rejects_non_positive_counts(field: str, value: int) -> None:
    """Zero or negative timestep/count fields must fail mapping-based parsing."""
    with pytest.raises(ValueError, match=f"{field}.*positive"):
        build_guarded_ppo_config({_FIELD_TO_CONFIG_KEY[field]: value})


@pytest.mark.parametrize("field", _NON_NEGATIVE_FIELDS)
def test_guarded_ppo_config_rejects_negative_clearance_like_fields(field: str) -> None:
    """Negative distance/clearance/margin/delta/speed/TTC fields must fail construction."""
    with pytest.raises(ValueError, match=f"{field}.*non-negative"):
        GuardedPPOConfig(**{field: -0.1})


@pytest.mark.parametrize("field", _NON_NEGATIVE_FIELDS)
def test_guarded_ppo_build_config_rejects_negative_clearance_like_fields(field: str) -> None:
    """Negative distance/clearance/margin/delta/speed/TTC fields must fail parsing."""
    with pytest.raises(ValueError, match=f"{field}.*non-negative"):
        build_guarded_ppo_config({_FIELD_TO_CONFIG_KEY[field]: -0.1})


@pytest.mark.parametrize(
    "field,value",
    [(field, value) for field in _UNIT_INTERVAL_FIELDS for value in (-0.01, 1.01)],
)
def test_guarded_ppo_config_rejects_out_of_unit_interval(field: str, value: float) -> None:
    """Probability and blend-weight fields outside [0.0, 1.0] must fail construction."""
    with pytest.raises(ValueError, match=rf"{field}.*\[0\.0, 1\.0\]"):
        GuardedPPOConfig(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [(field, value) for field in _UNIT_INTERVAL_FIELDS for value in (-0.01, 1.01)],
)
def test_guarded_ppo_build_config_rejects_out_of_unit_interval(field: str, value: float) -> None:
    """Probability and blend-weight fields outside [0.0, 1.0] must fail parsing."""
    with pytest.raises(ValueError, match=rf"{field}.*\[0\.0, 1\.0\]"):
        build_guarded_ppo_config({_FIELD_TO_CONFIG_KEY[field]: value})


def test_guarded_ppo_config_accepts_shipped_style_values() -> None:
    """Shipped camera-ready guard values must parse without error."""
    config = build_guarded_ppo_config(
        {
            "guard_rollout_dt": 0.2,
            "guard_rollout_steps": 6,
            "goal_tolerance": 0.25,
            "guard_near_field_distance": 2.0,
            "guard_hard_ped_clearance": 0.58,
            "guard_first_step_ped_clearance": 0.72,
            "guard_hard_obstacle_clearance": 0.30,
            "guard_min_ttc": 0.70,
            "guard_obstacle_threshold": 0.5,
            "guard_obstacle_search_cells": 12,
            "prior_blend_weight": 0.0,
            "prior_progress_margin": 0.05,
            "prior_residual_max_linear_delta": 0.25,
            "prior_residual_max_angular_delta": 0.35,
            "uncertainty_base_radius_m": 0.58,
            "uncertainty_conformal_radius_m": 0.25,
            "uncertainty_buffer_intrusion_threshold": 0.0,
            "uncertainty_collision_probability_threshold": 0.5,
            "uncertainty_slow_down_speed_m_s": 0.2,
        }
    )
    assert config.rollout_dt == 0.2
    assert config.goal_tolerance == 0.25
    assert config.obstacle_search_cells == 12


def test_guarded_ppo_config_accepts_boundary_values() -> None:
    """Zero for distance-like fields and unit-interval boundaries remain valid."""
    config = GuardedPPOConfig(
        goal_tolerance=0.0,
        near_field_distance=0.0,
        min_ttc=0.0,
        prior_blend_weight=1.0,
        obstacle_threshold=1.0,
        uncertainty_min_ttc_threshold_s=0.0,
    )
    assert config.prior_blend_weight == 1.0
    assert config.obstacle_threshold == 1.0
    assert config.goal_tolerance == 0.0


def test_surface_v2_guard_uses_body_to_body_clearance() -> None:
    """The candidate guard measures the free gap between the two physical discs."""
    guard = GuardedPPOAdapter(
        GuardedPPOConfig(
            rollout_dt=0.1,
            rollout_steps=1,
            clearance_model="surface_v2",
            robot_radius_m=1.0,
            pedestrian_radius_m=0.4,
            hard_ped_clearance=0.58,
            first_step_ped_clearance=0.72,
        )
    )
    unsafe = guard._evaluate_command(
        _obs(ped_positions=[(1.7, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.0, 0.0),
    )
    safe = guard._evaluate_command(
        _obs(ped_positions=[(2.2, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.0, 0.0),
    )

    # Include both braking steps: trapezoidal stopping displacement is 0.02 m.
    assert unsafe["min_ped_clear"] == pytest.approx(0.28)
    assert unsafe["safe"] is False
    assert safe["min_ped_clear"] == pytest.approx(0.78)
    assert safe["safe"] is True


def test_surface_v2_guard_reports_ttc_from_rollout_start() -> None:
    """TTC from each rollout sample includes the elapsed time to that sample."""
    guard = GuardedPPOAdapter(
        GuardedPPOConfig(
            rollout_dt=0.1,
            rollout_steps=1,
            clearance_model="surface_v2",
            robot_radius_m=1.0,
            pedestrian_radius_m=0.4,
            hard_ped_clearance=0.0,
            first_step_ped_clearance=0.0,
        )
    )
    result = guard._evaluate_command(
        _obs(ped_positions=[(3.0, 0.0)], ped_velocities=[(-1.0, 0.0)]),
        (0.0, 0.0),
    )
    # At t=.1, gap=3-.1-.015-1.4=1.485; relative speed=1+.1.
    assert result["min_ttc"] == pytest.approx(0.1 + 1.485 / 1.1)


def test_surface_v2_guard_requires_positive_body_radii() -> None:
    """Surface geometry cannot silently degrade to center-distance checks."""
    with pytest.raises(ValueError, match="robot_radius must be finite and positive"):
        GuardedPPOConfig(
            clearance_model="surface_v2",
            robot_radius_m=0.0,
            pedestrian_radius_m=0.4,
        )


def _adapter_residual_observation(*, speed=0.0, angular=0.0, pedestrian=None):
    """Build an unnormalised adapter observation without resetting a simulator."""
    positions = np.asarray([] if pedestrian is None else [pedestrian], dtype=float).reshape(-1, 2)
    return {
        "robot": {
            "position": np.zeros(2),
            "heading": np.zeros(1),
            "speed": np.array([speed]),
            "angular_velocity": np.array([angular]),
            "radius": np.array([1.0]),
        },
        "goal": {"current": np.array([10.0, 0.0])},
        "pedestrians": {
            "positions": positions,
            "velocities": np.zeros_like(positions),
            "count": np.array([len(positions)]),
            "radius": np.array([0.4]),
        },
        "sim": {"timestep": np.array([0.1])},
    }


def test_guard_checks_pedestrians_until_braking_finishes():
    """The reported .82 m horizon gap must include the unsafe .50 m stopping gap."""
    guard = GuardedPPOAdapter(
        GuardedPPOConfig(
            clearance_model="surface_v2", robot_radius_m=1.0, pedestrian_radius_m=0.4, min_ttc=0.0
        )
    )
    observation = _adapter_residual_observation(speed=2.0, pedestrian=(3.9, 0.0))
    result = guard._evaluate_command(observation, (0.0, 0.0))
    assert result["min_ped_clear"] == pytest.approx(0.5)
    assert not result["safe"]


@pytest.mark.parametrize("pedestrian, expected", [((1.0, 2.0), np.inf), ((1.0, 0.0), 0.42)])
def test_legacy_guard_ttc_is_first_contact_not_closest_approach(pedestrian, expected):
    """A near miss has no contact; head-on contact occurs at (1 - .58) / 1 seconds."""
    guard = GuardedPPOAdapter(GuardedPPOConfig(rollout_dt=0.1, rollout_steps=1))
    result = guard._evaluate_command(
        _adapter_residual_observation(pedestrian=pedestrian), (1.0, 0.0)
    )
    assert result["min_ttc"] == pytest.approx(expected)


def test_guard_grid_fallback_keeps_pedestrians_out_of_static_clearance():
    """Unbound geometry must read the static channel rather than combined occupancy."""
    guard = GuardedPPOAdapter()
    observation = _adapter_residual_observation()
    grid = np.zeros((4, 20, 20))
    grid[[1, 3], 10, 10] = 1.0
    meta = {
        "origin": [-1.0, -1.0],
        "resolution": [0.1],
        "size": [2.0, 2.0],
        "channel_indices": [0, 1, 2, 3],
    }
    assert np.isinf(
        guard._min_obstacle_clearance(np.zeros(2), observation, grid_payload=(grid, meta))
    )
    grid[0, 10, 10] = 1.0
    assert guard._min_obstacle_clearance(np.zeros(2), observation, grid_payload=(grid, meta)) == 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("raw_action", [False, True])
def test_sacadrl_nonfinite_scores_stop_and_record_fallback(monkeypatch, bad, raw_action):
    """Invalid model scores cannot select a full-speed action through argmax."""
    adapter = SACADRLPlannerAdapter()
    model = SimpleNamespace(
        actions=np.full((2, 2), bad) if raw_action else np.array([[1.0, -0.5], [0.5, 0.0]]),
        predict=lambda _: np.array([[0.1, 0.2]]) if raw_action else np.array([[bad, bad]]),
    )
    monkeypatch.setattr(adapter, "_ensure_model", lambda: model)
    monkeypatch.setattr(adapter, "_build_network_input", lambda _: (np.zeros(3), 1.0, 10.0))
    assert adapter.plan(_adapter_residual_observation()) == (0.0, 0.0)
    provenance = adapter.diagnostics()["checkpoint_provenance"]
    assert provenance["fallback_triggered"] is True
    assert provenance["fallback_reason"] == "nonfinite_model_output"
    model.actions = np.array([[1.0, -0.5], [0.5, 0.0]])
    model.predict = lambda _: np.array([[0.1, 0.2]])
    assert adapter.plan(_adapter_residual_observation()) == (0.5, 0.0)


def test_sacadrl_episode_reset_clears_transient_fallback_provenance(monkeypatch):
    """A nonfinite step taints only its episode, retaining checkpoint custody."""
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_metadata import (
        attach_planner_reset,
    )

    adapter = SACADRLPlannerAdapter(allow_fallback=True)
    model = SimpleNamespace(
        actions=np.array([[1.0, -0.5], [0.5, 0.0]]),
        predict=lambda _: np.array([[np.nan, np.nan]]),
    )
    monkeypatch.setattr(adapter, "_build_model", lambda: model)
    monkeypatch.setattr(adapter, "_build_network_input", lambda _: (np.zeros(3), 1.0, 10.0))
    adapter._checkpoint_provenance.update(checkpoint_sha256="checkpoint-custody")
    observation = _adapter_residual_observation()
    assert adapter.plan(observation) == (0.0, 0.0)
    model.predict = lambda _: np.array([[0.1, 0.2]])
    assert adapter.plan(observation) == (0.5, 0.0)
    assert adapter.diagnostics()["checkpoint_provenance"]["fallback_triggered"] is True

    def policy(_observation):
        return adapter.plan(_observation)

    attach_planner_reset(policy, adapter)
    # The episode runner tolerates adapters without a reset hook.
    reset = getattr(policy, "_planner_reset", None)
    if reset is not None:
        reset(seed=1001)
    assert policy(observation) == (0.5, 0.0)
    provenance = adapter.diagnostics()["checkpoint_provenance"]
    assert provenance["fallback_triggered"] is False
    assert "fallback_reason" not in provenance
    assert provenance["checkpoint_sha256"] == "checkpoint-custody"
    assert provenance["load_succeeded"] is True
    assert provenance["load_status"] == "loaded"
    assert adapter._model is model

    # A persistent load failure must still mark later episodes as degraded.
    adapter._model = None
    adapter._load_error = RuntimeError("checkpoint unavailable")
    adapter._checkpoint_provenance.update(
        load_succeeded=False,
        load_status="fallback",
        load_error="RuntimeError: checkpoint unavailable",
    )
    reset(seed=1002)
    assert adapter._ensure_model() is None
    provenance = adapter.diagnostics()["checkpoint_provenance"]
    assert provenance["fallback_triggered"] is True
    assert provenance["load_error"] == "RuntimeError: checkpoint unavailable"


@pytest.mark.parametrize("parser", ["map", "classic"])
@pytest.mark.parametrize(
    "payload",
    [
        '{"vx":0,"vy":1}',
        '{"v":1,"omega":0,"unexpected":1}',
        '{"v":1,"omega":0,"linear":2}',
    ],
)
def test_native_command_rejects_unknown_or_ambiguous_keys(parser, payload):
    """Both real parsers refuse holonomic aliases, unknown keys and conflicting pairs."""

    def parse(text):
        if parser == "map":
            return _parse_response(text)
        return _NativeCommandPolicy._parse_response(None, text)

    error = NativeCommandStepError if parser == "map" else ValueError
    with pytest.raises(error):
        parse(payload)
    finite = parse('{"v":0.5,"omega":0.25}')
    np.testing.assert_allclose(finite, [0.5, 0.25])


@pytest.mark.parametrize("rollout_dt", [0.1, 0.2])
def test_prediction_score_rolls_out_bound_drive_from_observed_velocity(monkeypatch, rollout_dt):
    """Scoring a command must use the same accelerated turning pose as the bound plant."""
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=rollout_dt))
    settings = DifferentialDriveSettings(max_linear_accel=0.4, max_angular_accel=0.3)
    drive = DifferentialDriveRobot(settings)
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[drive]))
    adapter.bind_env(env)
    config = SimpleNamespace(
        robot_config=drive.config, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    observation = _adapter_residual_observation(speed=0.4, angular=0.2)
    drive.state.velocity = (0.4, 0.2)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    expected = []
    for _ in range(3):
        for _ in range(round(rollout_dt / 0.1)):
            action = policy_command_to_env_action(env=env, config=config, command=(1.5, 0.8))
            drive.apply_action(tuple(action), 0.1)
        expected.append(drive.pos)

    def check_progress(*args, robot_traj, **kwargs):
        np.testing.assert_allclose(robot_traj, expected, atol=1e-14, rtol=0.0)
        return 0.0

    monkeypatch.setattr(adapter, "_goal_progress", check_progress)
    adapter._score_action(
        observation=observation,
        future_peds=np.zeros((0, 3, 2)),
        mask=np.zeros(0),
        v=1.5,
        w=0.8,
        steps=3,
    )


def test_prediction_sequence_rollout_uses_measured_drive_state(monkeypatch):
    """Sequence search must share the accelerated wheel odometry used by one-action scoring."""
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=0.1))
    drive = DifferentialDriveRobot(DifferentialDriveSettings(max_angular_accel=0.3))
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[drive]))
    adapter.bind_env(env)
    config = SimpleNamespace(
        robot_config=drive.config, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    drive.state.velocity = (0.4, 0.2)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    sequence = [(1.5, 0.8), (0.0, -0.5)]
    expected = []
    for command in np.repeat(sequence, 2, axis=0):
        action = policy_command_to_env_action(env=env, config=config, command=tuple(command))
        drive.apply_action(tuple(action), 0.1)
        expected.append(drive.pos)
    real_rollout = adapter._rollout_robot_sequence

    def check_rollout(**kwargs):
        positions, headings = real_rollout(**kwargs)
        np.testing.assert_allclose(positions, expected, atol=1e-14, rtol=0.0)
        return positions, headings

    monkeypatch.setattr(adapter, "_rollout_robot_sequence", check_rollout)
    adapter._score_action_sequence(
        observation=_adapter_residual_observation(speed=0.4, angular=0.2),
        future_peds=np.zeros((0, 4, 2)),
        mask=np.zeros(0),
        sequence=sequence,
        steps=4,
    )


@pytest.mark.parametrize("planner", ["guard", "sampler"])
def test_static_grid_fallback_refuses_an_unseparated_combined_channel(planner):
    """Combined-only occupancy cannot identify which cells are static obstacles."""
    observation = _adapter_residual_observation()
    observation["occupancy_grid"] = np.ones((1, 20, 20))
    observation["occupancy_grid_meta"] = {
        "origin": [-1.0, -1.0],
        "resolution": [0.1],
        "size": [2.0, 2.0],
        "channel_indices": [-1, -1, -1, 0],
    }
    guard = GuardedPPOAdapter()
    with pytest.raises(ValueError, match="static obstacle channel"):
        if planner == "guard":
            guard._min_obstacle_clearance(np.zeros(2), observation)
        else:
            sampling._ObstacleClearance(guard, observation)


@pytest.mark.parametrize("observed_speed", [-0.5, 2.0])
def test_bounded_sampler_preserves_measured_speed_and_braking_horizon(monkeypatch, observed_speed):
    """Preferred command bounds cannot erase feasible reverse or overspeed plant state."""
    config = SocNavPlannerConfig(
        socnav_sampling_version="bounded_v2",
        max_linear_speed=1.0,
        sampling_heading_candidates=1,
        occupancy_heading_sweep=0.0,
        sampling_speed_fractions=(1.0,),
        sampling_horizon_s=0.2,
        sampling_braking_envelope=True,
    )
    settings = DifferentialDriveSettings(
        max_linear_accel=0.6, max_linear_decel=0.7, max_angular_accel=0.35, allow_backwards=True
    )
    adapter = SamplingPlannerAdapter(config)
    adapter.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(robots=[DifferentialDriveRobot(settings)]),
            config=SimpleNamespace(robot_config=settings),
        )
    )
    real_rollout = sampling._rollout
    calls = []

    def check_rollout(*args, **kwargs):
        points, travelled = real_rollout(*args, **kwargs)
        drive = DifferentialDriveRobot(settings)
        drive.state.velocity = (observed_speed, 0.0)
        drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
        env = SimpleNamespace(simulator=SimpleNamespace(robots=[drive]))
        step_dt = args[6]
        conversion = SimpleNamespace(
            robot_config=settings, sim_config=SimpleNamespace(time_per_step_in_secs=step_dt)
        )
        expected = []
        for _ in points:
            action = policy_command_to_env_action(
                env=env, config=conversion, command=(args[4], 0.0)
            )
            drive.apply_action(tuple(action), step_dt)
            expected.append(drive.pos)
        np.testing.assert_allclose(points, expected, atol=1e-14, rtol=0.0)
        stopping_time = abs(observed_speed) / (0.6 if observed_speed < 0 else 0.7)
        assert len(points) * step_dt >= stopping_time
        calls.append(points)
        return points, travelled

    monkeypatch.setattr(sampling, "_rollout", check_rollout)
    command = adapter.plan(_adapter_residual_observation(speed=observed_speed))
    assert calls
    assert 0.0 <= command[0] <= 1.0


def test_bounded_sampler_binding_preserves_native_limited_reverse():
    """A bound reverse forecast must agree with native command execution."""
    settings = DifferentialDriveSettings(
        limited_reverse=True,
        max_reverse_speed=0.5,
        max_linear_accel=0.6,
        max_linear_decel=0.7,
        max_angular_accel=0.35,
    )
    drive = DifferentialDriveRobot(settings)
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(socnav_sampling_version="bounded_v2"))
    env = SimpleNamespace(
        simulator=SimpleNamespace(robots=[drive]), config=SimpleNamespace(robot_config=settings)
    )
    adapter.bind_env(env)
    forecast, _ = sampling._rollout(
        np.zeros(2),
        0.0,
        -0.5,
        0.0,
        1.0,
        1.0,
        0.1,
        (1.2, 1.0, 0.6, 0.7),
        settings=adapter._sampling_drive_settings,
    )
    drive.state.velocity = (-0.5, 0.0)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    config = SimpleNamespace(
        robot_config=settings, sim_config=SimpleNamespace(time_per_step_in_secs=0.1)
    )
    expected = []
    for _ in forecast:
        action = policy_command_to_env_action(env=env, config=config, command=(1.0, 0.0))
        drive.apply_action(tuple(action), 0.1)
        expected.append(drive.pos)
    np.testing.assert_allclose(forecast, expected, atol=1e-14, rtol=0.0)
    assert adapter._sampling_drive_settings.min_linear_speed == -0.5
    limits = adapter.diagnostics()["drive_limits"]
    assert limits["limited_reverse"] is True
    assert limits["max_reverse_speed"] == 0.5


def test_bounded_sampler_brakes_before_turning_from_measured_reverse():
    """The stop fallback cannot treat signed reverse motion as stationary."""
    adapter = SamplingPlannerAdapter(
        SocNavPlannerConfig(
            socnav_sampling_version="bounded_v2",
            sampling_speed_fractions=(0.0,),
            sampling_heading_candidates=1,
            occupancy_heading_sweep=0.0,
        )
    )
    settings = DifferentialDriveSettings(limited_reverse=True, max_reverse_speed=0.5)
    adapter.bind_env(
        SimpleNamespace(
            config=SimpleNamespace(robot_config=settings),
            simulator=SimpleNamespace(robots=[DifferentialDriveRobot(settings)]),
        )
    )
    observation = _adapter_residual_observation(speed=-0.5)
    observation["goal"]["current"] = np.array([0.0, 10.0])
    assert adapter.plan(observation) == (0.0, 0.0)
    assert adapter._last_sampling_v2["reason"] == "brake_straight"
    observation["robot"]["speed"] = np.zeros(1)
    assert adapter.plan(observation) == (0.0, 1.0)
    assert adapter._last_sampling_v2["reason"] == "turn_in_place"


def test_guard_compares_best_effort_commands_on_the_same_braking_clock():
    """A shorter stopped forecast must not outrank a better moving escape."""
    guard = GuardedPPOAdapter(
        GuardedPPOConfig(
            rollout_dt=0.1,
            rollout_steps=12,
            clearance_model="surface_v2",
            robot_radius_m=1.0,
            pedestrian_radius_m=0.4,
        ),
        fallback_adapter=_FallbackAdapter((0.55, 0.0)),
    )
    observation = _adapter_residual_observation(speed=0.5)
    observation["pedestrians"]["positions"] = np.array([[-2.4, 0.0], [1.6, 0.0]])
    observation["pedestrians"]["velocities"] = np.array([[1.2, 0.0], [0.4, 0.0]])
    observation["pedestrians"]["count"] = np.array([2])
    decision = guard.choose_command_decision(observation, (0.6, 0.0))
    command, label = decision.as_command_result()
    # The native 0.55 command covers 0.6575 m in 1.2 s, then 0.1525 m braking.
    # At 1.8 s the rear pedestrian is at -0.24 m: 0.81 + 0.24 - 1.4 = -0.35 m.
    # Braking immediately ends at 0.125 m, so its same-clock gap is -1.035 m.
    assert decision.selected_evaluation["min_ped_clear"] == pytest.approx(-0.35)
    assert command == pytest.approx((0.55, 0.0))
    assert label == "fallback_best_effort"


@pytest.mark.parametrize("indices", [[-1, 0, -1, -1], [3, 0, -1, -1]])
def test_guard_grid_refuses_missing_or_out_of_bounds_static_channel(indices):
    """A pedestrian-only or invalid static channel must not certify a wall-free path."""
    observation = _adapter_residual_observation()
    observation["occupancy_grid"] = np.ones((1, 20, 20))
    observation["occupancy_grid_meta"] = {
        "origin": [-1.0, -1.0],
        "resolution": [0.1],
        "size": [2.0, 2.0],
        "channel_indices": indices,
    }
    with pytest.raises(ValueError, match="static obstacle channel"):
        GuardedPPOAdapter()._min_obstacle_clearance(np.zeros(2), observation)


@pytest.mark.parametrize("helper", ["progress", "collision", "clearance", "ttc", "sequence"])
def test_bound_prediction_helpers_refuse_unspecified_measured_motion(helper):
    """Standalone bound forecasts must not silently substitute rest for unknown motion."""
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig())
    drive = DifferentialDriveRobot(DifferentialDriveSettings())
    adapter.bind_env(SimpleNamespace(simulator=SimpleNamespace(robots=[drive])))
    kwargs = {"future_peds": np.ones((1, 3, 2)), "mask": np.ones(1), "v": 1.0, "w": 0.0, "steps": 3}
    with pytest.raises(ValueError, match="measured-motion observation"):
        if helper == "progress":
            adapter._goal_progress({}, {"current": [2.0, 0.0]}, 1.0, 0.0, steps=3)
        elif helper == "collision":
            adapter._collision_cost(**kwargs)
        elif helper == "clearance":
            adapter._min_clearance(**kwargs)
        elif helper == "ttc":
            adapter._ttc_penalty(**kwargs)
        else:
            adapter._rollout_robot_sequence(sequence=[(1.0, 0.0)], segment_steps=3, dt=0.2)
