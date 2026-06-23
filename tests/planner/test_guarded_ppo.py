"""Tests for guarded PPO safety veto behavior."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.guarded_ppo import (
    GuardedPPOAdapter,
    build_guarded_ppo_config,
    build_guarded_ppo_fallback,
    build_guarded_ppo_prior,
)


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
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]

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
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]

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
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]

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
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]

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
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]

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
            {"safe": False, "min_ped_clear": 0.2},
            {"safe": True, "min_ped_clear": 0.9},
        ]
    )
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]

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
            {"safe": False, "min_ped_clear": 0.2},
            {"safe": True, "min_ped_clear": 0.9},
        ]
    )
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]

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
            {"safe": False, "min_ped_clear": 0.2},
            {"safe": False, "min_ped_clear": 0.8},
            {"safe": False, "min_ped_clear": 0.5},
        ]
    )
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]
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
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]
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

    guard._preferred_channel = lambda meta: 2  # type: ignore[method-assign]
    assert guard._min_obstacle_clearance(point, {}) == float("inf")

    guard._preferred_channel = lambda meta: 0  # type: ignore[method-assign]
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
            {"safe": False, "min_ped_clear": 0.6},
            {"safe": False, "min_ped_clear": 0.5},
            {"safe": False, "min_ped_clear": 0.7},
        ]
    )
    blocked_guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]
    command, decision = blocked_guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 0.0)
    assert decision == "stop_best_effort"
