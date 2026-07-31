"""Real-policy rollout for the Robot SF side of the cross-suite canary (#5842).

Issue #5842 requires that the pinned policy be *actually executed* on the Robot SF
path rather than a trajectory synthesized from a seed. This module rolls out the
pinned SocialForce policy headlessly (CPU, no GUI, no licensed data) through the
real Robot SF environment and returns the trajectory plus runtime policy provenance
derived from the executed configuration, not a copied metadata block.

The rollout is deterministic for a fixed seed and pinned config. It is the Robot SF
"native" execution path; the SocNavBench side consumes the exported scenario (see
``socnavbench_canary.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.planner.socnav import SocNavPlannerConfig, make_social_force_policy
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)

MODULE_ROOT = Path(__file__).resolve().parents[2]

# The canary pins one Robot SF scenario that resolves to a real, tracked SVG map so
# the policy executes a genuine navigation episode (no licensed external asset needed).
CANARY_SCENARIO_MANIFEST = MODULE_ROOT / "configs" / "scenarios" / "canary_corridor.yaml"
CANARY_SCENARIO_INDEX = 0
CANARY_SIM_HORIZON = 120

# Pinned SocialForce planner parameters (mirror of
# configs/algos/social_force_holonomic_tuned_tau_low.yaml, read at runtime for provenance).
SOCIAL_FORCE_TAU_LOW_KEYS = (
    "social_force_tau",
    "social_force_desired_speed",
    "social_force_repulsion_weight",
    "social_force_factor",
    "social_force_obstacle_factor",
)
_PINNED_PLANNER_CONFIG_KEYS = (
    "max_linear_speed",
    "max_angular_speed",
    "angular_gain",
    "goal_tolerance",
    *SOCIAL_FORCE_TAU_LOW_KEYS,
)


@dataclass(frozen=True, slots=True)
class PolicyRolloutResult:
    """Outcome of executing the pinned policy inside a real Robot SF episode."""

    robot_positions: list[tuple[float, float]]
    goal_position: tuple[float, float]
    reached_goal_step: int | None
    terminated: bool
    truncated: bool
    # Runtime policy provenance: the exact planner parameters that actually governed the
    # executed trajectory. Derived from the live SocNavPlannerConfig, not a static copy.
    planner_config: dict[str, float]
    scenario_id: str
    seed: int

    def trajectory_array(self) -> np.ndarray:
        """Return the robot trajectory as an (N, 2) numpy array."""
        return np.asarray(self.robot_positions, dtype=float)


def _resolve_pinned_planner_config(*, algo_config: Path) -> SocNavPlannerConfig:
    """Build the pinned SocialForce planner config from the tracked algo YAML.

    Reading the config at runtime (rather than hard-coding constants) gives the canary
    provenance that tracks the real source of truth; a drifted config changes the receipt.

    Returns:
        SocNavPlannerConfig with values from the tracked YAML.

    Raises:
        ValueError: If the YAML is not a mapping or omits a pinned parameter.
    """
    raw = yaml.safe_load(algo_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Pinned planner config root must be a mapping.")
    missing = [key for key in _PINNED_PLANNER_CONFIG_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Pinned planner config is missing required keys: {', '.join(missing)}")

    values: dict[str, float] = {}
    for key in _PINNED_PLANNER_CONFIG_KEYS:
        try:
            value = float(raw[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pinned planner config value for {key!r} must be numeric.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Pinned planner config value for {key!r} must be finite.")
        values[key] = value

    return SocNavPlannerConfig(
        max_linear_speed=values["max_linear_speed"],
        max_angular_speed=values["max_angular_speed"],
        angular_gain=values["angular_gain"],
        goal_tolerance=values["goal_tolerance"],
        social_force_tau=values["social_force_tau"],
        social_force_desired_speed=values["social_force_desired_speed"],
        social_force_repulsion_weight=values["social_force_repulsion_weight"],
        social_force_factor=values["social_force_factor"],
        social_force_obstacle_factor=values["social_force_obstacle_factor"],
    )


def _planner_config_provenance(config: SocNavPlannerConfig) -> dict[str, float]:
    """Return the pinned tau-low parameter subset as a provenance dict."""
    return {
        key: float(getattr(config, key))
        for key in SOCIAL_FORCE_TAU_LOW_KEYS
        if hasattr(config, key)
    }


def execute_pinned_policy(
    *,
    seed: int,
    algo_config: Path,
    scenario_manifest: Path = CANARY_SCENARIO_MANIFEST,
    scenario_index: int = CANARY_SCENARIO_INDEX,
    max_steps: int = CANARY_SIM_HORIZON,
) -> PolicyRolloutResult:
    """Roll out the pinned SocialForce policy on a real Robot SF scenario.

    The policy is genuinely executed: it plans an action each step from the live
    observation and the environment steps the simulation. The returned trajectory is
    therefore a native Robot SF execution artifact, and ``planner_config`` is the runtime
    provenance of the policy that produced it.

    Args:
        seed: Episode seed (determinism).
        algo_config: Path to the pinned SocialForce algo YAML (runtime provenance source).
        scenario_manifest: YAML manifest with a ``scenarios`` list resolving to a real SVG map.
        scenario_index: Index of the canary scenario within the manifest.
        max_steps: Episode horizon cap.

    Returns:
        The executed trajectory, goal, termination state, and runtime policy provenance.
    """
    scenarios = load_scenarios(str(scenario_manifest))
    scenario = select_scenario(scenarios, scenario_index)
    if not isinstance(scenario, dict):
        raise ValueError(
            f"Canary scenario at index {scenario_index} must be a mapping, got "
            f"{type(scenario).__name__}."
        )
    scenario_id = str(
        scenario.get("name") or scenario.get("scenario_id") or f"scenario[{scenario_index}]"
    )

    cfg = build_robot_config_from_scenario(scenario, scenario_path=scenario_manifest)
    cfg.observation_mode = ObservationMode.SOCNAV_STRUCT
    cfg.use_planner = False

    planner_config = _resolve_pinned_planner_config(algo_config=algo_config)
    policy = make_social_force_policy(planner_config)

    env = make_robot_env(config=cfg, seed=int(seed))
    try:
        obs, _ = env.reset(seed=int(seed))

        positions: list[tuple[float, float]] = [tuple(float(v) for v in env.simulator.robot_pos[0])]
        reached_goal_step: int | None = None
        terminated = False
        truncated = False
        for step_idx in range(int(max_steps)):
            action = policy.act(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            positions.append(tuple(float(v) for v in env.simulator.robot_pos[0]))
            if reached_goal_step is None and bool(info.get("is_success", False)):
                reached_goal_step = step_idx + 1
                break
            if terminated or truncated:
                break

        goal = tuple(float(v) for v in env.simulator.goal_pos[0])
        return PolicyRolloutResult(
            robot_positions=positions,
            goal_position=goal,
            reached_goal_step=reached_goal_step,
            terminated=terminated,
            truncated=truncated,
            planner_config=_planner_config_provenance(planner_config),
            scenario_id=scenario_id,
            seed=int(seed),
        )
    finally:
        env.close()


__all__ = [
    "CANARY_SCENARIO_MANIFEST",
    "PolicyRolloutResult",
    "execute_pinned_policy",
]
