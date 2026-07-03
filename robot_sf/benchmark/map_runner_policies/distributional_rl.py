"""Map-runner builder for issue #4016 distributional RL checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.baselines.distributional_rl import DistributionalRLPlanner

if TYPE_CHECKING:
    from .registry import PolicyBuildResult

DISTRIBUTIONAL_RL_KEYS = frozenset({"distributional_rl", "qr_dqn"})


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> PolicyBuildResult:
    """Build a distributional RL policy callable for map-runner.

    Returns:
        Policy callable and metadata consumed by ``map_runner._build_policy``.
    """

    del robot_command_mode, adapter_impact_eval
    planner = DistributionalRLPlanner(algo_config, seed=None)
    meta = planner.get_metadata()
    meta.setdefault("algorithm", algo_key)
    meta["profile"] = str(algo_config.get("profile", "experimental")).strip().lower()
    meta["runtime_adapter"] = "distributional_rl"
    meta["robot_kinematics"] = robot_kinematics

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        action = planner.step(obs)
        if not isinstance(action, dict):
            raise TypeError(f"distributional RL planner returned non-dict action: {type(action)}")
        if "v" not in action or "omega" not in action:
            raise ValueError("distributional RL action must contain 'v' and 'omega'")
        latest = planner.diagnostics()
        if latest:
            meta["diagnostics"] = latest
        return float(action["v"]), float(action["omega"])

    _policy._planner_close = planner.close  # type: ignore[attr-defined]
    _policy._planner_native_env_action = False  # type: ignore[attr-defined]
    return _policy, meta
