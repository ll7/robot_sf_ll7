"""Builder for the built-in goal/simple map-runner policy family.

First slice of the ``_build_policy`` decomposition (#3384 / #3400). Behavior is a
faithful move of the original ``if algo_key in {"goal", "simple", ...}`` branch from
``robot_sf.benchmark.map_runner`` — no semantic change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark import planner_command_contract as planner_commands
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.utils import _config_hash
from robot_sf.planner.kinematics_model import resolve_benchmark_kinematics_model

if TYPE_CHECKING:
    from collections.abc import Callable

#: Algorithm keys handled by this builder.
GOAL_ALGO_KEYS = frozenset({"goal", "simple", "goal_policy", "simple_policy"})


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build the built-in goal/simple policy and its metadata.

    Args:
        algo_key: Normalized algorithm key (one of :data:`GOAL_ALGO_KEYS`).
        algo_config: Algorithm configuration payload.
        robot_kinematics: Runtime robot kinematics label for metadata enrichment.
        robot_command_mode: Runtime robot command mode (for holonomic metadata labels).
        adapter_impact_eval: Unused compatibility hook for learned-policy builders.

    Returns:
        Policy callable and enriched metadata dictionary.
    """
    del adapter_impact_eval
    # Local import avoids a map_runner <-> map_runner_policies import cycle. _goal_policy
    # stays defined in map_runner (also used by other call sites there and imported
    # directly by tests/benchmark/test_map_runner_utils.py).
    from robot_sf.benchmark.map_runner import _goal_policy  # noqa: PLC0415

    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )

    goal_kinematics_model = resolve_benchmark_kinematics_model(
        robot_kinematics=robot_kinematics,
        command_limits=algo_config,
    )

    meta: dict[str, Any] = {
        "algorithm": algo_key,
        "status": "ok",
        "config": algo_config,
        "config_hash": _config_hash(algo_config),
    }
    meta = enrich_algorithm_metadata(
        algo=algo_key,
        metadata=meta,
        execution_mode="native",
        robot_kinematics=robot_kinematics,
    )
    planner_commands.init_feasibility_metadata(meta)
    planner_meta = meta.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["planner_command_space"] = planner_commands.default_robot_command_space(
            robot_kinematics,
            algo_config,
            robot_command_mode=normalized_robot_command_mode,
        )

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        """Run the built-in goal policy with feasibility projection.

        Returns:
            tuple[float, float]: Projected linear and angular command.
        """
        configured_max_speed = algo_config.get("max_speed")
        max_speed = 1.0 if configured_max_speed is None else float(configured_max_speed)
        linear, angular = _goal_policy(obs, max_speed=max_speed)
        return planner_commands.project_with_feasibility(
            model=goal_kinematics_model,
            command=(linear, angular),
            meta=meta,
        )

    return _policy, meta
