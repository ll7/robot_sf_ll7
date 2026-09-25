"""Builder for the deterministic stand-still map-runner reference policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark import planner_command_contract as planner_commands
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.utils import _config_hash

if TYPE_CHECKING:
    from collections.abc import Callable

STAND_STILL_ALGO_KEYS = frozenset({"stand_still"})


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build the zero-command policy and its native execution metadata.

    The callable deliberately does not inspect its observation or configuration.
    The runner records the active observation contract for comparability, while
    this reference always emits zero linear and angular velocity.

    Returns:
        Policy callable and metadata with native execution identity.
    """
    del adapter_impact_eval
    if algo_key not in STAND_STILL_ALGO_KEYS:
        supported = ", ".join(sorted(STAND_STILL_ALGO_KEYS))
        raise ValueError(f"Unsupported stand-still policy key '{algo_key}'. Expected: {supported}")

    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
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
    planner_meta = meta.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["planner_command_space"] = planner_commands.default_robot_command_space(
            robot_kinematics,
            algo_config,
            robot_command_mode=normalized_robot_command_mode,
        )

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        """Emit the stationary reference command without reading observations.

        Returns:
            tuple[float, float]: Exact zero linear and angular velocity.
        """
        del obs
        return 0.0, 0.0

    return _policy, meta
