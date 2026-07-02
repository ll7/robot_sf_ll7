"""Builder for the safety-barrier and grid-route map-runner policy families.

Second slice of the ``_build_policy`` decomposition (#3384). Behavior is a
faithful move of the original ``if algo_key == "safety_barrier"`` and
``if algo_key == "grid_route"`` branches from
``robot_sf.benchmark.map_runner`` — no semantic change.

Both branches construct an adapter and delegate to the shared
``build_adapter_policy`` helper in ``map_runner_policy_common``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

#: Algorithm keys handled by this builder.
SAFETY_BARRIER_ALGO_KEYS = frozenset({"safety_barrier"})
GRID_ROUTE_ALGO_KEYS = frozenset({"grid_route"})
ADAPTER_ALGO_KEYS = SAFETY_BARRIER_ALGO_KEYS | GRID_ROUTE_ALGO_KEYS


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build a safety-barrier or grid-route adapter policy and its metadata.

    Args:
        algo_key: Normalized algorithm key (one of :data:`ADAPTER_ALGO_KEYS`).
        algo_config: Algorithm configuration payload.
        robot_kinematics: Runtime robot kinematics label for metadata enrichment.
        robot_command_mode: Runtime robot command mode (for holonomic metadata labels).
        adapter_impact_eval: Unused compatibility hook for learned-policy builders.

    Returns:
        Policy callable and enriched metadata dictionary.
    """
    if algo_key not in ADAPTER_ALGO_KEYS:
        supported = ", ".join(sorted(ADAPTER_ALGO_KEYS))
        raise ValueError(
            f"Unsupported safety-barrier/grid-route policy algo_key {algo_key!r}; "
            f"expected one of: {supported}"
        )

    del adapter_impact_eval

    from robot_sf.benchmark.map_runner_policy_common import (  # noqa: PLC0415
        build_adapter_policy,
    )
    from robot_sf.planner.grid_route import (  # noqa: PLC0415
        GridRoutePlannerAdapter,
        build_grid_route_config,
    )
    from robot_sf.planner.lidar_occupancy import (  # noqa: PLC0415
        LidarOccupancyPlannerAdapter,
        build_lidar_occupancy_config,
    )
    from robot_sf.planner.safety_barrier import (  # noqa: PLC0415
        SafetyBarrierPlannerAdapter,
        build_safety_barrier_config,
    )

    meta: dict[str, Any] = {"algorithm": algo_key}
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )

    if algo_key in SAFETY_BARRIER_ALGO_KEYS:
        adapter: Any = SafetyBarrierPlannerAdapter(config=build_safety_barrier_config(algo_config))
        adapter_name = "SafetyBarrierPlannerAdapter"
        limitations = "static_obstacle_first_testing_only"
        if algo_config.get("lidar_occupancy_adapter"):
            adapter = LidarOccupancyPlannerAdapter(
                planner=adapter,
                config=build_lidar_occupancy_config(algo_config),
            )
            adapter_name = "LidarOccupancySafetyBarrierAdapter"
            limitations = "lidar_derived_ego_occupancy_testing_only"
            meta["lidar_occupancy_adapter"] = {
                "status": "enabled",
                "source": "lidar_rays",
                "output": "ego_occupancy_grid",
                "planner": "safety_barrier",
            }
        return build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name=adapter_name,
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations=limitations,
        )

    # grid_route
    adapter = GridRoutePlannerAdapter(config=build_grid_route_config(algo_config))
    return build_adapter_policy(
        algo_key=algo_key,
        algo_config=algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="GridRoutePlannerAdapter",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations="static_obstacle_first_testing_only",
    )
