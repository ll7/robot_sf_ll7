"""Builders for rule-stack and lidar-grid map-runner policy families.

This is a behavior-preserving continuation of the ``_build_policy``
decomposition tracked by #3384. The migrated branches only depend on neutral
planner modules and ``map_runner_policy_common.build_adapter_policy``, so they
can live outside ``map_runner.py`` without introducing a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_policy_common import build_adapter_policy
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from robot_sf.planner.lidar_occupancy_grid import build_lidar_grid_route_adapter
from robot_sf.planner.policy_stack_v1 import (
    PolicyStackV1Adapter,
    build_policy_stack_v1_build_config,
)
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter
from robot_sf.planner.topology_guided_local_policy import (
    TopologyGuidedHybridRulePlannerAdapter,
    build_topology_guided_local_policy_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable


POLICY_STACK_KEYS = frozenset({"policy_stack_v1"})
HYBRID_RULE_KEYS = frozenset(
    {
        "hybrid_rule_local_planner",
        "hybrid_rule_v0_minimal",
        "actuation_aware_hybrid_rule_v0",
    }
)
TOPOLOGY_GUIDED_KEYS = frozenset({"topology_guided_hybrid_rule_v0"})
LIDAR_GRID_ROUTE_KEYS = frozenset({"lidar_grid_route", "lidar_occupancy_grid_route"})

RULE_AND_GRID_KEYS = (
    POLICY_STACK_KEYS | HYBRID_RULE_KEYS | TOPOLOGY_GUIDED_KEYS | LIDAR_GRID_ROUTE_KEYS
)


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build migrated rule-stack and lidar-grid policy metadata.

    Returns:
        Policy callable and enriched metadata dictionary.
    """
    if algo_key not in RULE_AND_GRID_KEYS:
        supported = ", ".join(sorted(RULE_AND_GRID_KEYS))
        raise ValueError(
            f"Unsupported rule/grid policy builder key '{algo_key}'. Expected: {supported}"
        )

    del adapter_impact_eval
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    meta: dict[str, Any] = {"algorithm": algo_key}

    if algo_key == "policy_stack_v1":
        stack_cfg = build_policy_stack_v1_build_config(algo_config)
        adapter = PolicyStackV1Adapter(
            config=stack_cfg.policy_stack,
            risk_dwa=RiskDWAPlannerAdapter(config=stack_cfg.risk_dwa),
        )
        return build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="PolicyStackV1Adapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    if algo_key in HYBRID_RULE_KEYS:
        adapter = HybridRuleLocalPlannerAdapter(
            config=build_hybrid_rule_local_planner_config(algo_config)
        )
        return build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="HybridRuleLocalPlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    if algo_key == "topology_guided_hybrid_rule_v0":
        adapter = TopologyGuidedHybridRulePlannerAdapter(
            config=build_topology_guided_local_policy_config(algo_config)
        )
        meta["topology_guided_hybrid_rule"] = {
            "diagnostic_only": True,
            "claim_boundary": "diagnostic_only",
            "hypothesis_source": "masked_occupancy_grid_routes",
            "wrapped_planner": "HybridRuleLocalPlannerAdapter",
        }
        return build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="TopologyGuidedHybridRulePlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="diagnostic_only_topology_hypothesis_selector",
        )

    adapter = build_lidar_grid_route_adapter(algo_config)
    return build_adapter_policy(
        algo_key="lidar_grid_route",
        algo_config=algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="LidarOccupancyGridRouteAdapter",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations="lidar_ego_occupancy_grid_route_testing_only",
    )
