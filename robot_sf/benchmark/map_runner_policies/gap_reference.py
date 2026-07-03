"""Builders for gap and reference adapter map-runner policy families.

Behavior-preserving continuation of the ``_build_policy`` decomposition
tracked in #3384. These branches only depend on neutral map-runner helpers and
planner modules, so they can live in the registry without importing the
monolithic ``map_runner`` module.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_policy_common import build_adapter_policy
from robot_sf.benchmark.map_runner_policy_resolution import _build_socnav_config
from robot_sf.benchmark.scenario_belief_policy_hook import (
    BELIEF_MODES,
    DEFAULT_FOV_DEGREES,
    DEFAULT_MAX_PEDESTRIANS,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_PED_RADIUS,
    BeliefModeStreamGapAdapter,
)
from robot_sf.planner.gap_prediction import (
    GapAwarePredictionAdapter,
    build_gap_prediction_config,
)
from robot_sf.planner.socnav import TrivialReferencePlannerAdapter
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, build_stream_gap_config

if TYPE_CHECKING:
    from collections.abc import Callable


TRIVIAL_REFERENCE_KEYS = frozenset({"trivial_reference", "reference_adapter"})
STREAM_GAP_KEYS = frozenset({"stream_gap"})
GAP_PREDICTION_KEYS = frozenset({"gap_prediction"})
GAP_REFERENCE_KEYS = TRIVIAL_REFERENCE_KEYS | STREAM_GAP_KEYS | GAP_PREDICTION_KEYS


def _map_runner_compat_attr(name: str, default: Any) -> Any:
    """Return old ``map_runner`` private test hook when monkeypatched."""
    map_runner_module = sys.modules.get("robot_sf.benchmark.map_runner")
    if map_runner_module is None:
        return default
    return getattr(map_runner_module, name, default)


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float] | dict[str, Any]], dict[str, Any]]:
    """Build one migrated reference/gap policy family.

    Returns:
        Policy callable and algorithm metadata.
    """
    del adapter_impact_eval

    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    meta: dict[str, Any] = {"algorithm": algo_key}

    if algo_key in TRIVIAL_REFERENCE_KEYS:
        adapter = TrivialReferencePlannerAdapter(config=_build_socnav_config(algo_config))
        meta["algorithm"] = "trivial_reference"
        return build_adapter_policy(
            algo_key="trivial_reference",
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="TrivialReferencePlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="Diagnostic adapter template only; do not use as benchmark planner evidence.",
        )

    if algo_key == "stream_gap":
        belief_mode = str(algo_config.get("belief_mode") or "").strip().lower() or None
        stream_gap_config = algo_config
        limitations: str | None = None
        if belief_mode in BELIEF_MODES:
            # Belief mode owns the uncertainty gate; only dropping mode enables it.
            stream_gap_config = {
                **algo_config,
                "uncertainty_gating_enabled": BELIEF_MODES[belief_mode]["gate"],
            }
            limitations = f"scenario_belief_mode={belief_mode}_diagnostic"
        stream_gap_adapter_cls = _map_runner_compat_attr(
            "StreamGapPlannerAdapter",
            StreamGapPlannerAdapter,
        )
        adapter: Any = stream_gap_adapter_cls(config=build_stream_gap_config(stream_gap_config))
        if belief_mode in BELIEF_MODES:
            adapter = BeliefModeStreamGapAdapter(
                adapter,
                mode=belief_mode,
                fov_degrees=float(algo_config.get("belief_fov_degrees", DEFAULT_FOV_DEGREES)),
                max_range_m=algo_config.get("belief_max_range_m", DEFAULT_MAX_RANGE_M),
                ped_radius=float(algo_config.get("belief_ped_radius", DEFAULT_PED_RADIUS)),
                max_pedestrians=int(
                    algo_config.get("belief_max_pedestrians", DEFAULT_MAX_PEDESTRIANS)
                ),
            )
        return build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="StreamGapPlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations=limitations,
        )

    if algo_key == "gap_prediction":
        gap_prediction_adapter_cls = _map_runner_compat_attr(
            "GapAwarePredictionAdapter",
            GapAwarePredictionAdapter,
        )
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = gap_prediction_adapter_cls(
            config=build_gap_prediction_config(algo_config),
            allow_fallback=allow_fallback,
        )
        return build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="GapAwarePredictionAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    raise ValueError(f"Unsupported gap/reference policy builder key: {algo_key}")
