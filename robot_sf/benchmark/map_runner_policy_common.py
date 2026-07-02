"""Shared policy-construction helpers for the map-runner benchmark.

These helpers were previously defined inside ``robot_sf.benchmark.map_runner`` and
called from ~20 of ``_build_policy``'s adapter branches. Hosting them in a neutral
module lets the per-algorithm builder modules in ``map_runner_policies/`` reuse them
without an import cycle back into ``map_runner`` (see #3403, follow-up to #3400/#3384).

No behavior change: ``build_adapter_policy`` is a verbatim move of the former
``map_runner._build_policy`` helper ``_build_adapter_policy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark import planner_command_contract as _planner_commands
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.map_runner_policy_metadata import (
    attach_planner_reset as _attach_planner_reset,
)
from robot_sf.benchmark.utils import _config_hash
from robot_sf.planner.cbf_safety_filter import (
    build_cbf_safety_filter,
    build_cbf_safety_filter_config,
)
from robot_sf.planner.kinematics_model import resolve_benchmark_kinematics_model
from robot_sf.planner.safety_shield import (
    new_shield_stats,
    shield_contract_metadata,
    update_shield_stats,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_default_robot_command_space = _planner_commands.default_robot_command_space
_init_feasibility_metadata = _planner_commands.init_feasibility_metadata
_project_with_feasibility = _planner_commands.project_with_feasibility


def build_adapter_policy(  # noqa: C901
    *,
    algo_key: str,
    algo_config: dict[str, Any],
    meta: dict[str, Any],
    adapter: Any,
    adapter_name: str,
    robot_kinematics: str | None,
    normalized_robot_command_mode: str | None,
    limitations: str | None = None,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Construct a projected adapter policy with standard metadata wiring.

    Returns:
        tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
            Projected policy callable and populated benchmark metadata.
    """
    meta.update(
        {
            "status": "ok",
            "config": algo_config,
            "config_hash": _config_hash(algo_config),
        }
    )
    meta = enrich_algorithm_metadata(
        algo=algo_key,
        metadata=meta,
        execution_mode="adapter",
        adapter_name=adapter_name,
        robot_kinematics=robot_kinematics,
    )
    _init_feasibility_metadata(meta)
    planner_meta = meta.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["planner_command_space"] = _default_robot_command_space(
            robot_kinematics,
            algo_config,
            robot_command_mode=normalized_robot_command_mode,
        )
        if limitations is not None:
            planner_meta["limitations"] = limitations
    adapter_kinematics_model = resolve_benchmark_kinematics_model(
        robot_kinematics=robot_kinematics,
        command_limits=algo_config,
    )
    cbf_config = build_cbf_safety_filter_config(algo_config.get("cbf_safety_filter"))
    cbf_filter: Any | None = None
    if cbf_config.enabled:
        cbf_filter = build_cbf_safety_filter(cbf_config)
        meta["cbf_safety_filter"] = {
            "status": "enabled",
            "variant": cbf_config.variant,
            "prediction_source": "current_state",
            "wrapped_planner": adapter_name,
            "claim_boundary": "planner-comparison baseline only; not a formal safety certificate",
        }
        meta["safety_shield_contract"] = shield_contract_metadata(
            shield_name=type(cbf_filter).__name__,
            prediction_source="current_state",
            fallback_policy="qp_projection_best_effort",
        )
        meta["shield_stats"] = new_shield_stats()

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        """Run an adapter-backed planner and project command feasibility.

        Returns:
            tuple[float, float]: Projected linear and angular command.
        """
        linear, angular = adapter.plan(obs)
        if cbf_filter is not None:
            shield_decision = cbf_filter.filter_command(obs, (float(linear), float(angular)))
            shield_stats = meta.get("shield_stats")
            if isinstance(shield_stats, dict):
                update_shield_stats(shield_stats, shield_decision)
            linear, angular = shield_decision.filtered_action
        return _project_with_feasibility(
            model=adapter_kinematics_model,
            command=(float(linear), float(angular)),
            meta=meta,
        )

    _attach_planner_reset(_policy, adapter)
    _policy._planner_adapter = adapter
    if hasattr(adapter, "bind_env"):
        _policy._planner_bind_env = adapter.bind_env
    if hasattr(adapter, "close"):
        _policy._planner_close = adapter.close
    if hasattr(adapter, "diagnostics") or cbf_filter is not None:

        def _planner_stats() -> dict[str, Any]:
            """Expose adapter diagnostics for episode metadata.

            Returns:
                dict[str, Any]: Adapter diagnostic payload.
            """
            diagnostics = adapter.diagnostics() if hasattr(adapter, "diagnostics") else {}
            if cbf_filter is None:
                return diagnostics
            return {
                **diagnostics,
                "cbf_safety_filter": cbf_filter.diagnostics(),
            }

        _policy._planner_stats = _planner_stats

    return _policy, meta
