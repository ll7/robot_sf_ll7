"""Batch-summary metadata helpers for map-based benchmark runs."""

from __future__ import annotations

from typing import Any, NamedTuple


def _float_metadata_value(value: Any, default: float = 0.0) -> float:
    """Convert optional numeric metadata without treating valid falsy values as missing.

    Returns:
        Float-converted ``value`` or ``default`` when the value is explicitly ``None``.
    """
    return float(default if value is None else value)


def accumulate_batch_metadata(
    rec: dict[str, Any],
    *,
    feasibility_totals: dict[str, float],
) -> tuple[bool, int, int]:
    """Aggregate adapter-impact and feasibility counters from one episode record.

    Returns:
        tuple[bool, int, int]: ``(adapter_requested_seen, native_steps, adapted_steps)`` deltas.
    """
    impact_meta = (rec.get("algorithm_metadata") or {}).get("adapter_impact") or {}
    feasibility_meta = (rec.get("algorithm_metadata") or {}).get("kinematics_feasibility") or {}
    adapter_requested_seen = False
    adapter_native_steps = 0
    adapter_adapted_steps = 0
    if isinstance(impact_meta, dict):
        adapter_requested_seen = bool(impact_meta.get("requested", False))
        adapter_native_steps = int(impact_meta.get("native_steps", 0) or 0)
        adapter_adapted_steps = int(impact_meta.get("adapted_steps", 0) or 0)
    if isinstance(feasibility_meta, dict):
        commands_evaluated = int(feasibility_meta.get("commands_evaluated", 0) or 0)
        feasibility_totals["commands_evaluated"] += commands_evaluated
        feasibility_totals["infeasible_native_count"] += int(
            feasibility_meta.get("infeasible_native_count", 0) or 0
        )
        feasibility_totals["projected_count"] += int(
            feasibility_meta.get("projected_count", 0) or 0
        )
        mean_abs_delta_linear = _float_metadata_value(feasibility_meta.get("mean_abs_delta_linear"))
        mean_abs_delta_angular = _float_metadata_value(
            feasibility_meta.get("mean_abs_delta_angular")
        )
        feasibility_totals["sum_abs_delta_linear"] += mean_abs_delta_linear * commands_evaluated
        feasibility_totals["sum_abs_delta_angular"] += mean_abs_delta_angular * commands_evaluated
        feasibility_totals["max_abs_delta_linear"] = max(
            float(feasibility_totals["max_abs_delta_linear"]),
            _float_metadata_value(feasibility_meta.get("max_abs_delta_linear")),
        )
        feasibility_totals["max_abs_delta_angular"] = max(
            float(feasibility_totals["max_abs_delta_angular"]),
            _float_metadata_value(feasibility_meta.get("max_abs_delta_angular")),
        )
    return adapter_requested_seen, adapter_native_steps, adapter_adapted_steps


class WorkerMetadataBridgeUpdate(NamedTuple):
    """Normalized metadata update emitted by either direct or serialized worker paths."""

    runtime_algorithm_contract: dict[str, Any]
    adapter_requested_seen: bool
    adapter_native_steps: int
    adapter_adapted_steps: int


def apply_worker_metadata_bridge(
    rec: dict[str, Any],
    *,
    feasibility_totals: dict[str, float],
    runtime_algorithm_contract: dict[str, Any] | None,
) -> WorkerMetadataBridgeUpdate:
    """Fold one worker episode record into the batch-level metadata contract.

    This is the explicit bridge between per-episode worker payloads and the batch summary.  It is
    used by both direct function-call execution and serialized worker execution so metadata-only
    additions have one testable hop.

    Returns:
        Worker metadata update with merged runtime contract and adapter-impact deltas.
    """
    requested_seen, native_steps, adapted_steps = accumulate_batch_metadata(
        rec,
        feasibility_totals=feasibility_totals,
    )
    merged_runtime_contract = merge_runtime_algorithm_contract(
        runtime_algorithm_contract or {},
        rec.get("algorithm_metadata"),
    )
    return WorkerMetadataBridgeUpdate(
        runtime_algorithm_contract=merged_runtime_contract,
        adapter_requested_seen=requested_seen,
        adapter_native_steps=native_steps,
        adapter_adapted_steps=adapted_steps,
    )


def merge_runtime_algorithm_contract(  # noqa: C901
    base_contract: dict[str, Any],
    runtime_algorithm_metadata: Any,
) -> dict[str, Any]:
    """Merge runtime-resolved algorithm contract fields into a batch summary contract.

    Returns:
        dict[str, Any]: The merged contract mapping, or the original input on mismatch.
    """
    if not isinstance(base_contract, dict) or not isinstance(runtime_algorithm_metadata, dict):
        return base_contract

    def _merge_mapping(target: dict[str, Any], source: dict[str, Any]) -> None:
        """Merge authoritative runtime contract values into a nested mapping."""
        authoritative_keys = {
            "robot_kinematics",
            "execution_mode",
            "adapter_name",
            "planner_command_space",
            "benchmark_command_space",
            "projection_policy",
            "execution_detail",
            "adapter_boundary",
        }

        def _is_placeholder(value: Any) -> bool:
            """Return whether a contract value should be replaced by runtime data."""
            if value is None:
                return True
            if isinstance(value, str):
                normalized = value.strip().lower()
                return normalized in {"", "unknown", "unspecified", "mixed"}
            return False

        for key, value in source.items():
            current = target.get(key)
            if _is_placeholder(current):
                target[key] = value
                continue
            if isinstance(current, dict) and isinstance(value, dict):
                _merge_mapping(current, value)
                continue
            if _is_placeholder(value) or current == value:
                continue
            if key in authoritative_keys:
                target[key] = value
                continue
            target[key] = "mixed"

    runtime_planner_kinematics = runtime_algorithm_metadata.get("planner_kinematics")
    if isinstance(runtime_planner_kinematics, dict):
        planner_kinematics = base_contract.get("planner_kinematics")
        if not isinstance(planner_kinematics, dict):
            planner_kinematics = {}
            base_contract["planner_kinematics"] = planner_kinematics
        _merge_mapping(planner_kinematics, runtime_planner_kinematics)

    runtime_upstream_reference = runtime_algorithm_metadata.get("upstream_reference")
    if isinstance(runtime_upstream_reference, dict):
        upstream_reference = base_contract.get("upstream_reference")
        if not isinstance(upstream_reference, dict):
            upstream_reference = {}
            base_contract["upstream_reference"] = upstream_reference
        _merge_mapping(upstream_reference, runtime_upstream_reference)

    return base_contract
