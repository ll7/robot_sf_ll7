"""Batch-summary metadata helpers for map-based benchmark runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
)
from robot_sf.benchmark.fallback_policy import availability_payload
from robot_sf.benchmark.latency_stress import not_available_latency_metrics
from robot_sf.benchmark.map_runner_metrics import summarize_collision_metrics
from robot_sf.benchmark.map_runner_provenance import map_result_provenance
from robot_sf.benchmark.utils import attach_track_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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
    ammv_meta = (rec.get("algorithm_metadata") or {}).get("ammv_feasibility") or {}
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
    if isinstance(ammv_meta, dict):
        feasibility_totals.setdefault("ammv_episode_count", 0)
        feasibility_totals.setdefault("ammv_commands_evaluated", 0)
        feasibility_totals.setdefault("ammv_curvature_violation_count", 0)
        feasibility_totals.setdefault("ammv_feasible_episode_count", 0)
        feasibility_totals.setdefault("ammv_tip_over_episode_count", 0)
        feasibility_totals.setdefault("ammv_min_stability_margin", float("inf"))
        feasibility_totals["ammv_episode_count"] += 1
        feasibility_totals["ammv_commands_evaluated"] += int(ammv_meta.get("n_commands", 0) or 0)
        feasibility_totals["ammv_curvature_violation_count"] += int(
            ammv_meta.get("n_curvature_violations", 0) or 0
        )
        if bool(ammv_meta.get("feasible", False)):
            feasibility_totals["ammv_feasible_episode_count"] += 1
        if bool(ammv_meta.get("tip_over_violation", False)):
            feasibility_totals["ammv_tip_over_episode_count"] += 1
        margin = ammv_meta.get("min_stability_margin")
        if margin is not None:
            feasibility_totals["ammv_min_stability_margin"] = min(
                float(feasibility_totals["ammv_min_stability_margin"]),
                float(margin),
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

    runtime_planner = runtime_algorithm_metadata.get("planner_runtime")
    if isinstance(runtime_planner, dict):
        runtime_checkpoint = runtime_planner.get("checkpoint_provenance")
        if isinstance(runtime_checkpoint, dict):
            checkpoint = base_contract.get("checkpoint_provenance")
            if not isinstance(checkpoint, dict):
                checkpoint = {}
                base_contract["checkpoint_provenance"] = checkpoint
            _merge_mapping(checkpoint, runtime_checkpoint)

    return base_contract


def build_ammv_feasibility_summary(feasibility_totals: Mapping[str, float]) -> dict[str, Any]:
    """Fold per-episode AMMV feasibility totals into the batch-summary artifact block (issue #3466).

    The per-episode ``ammv_feasibility.v1`` payload carries the claim-boundary markers
    (``evidence_kind`` / ``proxy_kind`` / ``status``) that keep this internal proxy from being read
    as calibrated hardware AMMV safety evidence. This helper re-attaches the same markers to the
    batch aggregate so a consumer that reads only the summary sees the identical boundary, then folds
    the four acceptance fields conservatively:

    - ``min_stability_margin`` is the worst-case (minimum) margin across contributing episodes, not a
      mean, so a single tip-over-prone episode is never averaged away.
    - ``tip_over_violation`` / ``feasible`` are worst-case AND/OR reductions: any tip-over marks the
      batch, and ``feasible`` requires every contributing episode to be feasible.
    - ``status`` distinguishes a real aggregate from the fail-closed "no eligible episodes" case, so
      a ``None`` margin with ``feasible: false`` is interpretable rather than silently ambiguous.

    Returns:
        The versioned ``ammv_feasibility.v1`` batch-summary block marked ``internal_non_hardware``.
    """
    ammv_episode_count = int(feasibility_totals.get("ammv_episode_count", 0))
    ammv_margin = float(feasibility_totals.get("ammv_min_stability_margin", float("inf")))
    return {
        "schema_version": "ammv_feasibility.v1",
        "evidence_kind": "diagnostic_proxy",
        "proxy_kind": "internal_non_hardware",
        "status": "available" if ammv_episode_count > 0 else "no_ammv_episodes",
        "episode_count": ammv_episode_count,
        "commands_evaluated": int(feasibility_totals.get("ammv_commands_evaluated", 0)),
        "min_stability_margin": (
            ammv_margin if ammv_episode_count > 0 and ammv_margin != float("inf") else None
        ),
        "tip_over_violation": bool(
            int(feasibility_totals.get("ammv_tip_over_episode_count", 0)) > 0
        ),
        "n_curvature_violations": int(feasibility_totals.get("ammv_curvature_violation_count", 0)),
        "feasible": bool(
            ammv_episode_count > 0
            and int(feasibility_totals.get("ammv_feasible_episode_count", 0)) == ammv_episode_count
        ),
    }


def build_completed_batch_summary(  # noqa: PLR0913
    *,
    algo_contract: dict[str, Any],
    runtime_algorithm_contract: dict[str, Any],
    preflight: dict[str, Any],
    feasibility_totals: dict[str, float],
    adapter_samples_seen: bool,
    adapter_native_steps: int,
    adapter_adapted_steps: int,
    planner_command_space_fallback: str,
    total_jobs: int,
    workers: int,
    batch_runtime_sec: float,
    wrote: int,
    failures: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    preflight_skipped_jobs: int,
    out_path: Path,
    readiness: Any,
    algo: str,
    benchmark_profile: str,
    noise_spec: dict[str, Any],
    noise_hash: str,
    tracking_precision_spec: dict[str, Any],
    tracking_precision_hash: str,
    active_observation_mode: str | None,
    active_observation_level: str | None,
    actuation_profile_metadata: dict[str, Any] | None,
    latency_profile_metadata: dict[str, Any] | None,
    benchmark_track: str | None,
    track_schema_version: str | None,
    schema_path: str | Path,
    scenario_path: Path,
    scenarios: list[dict[str, Any]],
    algo_config_path: str | None,
    suite_key: str,
    kinematics_tag: str,
    metric_affecting_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the final successful batch summary after all map jobs finish.

    Returns:
        Batch summary payload with metadata contracts, provenance, and availability.
    """
    impact_contract = algo_contract.get("adapter_impact")
    if (
        isinstance(impact_contract, dict)
        and bool(impact_contract.get("requested", False))
        and adapter_samples_seen
    ):
        impact_contract["native_steps"] = int(adapter_native_steps)
        impact_contract["adapted_steps"] = int(adapter_adapted_steps)
        total_steps = adapter_native_steps + adapter_adapted_steps
        if total_steps > 0:
            execution_mode = infer_execution_mode_from_counts(
                adapter_native_steps, adapter_adapted_steps
            )
            impact_contract["status"] = "complete"
            impact_contract["execution_mode"] = execution_mode
            impact_contract["adapter_fraction"] = float(adapter_adapted_steps / total_steps)
            algo_contract = enrich_algorithm_metadata(
                algo=algo,
                metadata=algo_contract,
                execution_mode=execution_mode,
                robot_kinematics=kinematics_tag,
                observation_mode=active_observation_mode,
                observation_level=active_observation_level,
            )
            attach_track_metadata(
                algo_contract,
                benchmark_track=benchmark_track,
                track_schema_version=track_schema_version,
                observation_level=active_observation_level,
                observation_mode=active_observation_mode,
            )
        else:
            impact_contract["status"] = "not_applicable"
            impact_contract["adapter_fraction"] = 0.0

    algo_contract = merge_runtime_algorithm_contract(
        algo_contract,
        runtime_algorithm_contract,
    )
    preflight["algorithm_metadata_contract"] = algo_contract
    planner_contract = algo_contract.get("planner_kinematics")
    if isinstance(planner_contract, dict) and planner_contract.get("planner_command_space") in {
        None,
        "unknown",
    }:
        planner_contract["planner_command_space"] = planner_command_space_fallback
    total_commands = int(feasibility_totals["commands_evaluated"])
    algo_contract["kinematics_feasibility"] = {
        "commands_evaluated": total_commands,
        "infeasible_native_count": int(feasibility_totals["infeasible_native_count"]),
        "projected_count": int(feasibility_totals["projected_count"]),
        "projection_rate": (
            float(feasibility_totals["projected_count"] / total_commands)
            if total_commands > 0
            else 0.0
        ),
        "infeasible_rate": (
            float(feasibility_totals["infeasible_native_count"] / total_commands)
            if total_commands > 0
            else 0.0
        ),
        "mean_abs_delta_linear": (
            float(feasibility_totals["sum_abs_delta_linear"] / total_commands)
            if total_commands > 0
            else 0.0
        ),
        "mean_abs_delta_angular": (
            float(feasibility_totals["sum_abs_delta_angular"] / total_commands)
            if total_commands > 0
            else 0.0
        ),
        "max_abs_delta_linear": float(feasibility_totals["max_abs_delta_linear"]),
        "max_abs_delta_angular": float(feasibility_totals["max_abs_delta_angular"]),
    }
    algo_contract["ammv_feasibility"] = build_ammv_feasibility_summary(feasibility_totals)

    summary = {
        "total_jobs": total_jobs,
        "workers": int(workers),
        "parallel_execution": bool(workers > 1),
        "batch_runtime_sec": batch_runtime_sec,
        "written": wrote,
        "successful_jobs": wrote,
        "failed_jobs": len(failures),
        "skipped_jobs": preflight_skipped_jobs,
        "failures": failures,
        "out_path": str(out_path),
        "algorithm_readiness": {
            "name": readiness.canonical_name if readiness is not None else algo,
            "tier": readiness.tier if readiness is not None else "unknown",
            "profile": benchmark_profile,
        },
        "algorithm_metadata_contract": algo_contract,
        "preflight": preflight,
        "observation_noise": noise_spec,
        "observation_noise_hash": noise_hash,
        "tracking_precision": tracking_precision_spec,
        "tracking_precision_hash": tracking_precision_hash,
        "observation_level": active_observation_level,
        "metrics": summarize_collision_metrics(episode_records),
        "synthetic_actuation_profile": actuation_profile_metadata,
        "latency_stress_profile": latency_profile_metadata,
        "latency_stress_metrics": (
            not_available_latency_metrics() if latency_profile_metadata is not None else None
        ),
    }
    if benchmark_track is not None:
        summary["benchmark_track"] = benchmark_track
    if track_schema_version is not None:
        summary["track_schema_version"] = track_schema_version
    artifact_pointer_status = "local_jsonl_present" if out_path.exists() else "not_available"
    summary["provenance"] = map_result_provenance(
        schema_path=schema_path,
        scenario_path=scenario_path,
        scenarios=scenarios,
        algo=algo,
        algo_config_path=algo_config_path,
        benchmark_profile=benchmark_profile,
        suite_key=suite_key,
        total_jobs=total_jobs,
        written=wrote,
        artifact_pointer_status=artifact_pointer_status,
        metric_affecting_config=metric_affecting_config,
    )
    summary["benchmark_availability"] = availability_payload(summary)
    return summary
