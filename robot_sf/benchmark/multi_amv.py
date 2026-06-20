"""Minimal multi-AMV benchmark helpers.

This module provides the first narrow multi-robot benchmark slice: scenario
settings parsing and inter-robot metric computation. It intentionally avoids a
fleet-optimization abstraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MultiAmvSettings:
    """Scenario-level settings for the minimal multi-AMV benchmark slice."""

    num_robots: int = 1
    near_miss_distance_m: float = 1.0
    collision_distance_m: float = 0.4
    deadlock_speed_mps: float = 0.05
    deadlock_window_steps: int = 10


class MultiAmvPlannerSupportStatus(StrEnum):
    """Support status for running a planner family in multi-AMV scenarios."""

    NATIVE = "native"
    ADAPTER = "adapter"
    NOT_AVAILABLE = "not_available"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True)
class MultiAmvPlannerSupport:
    """Planner-family support classification for multi-AMV execution."""

    planner_family: str
    support_status: MultiAmvPlannerSupportStatus
    contract_kind: str
    action_shape: str
    robot_identity: str
    collision_responsibility: str
    metadata_reporting: str
    rationale: str

    def to_json_dict(self) -> dict[str, str]:
        """Return JSON-compatible support metadata.

        Returns
        -------
        dict[str, str]
            Planner support classification and minimum contract fields.
        """
        return {
            "planner_family": self.planner_family,
            "support_status": self.support_status.value,
            "contract_kind": self.contract_kind,
            "action_shape": self.action_shape,
            "robot_identity": self.robot_identity,
            "collision_responsibility": self.collision_responsibility,
            "metadata_reporting": self.metadata_reporting,
            "rationale": self.rationale,
        }


_MULTI_AMV_PLANNER_SUPPORT: dict[str, MultiAmvPlannerSupport] = {
    "goal_controller_smoke": MultiAmvPlannerSupport(
        planner_family="goal_controller_smoke",
        support_status=MultiAmvPlannerSupportStatus.NATIVE,
        contract_kind="goal_controller_smoke",
        action_shape="array[num_robots, 2] unicycle velocity commands",
        robot_identity="implicit row order from MultiRobotEnv simulators",
        collision_responsibility="smoke runner only; not a coordinated fleet planner",
        metadata_reporting="planner_support block plus inter-robot metrics",
        rationale=(
            "Supported only as the minimal smoke controller used to exercise multi-AMV "
            "episode records; it is not benchmark-comparable planner-family support."
        ),
    ),
    "goal": MultiAmvPlannerSupport(
        planner_family="goal",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_planner",
        action_shape="single robot action; no fleet action tensor",
        robot_identity="missing",
        collision_responsibility="missing inter-robot coordination contract",
        metadata_reporting="not available",
        rationale="The single-robot goal planner does not define multi-robot identity or actions.",
    ),
    "social_force": MultiAmvPlannerSupport(
        planner_family="social_force",
        support_status=MultiAmvPlannerSupportStatus.RESEARCH_ONLY,
        contract_kind="pedestrian_dynamics_or_single_robot_baseline",
        action_shape="not defined for coordinated robot fleet control",
        robot_identity="missing",
        collision_responsibility="research question; no benchmark contract yet",
        metadata_reporting="not available",
        rationale="Social-force behavior can inform research, but no multi-AMV planner adapter exists.",
    ),
    "orca": MultiAmvPlannerSupport(
        planner_family="orca",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_or_pairwise_adapter_missing",
        action_shape="not defined for all robots with stable robot ids",
        robot_identity="missing",
        collision_responsibility="missing fleet-level responsibility and metadata contract",
        metadata_reporting="not available",
        rationale=(
            "ORCA is the first plausible non-trivial candidate, but it needs an explicit "
            "multi-robot adapter before benchmark use."
        ),
    ),
    "ppo": MultiAmvPlannerSupport(
        planner_family="ppo",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_policy",
        action_shape="single policy action; no multi-robot observation/action schema",
        robot_identity="missing",
        collision_responsibility="missing learned fleet-control contract",
        metadata_reporting="not available",
        rationale="Existing PPO checkpoints are trained for single-robot environment contracts.",
    ),
    "guarded_ppo": MultiAmvPlannerSupport(
        planner_family="guarded_ppo",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_policy_with_guard",
        action_shape="single policy action; no multi-robot observation/action schema",
        robot_identity="missing",
        collision_responsibility="missing learned fleet-control contract",
        metadata_reporting="not available",
        rationale="Guarded PPO inherits the single-robot PPO contract and is not fleet-aware.",
    ),
    "sacadrl": MultiAmvPlannerSupport(
        planner_family="sacadrl",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_policy",
        action_shape="single robot action",
        robot_identity="missing",
        collision_responsibility="missing fleet-control contract",
        metadata_reporting="not available",
        rationale="SA-CADRL support is single-robot and has no multi-AMV adapter.",
    ),
    "teb": MultiAmvPlannerSupport(
        planner_family="teb",
        support_status=MultiAmvPlannerSupportStatus.RESEARCH_ONLY,
        contract_kind="testing_only_single_robot_planner",
        action_shape="single robot trajectory command",
        robot_identity="missing",
        collision_responsibility="research-only until a fleet adapter exists",
        metadata_reporting="not available",
        rationale="TEB is testing-only in this repo and not a coordinated multi-AMV planner.",
    ),
}


def multi_amv_planner_support_inventory() -> dict[str, dict[str, str]]:
    """Return planner-family multi-AMV support inventory.

    Returns
    -------
    dict[str, dict[str, str]]
        JSON-compatible support records keyed by planner family.
    """
    return {
        planner_family: support.to_json_dict()
        for planner_family, support in sorted(_MULTI_AMV_PLANNER_SUPPORT.items())
    }


def multi_amv_planner_support(planner_family: str) -> MultiAmvPlannerSupport:
    """Return the multi-AMV support classification for a planner family.

    Returns
    -------
    MultiAmvPlannerSupport
        Support classification for the requested planner family.
    """
    normalized = str(planner_family).strip().lower().replace("-", "_")
    try:
        return _MULTI_AMV_PLANNER_SUPPORT[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(_MULTI_AMV_PLANNER_SUPPORT))
        raise ValueError(
            f"unknown multi-AMV planner family {planner_family!r}; known: {known}"
        ) from exc


def ensure_multi_amv_planner_supported(
    planner_family: str,
    *,
    require_non_smoke: bool = False,
) -> MultiAmvPlannerSupport:
    """Fail closed when a planner family lacks a multi-AMV execution contract.

    Returns
    -------
    MultiAmvPlannerSupport
        Support classification for allowed planner families.
    """
    support = multi_amv_planner_support(planner_family)
    if support.support_status not in {
        MultiAmvPlannerSupportStatus.NATIVE,
        MultiAmvPlannerSupportStatus.ADAPTER,
    }:
        raise ValueError(
            f"planner family {support.planner_family!r} is {support.support_status.value} "
            f"for multi-AMV scenarios: {support.rationale}"
        )
    if require_non_smoke and support.contract_kind == "goal_controller_smoke":
        raise ValueError(
            "multi-AMV goal_controller_smoke is a smoke controller, not non-trivial "
            "planner-family support"
        )
    return support


def multi_amv_settings_from_scenario(scenario: dict[str, Any]) -> MultiAmvSettings:
    """Parse the optional ``multi_amv`` scenario block.

    Returns:
        MultiAmvSettings: Validated settings for the minimal multi-AMV slice.
    """
    raw = scenario.get("multi_amv")
    if raw is None:
        return MultiAmvSettings()
    if not isinstance(raw, dict):
        raise ValueError("multi_amv must be a mapping.")
    allowed = {
        "num_robots",
        "near_miss_distance_m",
        "collision_distance_m",
        "deadlock_speed_mps",
        "deadlock_window_steps",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"multi_amv contains unknown keys: {', '.join(unknown)}.")
    settings = MultiAmvSettings(
        num_robots=int(raw.get("num_robots", 1)),
        near_miss_distance_m=float(raw.get("near_miss_distance_m", 1.0)),
        collision_distance_m=float(raw.get("collision_distance_m", 0.4)),
        deadlock_speed_mps=float(raw.get("deadlock_speed_mps", 0.05)),
        deadlock_window_steps=int(raw.get("deadlock_window_steps", 10)),
    )
    if settings.num_robots < 1:
        raise ValueError("multi_amv.num_robots must be >= 1.")
    if settings.collision_distance_m <= 0.0:
        raise ValueError("multi_amv.collision_distance_m must be > 0.")
    if settings.near_miss_distance_m <= settings.collision_distance_m:
        raise ValueError("multi_amv.near_miss_distance_m must be > collision_distance_m.")
    if settings.deadlock_speed_mps < 0.0:
        raise ValueError("multi_amv.deadlock_speed_mps must be >= 0.")
    if settings.deadlock_window_steps < 1:
        raise ValueError("multi_amv.deadlock_window_steps must be >= 1.")
    return settings


def inter_robot_metrics(
    robot_positions: np.ndarray,
    *,
    dt: float,
    settings: MultiAmvSettings,
) -> dict[str, float | bool]:
    """Compute minimal inter-robot safety/deadlock metrics from trajectories.

    Args:
        robot_positions: Array shaped ``(steps, robots, 2)``.
        dt: Simulation step duration in seconds.
        settings: Multi-AMV metric thresholds.

    Returns:
        dict[str, float | bool]: JSON-safe inter-robot metrics where collision/near-miss
        events count contiguous encounter runs, and deadlock detection is fleet-wide for this
        first slice.
    """
    positions = np.asarray(robot_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("robot_positions must have shape (steps, robots, 2).")
    steps, robots, _ = positions.shape
    pair_count = (robots * (robots - 1)) // 2
    if robots < 2:
        return {
            "robot_count": float(robots),
            "pair_count": 0.0,
            "min_inter_robot_distance_m": float("nan"),
            "inter_robot_collision_events": 0.0,
            "inter_robot_near_miss_events": 0.0,
            "deadlock_steps": 0.0,
            "deadlock_detected": False,
        }
    if steps == 0:
        return {
            "robot_count": float(robots),
            "pair_count": float(pair_count),
            "min_inter_robot_distance_m": float("nan"),
            "inter_robot_collision_events": 0.0,
            "inter_robot_near_miss_events": 0.0,
            "deadlock_steps": 0.0,
            "deadlock_detected": False,
        }

    pair_distances = []
    for i in range(robots):
        for j in range(i + 1, robots):
            pair_distances.append(np.linalg.norm(positions[:, i, :] - positions[:, j, :], axis=1))
    distances = np.stack(pair_distances, axis=1)
    min_per_step = np.min(distances, axis=1)
    collision_events = _count_true_runs(min_per_step < settings.collision_distance_m)
    near_miss_mask = (min_per_step >= settings.collision_distance_m) & (
        min_per_step < settings.near_miss_distance_m
    )
    near_miss_events = _count_true_runs(near_miss_mask)

    deadlock_steps = 0
    deadlock_detected = False
    if steps >= 2:
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=2) / max(float(dt), 1e-9)
        # This minimal first slice only marks deadlock when the whole fleet stays slow.
        all_slow = np.all(speeds <= settings.deadlock_speed_mps, axis=1)
        deadlock_steps = int(np.count_nonzero(all_slow))
        deadlock_detected = _has_consecutive_true(all_slow, settings.deadlock_window_steps)

    return {
        "robot_count": float(robots),
        "pair_count": float(pair_count),
        "min_inter_robot_distance_m": float(np.min(min_per_step)),
        "inter_robot_collision_events": float(collision_events),
        "inter_robot_near_miss_events": float(near_miss_events),
        "deadlock_steps": float(deadlock_steps),
        "deadlock_detected": bool(deadlock_detected),
    }


def multi_amv_episode_extension(
    *,
    settings: MultiAmvSettings,
    inter_robot: dict[str, float | bool],
    planner_family: str,
    planner_status: str,
    planner_note: str | None = None,
) -> dict[str, Any]:
    """Build an additive episode-record block for multi-AMV benchmark outputs.

    The block is intentionally namespaced under ``multi_amv`` so single-robot
    episode consumers can ignore it without schema migration.

    Returns
    -------
    dict[str, Any]
        Namespaced episode extension containing settings, planner status, and inter-robot metrics.
    """
    if settings.num_robots < 2:
        raise ValueError("multi-AMV episode extension requires at least two robots")
    if not inter_robot:
        raise ValueError("inter_robot metrics must be non-empty")
    planner_support = multi_amv_planner_support(planner_family)
    payload: dict[str, Any] = {
        "multi_amv": {
            "enabled": True,
            "settings": {
                "num_robots": int(settings.num_robots),
                "near_miss_distance_m": float(settings.near_miss_distance_m),
                "collision_distance_m": float(settings.collision_distance_m),
                "deadlock_speed_mps": float(settings.deadlock_speed_mps),
                "deadlock_window_steps": int(settings.deadlock_window_steps),
            },
            "planner_family": planner_support.planner_family,
            "planner_status": str(planner_status),
            "planner_support": planner_support.to_json_dict(),
        }
    }
    if planner_note is not None:
        payload["multi_amv"]["planner_note"] = planner_note
    return payload


def paired_actuation_feasibility_ranking(
    records: list[dict[str, Any]],
    *,
    baseline_variant: str,
    intervention_variant: str,
) -> dict[str, Any]:
    """Summarize a paired AMV actuation-feasibility ranking slice.

    The helper is deliberately diagnostic: it only pairs rows with identical
    ``(scenario_id, seed)`` keys, excludes fallback/degraded/unavailable rows,
    and reports mechanism deltas without promoting them to benchmark-strength
    claims.

    Returns:
        Diagnostic ranking summary with pair rows, uncertainty, and exclusions.
    """
    baseline_variant = str(baseline_variant)
    intervention_variant = str(intervention_variant)
    excluded_rows: list[dict[str, Any]] = []
    rows_by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        variant = _record_variant(record)
        scenario_id = str(record.get("scenario_id", ""))
        seed = int(record.get("seed", -1))
        if variant not in {baseline_variant, intervention_variant}:
            continue
        row_status = _record_success_status(record)
        if row_status["excluded"]:
            excluded_rows.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "variant": variant,
                    "reason": row_status["reason"],
                }
            )
            continue
        rows_by_pair.setdefault((scenario_id, seed), {})[variant] = record

    pair_rows: list[dict[str, Any]] = []
    incomplete_pairs: list[dict[str, Any]] = []
    for (scenario_id, seed), variants in sorted(rows_by_pair.items()):
        if baseline_variant not in variants or intervention_variant not in variants:
            incomplete_pairs.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "present_variants": sorted(variants),
                    "missing_variants": sorted(
                        {baseline_variant, intervention_variant} - set(variants)
                    ),
                }
            )
            continue
        baseline = _ranking_mechanism_row(variants[baseline_variant])
        intervention = _ranking_mechanism_row(variants[intervention_variant])
        deltas = {
            key: _numeric_delta(intervention.get(key), baseline.get(key))
            for key in (
                "success",
                "collisions",
                "command_clip_fraction",
                "yaw_rate_saturation_fraction",
                "signed_braking_peak_m_s2",
                "final_route_progress_m",
                "final_distance_to_goal_m",
            )
        }
        pair_rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "baseline": baseline,
                "intervention": intervention,
                "deltas_intervention_minus_baseline": deltas,
                "disagreement_cases": _pair_disagreement_cases(deltas),
            }
        )

    clip_deltas = [
        row["deltas_intervention_minus_baseline"]["command_clip_fraction"]
        for row in pair_rows
        if _is_finite_number(row["deltas_intervention_minus_baseline"]["command_clip_fraction"])
    ]
    scenario_ids = sorted({row["scenario_id"] for row in pair_rows})
    seeds = sorted({int(row["seed"]) for row in pair_rows})
    ranking_supported = bool(
        pair_rows
        and len(scenario_ids) >= 2
        and len(seeds) >= 2
        and clip_deltas
        and all(delta <= 0.0 for delta in clip_deltas)
        and any(delta < 0.0 for delta in clip_deltas)
        and not excluded_rows
        and not incomplete_pairs
    )
    classification = (
        "bounded_diagnostic_feasibility_direction"
        if ranking_supported
        else "diagnostic_only_inconclusive"
    )
    return {
        "schema_version": "paired-amv-actuation-feasibility-ranking.v1",
        "claim_boundary": "diagnostic-only; not benchmark-strength, paper-facing, or hardware-calibrated AMV evidence",
        "baseline_variant": baseline_variant,
        "intervention_variant": intervention_variant,
        "classification": classification,
        "ranking_supported": ranking_supported,
        "uncertainty": {
            "paired_rows": len(pair_rows),
            "scenario_count": len(scenario_ids),
            "seed_count": len(seeds),
            "command_clip_delta_mean": _finite_mean(clip_deltas),
            "command_clip_delta_min": min(clip_deltas) if clip_deltas else None,
            "command_clip_delta_max": max(clip_deltas) if clip_deltas else None,
            "note": "small paired diagnostic slice; uncertainty remains high below broader benchmark scale",
        },
        "scenarios": scenario_ids,
        "seeds": seeds,
        "excluded_rows": excluded_rows,
        "incomplete_pairs": incomplete_pairs,
        "pairs": pair_rows,
        "disagreement_cases": _summary_disagreement_cases(
            pair_rows=pair_rows,
            excluded_rows=excluded_rows,
            incomplete_pairs=incomplete_pairs,
        ),
    }


def _count_true_runs(values: np.ndarray) -> int:
    """Count contiguous true runs in a boolean sequence.

    Returns:
        Number of distinct true runs in the sequence.
    """

    mask = np.asarray(values, dtype=bool)
    if mask.size == 0:
        return 0
    padded = np.concatenate((np.array([False]), mask, np.array([False])))
    starts = np.logical_not(padded[:-1]) & padded[1:]
    return int(np.count_nonzero(starts))


def _has_consecutive_true(values: np.ndarray, window: int) -> bool:
    """Return whether a boolean sequence contains ``window`` consecutive true values."""
    run = 0
    for value in values:
        run = run + 1 if bool(value) else 0
        if run >= window:
            return True
    return False


def _record_variant(record: dict[str, Any]) -> str:
    """Return the best available planner/candidate identifier for a record."""
    metadata = record.get("algorithm_metadata")
    if isinstance(metadata, dict):
        config = metadata.get("config")
        if isinstance(config, dict):
            planner_variant = str(config.get("planner_variant", ""))
            if planner_variant == "actuation_aware_hybrid_rule_v0":
                return planner_variant
            if (
                str(metadata.get("algorithm", "")) == "hybrid_rule_local_planner"
                and planner_variant == "hybrid_rule_v3_teb_like_rollout"
                and _number_or_none(config.get("max_linear_speed")) == 3.0
                and _number_or_none(config.get("max_linear_accel")) == 3.0
            ):
                return "hybrid_rule_v3_fast_progress"
        for key in ("planner_variant", "algorithm", "canonical_algorithm", "planner_key"):
            value = metadata.get(key)
            if value:
                return str(value)
    scenario_params = record.get("scenario_params")
    if isinstance(scenario_params, dict):
        for key in ("planner_variant", "candidate", "algo"):
            value = scenario_params.get(key)
            if value:
                return str(value)
    return str(record.get("planner_key") or record.get("algo") or "")


def _record_success_status(record: dict[str, Any]) -> dict[str, Any]:
    """Return fail-closed availability/readiness classification for one record."""
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    planner_kinematics = metadata.get("planner_kinematics")
    planner_kinematics = planner_kinematics if isinstance(planner_kinematics, dict) else {}
    values = {
        "status": str(record.get("status", metadata.get("status", "ok"))),
        "readiness_status": str(
            record.get("readiness_status", metadata.get("readiness_status", ""))
        ),
        "availability_status": str(
            record.get("availability_status", metadata.get("availability_status", ""))
        ),
        "execution_mode": str(
            record.get(
                "execution_mode",
                metadata.get("execution_mode", planner_kinematics.get("execution_mode", "")),
            )
        ),
    }
    excluded_tokens = {
        "fallback",
        "degraded",
        "failed",
        "not_available",
        "not-available",
        "partial-failure",
        "partial_failure",
    }
    for key, value in values.items():
        normalized = value.strip().lower()
        if normalized in excluded_tokens:
            return {"excluded": True, "reason": f"{key}={value}"}
    readiness_status = values["readiness_status"].strip().lower()
    availability_status = values["availability_status"].strip().lower()
    execution_mode = values["execution_mode"].strip().lower()
    if readiness_status not in {"native", "adapter"}:
        return {"excluded": True, "reason": f"readiness_status={values['readiness_status']}"}
    if availability_status != "available":
        return {"excluded": True, "reason": f"availability_status={values['availability_status']}"}
    if execution_mode not in {"native", "adapter", "mixed"}:
        return {"excluded": True, "reason": f"execution_mode={values['execution_mode']}"}
    return {"excluded": False, "reason": "available"}


def _ranking_mechanism_row(record: dict[str, Any]) -> dict[str, Any]:
    """Extract terminal and mechanism fields used by the diagnostic ranking.

    Returns:
        JSON-safe terminal and mechanism fields for one variant row.
    """
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    outcome = record.get("outcome")
    outcome = outcome if isinstance(outcome, dict) else {}
    return {
        "variant": _record_variant(record),
        "success": _number_or_none(metrics.get("success", outcome.get("route_complete"))),
        "collisions": _number_or_none(
            metrics.get("collisions", metrics.get("total_collision_count"))
        ),
        "timeout_mode": str(
            record.get("timeout_mode")
            or record.get("termination_reason")
            or metrics.get("timeout_mode")
            or ""
        ),
        "command_clip_fraction": _number_or_none(metrics.get("command_clip_fraction")),
        "yaw_rate_saturation_fraction": _number_or_none(
            metrics.get("yaw_rate_saturation_fraction")
        ),
        "signed_braking_peak_m_s2": _number_or_none(metrics.get("signed_braking_peak_m_s2")),
        "final_route_progress_m": _final_route_progress(record),
        "final_distance_to_goal_m": _final_distance_to_goal(record),
    }


def _final_route_progress(record: dict[str, Any]) -> float | None:
    """Extract final synthetic-actuation route progress from known record shapes.

    Returns:
        Final route-progress value when present, otherwise ``None``.
    """
    for source in (record, record.get("metrics")):
        if isinstance(source, dict) and _is_finite_number(source.get("final_route_progress_m")):
            return float(source["final_route_progress_m"])
    final_step = _final_synthetic_actuation_step(record)
    if isinstance(final_step, dict) and _is_finite_number(
        final_step.get("route_progress_from_start_m")
    ):
        return float(final_step["route_progress_from_start_m"])
    return None


def _final_distance_to_goal(record: dict[str, Any]) -> float | None:
    """Extract final distance-to-goal from known record shapes.

    Returns:
        Final distance-to-goal value when present, otherwise ``None``.
    """
    if _is_finite_number(record.get("final_distance_to_goal_m")):
        return float(record["final_distance_to_goal_m"])
    delta = record.get("distance_to_goal_delta")
    if isinstance(delta, dict) and _is_finite_number(delta.get("final_distance_to_goal_m")):
        return float(delta["final_distance_to_goal_m"])
    metrics = record.get("metrics")
    if isinstance(metrics, dict) and _is_finite_number(metrics.get("final_distance_to_goal_m")):
        return float(metrics["final_distance_to_goal_m"])
    final_step = _final_synthetic_actuation_step(record)
    if isinstance(final_step, dict) and _is_finite_number(final_step.get("distance_to_goal_m")):
        return float(final_step["distance_to_goal_m"])
    return None


def _final_synthetic_actuation_step(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the last synthetic-actuation trace step when present."""
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, dict):
        return None
    synthetic_actuation = metadata.get("synthetic_actuation")
    if not isinstance(synthetic_actuation, dict):
        return None
    trace = synthetic_actuation.get("trace")
    if not isinstance(trace, dict):
        return None
    steps = trace.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[-1], dict):
        return steps[-1]
    return None


def _pair_disagreement_cases(deltas: dict[str, float | None]) -> list[dict[str, str]]:
    """Classify within-pair terminal/mechanism disagreements.

    Returns:
        List of disagreement case summaries for one paired row.
    """
    cases: list[dict[str, str]] = []
    success_delta = deltas.get("success")
    clip_delta = deltas.get("command_clip_fraction")
    if success_delta == 0.0 and _is_finite_number(clip_delta) and clip_delta != 0.0:
        cases.append(
            {
                "type": "feasibility_success_divergence",
                "summary": "command clipping changed while success stayed tied",
            }
        )
    progress_delta = deltas.get("final_route_progress_m")
    if _is_finite_number(clip_delta) and clip_delta < 0.0 and _is_finite_number(progress_delta):
        if progress_delta < 0.0:
            cases.append(
                {
                    "type": "feasibility_progress_tradeoff",
                    "summary": "command clipping improved while final route progress worsened",
                }
            )
    return cases


def _summary_disagreement_cases(
    *,
    pair_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    incomplete_pairs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Summarize cross-slice blockers and mechanism disagreements.

    Returns:
        List of summary-level disagreement or blocker cases.
    """
    cases: list[dict[str, str]] = []
    if excluded_rows:
        cases.append(
            {
                "type": "fallback_degraded_exclusion",
                "summary": f"{len(excluded_rows)} fallback/degraded/unavailable rows excluded",
            }
        )
    if incomplete_pairs:
        cases.append(
            {
                "type": "incomplete_pairing",
                "summary": f"{len(incomplete_pairs)} scenario/seed pairs missing one variant",
            }
        )
    pair_case_types = sorted(
        {
            case["type"]
            for row in pair_rows
            for case in row.get("disagreement_cases", [])
            if isinstance(case, dict) and "type" in case
        }
    )
    for case_type in pair_case_types:
        cases.append(
            {
                "type": case_type,
                "summary": "observed in at least one paired scenario/seed row",
            }
        )
    if not cases and pair_rows:
        cases.append(
            {
                "type": "no_pairwise_disagreement",
                "summary": "paired rows did not expose terminal/mechanism disagreement",
            }
        )
    return cases


def _numeric_delta(value: Any, baseline: Any) -> float | None:
    """Return ``value - baseline`` when both values are finite numbers."""
    if _is_finite_number(value) and _is_finite_number(baseline):
        return float(value) - float(baseline)
    return None


def _number_or_none(value: Any) -> float | None:
    """Convert finite numeric and boolean values to float, otherwise return ``None``.

    Returns:
        Finite float value when possible, otherwise ``None``.
    """
    if isinstance(value, bool):
        return float(value)
    if _is_finite_number(value):
        return float(value)
    return None


def _is_finite_number(value: Any) -> bool:
    """Return whether ``value`` can be interpreted as a finite float."""
    if isinstance(value, bool):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_mean(values: list[float]) -> float | None:
    """Return the finite arithmetic mean for ``values`` or ``None``."""
    finite_values = [float(value) for value in values if _is_finite_number(value)]
    return float(fmean(finite_values)) if finite_values else None
