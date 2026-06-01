#!/usr/bin/env python3
"""Extract closest-approach trace slices for paired scenario perturbations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.algorithm_metadata import resolve_observation_mode
from robot_sf.benchmark.map_runner import (
    _apply_active_observation_mode_to_env_config,
    _apply_planner_selector_v2_context,
    _build_env_config,
    _build_policy,
    _parse_algo_config,
    _policy_command_to_env_action,
    _resolve_policy_search_candidate_runtime,
    _robot_kinematics_label,
    _scenario_with_episode_seed_defaults,
    _validate_planner_contract,
    _validate_sensor_fusion_adapter_config,
)
from robot_sf.benchmark.termination_reason import (
    collision_event,
    resolve_termination_reason,
    route_complete_success,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.scenario_certification import materialize_perturbation_pilot_matrix
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_scenario_perturbation_criticality_pilot import (
    PlannerRunSpec,
    _planner_output_stem,
    _scenario_metadata,
    resolve_planner_run_spec,
)

SCHEMA_VERSION = "scenario_perturbation_trace_response.v1"


@dataclass(frozen=True)
class TraceRun:
    """One diagnostic rollout trace."""

    planner: str
    scenario_id: str
    source_scenario_id: str
    variant_id: str
    family: str
    seed: int
    termination_reason: str
    frames: list[dict[str, Any]]


def _build_parser() -> argparse.ArgumentParser:
    """Build the trace-response CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Scenario perturbation manifest YAML.")
    parser.add_argument(
        "--materialized-output-dir",
        type=Path,
        required=True,
        help="Ignored output/ directory for generated scenario matrix and route overrides.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Tracked compact JSON report path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional tracked Markdown report path.",
    )
    parser.add_argument(
        "--source-scenario-id",
        default="classic_head_on_corridor_low",
        help="Source scenario to inspect.",
    )
    parser.add_argument(
        "--perturbed-family",
        default="pedestrian_route_offset",
        help="Perturbation family to pair against the no-op variant.",
    )
    parser.add_argument(
        "--perturbed-variant-id",
        help=(
            "Optional exact perturbed variant_id to pair against the no-op. "
            "When omitted, the first variant matching --perturbed-family is used."
        ),
    )
    parser.add_argument(
        "--planner",
        action="append",
        dest="planners",
        help="Planner algorithm or policy-search candidate key. Defaults to goal, orca, and the promoted collision guard.",
    )
    parser.add_argument(
        "--planner-candidate-registry",
        type=Path,
        default=Path("docs/context/policy_search/candidate_registry.yaml"),
        help="Policy-search candidate registry used to resolve planner keys.",
    )
    parser.add_argument("--seed-limit", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--slice-window",
        type=int,
        default=2,
        help="Frames to include before and after each closest approach.",
    )
    return parser


def _round_float(value: float | None, *, digits: int = 6) -> float | None:
    """Round finite floats for compact JSON output."""
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _xy(value: Any) -> list[float]:
    """Return a compact ``[x, y]`` pair."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 2:
        return [0.0, 0.0]
    return [_round_float(float(arr[0])) or 0.0, _round_float(float(arr[1])) or 0.0]


def _closest_pedestrian_payload(
    *,
    robot_position: np.ndarray,
    pedestrian_positions: np.ndarray,
    robot_radius: float,
    ped_radius: float,
) -> dict[str, Any] | None:
    """Return the closest pedestrian payload for one frame."""
    if pedestrian_positions.size == 0:
        return None
    if pedestrian_positions.ndim != 2 or pedestrian_positions.shape[1] < 2:
        return None
    distances = np.linalg.norm(pedestrian_positions[:, :2] - robot_position[:2], axis=1)
    if distances.size == 0 or not np.isfinite(distances).any():
        return None
    ped_index = int(np.nanargmin(distances))
    center_distance = float(distances[ped_index])
    clearance = center_distance - float(robot_radius + ped_radius)
    return {
        "pedestrian_index": ped_index,
        "pedestrian_position": _xy(pedestrian_positions[ped_index]),
        "center_distance_m": _round_float(center_distance),
        "clearance_m": _round_float(clearance),
    }


def closest_approach_slice(
    frames: list[dict[str, Any]],
    *,
    window: int = 2,
) -> dict[str, Any]:
    """Return the closest-approach frame and a compact local slice."""
    candidates = [
        (index, frame)
        for index, frame in enumerate(frames)
        if isinstance(frame.get("closest_pedestrian"), dict)
        and frame["closest_pedestrian"].get("center_distance_m") is not None
    ]
    if not candidates:
        return {"status": "no_pedestrians", "closest": None, "frames": []}
    closest_index, closest = min(
        candidates,
        key=lambda item: float(item[1]["closest_pedestrian"]["center_distance_m"]),
    )
    start = max(0, closest_index - max(0, int(window)))
    stop = min(len(frames), closest_index + max(0, int(window)) + 1)
    return {
        "status": "ok",
        "closest": closest,
        "frames": frames[start:stop],
    }


def _delta(after: float | int | None, before: float | int | None) -> float | None:
    """Return a rounded numeric delta when both sides are present."""
    if after is None or before is None:
        return None
    return _round_float(float(after) - float(before))


def build_trace_pair_summary(
    *,
    planner: str,
    seed: int,
    noop: TraceRun,
    perturbed: TraceRun,
    window: int,
) -> dict[str, Any]:
    """Build one paired closest-approach summary."""
    noop_slice = closest_approach_slice(noop.frames, window=window)
    perturbed_slice = closest_approach_slice(perturbed.frames, window=window)
    noop_closest = noop_slice.get("closest") if isinstance(noop_slice.get("closest"), dict) else {}
    perturbed_closest = (
        perturbed_slice.get("closest") if isinstance(perturbed_slice.get("closest"), dict) else {}
    )
    noop_ped = (
        noop_closest.get("closest_pedestrian") if isinstance(noop_closest, dict) else None
    ) or {}
    perturbed_ped = (
        perturbed_closest.get("closest_pedestrian") if isinstance(perturbed_closest, dict) else None
    ) or {}
    return {
        "planner": planner,
        "seed": int(seed),
        "source_scenario_id": noop.source_scenario_id,
        "noop_variant_id": noop.variant_id,
        "perturbed_variant_id": perturbed.variant_id,
        "perturbed_family": perturbed.family,
        "pair_status": (
            "completed"
            if noop_slice["status"] == "ok" and perturbed_slice["status"] == "ok"
            else "excluded"
        ),
        "noop_termination_reason": noop.termination_reason,
        "perturbed_termination_reason": perturbed.termination_reason,
        "closest_approach_delta": {
            "time_s": _delta(perturbed_closest.get("time_s"), noop_closest.get("time_s")),
            "center_distance_m": _delta(
                perturbed_ped.get("center_distance_m"),
                noop_ped.get("center_distance_m"),
            ),
            "clearance_m": _delta(perturbed_ped.get("clearance_m"), noop_ped.get("clearance_m")),
            "goal_distance_m": _delta(
                perturbed_closest.get("goal_distance_m"),
                noop_closest.get("goal_distance_m"),
            ),
            "progress_m": _delta(
                perturbed_closest.get("progress_m"),
                noop_closest.get("progress_m"),
            ),
        },
        "noop_closest_approach": noop_slice,
        "perturbed_closest_approach": perturbed_slice,
    }


def _summarize_trace_pair_subset(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one subset of trace-pair deltas."""
    status_counts: dict[str, int] = defaultdict(int)
    delta_values: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        status_counts[str(row.get("pair_status") or "unknown")] += 1
        if row.get("pair_status") != "completed":
            continue
        deltas = row.get("closest_approach_delta")
        if not isinstance(deltas, dict):
            continue
        for field, value in deltas.items():
            if isinstance(value, int | float):
                delta_values[field].append(float(value))
    return {
        "pairs": len(pair_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "mean_closest_approach_deltas_completed_pairs": {
            field: _round_float(sum(values) / len(values))
            for field, values in sorted(delta_values.items())
            if values
        },
    }


def summarize_trace_pairs(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate trace-pair deltas overall and by planner."""
    by_planner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_planner[str(row.get("planner") or "unknown")].append(row)
    return {
        **_summarize_trace_pair_subset(pair_rows),
        "by_planner": {
            planner: _summarize_trace_pair_subset(rows)
            for planner, rows in sorted(by_planner.items())
        },
    }


def _frame_payload(
    *,
    step: int,
    dt: float,
    robot_position: np.ndarray,
    pedestrian_positions: np.ndarray,
    goal_position: np.ndarray,
    start_goal_distance: float,
    robot_radius: float,
    ped_radius: float,
) -> dict[str, Any]:
    """Build one compact diagnostic frame."""
    goal_distance = float(np.linalg.norm(goal_position[:2] - robot_position[:2]))
    return {
        "step": int(step),
        "time_s": _round_float((int(step) + 1) * float(dt)),
        "robot_position": _xy(robot_position),
        "goal_distance_m": _round_float(goal_distance),
        "progress_m": _round_float(start_goal_distance - goal_distance),
        "closest_pedestrian": _closest_pedestrian_payload(
            robot_position=robot_position,
            pedestrian_positions=pedestrian_positions,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
        ),
    }


def _run_trace_episode(
    scenario: dict[str, Any],
    *,
    seed: int,
    planner_spec: PlannerRunSpec,
    horizon: int,
    dt: float,
    scenario_path: Path,
) -> TraceRun:
    """Run one local diagnostic episode and capture compact per-step geometry."""
    scenario = _scenario_with_episode_seed_defaults(scenario, seed=seed)
    metadata = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    perturbation = metadata.get("scenario_perturbation") if isinstance(metadata, dict) else {}
    config = _build_env_config(scenario, scenario_path=scenario_path)
    config.sim_config.time_per_step_in_secs = float(dt)
    raw_policy_cfg = (
        _parse_algo_config(planner_spec.algo_config_path.as_posix())
        if planner_spec.algo_config_path is not None
        else {}
    )
    algo, policy_cfg = _resolve_policy_search_candidate_runtime(
        default_algo=planner_spec.algo,
        algo_config_path=(
            planner_spec.algo_config_path.as_posix()
            if planner_spec.algo_config_path is not None
            else None
        ),
        algo_config=raw_policy_cfg,
        scenario=scenario,
    )
    policy_cfg = _apply_planner_selector_v2_context(
        algo,
        policy_cfg,
        scenario=scenario,
        seed=int(seed),
    )
    observation_mode = resolve_observation_mode(algo, None, observation_level=None)
    _apply_active_observation_mode_to_env_config(
        config,
        active_observation_mode=observation_mode,
    )
    _validate_sensor_fusion_adapter_config(
        algo=algo,
        active_observation_mode=observation_mode,
        algo_config=policy_cfg,
    )
    robot_kinematics = _robot_kinematics_label(config)
    robot_command_mode = str(getattr(config.robot_config, "command_mode", "vx_vy")).strip().lower()
    _validate_planner_contract(
        algo=algo,
        robot_kinematics=robot_kinematics,
        algo_config=policy_cfg,
        observation_mode=observation_mode,
        observation_level=None,
    )
    policy_fn, _algo_meta = _build_policy(
        algo,
        policy_cfg,
        robot_kinematics=robot_kinematics,
        robot_command_mode=robot_command_mode,
        adapter_impact_eval=False,
    )
    planner_close = getattr(policy_fn, "_planner_close", None)
    planner_reset = getattr(policy_fn, "_planner_reset", None)
    planner_bind_env = getattr(policy_fn, "_planner_bind_env", None)
    planner_native_action = getattr(policy_fn, "_planner_native_env_action", False)

    env = make_robot_env(config=config, seed=int(seed), debug=False)
    frames: list[dict[str, Any]] = []
    termination_reason = "max_steps"
    try:
        obs, _ = env.reset(seed=int(seed))
        if callable(planner_bind_env):
            planner_bind_env(env)
        if callable(planner_reset):
            planner_reset(seed=int(seed))
        goal_position = np.asarray(env.simulator.goal_pos[0], dtype=float)
        robot_start = np.asarray(env.simulator.robot_pos[0], dtype=float)
        start_goal_distance = float(np.linalg.norm(goal_position[:2] - robot_start[:2]))
        robot_radius = float(getattr(config.robot_config, "radius", 1.0))
        ped_radius = float(getattr(config.sim_config, "ped_radius", 0.4))

        for step_idx in range(int(horizon)):
            policy_command = policy_fn(obs)
            step_is_native = getattr(policy_fn, "_last_step_native", planner_native_action)
            if step_is_native:
                action = np.asarray(policy_command, dtype=np.float32)
            else:
                action = _policy_command_to_env_action(
                    env=env,
                    config=config,
                    command=policy_command,
                )
            obs, _reward, terminated, truncated, info = env.step(action)
            robot_position = np.asarray(env.simulator.robot_pos[0], dtype=float)
            pedestrian_positions = np.asarray(env.simulator.ped_pos, dtype=float)
            frames.append(
                _frame_payload(
                    step=step_idx,
                    dt=dt,
                    robot_position=robot_position,
                    pedestrian_positions=pedestrian_positions,
                    goal_position=goal_position,
                    start_goal_distance=start_goal_distance,
                    robot_radius=robot_radius,
                    ped_radius=ped_radius,
                )
            )

            step_collision = collision_event(info)
            step_success = route_complete_success(info) and not step_collision
            if step_success:
                termination_reason = resolve_termination_reason(
                    terminated=True,
                    truncated=False,
                    success=True,
                    collision=step_collision,
                )
                break
            if terminated or truncated:
                termination_reason = resolve_termination_reason(
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    success=step_success,
                    collision=step_collision,
                )
                break
    finally:
        if callable(planner_close):
            planner_close()
        env.close()

    scenario_id = str(scenario.get("scenario_id") or scenario.get("name") or "")
    return TraceRun(
        planner=planner_spec.label,
        scenario_id=scenario_id,
        source_scenario_id=str(perturbation.get("source_scenario_id") or scenario_id),
        variant_id=str(perturbation.get("variant_id") or scenario_id),
        family=str(perturbation.get("family") or ""),
        seed=int(seed),
        termination_reason=termination_reason,
        frames=frames,
    )


def _select_variant_pair(
    scenarios: list[dict[str, Any]],
    *,
    source_scenario_id: str,
    perturbed_family: str,
    perturbed_variant_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the no-op and perturbed scenario entries for one source scenario."""
    noop: dict[str, Any] | None = None
    perturbed: dict[str, Any] | None = None
    for scenario in scenarios:
        metadata = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
        payload = metadata.get("scenario_perturbation") if isinstance(metadata, dict) else None
        if not isinstance(payload, dict):
            continue
        if str(payload.get("source_scenario_id") or "") != source_scenario_id:
            continue
        family = str(payload.get("family") or "")
        variant_id = str(payload.get("variant_id") or "")
        if family == "noop" and noop is None:
            noop = scenario
        elif (
            family == perturbed_family
            and perturbed is None
            and (perturbed_variant_id is None or variant_id == perturbed_variant_id)
        ):
            perturbed = scenario
    if noop is None:
        raise ValueError(f"No no-op variant found for source scenario {source_scenario_id!r}")
    if perturbed is None:
        if perturbed_variant_id is not None:
            raise ValueError(
                f"No {perturbed_family!r} variant with variant_id {perturbed_variant_id!r} "
                f"found for source scenario {source_scenario_id!r}"
            )
        raise ValueError(
            f"No {perturbed_family!r} variant found for source scenario {source_scenario_id!r}"
        )
    return noop, perturbed


def _scenario_seeds(scenario: dict[str, Any]) -> list[int]:
    """Return explicit scenario seeds."""
    raw = scenario.get("seeds")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Scenario {scenario.get('scenario_id')} has no explicit seeds")
    return [int(seed) for seed in raw]


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown trace-response report."""
    summary = report["pair_summary"]
    lines = [
        "# Scenario Perturbation Trace Response",
        "",
        "## Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Scope",
        "",
        f"- Source scenario: `{report['source_scenario_id']}`",
        f"- Perturbed family: `{report['perturbed_family']}`",
        f"- Perturbed variant: `{report['perturbed_variant_id']}`",
        f"- Planners: {', '.join(f'`{planner}`' for planner in report['planners'])}",
        f"- Pair rows: `{summary['pairs']}`",
        f"- Pair statuses: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
        "",
        "## Mean Closest-Approach Deltas",
        "",
    ]
    deltas = summary["mean_closest_approach_deltas_completed_pairs"]
    if deltas:
        for field, value in deltas.items():
            lines.append(f"- `{field}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Positive clearance/center-distance deltas mean the perturbed run was farther from the closest pedestrian at its closest approach.",
            "",
            "## By Planner",
            "",
            "| Planner | Pairs | Center Distance Delta | Progress Delta | Time Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for planner, planner_summary in sorted(summary.get("by_planner", {}).items()):
        planner_deltas = planner_summary.get("mean_closest_approach_deltas_completed_pairs", {})
        lines.append(
            "| "
            f"`{planner}` | "
            f"{planner_summary.get('pairs', 0)} | "
            f"`{planner_deltas.get('center_distance_m')}` | "
            f"`{planner_deltas.get('progress_m')}` | "
            f"`{planner_deltas.get('time_s')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the trace-response diagnostic."""
    args = _build_parser().parse_args()
    planners = args.planners or [
        "goal",
        "orca",
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
    ]
    planner_specs = [
        resolve_planner_run_spec(planner, candidate_registry_path=args.planner_candidate_registry)
        for planner in planners
    ]
    materialized = materialize_perturbation_pilot_matrix(
        args.manifest,
        output_dir=args.materialized_output_dir,
        seed_limit=args.seed_limit,
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    scenarios = load_scenarios(matrix_path)
    metadata = _scenario_metadata(scenarios)
    noop_scenario, perturbed_scenario = _select_variant_pair(
        scenarios,
        source_scenario_id=args.source_scenario_id,
        perturbed_family=args.perturbed_family,
        perturbed_variant_id=args.perturbed_variant_id,
    )
    perturbed_metadata = (
        perturbed_scenario.get("metadata")
        if isinstance(perturbed_scenario.get("metadata"), dict)
        else {}
    )
    perturbed_payload = (
        perturbed_metadata.get("scenario_perturbation")
        if isinstance(perturbed_metadata, dict)
        else {}
    )
    selected_perturbed_variant_id = (
        str(perturbed_payload.get("variant_id") or perturbed_scenario.get("scenario_id") or "")
        if isinstance(perturbed_payload, dict)
        else str(perturbed_scenario.get("scenario_id") or "")
    )
    seeds = sorted(set(_scenario_seeds(noop_scenario)) & set(_scenario_seeds(perturbed_scenario)))
    pair_rows: list[dict[str, Any]] = []
    for planner_spec in planner_specs:
        for seed in seeds:
            noop_trace = _run_trace_episode(
                noop_scenario,
                seed=seed,
                planner_spec=planner_spec,
                horizon=args.horizon,
                dt=args.dt,
                scenario_path=matrix_path,
            )
            perturbed_trace = _run_trace_episode(
                perturbed_scenario,
                seed=seed,
                planner_spec=planner_spec,
                horizon=args.horizon,
                dt=args.dt,
                scenario_path=matrix_path,
            )
            pair_rows.append(
                build_trace_pair_summary(
                    planner=planner_spec.label,
                    seed=seed,
                    noop=noop_trace,
                    perturbed=perturbed_trace,
                    window=args.slice_window,
                )
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "manifest": args.manifest.as_posix(),
        "source_scenario_id": args.source_scenario_id,
        "perturbed_family": args.perturbed_family,
        "perturbed_variant_id": selected_perturbed_variant_id,
        "requested_perturbed_variant_id": args.perturbed_variant_id,
        "planners": [spec.label for spec in planner_specs],
        "horizon": args.horizon,
        "dt": args.dt,
        "seed_limit": args.seed_limit,
        "slice_window": args.slice_window,
        "materialization": {
            "schema_version": materialized.schema_version,
            "manifest_id": materialized.manifest_id,
            "included_variants": list(materialized.included_variants),
            "excluded_variants": list(materialized.excluded_variants),
            "variant_count": len(materialized.included_variants),
            "local_artifact_boundary": (
                "materialized scenario matrix and route overrides remain ignored local outputs "
                "reproducible from the tracked manifest and command"
            ),
        },
        "variant_metadata": {
            key: value
            for key, value in sorted(metadata.items())
            if value.get("source_scenario_id") == args.source_scenario_id
            and value.get("family") in {"noop", args.perturbed_family}
        },
        "planner_runs": {
            spec.label: {
                "algo": spec.algo,
                "algo_config_path": (
                    spec.algo_config_path.as_posix() if spec.algo_config_path is not None else None
                ),
                "source": spec.source,
                "output_stem": _planner_output_stem(spec.label),
            }
            for spec in planner_specs
        },
        "pair_summary": summarize_trace_pairs(pair_rows),
        "trace_pairs": pair_rows,
        "claim_boundary": (
            "diagnostic local trace inspection only; not benchmark-strength or paper-facing evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        _write_markdown(report, args.markdown_output)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "markdown_output": (
                    args.markdown_output.as_posix() if args.markdown_output is not None else None
                ),
                "pair_summary": report["pair_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
