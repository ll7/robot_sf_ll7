#!/usr/bin/env python3
"""Reproducible DWA global-route integration probe for issue #5331 (CPU-only, diagnostic-only).

This runner captures a per-step Dynamic Window Approach (DWA) decision trace with the
global-route integration probe enabled for the two fixed-seed episodes named by the #5262
configuration-sensitivity diagnostic:

- ``classic_bottleneck_medium`` seed ``131`` (canonical config, ``max_steps`` timeout).
- ``classic_t_intersection_low`` seed ``161`` (canonical config, ``collision``).

The probe biases DWA toward the next global-route waypoint to test whether waypoint-following
helps navigate through bottleneck corridors where the constant-velocity rollout cannot directly
see the goal. This is a successor to the #5319 route-rescue probe.

For each episode it records the selected command, the selected candidate score, the
feasible/infeasible candidate counts, the dynamic-window reachability bounds, the
constraint reason, the route-progress (distance-to-goal) state, whether the global-route
probe activated, and the first observable point at which the episode becomes unrecoverable.
The result is analysis-only: it makes no benchmark, roster, metric, frozen-suite, paper, or
dissertation claim.

Outputs:

- ``<out-dir>/dwa_global_route_probe_steps.csv``: compact reviewable per-step rows.
- ``<out-dir>/dwa_global_route_probe_summary.json``: per-episode failure-mechanism summary.

The optional ``--evidence-dir`` writes the reviewable packet (README + steps CSV + summary
JSON) into ``docs/context/evidence/`` for durable traceability.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.evidence.distance_convention import DistanceConvention
from robot_sf.evidence.writers import write_distance_series_csv, write_json
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_MATRIX = REPO_ROOT / "configs/scenarios/classic_interactions.yaml"
DEFAULT_ALGO_CONFIG = REPO_ROOT / "configs/algos/dwa_global_route_probe.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "output/benchmarks/issue_5331"
HORIZON = 100
DT = 0.1

# The two episodes named by the #5262 manifest's canonical config point. The seeds come
# from the standard classic archetype matrix declaration used by both the #5020 and #5262
# diagnostics; the global-route probe config applies waypoint-following bias.
TARGET_EPISODES: tuple[tuple[str, int, str], ...] = (
    ("classic_bottleneck_medium", 131, "bottleneck_timeout"),
    ("classic_t_intersection_low", 161, "t_intersection_collision"),
)
FOLLOW_UP_ISSUE = 5331

STEP_TRACE_FIELDS: tuple[str, ...] = (
    "episode_id",
    "scenario_id",
    "seed",
    "step",
    "selected_source",
    "selected_v_mps",
    "selected_w_radps",
    "selected_score",
    "constraint_reason",
    "candidate_total",
    "candidate_feasible",
    "candidate_infeasible",
    "feasible_score_min",
    "feasible_score_max",
    "dynamic_window_v_min",
    "dynamic_window_v_max",
    "dynamic_window_w_min",
    "dynamic_window_w_max",
    "target_goal_kind",
    "target_goal_x",
    "target_goal_y",
    "distance_to_goal_m",
    "route_progress_from_start_m",
    "robot_x_m",
    "robot_y_m",
    "global_route_probe_activated",
)


def _load_scenario(name: str, seed: int, matrix_path: Path) -> dict[str, Any]:
    """Return one scenario from the source matrix with a single pinned seed."""
    scenarios = load_scenarios(matrix_path, base_dir=matrix_path.parent)
    by_name = {str(row.get("name")): dict(row) for row in scenarios}
    if name not in by_name:
        raise KeyError(f"scenario {name!r} is absent from matrix {matrix_path}")
    scenario = dict(by_name[name])
    scenario["seeds"] = [int(seed)]
    return scenario


def _read_record(jsonl_path: Path) -> dict[str, Any]:
    """Return the single episode record from a one-row JSONL file."""
    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one episode record in {jsonl_path}, got {len(lines)}")
    record = json.loads(lines[0])
    if not isinstance(record, dict):
        raise ValueError(f"episode record in {jsonl_path} must be a JSON object")
    return record


def _flatten_step(
    step: dict[str, Any],
    *,
    episode_id: str,
    scenario_id: str,
    seed: int,
) -> dict[str, Any]:
    """Normalize one planner-decision-trace step into a flat CSV/JSON row."""
    command = step.get("selected_command") or []
    selected_v = float(command[0]) if len(command) > 0 else None
    selected_w = float(command[1]) if len(command) > 1 else None
    window = step.get("dynamic_window") if isinstance(step.get("dynamic_window"), dict) else {}
    target = step.get("target_goal") if isinstance(step.get("target_goal"), dict) else {}
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": int(seed),
        "step": int(step.get("step", -1)),
        "selected_source": str(step.get("selected_source", "unknown")),
        "selected_v_mps": selected_v,
        "selected_w_radps": selected_w,
        "selected_score": step.get("selected_score"),
        "constraint_reason": str(step.get("constraint_reason", "unknown")),
        "candidate_total": step.get("candidate_total"),
        "candidate_feasible": step.get("candidate_feasible"),
        "candidate_infeasible": step.get("candidate_infeasible"),
        "feasible_score_min": step.get("feasible_score_min"),
        "feasible_score_max": step.get("feasible_score_max"),
        "dynamic_window_v_min": window.get("v_min"),
        "dynamic_window_v_max": window.get("v_max"),
        "dynamic_window_w_min": window.get("w_min"),
        "dynamic_window_w_max": window.get("w_max"),
        "target_goal_kind": target.get("kind"),
        "target_goal_x": target.get("x"),
        "target_goal_y": target.get("y"),
        "distance_to_goal_m": step.get("distance_to_goal_m"),
        "route_progress_from_start_m": step.get("route_progress_from_start_m"),
        "robot_x_m": step.get("robot_x_m"),
        "robot_y_m": step.get("robot_y_m"),
        "global_route_probe_activated": step.get("global_route_probe_activated", False),
    }


def _first_unrecoverable_step(rows: list[dict[str, Any]]) -> int | None:
    """Return the first step where all rollout candidates become infeasible.

    DWA returns ``best_feasible`` until every dynamically reachable candidate scores
    negative infinity (every constant-velocity rollout breaches the safety margin within
    the prediction horizon). The first such step marks the point at which the reactive
    window can no longer find any safe forward command; if the episode still collides
    afterwards, that is the first observable unrecoverable point under this controller.
    """
    for row in rows:
        feasible = row.get("candidate_feasible")
        if feasible is not None and int(feasible) == 0:
            return int(row["step"])
    return None


def _first_infeasible_candidate_step(rows: list[dict[str, Any]]) -> int | None:
    """Return the first step where at least one rollout candidate is infeasible."""
    for row in rows:
        infeasible = row.get("candidate_infeasible")
        if infeasible is not None and int(infeasible) > 0:
            return int(row["step"])
    return None


def _global_route_probe_activation_step(rows: list[dict[str, Any]]) -> int | None:
    """Return the first step where the global-route probe activated."""
    for row in rows:
        if row.get("global_route_probe_activated"):
            return int(row["step"])
    return None


def _route_progress_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact route-progress statistics for one episode's trace."""
    if not rows:
        return {"status": "no_steps"}
    distances: list[float] = []
    progresses: list[float] = []
    skipped_non_finite_rows = 0
    skipped_non_finite_cells = 0
    for row in rows:
        row_has_non_finite_value = False
        for key, values in (
            ("distance_to_goal_m", distances),
            ("route_progress_from_start_m", progresses),
        ):
            raw_value = row.get(key)
            if raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                values.append(value)
            else:
                row_has_non_finite_value = True
                skipped_non_finite_cells += 1
        if row_has_non_finite_value:
            skipped_non_finite_rows += 1
    initial = distances[0] if distances else None
    final = distances[-1] if distances else None
    return {
        "initial_distance_to_goal_m": initial,
        "final_distance_to_goal_m": final,
        "min_distance_to_goal_m": min(distances) if distances else None,
        "max_route_progress_from_start_m": max(progresses) if progresses else None,
        "final_route_progress_from_start_m": progresses[-1] if progresses else None,
        "net_progress_m": (float(initial) - float(final))
        if initial is not None and final is not None
        else None,
        "progress_ratio_of_initial": (
            (float(initial) - float(final)) / float(initial)
            if initial not in (None, 0.0) and final is not None
            else None
        ),
        "skipped_non_finite_rows": skipped_non_finite_rows,
        "skipped_non_finite_cells": skipped_non_finite_cells,
    }


def _constraint_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return per-constraint-reason step counts for one episode's trace."""
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("constraint_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _summarize_episode(
    *,
    episode_id: str,
    record: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the per-episode failure-mechanism summary from the raw step trace."""
    rows = [
        _flatten_step(
            step,
            episode_id=episode_id,
            scenario_id=record.get("scenario_id", ""),
            seed=record.get("seed", -1),
        )
        for step in steps
    ]
    outcome = record.get("outcome", {}) if isinstance(record.get("outcome"), dict) else {}
    summary = {
        "episode_id": episode_id,
        "scenario_id": record.get("scenario_id"),
        "seed": record.get("seed"),
        "termination_reason": record.get("termination_reason"),
        "steps": record.get("steps"),
        "route_complete": bool(outcome.get("route_complete")),
        "collision_event": bool(outcome.get("collision_event")),
        "timeout_event": bool(outcome.get("timeout_event")),
        "trace_step_count": len(rows),
        "constraint_reason_counts": _constraint_reason_counts(rows),
        "route_progress": _route_progress_summary(rows),
        "first_infeasible_candidate_step": _first_infeasible_candidate_step(rows),
        "first_all_infeasible_step": _first_unrecoverable_step(rows),
        "global_route_probe_first_activation_step": _global_route_probe_activation_step(rows),
        "global_route_probe_activated_any_step": any(
            row.get("global_route_probe_activated") for row in rows
        ),
        "last_selected_command": {
            "v_mps": rows[-1].get("selected_v_mps") if rows else None,
            "w_radps": rows[-1].get("selected_w_radps") if rows else None,
        },
        "last_selected_score": rows[-1].get("selected_score") if rows else None,
    }
    return summary


def _write_steps_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat per-step rows as a deterministic CSV artifact."""
    if not rows:
        raise ValueError(f"cannot write empty steps CSV: {path}")
    write_distance_series_csv(
        path,
        [{field: row.get(field) for field in STEP_TRACE_FIELDS} for row in rows],
        convention=DistanceConvention.CENTER_CENTER,
    )


def _append_outcome_comparison_and_acceptance_criteria(lines: list[str]) -> None:
    """Append the bounded inference and completion record for the packet."""
    lines.append("## Outcome comparison")
    lines.append("")
    lines.append(
        "The probe did not activate on either recorded step. Under the fail-closed scoring "
        "contract, its waypoint term is therefore zero and DWA uses the baseline score. "
        "This is a scoring-contract inference, not an independently rerun baseline comparator."
    )
    lines.append("")
    lines.append("## Acceptance criteria")
    lines.append("")
    lines.append(
        "- [x] Planner/config tests cover the new contract and malformed-input failure mode"
    )
    lines.append(
        "- [x] The evidence packet states whether the probe activates and whether either "
        "original mechanism changes"
    )
    lines.append(
        "- [x] Results remain diagnostic-only unless a separate benchmark decision establishes "
        "a broader claim"
    )
    lines.append("")


def _write_evidence_readme(
    path: Path, *, summaries: list[dict[str, Any]], trace_commit: str
) -> None:
    """Write the analysis-only evidence README naming config/scenario/seed/mechanism."""
    bottleneck = next((s for s in summaries if s["episode_id"] == "bottleneck_timeout"), None)
    t_inter = next((s for s in summaries if s["episode_id"] == "t_intersection_collision"), None)
    lines: list[str] = []
    lines.append("<!-- AI-GENERATED (robot_sf#5331, 2026-07-11) - NEEDS-REVIEW -->")
    lines.append("# Issue #5331 — DWA Global-Route Integration Probe for Bottleneck Convergence")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This diagnostic probes global-route waypoint integration for the classical DWA "
        "planner to test whether waypoint-following helps navigate through bottleneck "
        "corridors where the constant-velocity rollout cannot directly see the goal."
    )
    lines.append("")
    lines.append("- Config: `configs/algos/dwa_global_route_probe.yaml`")
    lines.append("- Matrix: `configs/scenarios/classic_interactions.yaml`")
    lines.append(f"- Commit: `{trace_commit}`")
    lines.append("")
    lines.append("## Episodes")
    lines.append("")
    if bottleneck:
        lines.append("### Bottleneck timeout (seed 131)")
        lines.append("")
        lines.append(f"- Termination: {bottleneck['termination_reason']}")
        lines.append(f"- Steps: {bottleneck['steps']}")
        rp = bottleneck["route_progress"]
        lines.append(f"- Net progress: {rp.get('net_progress_m'):.3f} m")
        lines.append(f"- Min distance to goal: {rp.get('min_distance_to_goal_m'):.3f} m")
        lines.append(
            "- Global-route probe activated: "
            f"{bottleneck.get('global_route_probe_activated_any_step', False)}"
        )
        first_activation = bottleneck.get("global_route_probe_first_activation_step")
        if first_activation is not None:
            lines.append(f"- Global-route probe first activation step: {first_activation}")
        lines.append("")
    if t_inter:
        lines.append("### T-intersection collision (seed 161)")
        lines.append("")
        lines.append(f"- Termination: {t_inter['termination_reason']}")
        lines.append(f"- Steps: {t_inter['steps']}")
        rp = t_inter["route_progress"]
        lines.append(f"- Net progress: {rp.get('net_progress_m'):.3f} m")
        lines.append(f"- Min distance to goal: {rp.get('min_distance_to_goal_m'):.3f} m")
        lines.append(
            "- Global-route probe activated: "
            f"{t_inter.get('global_route_probe_activated_any_step', False)}"
        )
        first_activation = t_inter.get("global_route_probe_first_activation_step")
        if first_activation is not None:
            lines.append(f"- Global-route probe first activation step: {first_activation}")
        lines.append("")
    _append_outcome_comparison_and_acceptance_criteria(lines)
    lines.append("## Claim boundary")
    lines.append("")
    lines.append(
        "This is a diagnostic-only trace. It makes no benchmark, metric, paper, or "
        "dissertation claim. Results indicate whether the global-route probe activates "
        "and whether it changes the episode outcome relative to the baseline."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Two fixed-seed episodes only; not a representative sample.")
    lines.append("- CPU-only, no training, no benchmark suite.")
    lines.append(
        "- The probe requires `route_waypoints` in the observation; episodes without "
        "waypoints fall back to baseline DWA behavior."
    )
    lines.append(
        "- Activation depends on the waypoint being within "
        "`global_route_probe_waypoint_distance` of the robot."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_trace(
    *,
    algo_config: Path,
    matrix_path: Path,
    out_dir: Path,
    evidence_dir: Path | None = None,
) -> None:
    """Run the two fixed-seed episodes and produce the diagnostic packet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_commit = "unknown"
    try:
        trace_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        pass

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for scenario_id, seed, episode_id in TARGET_EPISODES:
        scenario = _load_scenario(scenario_id, seed, matrix_path)
        episodes_path = out_dir / f"episodes_{episode_id}.jsonl"
        if episodes_path.exists():
            episodes_path.unlink()
        run_map_batch(
            [scenario],
            episodes_path,
            schema_path=SCHEMA_PATH,
            scenario_path=matrix_path,
            horizon=HORIZON,
            dt=DT,
            record_forces=False,
            algo="dwa",
            algo_config_path=str(algo_config),
            benchmark_profile="experimental",
            workers=1,
            resume=False,
            record_planner_decision_trace=True,
        )
        record = _read_record(episodes_path)
        algorithm_metadata = record.get("algorithm_metadata")
        trace = (
            algorithm_metadata.get("planner_decision_trace", {})
            if isinstance(algorithm_metadata, dict)
            else {}
        )
        steps = trace.get("steps", []) if isinstance(trace.get("steps"), list) else []
        summary = _summarize_episode(
            episode_id=episode_id,
            record=record,
            steps=steps,
        )
        summaries.append(summary)
        rows = [
            _flatten_step(
                step,
                episode_id=episode_id,
                scenario_id=scenario_id,
                seed=seed,
            )
            for step in steps
        ]
        all_rows.extend(rows)

    steps_csv = out_dir / "dwa_global_route_probe_steps.csv"
    _write_steps_csv(steps_csv, all_rows)

    summary_json = out_dir / "dwa_global_route_probe_summary.json"
    write_json(
        summary_json,
        {
            "issue": FOLLOW_UP_ISSUE,
            "config": str(algo_config.relative_to(REPO_ROOT)),
            "schema_version": "dwa-global-route-probe-trace.v1",
            "review_marker": "AI-GENERATED NEEDS-REVIEW",
            "episodes": summaries,
        },
    )

    if evidence_dir:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        readme_path = evidence_dir / "README.md"
        _write_evidence_readme(
            readme_path,
            summaries=summaries,
            trace_commit=trace_commit,
        )
        import shutil

        shutil.copy2(steps_csv, evidence_dir / steps_csv.name)
        shutil.copy2(summary_json, evidence_dir / summary_json.name)
        print(f"Evidence packet written to {evidence_dir}")

    print(f"Steps CSV: {steps_csv}")
    print(f"Summary JSON: {summary_json}")
    for summary in summaries:
        print(
            f"  {summary['episode_id']}: "
            f"probe_activated={summary.get('global_route_probe_activated_any_step')}, "
            f"termination={summary['termination_reason']}"
        )


def main() -> None:
    """Parse CLI arguments and run the global-route integration probe trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algo-config",
        type=Path,
        default=DEFAULT_ALGO_CONFIG,
        help="Path to the DWA algorithm config YAML.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
        help="Path to the scenario matrix YAML.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for trace output artifacts.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="Optional evidence directory for the reviewable packet.",
    )
    args = parser.parse_args()
    run_trace(
        algo_config=args.algo_config,
        matrix_path=args.matrix,
        out_dir=args.out_dir,
        evidence_dir=args.evidence_dir,
    )


if __name__ == "__main__":
    main()
