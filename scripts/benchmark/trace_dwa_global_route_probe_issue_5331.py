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
from pathlib import Path
from typing import Any

from robot_sf.benchmark.dwa_diagnostic_harness import (
    DwaDiagnosticRequest,
    collect_episode,
    constraint_reason_counts,
    first_infeasible_candidate_step,
    first_unrecoverable_step,
    flatten_trace_step,
    load_scenario,
    read_single_episode_record,
    repo_relative_path,
    route_progress_summary,
    summarize_episode,
    trace_commit,
    write_json_atomic,
    write_markdown_atomic,
    write_steps_csv,
)
from robot_sf.benchmark.map_runner.map_runner import run_map_batch
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
    """Compatibility adapter for the shared deterministic scenario loader."""
    return load_scenario(name, seed, matrix_path, load_scenarios_fn=load_scenarios)


_read_record = read_single_episode_record


def _flatten_step(
    step: dict[str, Any],
    *,
    episode_id: str,
    scenario_id: str,
    seed: int,
) -> dict[str, Any]:
    """Normalize one planner-decision-trace step into a flat CSV/JSON row."""
    return flatten_trace_step(
        step,
        episode_id=episode_id,
        scenario_id=scenario_id,
        seed=seed,
        extra_fields={
            "global_route_probe_activated": step.get("global_route_probe_activated", False),
        },
    )


_first_unrecoverable_step = first_unrecoverable_step
_first_infeasible_candidate_step = first_infeasible_candidate_step


def _global_route_probe_activation_step(rows: list[dict[str, Any]]) -> int | None:
    """Return the first step where the global-route probe activated."""
    for row in rows:
        if row.get("global_route_probe_activated"):
            return int(row["step"])
    return None


_route_progress_summary = route_progress_summary
_constraint_reason_counts = constraint_reason_counts


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
    return summarize_episode(
        episode_id=episode_id,
        record=record,
        rows=rows,
        extra_fields={
            "global_route_probe_first_activation_step": _global_route_probe_activation_step(rows),
            "global_route_probe_activated_any_step": any(
                row.get("global_route_probe_activated") for row in rows
            ),
        },
    )


def _write_steps_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat per-step rows as a deterministic CSV artifact."""
    if not rows:
        raise ValueError(f"cannot write empty steps CSV: {path}")
    write_steps_csv(path, rows, STEP_TRACE_FIELDS)


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
    write_markdown_atomic(path, "\n".join(lines) + "\n")


def run_trace(
    *,
    algo_config: Path,
    matrix_path: Path,
    out_dir: Path,
    evidence_dir: Path | None = None,
) -> None:
    """Run the two fixed-seed episodes and produce the diagnostic packet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    current_commit = trace_commit()

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for scenario_id, seed, episode_id in TARGET_EPISODES:
        episode = collect_episode(
            DwaDiagnosticRequest(
                config_path=algo_config,
                scenario=scenario_id,
                seed=seed,
                algorithm="dwa",
                output_dir=out_dir,
                episode_id=episode_id,
                matrix_path=matrix_path,
                schema_path=SCHEMA_PATH,
                horizon=HORIZON,
                dt=DT,
            ),
            run_map_batch_fn=run_map_batch,
            load_scenario_fn=_load_scenario,
        )
        record = dict(episode.episode_row)
        steps = list(episode.steps)
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
    write_json_atomic(
        summary_json,
        {
            "issue": FOLLOW_UP_ISSUE,
            "config": repo_relative_path(algo_config),
            "schema_version": "dwa-global-route-probe-trace.v1",
            "review_marker": "AI-GENERATED NEEDS-REVIEW",
            "episodes": summaries,
        },
        review_marker=True,
    )

    if evidence_dir:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        readme_path = evidence_dir / "README.md"
        _write_evidence_readme(
            readme_path,
            summaries=summaries,
            trace_commit=current_commit,
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
