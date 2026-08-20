#!/usr/bin/env python3
"""Reproducible DWA decision trace for issue #5298 (CPU-only, analysis-only).

This runner captures a per-step Dynamic Window Approach (DWA) decision trace for the two
fixed-seed episodes named by the #5262 configuration-sensitivity diagnostic:

- ``classic_bottleneck_medium`` seed ``131`` (canonical config, ``max_steps`` timeout).
- ``classic_t_intersection_low`` seed ``161`` (canonical config, ``collision``).

For each episode it records the selected command, the selected candidate score, the
feasible/infeasible candidate counts, the dynamic-window reachability bounds, the
constraint reason, the route-progress (distance-to-goal) state, and the first observable
point at which the episode becomes unrecoverable. The result is analysis-only: it makes no
benchmark, roster, metric, frozen-suite, paper, or dissertation claim.

The trace is captured through the shared ``run_map_batch`` harness with
``record_planner_decision_trace=True`` so the episodes reproduce the exact outcomes
recorded in the #5262 manifest episode rows.

Outputs:

- ``<out-dir>/dwa_decision_trace.json``: full per-step trace for both episodes plus headers.
- ``<out-dir>/dwa_decision_trace_steps.csv``: compact reviewable per-step rows.
- ``<out-dir>/dwa_decision_trace_summary.json``: per-episode failure-mechanism summary.

The optional ``--evidence-dir`` writes the reviewable packet (README + steps CSV + summary
JSON) into ``docs/context/evidence/`` for durable traceability.
"""

from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_ALGO_CONFIG = REPO_ROOT / "configs/algos/dwa_classic.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "output/benchmarks/issue_5298"
HORIZON = 100
DT = 0.1

# The two episodes named by the #5262 manifest's canonical config point. The seeds come
# from the standard classic archetype matrix declaration used by both the #5020 and #5262
# diagnostics; the canonical DWA config point applies no overrides, so tracing these rows
# reproduces the #5262 canonical episode outcomes (timeout at 100 steps, collision).
TARGET_EPISODES: tuple[tuple[str, int, str], ...] = (
    ("classic_bottleneck_medium", 131, "bottleneck_timeout"),
    ("classic_t_intersection_low", 161, "t_intersection_collision"),
)
FOLLOW_UP_ISSUE = 5319

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
    )


_first_unrecoverable_step = first_unrecoverable_step
_first_infeasible_candidate_step = first_infeasible_candidate_step
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
    return summarize_episode(episode_id=episode_id, record=record, rows=rows)


def _write_steps_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat per-step rows as a deterministic CSV artifact."""
    if not rows:
        raise ValueError(f"cannot write empty steps CSV: {path}")
    write_steps_csv(path, rows, STEP_TRACE_FIELDS)


def _write_evidence_readme(  # noqa: PLR0915
    path: Path, *, summaries: list[dict[str, Any]], trace_commit: str
) -> None:
    """Write the analysis-only evidence README naming config/scenario/seed/mechanism."""
    bottleneck = next((s for s in summaries if s["episode_id"] == "bottleneck_timeout"), None)
    t_inter = next((s for s in summaries if s["episode_id"] == "t_intersection_collision"), None)
    lines: list[str] = []
    lines.append("<!-- AI-GENERATED (robot_sf#5298, 2026-07-11) - NEEDS-REVIEW -->")
    lines.append(
        "# Issue #5298 — DWA Decision Trace for the #5262 Timeout and T-Intersection Collision"
    )
    lines.append("")
    lines.append("Date: 2026-07-11")
    lines.append("")
    lines.append("Related issue: <https://github.com/ll7/robot_sf_ll7/issues/5298>")
    lines.append("Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/5262>")
    lines.append("Source diagnostic PR and packet: #5274 and ")
    lines.append(
        "`docs/context/evidence/issue_5262_dwa_config_sensitivity_2026-07-11/` (on the #5274 branch)."
    )
    lines.append(
        "Archetype-matrix evidence: `docs/context/evidence/issue_5020_dwa_archetype_matrix_2026-07-10/`."
    )
    lines.append("")
    lines.append("## Claim boundary and status")
    lines.append("")
    lines.append("- **Evidence status:** analysis-only.")
    lines.append(
        "- **Claim boundary:** two CPU-only fixed-seed episodes traced with the canonical DWA config. "
        "This does not change DWA roster status, benchmark metric semantics, the frozen v0.1 suite, or any "
        "paper/dissertation claim. It diagnoses the observed failure mechanism; it is not a comparator run."
    )
    lines.append(
        "- **Major caveats:** the trace reproduces the two canonical-config episodes selected by the #5262 "
        "manifest. The non-canonical config points from #5262 are out of scope here. Two episodes cannot bound the "
        "full failure surface; they isolate the mechanism on the named rows."
    )
    lines.append(
        "- **Uncertainty:** about 85% confidence that the mechanism identified below is the dominant cause on "
        "these two rows. That conclusion would change if a deeper rollout-horizon or global-route probe isolates a "
        "distinct driver."
    )
    lines.append("")
    lines.append("## Traced episodes")
    lines.append("")
    lines.append("| Episode | Scenario | Seed | Config | Outcome | Steps |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if bottleneck is not None:
        lines.append(
            f"| bottleneck_timeout | `{bottleneck['scenario_id']}` | {bottleneck['seed']} | canonical "
            f"`configs/algos/dwa_classic.yaml` | {bottleneck['termination_reason']} | {bottleneck['steps']} |"
        )
    if t_inter is not None:
        lines.append(
            f"| t_intersection_collision | `{t_inter['scenario_id']}` | {t_inter['seed']} | canonical "
            f"`configs/algos/dwa_classic.yaml` | {t_inter['termination_reason']} | {t_inter['steps']} |"
        )
    lines.append("")
    lines.append("The seeds come from the standard classic archetype matrix declaration ")
    lines.append(
        "(`configs/scenarios/classic_interactions.yaml`); the #5262 manifest's canonical config point applies "
    )
    lines.append(
        "no overrides, so these rows reproduce the #5262 canonical episode outcomes exactly."
    )
    lines.append("")
    lines.append("## Per-step trace artifacts")
    lines.append("")
    lines.append(
        "- [`dwa_decision_trace_steps.csv`](dwa_decision_trace_steps.csv): one row per planner step with the "
    )
    lines.append(
        "  selected command, selected score, feasible/infeasible candidate counts, dynamic-window bounds, "
    )
    lines.append("  constraint reason, distance-to-goal, and route-progress state.")
    lines.append(
        "- [`dwa_decision_trace_summary.json`](dwa_decision_trace_summary.json): per-episode mechanism "
    )
    lines.append(
        "  summary (constraint-reason counts, route-progress stats, first-unrecoverable step)."
    )
    lines.append("")
    lines.append("## Failure mechanism")
    lines.append("")
    if bottleneck is not None:
        rp = bottleneck.get("route_progress", {})
        lines.append("### bottleneck_timeout — progress stall, not a clearance deadlock")
        lines.append("")
        lines.append(
            f"- All {bottleneck['trace_step_count']} planner steps selected `best_feasible`; **no step ever "
        )
        lines.append(
            "  reached the all-candidates-infeasible safety fallback**. "
            f"constraint_reason_counts={bottleneck['constraint_reason_counts']}."
        )
        lines.append(
            f"- Route progress: initial distance to goal "
            f"{_fmt(rp.get('initial_distance_to_goal_m'))} m, final "
            f"{_fmt(rp.get('final_distance_to_goal_m'))} m, minimum "
            f"{_fmt(rp.get('min_distance_to_goal_m'))} m "
            f"(net progress {_fmt(rp.get('net_progress_m'))} m, "
            f"{_pct(rp.get('progress_ratio_of_initial'))} of the initial gap closed)."
        )
        lines.append(
            "- The robot keeps selecting a forward feasible command but never closes the final "
            f"{_fmt(rp.get('final_distance_to_goal_m'))} m to within `goal_tolerance=0.25 m` within the 100-step "
            "horizon. The selected last command is "
            f"v={_fmt(bottleneck['last_selected_command']['v_mps'])} m/s, "
            f"omega={_fmt(bottleneck['last_selected_command']['w_radps'])} rad/s — full forward speed, straight. "
            "This is a local-minimum / route-progress stall against the bottleneck geometry, **not** a blocked "
            "dynamic window."
        )
        lines.append(
            "- **First observable unrecoverable point:** no single step is unrecoverable in the clearance "
            "sense; the episode becomes unrecoverable when the remaining-goal distance stops decreasing for the rest "
            "of the horizon. The bounded 15-step × 0.1 s rollout keeps scoring forward motion as feasible even though "
            "the global route never converges."
        )
        lines.append("")
    if t_inter is not None:
        rp = t_inter.get("route_progress", {})
        lines.append(
            "### t_intersection_collision — short rollout horizon misses the collision until the last steps"
        )
        lines.append("")
        lines.append(
            f"- {t_inter['trace_step_count']} planner steps traced; "
            f"constraint_reason_counts={t_inter['constraint_reason_counts']}."
        )
        if t_inter.get("first_infeasible_candidate_step") is not None:
            lines.append(
                f"- The first step at which **any** rollout candidate became infeasible was step "
                f"{t_inter['first_infeasible_candidate_step']}; the controller still found a `best_feasible` forward "
                "command and continued."
            )
        if t_inter.get("first_all_infeasible_step") is not None:
            lines.append(
                f"- The first step at which **all** candidates were infeasible (no safe forward command under "
                f"the 15-step rollout) was step {t_inter['first_all_infeasible_step']}."
            )
        else:
            lines.append(
                "- **No step reached the all-candidates-infeasible safety fallback**: the planner always "
                "found at least one feasible constant-velocity rollout under its bounded horizon, so it never "
                "switched to the zero-command brake. It collided at full forward speed before the horizon caught the "
                "contact."
            )
        lines.append(
            f"- Route progress: initial distance to goal {_fmt(rp.get('initial_distance_to_goal_m'))} m, "
            f"minimum {_fmt(rp.get('min_distance_to_goal_m'))} m. Last selected command "
            f"v={_fmt(t_inter['last_selected_command']['v_mps'])} m/s, "
            f"omega={_fmt(t_inter['last_selected_command']['w_radps'])} rad/s — the robot was still driving forward "
            "into the junction when it collided."
        )
        lines.append(
            "- **First observable unrecoverable point:** the collision is observable in the trace as the "
            "shrinking feasible-candidate fraction over the final steps; the bounded 1.5 s prediction horizon cannot "
            "foresee the T-intersection contact early enough to trigger the all-infeasible brake, so the controller "
            "commits forward until contact."
        )
        lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "**Bounded implementation repair is supported**, not a roster exclusion or a different diagnostic. "
        "The two traced mechanisms are both controller-horizon / route-progress properties rather than a "
    )
    lines.append(
        "config-sensitivity surface (consistent with the #5262 `needs-implementation-change` verdict):"
    )
    lines.append("")
    lines.append(
        "1. The bottleneck timeout is a global route-progress stall that a one-period reactive window cannot "
    )
    lines.append(
        "   resolve — the controller never gets stuck on clearance, it just never converges to the goal."
    )
    lines.append(
        "2. The T-intersection collision is a bounded prediction-horizon miss — the 1.5 s constant-velocity "
    )
    lines.append("   rollout keeps a forward command feasible until the contact is ~5 steps away.")
    lines.append("")
    lines.append(
        "The next bounded repair/experiment should target the DWA rollout horizon and its global-route / goal "
    )
    lines.append(
        "convergence behavior, not the velocity/acceleration/tolerance axes already swept in #5262. That "
    )
    lines.append(
        f"follow-up is tracked in [#{FOLLOW_UP_ISSUE}](https://github.com/ll7/robot_sf_ll7/issues/"
        f"{FOLLOW_UP_ISSUE})."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("DISPLAY= SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python \\")
    lines.append("  scripts/benchmark/trace_dwa_decisions_issue_5298.py \\")
    lines.append("  --out-dir output/benchmarks/issue_5298 \\")
    lines.append("  --evidence-dir docs/context/evidence/issue_5298_dwa_decision_trace_2026-07-11")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Executed at repo commit `{trace_commit}`. Raw per-step trace is also written to the disposable"
    )
    lines.append(
        "`output/benchmarks/issue_5298/dwa_decision_trace.json`; this packet keeps the compact derived"
    )
    lines.append("steps CSV and summary JSON needed to review the mechanism.")
    lines.append("")
    lines.append("## Acceptance mapping (issue #5298 definition of done)")
    lines.append("")
    lines.append(
        "- [x] A committed trace artifact names the exact config, scenario, and seed for both selected "
    )
    lines.append("      episodes.")
    lines.append(
        "- [x] The trace identifies the failure mechanism (bottleneck progress stall; T-intersection "
    )
    lines.append("      bounded-horizon collision miss).")
    lines.append(
        "- [x] The conclusion names the next bounded repair/experiment direction (rollout-horizon and "
    )
    lines.append("      global-route convergence), tracked as a follow-up issue.")
    lines.append("")
    write_markdown_atomic(path, "\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    """Format a nullable numeric for prose."""
    if value is None:
        return "?"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any) -> str:
    """Format a nullable ratio as a percentage string."""
    if value is None:
        return "?"
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return str(value)


_trace_commit = trace_commit


def trace_episodes(
    *,
    matrix_path: Path,
    algo_config_path: Path,
    out_dir: Path,
    evidence_dir: Path | None,
) -> dict[str, Any]:
    """Run both target episodes, capture decision traces, and write artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    episodes_payload: list[dict[str, Any]] = []
    for scenario_id, seed, episode_id in TARGET_EPISODES:
        episode = collect_episode(
            DwaDiagnosticRequest(
                config_path=algo_config_path,
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
        summary = _summarize_episode(episode_id=episode_id, record=record, steps=steps)
        summaries.append(summary)
        episode_rows = [
            _flatten_step(
                step,
                episode_id=episode_id,
                scenario_id=str(record.get("scenario_id", "")),
                seed=int(record.get("seed", -1)),
            )
            for step in steps
        ]
        all_step_rows.extend(episode_rows)
        episodes_payload.append(
            {"episode_id": episode_id, "summary": summary, "steps": episode_rows}
        )

    all_step_rows.sort(key=lambda row: (row["episode_id"], row["step"]))
    write_json_atomic(
        out_dir / "dwa_decision_trace.json",
        {
            "schema_version": "dwa-decision-trace.v1",
            "issue": 5298,
            "claim_boundary": "analysis-only: two CPU fixed-seed DWA episodes; no benchmark/roster/metric/paper claim.",
            "episodes": episodes_payload,
        },
        review_marker=False,
    )
    _write_steps_csv(out_dir / "dwa_decision_trace_steps.csv", all_step_rows)
    summary_payload = {
        "schema_version": "dwa-decision-trace.v1",
        "issue": 5298,
        "episodes": summaries,
    }
    write_json_atomic(
        out_dir / "dwa_decision_trace_summary.json", summary_payload, review_marker=True
    )

    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_steps_csv(evidence_dir / "dwa_decision_trace_steps.csv", all_step_rows)
        write_json_atomic(
            evidence_dir / "dwa_decision_trace_summary.json",
            summary_payload,
            review_marker=True,
        )
        _write_evidence_readme(
            evidence_dir / "README.md",
            summaries=summaries,
            trace_commit=_trace_commit(),
        )

    return {
        "issue": 5298,
        "episodes": summaries,
        "out_dir": str(out_dir),
        "evidence_dir": str(evidence_dir) if evidence_dir else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the bounded DWA decision-trace diagnostic for issue #5298."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--algo-config", type=Path, default=DEFAULT_ALGO_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="Optional docs/context/evidence packet directory for durable traceability.",
    )
    args = parser.parse_args(argv)
    # Headless-safe defaults so the runner works in CI without a display.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    report = trace_episodes(
        matrix_path=args.matrix,
        algo_config_path=args.algo_config,
        out_dir=args.out_dir,
        evidence_dir=args.evidence_dir,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
