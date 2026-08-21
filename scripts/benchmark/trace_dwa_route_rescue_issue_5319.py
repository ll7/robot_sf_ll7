#!/usr/bin/env python3
"""Reproducible DWA route-rescue diagnostic trace for issue #5319 (CPU-only, analysis-only).

This runner captures a per-step DWA decision trace with the route-rescue and
feasibility-slowdown interventions enabled, using the same two fixed-seed
episodes from the #5298 baseline trace:

- ``classic_bottleneck_medium`` seed ``131`` (canonical config, ``max_steps`` timeout).
- ``classic_t_intersection_low`` seed ``161`` (canonical config, ``collision``).

It runs each episode with the ``dwa_route_rescue.yaml`` config and records
per-step diagnostics including rescue activation state. The result is
diagnostic-only: it makes no benchmark, roster, metric, frozen-suite, paper,
or dissertation claim. It identifies whether the intervention improves, fails
to improve, or changes the mechanism on both rows.

Outputs:

- ``<out-dir>/dwa_route_rescue_trace.json``: full per-step trace plus headers.
- ``<out-dir>/dwa_route_rescue_steps.csv``: compact reviewable per-step rows.
- ``<out-dir>/dwa_route_rescue_summary.json``: per-episode failure-mechanism summary.

The optional ``--evidence-dir`` writes the reviewable packet (README + steps CSV
+ summary JSON) into ``docs/context/evidence/`` for durable traceability.
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
    repo_relative_path,
    route_progress_summary,
    sha256_file,
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
DEFAULT_ALGO_CONFIG = REPO_ROOT / "configs/algos/dwa_route_rescue.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "output/benchmarks/issue_5319"
BASELINE_SUMMARY = (
    REPO_ROOT
    / "docs/context/evidence/issue_5298_dwa_decision_trace_2026-07-11"
    / "dwa_decision_trace_summary.json"
)
HORIZON = 100
DT = 0.1

TARGET_EPISODES: tuple[tuple[str, int, str], ...] = (
    ("classic_bottleneck_medium", 131, "bottleneck_timeout"),
    ("classic_t_intersection_low", 161, "t_intersection_collision"),
)
BASELINE_ISSUE = 5298

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
    "route_rescue_active",
    "route_rescue_type",
    "feasibility_slowdown_active",
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
            "route_rescue_active": step.get("route_rescue_active", False),
            "route_rescue_type": step.get("route_rescue_type"),
            "feasibility_slowdown_active": step.get("feasibility_slowdown_active", False),
        },
    )


_first_unrecoverable_step = first_unrecoverable_step
_first_infeasible_candidate_step = first_infeasible_candidate_step


def _route_rescue_activation_steps(rows: list[dict[str, Any]]) -> list[int]:
    """Return steps where route-rescue was active."""
    return [int(row["step"]) for row in rows if row.get("route_rescue_active")]


def _feasibility_slowdown_steps(rows: list[dict[str, Any]]) -> list[int]:
    """Return steps where feasibility-slowdown was active."""
    return [int(row["step"]) for row in rows if row.get("feasibility_slowdown_active")]


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
    rescue_steps = _route_rescue_activation_steps(rows)
    slowdown_steps = _feasibility_slowdown_steps(rows)
    return summarize_episode(
        episode_id=episode_id,
        record=record,
        rows=rows,
        extra_fields={
            "route_rescue_active_step_count": len(rescue_steps),
            "route_rescue_first_active_step": rescue_steps[0] if rescue_steps else None,
            "feasibility_slowdown_active_step_count": len(slowdown_steps),
            "feasibility_slowdown_first_active_step": slowdown_steps[0] if slowdown_steps else None,
        },
    )


def _write_steps_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat per-step rows as a deterministic CSV artifact."""
    if not rows:
        raise ValueError(f"cannot write empty steps CSV: {path}")
    write_steps_csv(path, rows, STEP_TRACE_FIELDS)


def _write_evidence_readme(  # noqa: PLR0915
    path: Path,
    *,
    summaries: list[dict[str, Any]],
    baseline_summaries: list[dict[str, Any]],
    trace_commit: str,
    config_sha256: str,
) -> None:
    """Write the diagnostic evidence README comparing rescue vs. baseline."""
    rescue_bottleneck = next(
        (s for s in summaries if s["episode_id"] == "bottleneck_timeout"), None
    )
    rescue_t_inter = next(
        (s for s in summaries if s["episode_id"] == "t_intersection_collision"), None
    )
    base_bottleneck = next(
        (s for s in baseline_summaries if s["episode_id"] == "bottleneck_timeout"), None
    )
    base_t_inter = next(
        (s for s in baseline_summaries if s["episode_id"] == "t_intersection_collision"), None
    )
    lines: list[str] = []
    lines.append("<!-- AI-GENERATED (robot_sf#5319, 2026-07-11) - NEEDS-REVIEW -->")
    lines.append("# Issue #5319 — DWA Route-Rescue Diagnostic Probe")
    lines.append("")
    lines.append("Date: 2026-07-11")
    lines.append("")
    lines.append("Issue: <https://github.com/ll7/robot_sf_ll7/issues/5319>")
    lines.append(
        "Baseline trace: <https://github.com/ll7/robot_sf_ll7/issues/5298> "
        "and `docs/context/evidence/issue_5298_dwa_decision_trace_2026-07-11/`."
    )
    lines.append("")
    lines.append("## Claim boundary and status")
    lines.append("")
    lines.append("- **Evidence status:** diagnostic-only.")
    lines.append(
        "- **Claim boundary:** two CPU-only fixed-seed episodes traced with the DWA "
        "route-rescue config (`configs/algos/dwa_route_rescue.yaml`). This does not change "
        "DWA roster status, benchmark metric semantics, the frozen v0.1 suite, or any "
        "paper/dissertation claim."
    )
    lines.append(
        "- **Interventions:** (1) route-rescue extends the rollout horizon and boosts progress "
        "weight when the robot stalls for `route_rescue_patience` steps; (2) feasibility-slowdown "
        "reduces linear speed when infeasible-candidate fraction exceeds a threshold."
    )
    lines.append(
        "- **Caveat:** this is a diagnostic probe, not a comparator benchmark. Two episodes cannot "
        "bound the full failure surface. The intervention may not generalize beyond these rows."
    )
    lines.append(f"- **Seeds:** {', '.join(str(seed) for _, seed, _ in TARGET_EPISODES)}.")
    lines.append(
        f"- **Config SHA-256 hash:** `{config_sha256}` for `configs/algos/dwa_route_rescue.yaml`."
    )
    lines.append("")
    lines.append("## Episodes traced")
    lines.append("")
    lines.append("| Episode | Scenario | Seed | Config | Outcome | Steps |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if rescue_bottleneck is not None:
        lines.append(
            f"| bottleneck_timeout | `{rescue_bottleneck['scenario_id']}` | "
            f"{rescue_bottleneck['seed']} | route-rescue | "
            f"{rescue_bottleneck['termination_reason']} | {rescue_bottleneck['steps']} |"
        )
    if rescue_t_inter is not None:
        lines.append(
            f"| t_intersection_collision | `{rescue_t_inter['scenario_id']}` | "
            f"{rescue_t_inter['seed']} | route-rescue | "
            f"{rescue_t_inter['termination_reason']} | {rescue_t_inter['steps']} |"
        )
    lines.append("")
    lines.append("## Comparison with baseline")
    lines.append("")
    lines.append("| Episode | Metric | Baseline (#5298) | Route-rescue | Delta |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, rescue, base in [
        ("bottleneck", rescue_bottleneck, base_bottleneck),
        ("t_intersection", rescue_t_inter, base_t_inter),
    ]:
        if rescue is None or base is None:
            continue
        rp_r = rescue.get("route_progress", {})
        rp_b = base.get("route_progress", {})
        base_steps = base.get("steps")
        rescue_steps = rescue.get("steps")
        base_termination = base.get("termination_reason", "?")
        rescue_termination = rescue.get("termination_reason", "?")
        outcome_changed = base_termination != rescue_termination
        steps_delta = rescue_steps - base_steps if base_steps and rescue_steps else None
        lines.append(f"| {label} | steps | {base_steps} | {rescue_steps} | {_fmt(steps_delta)} |")
        lines.append(
            f"| {label} | termination | {base_termination} | {rescue_termination} "
            f"| {'changed' if outcome_changed else 'same'} |"
        )
        base_min = rp_b.get("min_distance_to_goal_m")
        rescue_min = rp_r.get("min_distance_to_goal_m")
        min_delta = None
        if base_min is not None and rescue_min is not None:
            min_delta = rescue_min - base_min
        lines.append(
            f"| | min_distance | {_fmt(base_min)} m | {_fmt(rescue_min)} m | {_fmt(min_delta)} m |"
        )
        base_net = rp_b.get("net_progress_m")
        rescue_net = rp_r.get("net_progress_m")
        net_delta = None
        if base_net is not None and rescue_net is not None:
            net_delta = rescue_net - base_net
        lines.append(
            f"| | net_progress | {_fmt(base_net)} m | {_fmt(rescue_net)} m | {_fmt(net_delta)} m |"
        )
        lines.append(
            f"| | rescue_steps | 0 | {rescue.get('route_rescue_active_step_count', 0)} | — |"
        )
        lines.append(
            f"| | slowdown_steps | 0 | "
            f"{rescue.get('feasibility_slowdown_active_step_count', 0)} | — |"
        )
    lines.append("")
    lines.append("## Mechanism analysis")
    lines.append("")
    if rescue_bottleneck is not None:
        rp = rescue_bottleneck.get("route_progress", {})
        lines.append("### bottleneck_timeout")
        lines.append("")
        lines.append(
            f"- Termination: {rescue_bottleneck['termination_reason']} "
            f"after {rescue_bottleneck['steps']} steps."
        )
        lines.append(
            f"- Route progress: initial {_fmt(rp.get('initial_distance_to_goal_m'))} m, "
            f"final {_fmt(rp.get('final_distance_to_goal_m'))} m, "
            f"minimum {_fmt(rp.get('min_distance_to_goal_m'))} m, "
            f"net {_fmt(rp.get('net_progress_m'))} m."
        )
        lines.append(
            f"- Route-rescue was active for "
            f"{rescue_bottleneck.get('route_rescue_active_step_count', 0)} steps "
            f"(first activation step: "
            f"{rescue_bottleneck.get('route_rescue_first_active_step', 'never')})."
        )
        lines.append(
            f"- Feasibility-slowdown was active for "
            f"{rescue_bottleneck.get('feasibility_slowdown_active_step_count', 0)} steps."
        )
        lines.append("")
    if rescue_t_inter is not None:
        rp = rescue_t_inter.get("route_progress", {})
        lines.append("### t_intersection_collision")
        lines.append("")
        lines.append(
            f"- Termination: {rescue_t_inter['termination_reason']} "
            f"after {rescue_t_inter['steps']} steps."
        )
        lines.append(
            f"- Route progress: initial {_fmt(rp.get('initial_distance_to_goal_m'))} m, "
            f"final {_fmt(rp.get('final_distance_to_goal_m'))} m, "
            f"minimum {_fmt(rp.get('min_distance_to_goal_m'))} m, "
            f"net {_fmt(rp.get('net_progress_m'))} m."
        )
        lines.append(
            f"- Route-rescue was active for "
            f"{rescue_t_inter.get('route_rescue_active_step_count', 0)} steps "
            f"(first activation step: "
            f"{rescue_t_inter.get('route_rescue_first_active_step', 'never')})."
        )
        lines.append(
            f"- Feasibility-slowdown was active for "
            f"{rescue_t_inter.get('feasibility_slowdown_active_step_count', 0)} steps."
        )
        if rescue_t_inter.get("first_infeasible_candidate_step") is not None:
            lines.append(
                f"- First infeasible candidate at step "
                f"{rescue_t_inter['first_infeasible_candidate_step']}."
            )
        lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "This is a diagnostic probe, not a success/fail benchmark claim. The table above "
        "identifies whether the route-rescue/feasibility-slowdown intervention improves, "
        "fails to improve, or changes the mechanism on both rows."
    )
    lines.append("")
    lines.append(
        "If either row remains unresolved or the intervention alters the contract unsafely, "
        "the diagnostic classification is retained and the next smallest probe is named."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("DISPLAY= SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python \\")
    lines.append("  scripts/benchmark/trace_dwa_route_rescue_issue_5319.py \\")
    lines.append("  --out-dir output/benchmarks/issue_5319 \\")
    lines.append("  --evidence-dir docs/context/evidence/issue_5319_dwa_route_rescue_2026-07-11")
    lines.append("```")
    lines.append("")
    lines.append(f"Executed at repo commit `{trace_commit}`.")
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


_trace_commit = trace_commit
_sha256_file = sha256_file
_repo_relative_path = repo_relative_path


def trace_episodes(
    *,
    matrix_path: Path,
    algo_config_path: Path,
    out_dir: Path,
    evidence_dir: Path | None,
) -> dict[str, Any]:
    """Run both target episodes with route-rescue config and capture traces."""
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
        out_dir / "dwa_route_rescue_trace.json",
        {
            "schema_version": "dwa-route-rescue-trace.v1",
            "issue": 5319,
            "claim_boundary": "diagnostic-only: two CPU fixed-seed DWA episodes with route-rescue; no benchmark/roster/metric/paper claim.",
            "config": _repo_relative_path(algo_config_path),
            "episodes": episodes_payload,
        },
        review_marker=True,
    )
    _write_steps_csv(out_dir / "dwa_route_rescue_steps.csv", all_step_rows)
    summary_payload = {
        "schema_version": "dwa-route-rescue-trace.v1",
        "issue": 5319,
        "config": _repo_relative_path(algo_config_path),
        "episodes": summaries,
    }
    write_json_atomic(
        out_dir / "dwa_route_rescue_summary.json", summary_payload, review_marker=True
    )

    baseline_path = BASELINE_SUMMARY
    if not baseline_path.exists():
        candidate = out_dir.parent / "issue_5298" / "dwa_decision_trace_summary.json"
        baseline_path = candidate if candidate.exists() else None
    baseline_summaries: list[dict[str, Any]] = []
    if baseline_path and baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_summaries = baseline_data.get("episodes", [])

    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_steps_csv(evidence_dir / "dwa_route_rescue_steps.csv", all_step_rows)
        write_json_atomic(
            evidence_dir / "dwa_route_rescue_summary.json",
            summary_payload,
            review_marker=True,
        )
        _write_evidence_readme(
            evidence_dir / "README.md",
            summaries=summaries,
            baseline_summaries=baseline_summaries,
            trace_commit=_trace_commit(),
            config_sha256=_sha256_file(algo_config_path),
        )

    return {
        "issue": 5319,
        "episodes": summaries,
        "out_dir": str(out_dir),
        "evidence_dir": str(evidence_dir) if evidence_dir else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the bounded DWA route-rescue diagnostic for issue #5319."""
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
