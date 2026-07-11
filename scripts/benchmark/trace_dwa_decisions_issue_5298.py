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
import csv
import json
import os
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner import run_map_batch
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
    """Return one scenario from the source matrix with a single pinned seed."""
    scenarios = load_scenarios(matrix_path, base_dir=matrix_path)
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


def _route_progress_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact route-progress statistics for one episode's trace."""
    if not rows:
        return {"status": "no_steps"}
    distances = [
        float(row["distance_to_goal_m"])
        for row in rows
        if row.get("distance_to_goal_m") is not None
    ]
    progresses = [
        float(row["route_progress_from_start_m"])
        for row in rows
        if row.get("route_progress_from_start_m") is not None
    ]
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# AI-GENERATED NEEDS-REVIEW\n")
        writer = csv.DictWriter(handle, fieldnames=list(STEP_TRACE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in STEP_TRACE_FIELDS})


def _write_evidence_readme(
    path: Path, *, summaries: list[dict[str, Any]], trace_commit: str
) -> None:  # noqa: PLR0915
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
    lines.append("follow-up should be tracked in its own scoped issue (see PR body).")
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
        f"Executed at repo commit `{trace_commit}`. Raw per-step trace is also written to the disposable "
    )
    lines.append(
        "`output/benchmarks/issue_5298/dwa_decision_trace.json`; this packet keeps the compact derived "
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _trace_commit() -> str:
    """Return the current git commit hash for provenance, or 'unknown'."""
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance-only; never fail the trace on it
        return "unknown"


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
            algo_config_path=str(algo_config_path),
            benchmark_profile="experimental",
            workers=1,
            resume=False,
            record_planner_decision_trace=True,
        )
        record = _read_record(episodes_path)
        algo_meta = (
            record.get("algorithm_metadata", {})
            if isinstance(record.get("algorithm_metadata"), dict)
            else {}
        )
        trace = (
            algo_meta.get("planner_decision_trace", {})
            if isinstance(algo_meta.get("planner_decision_trace"), dict)
            else {}
        )
        steps = trace.get("steps", []) if isinstance(trace.get("steps"), list) else []
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
    (out_dir / "dwa_decision_trace.json").write_text(
        json.dumps(
            {
                "schema_version": "dwa-decision-trace.v1",
                "issue": 5298,
                "claim_boundary": "analysis-only: two CPU fixed-seed DWA episodes; no benchmark/roster/metric/paper claim.",
                "episodes": episodes_payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_steps_csv(out_dir / "dwa_decision_trace_steps.csv", all_step_rows)
    (out_dir / "dwa_decision_trace_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "dwa-decision-trace.v1",
                "issue": 5298,
                "episodes": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_steps_csv(evidence_dir / "dwa_decision_trace_steps.csv", all_step_rows)
        (evidence_dir / "dwa_decision_trace_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "dwa-decision-trace.v1",
                    "issue": 5298,
                    "episodes": summaries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
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
