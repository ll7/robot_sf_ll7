#!/usr/bin/env python3
"""Build the diagnostic S30 interpretation packet for issue #4882."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.rank_metrics import kendall_tau, rank_by, rank_order

CLAIM_BOUNDARY = (
    "Diagnostic-only S30 interpretation. This packet does not promote a paper, dissertation, "
    "record-breaking, universal-planner, or real-world claim."
)
FIVE_ARM_KEYS = (
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "orca",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
)
ALL_ARM_KEYS = (*FIVE_ARM_KEYS[:2], "orca", "ppo", *FIVE_ARM_KEYS[3:])
TARGET_HYBRID = "hybrid_rule_v3_fast_progress_static_escape_continuous"
METRICS = (
    "success",
    "collision_event",
    "near_miss_episode",
    "near_misses",
    "time_to_goal_norm",
    "path_efficiency",
    "snqi",
    "wall_time_sec",
)


@dataclass(frozen=True)
class ArmInput:
    """One frozen planner arm and its source provenance."""

    planner_key: str
    source_job: str
    source_commit: str
    execution_mode: str
    path: Path
    rows: list[dict[str, Any]]
    raw_rows: int
    raw_commits: dict[str, int]


def _json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _row_commit(row: dict[str, Any]) -> str:
    provenance = row.get("result_provenance") or {}
    return str(row.get("git_hash") or provenance.get("repo_commit") or "")


def _identity(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["scenario_id"]), int(row["seed"])


def _metric(row: dict[str, Any], metric: str) -> float:
    metrics = row.get("metrics") or {}
    if metric == "success":
        if "success" not in metrics:
            raise ValueError("metrics.success is required")
        return float(bool(metrics["success"]))
    if metric == "collision_event":
        outcome = row.get("outcome") or {}
        if "collision_event" not in outcome:
            raise ValueError("outcome.collision_event is required")
        return float(bool(outcome["collision_event"]))
    if metric == "near_miss_episode":
        return float(float(metrics.get("near_misses", 0.0)) > 0.0)
    if metric == "wall_time_sec":
        return float(row.get("wall_time_sec", 0.0))
    if metric not in metrics or metrics[metric] is None:
        raise ValueError(f"metrics.{metric} is required")
    return float(metrics[metric])


def _execution_mode(manifest: dict[str, Any], planner_key: str) -> str:
    for planner in manifest.get("planners", []):
        if planner.get("key") != planner_key:
            continue
        group = str(planner.get("planner_group") or "")
        return "native" if planner_key == "ppo" else "adapter" if group else "unknown"
    raise ValueError(f"Planner {planner_key!r} absent from campaign manifest")


def load_arm(
    root: Path,
    planner_key: str,
    *,
    source_job: str,
    source_commit: str,
) -> ArmInput:
    """Load one planner arm and select only the frozen source commit."""

    path = root / "runs" / f"{planner_key}__differential_drive" / "episodes.jsonl"
    raw_rows = _read_rows(path)
    commits: dict[str, int] = defaultdict(int)
    for row in raw_rows:
        commits[_row_commit(row)] += 1
    selected = [row for row in raw_rows if _row_commit(row) == source_commit]
    if not selected:
        raise ValueError(f"No rows for {planner_key} at frozen commit {source_commit}")
    manifest = _json(root / "campaign_manifest.json")
    return ArmInput(
        planner_key=planner_key,
        source_job=source_job,
        source_commit=source_commit,
        execution_mode=_execution_mode(manifest, planner_key),
        path=path,
        rows=selected,
        raw_rows=len(raw_rows),
        raw_commits=dict(sorted(commits.items())),
    )


def validate_grid(
    arms: list[ArmInput],
    *,
    expected_scenarios: int,
    expected_seeds: list[int],
    expected_horizon: int = 600,
) -> tuple[list[str], list[int]]:
    """Require the exact scenario-by-seed grid and exclude degraded modes."""

    reference_scenarios: list[str] | None = None
    expected = expected_scenarios * len(expected_seeds)
    for arm in arms:
        identities = [_identity(row) for row in arm.rows]
        if len(identities) != expected:
            raise ValueError(
                f"{arm.planner_key}: selected rows {len(identities)} != expected {expected}"
            )
        if len(set(identities)) != expected:
            raise ValueError(f"{arm.planner_key}: duplicate selected scenario/seed identities")
        scenarios = sorted({scenario for scenario, _seed in identities})
        seeds = sorted({seed for _scenario, seed in identities})
        if len(scenarios) != expected_scenarios or seeds != expected_seeds:
            raise ValueError(f"{arm.planner_key}: selected grid does not match the frozen contract")
        if reference_scenarios is None:
            reference_scenarios = scenarios
        elif scenarios != reference_scenarios:
            raise ValueError(f"{arm.planner_key}: scenario roster differs across arms")
        horizons = {
            int(row.get("horizon") or (row.get("scenario_params") or {}).get("run_horizon") or 0)
            for row in arm.rows
        }
        if horizons != {expected_horizon}:
            raise ValueError(
                f"{arm.planner_key}: horizons {sorted(horizons)} != {expected_horizon}"
            )
        modes = {
            str((row.get("pedestrian_model") or {}).get("fallback_degraded_status") or "native")
            for row in arm.rows
        }
        if modes - {"native", "adapter"}:
            raise ValueError(f"{arm.planner_key}: fallback/degraded rows present: {sorted(modes)}")
    return reference_scenarios or [], expected_seeds


def _seed_means(rows: list[dict[str, Any]], metric: str, seeds: list[int]) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if int(row["seed"]) in seeds:
            grouped[int(row["seed"])].append(_metric(row, metric))
    if sorted(grouped) != seeds:
        raise ValueError(f"Missing seed blocks for metric {metric}")
    return np.asarray([np.mean(grouped[seed]) for seed in seeds], dtype=float)


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if values.size == 0 or samples <= 0:
        raise ValueError("Bootstrap requires values and a positive sample count")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def arm_statistics(
    arms: list[ArmInput],
    seeds: list[int],
    *,
    samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Compute seed-block means and percentile intervals for every arm."""

    output: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(arms):
        row: dict[str, Any] = {
            "planner_key": arm.planner_key,
            "source_job": arm.source_job,
            "source_commit": arm.source_commit,
            "execution_mode": arm.execution_mode,
            "episodes": len(arm.rows),
            "scenarios": len({_identity(item)[0] for item in arm.rows}),
            "seeds": len(seeds),
        }
        for metric_index, metric in enumerate(METRICS):
            values = _seed_means(arm.rows, metric, seeds)
            low, high = _bootstrap_ci(
                values,
                samples=samples,
                seed=bootstrap_seed + arm_index * 100 + metric_index,
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        output.append(row)
    return output


def classify_branch(
    arms: list[ArmInput],
    seeds: list[int],
    *,
    samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Apply the preregistered paired success-separation rule."""

    by_key = {arm.planner_key: arm for arm in arms}
    orca = by_key["orca"]
    comparisons: list[dict[str, Any]] = []
    for index, planner_key in enumerate(key for key in ALL_ARM_KEYS if "hybrid" in key):
        candidate = by_key[planner_key]
        success_delta = _seed_means(candidate.rows, "success", seeds) - _seed_means(
            orca.rows, "success", seeds
        )
        collision_delta = _seed_means(candidate.rows, "collision_event", seeds) - _seed_means(
            orca.rows, "collision_event", seeds
        )
        success_ci = _bootstrap_ci(
            success_delta, samples=samples, seed=bootstrap_seed + 1000 + index
        )
        collision_ci = _bootstrap_ci(
            collision_delta, samples=samples, seed=bootstrap_seed + 2000 + index
        )
        comparisons.append(
            {
                "planner_key": planner_key,
                "comparator": "orca",
                "success_delta": float(success_delta.mean()),
                "success_delta_ci": list(success_ci),
                "collision_delta": float(collision_delta.mean()),
                "collision_delta_ci": list(collision_ci),
                "success_separates_positive": success_ci[0] > 0.0,
                "collision_supports_lower": collision_ci[1] < 0.0,
            }
        )
    target = next(item for item in comparisons if item["planner_key"] == TARGET_HYBRID)
    verdict = "branch_a_separation" if target["success_separates_positive"] else "branch_b_boundary"
    return {
        "schema_version": "issue_4882_branch_verdict.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_status": "diagnostic-only",
        "verdict": verdict,
        "target_pair": [TARGET_HYBRID, "orca"],
        "primary_rule": "paired seed-block 95% CI for success delta excludes zero above zero",
        "secondary_check": "paired seed-block collision-event delta CI excludes zero below zero",
        "collision_field": "outcome.collision_event",
        "success_field": "metrics.success",
        "comparisons": comparisons,
    }


def rank_stability(arms: list[ArmInput], seeds: list[int]) -> dict[str, Any]:
    """Compare the preregistered S20 seed prefix with the full S30 schedule."""

    s20 = seeds[:20] if len(seeds) >= 20 else seeds[:-1]
    if len(s20) < 2:
        raise ValueError("Rank stability needs at least two S20-prefix seeds")
    output: dict[str, Any] = {
        "schema_version": "issue_4882_s20_s30_rank_stability.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_status": "diagnostic-only",
        "s20_surface": "derived prefix of the same frozen S30 rows",
        "s20_seeds": s20,
        "s30_seeds": seeds,
        "independence_caveat": (
            "S20 is the preregistered 111-130 prefix, not an independent rerun; this isolates "
            "seed-budget sensitivity on the identical h600 matrix."
        ),
        "metrics": {},
    }
    for metric, higher_is_better in (
        ("success", True),
        ("collision_event", False),
        ("time_to_goal_norm", False),
    ):
        s20_values = {
            arm.planner_key: float(_seed_means(arm.rows, metric, s20).mean()) for arm in arms
        }
        s30_values = {
            arm.planner_key: float(_seed_means(arm.rows, metric, seeds).mean()) for arm in arms
        }
        order20 = [str(item) for item in rank_order(s20_values, higher_is_better=higher_is_better)]
        order30 = [str(item) for item in rank_order(s30_values, higher_is_better=higher_is_better)]
        ranks20 = rank_by(s20_values, higher_is_better=higher_is_better)
        ranks30 = rank_by(s30_values, higher_is_better=higher_is_better)
        output["metrics"][metric] = {
            "higher_is_better": higher_is_better,
            "s20_order": order20,
            "s30_order": order30,
            "kendall_tau": kendall_tau(order20, order30),
            "top_changed": order20[0] != order30[0],
            "planners": [
                {
                    "planner_key": key,
                    "s20_value": s20_values[key],
                    "s30_value": s30_values[key],
                    "metric_delta": s30_values[key] - s20_values[key],
                    "s20_rank": ranks20[key],
                    "s30_rank": ranks30[key],
                    "rank_delta": ranks30[key] - ranks20[key],
                }
                for key in sorted(s20_values)
            ],
        }
    return output


def determine_mode(job_state: str, supplemental_ppo_available: bool) -> str:
    """Route a failed campaign to triage unless supplemental evidence completes it."""

    normalized = job_state.upper()
    if normalized == "COMPLETED" or supplemental_ppo_available:
        return "interpretation"
    if normalized in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
        return "failure_triage"
    return "blocked_incomplete_evidence"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_branch(data: dict[str, Any]) -> str:
    lines = [
        "# S30 hybrid-versus-ORCA branch verdict",
        "",
        f"**Claim boundary:** {CLAIM_BOUNDARY}",
        "",
        f"**Verdict:** `{data['verdict']}`",
        "",
        "The primary decision uses the paired seed-block 95% interval for success. Collision-event "
        "separation is a secondary safety check. Adapter/native and simulator-only limitations remain.",
        "",
        "| Hybrid arm | Success delta (95% CI) | Collision-event delta (95% CI) |",
        "|---|---:|---:|",
    ]
    for row in data["comparisons"]:
        slo, shi = row["success_delta_ci"]
        clo, chi = row["collision_delta_ci"]
        lines.append(
            f"| `{row['planner_key']}` | {row['success_delta']:+.4f} "
            f"[{slo:+.4f}, {shi:+.4f}] | {row['collision_delta']:+.4f} "
            f"[{clo:+.4f}, {chi:+.4f}] |"
        )
    return "\n".join(lines) + "\n"


def _render_rank(data: dict[str, Any]) -> str:
    lines = [
        "# S20-prefix versus S30 rank stability",
        "",
        f"**Claim boundary:** {CLAIM_BOUNDARY}",
        "",
        data["independence_caveat"],
        "",
    ]
    for metric, report in data["metrics"].items():
        lines.extend(
            [
                f"## `{metric}`",
                "",
                f"- Kendall tau: `{report['kendall_tau']:.4f}`",
                f"- Top planner changed: `{str(report['top_changed']).lower()}`",
                f"- S20 order: `{' > '.join(report['s20_order'])}`",
                f"- S30 order: `{' > '.join(report['s30_order'])}`",
                "",
            ]
        )
    return "\n".join(lines)


def _render_readme(branch: dict[str, Any], ranks: dict[str, Any]) -> str:
    success_tau = ranks["metrics"]["success"]["kendall_tau"]
    target = next(item for item in branch["comparisons"] if item["planner_key"] == TARGET_HYBRID)
    return f"""# Issue #4882 S30 interpretation packet

**Claim boundary:** {CLAIM_BOUNDARY}

**Evidence status:** `diagnostic-only`. The packet composes the scheduler-frozen final-attempt
five-arm slice from job 13376 with the clean PPO-only job 13388. It excludes the contaminated
job 13378 aggregate and excludes SNQI ranking because the SNQI contract failed.

## Result

- Branch verdict: `{branch["verdict"]}`.
- Hybrid v3 continuous versus ORCA success delta: `{target["success_delta"]:+.4f}` with 95% CI
  `[{target["success_delta_ci"][0]:+.4f}, {target["success_delta_ci"][1]:+.4f}]`.
- Collision-event delta: `{target["collision_delta"]:+.4f}` with 95% CI
  `[{target["collision_delta_ci"][0]:+.4f}, {target["collision_delta_ci"][1]:+.4f}]`.
- Success-rank Kendall tau from the preregistered S20 prefix to S30: `{success_tau:.4f}`.

## Evidence boundaries

- Success is `metrics.success`; collision is `outcome.collision_event`.
- Intervals are paired or per-arm percentile bootstraps over seed-level means, not individual rows.
- S20 is seeds 111-130 derived from the identical frozen S30 rows, not an independent rerun.
- Hybrid/ORCA arms are adapter-mode; PPO is native. No fallback/degraded rows are admitted.
- The scenario-normalized quality index (SNQI) contract failed and is descriptive only.
- There is no independent clean-machine or real-world validation.

See `input_audit.json`, `campaign_crosscheck.json`, and `SHA256SUMS` for provenance and integrity.
"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Build all compact packet artifacts from two frozen campaign roots."""

    five_root = args.five_arm_root.resolve()
    ppo_root = args.ppo_root.resolve()
    mode = determine_mode(args.five_arm_job_state, supplemental_ppo_available=ppo_root.exists())
    if mode != "interpretation":
        raise ValueError(f"Cannot build interpretation packet: {mode}")
    arms = [
        load_arm(
            five_root,
            planner_key,
            source_job="13376",
            source_commit=args.five_arm_commit,
        )
        for planner_key in FIVE_ARM_KEYS
    ]
    arms.append(
        load_arm(
            ppo_root,
            "ppo",
            source_job="13388",
            source_commit=args.ppo_commit,
        )
    )
    arms.sort(key=lambda arm: ALL_ARM_KEYS.index(arm.planner_key))
    expected_seeds = list(range(args.seed_start, args.seed_start + args.expected_seeds))
    scenarios, seeds = validate_grid(
        arms,
        expected_scenarios=args.expected_scenarios,
        expected_seeds=expected_seeds,
        expected_horizon=args.expected_horizon,
    )
    manifest76 = _json(five_root / "campaign_manifest.json")
    manifest88 = _json(ppo_root / "campaign_manifest.json")
    if manifest76.get("scenario_matrix_hash") != manifest88.get("scenario_matrix_hash"):
        raise ValueError("Scenario matrix hashes differ across composed sources")

    statistics = arm_statistics(
        arms,
        seeds,
        samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    branch = classify_branch(
        arms,
        seeds,
        samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    ranks = rank_stability(arms, seeds)
    analyzer76 = _json(args.job13376_analysis_json) if args.job13376_analysis_json else {}
    analyzer78 = _json(args.job13378_analysis_json) if args.job13378_analysis_json else {}
    input_audit = {
        "schema_version": "issue_4882_input_audit.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "scenario_matrix_hash": manifest76.get("scenario_matrix_hash"),
        "scenario_count": len(scenarios),
        "seeds": seeds,
        "horizon": args.expected_horizon,
        "sources": [
            {
                "planner_key": arm.planner_key,
                "job_id": arm.source_job,
                "selected_commit": arm.source_commit,
                "execution_mode": arm.execution_mode,
                "episodes_path_sha256": _sha256(arm.path),
                "raw_rows": arm.raw_rows,
                "raw_commit_counts": arm.raw_commits,
                "selected_rows": len(arm.rows),
                "selected_unique_identities": len({_identity(row) for row in arm.rows}),
                "selected_timestamp_min": min(
                    str((row.get("timestamps") or {}).get("start") or "") for row in arm.rows
                ),
                "selected_timestamp_max": max(
                    str((row.get("timestamps") or {}).get("start") or "") for row in arm.rows
                ),
                "scheduler_state": "FAILED" if arm.source_job == "13376" else "COMPLETED",
            }
            for arm in arms
        ],
        "selection_rule": (
            "For job 13376, select the exact public commit launched by scheduler job 13376; its "
            "rows are the only commit block timestamped inside the scheduler interval. For job "
            "13388, require the single clean-root public commit."
        ),
    }
    crosscheck = {
        "schema_version": "issue_4882_campaign_crosscheck.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "status": "pass_with_documented_raw_append_history",
        "composed_rows": sum(len(arm.rows) for arm in arms),
        "expected_rows": len(arms) * len(scenarios) * len(seeds),
        "fallback_or_degraded_rows": 0,
        "matrix_hash_match": True,
        "job13376_analyzer_findings": analyzer76.get("findings", []),
        "job13378_analyzer_findings": analyzer78.get("findings", []),
        "contaminated_aggregate_excluded": True,
        "snqi_contract_status": "fail_excluded_from_rank_verdict",
        "amv_coverage_status": "incomplete",
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "input_audit.json", input_audit)
    _write_json(output / "s30_arm_statistics.json", statistics)
    _write_stats_csv(output / "s30_arm_statistics.csv", statistics)
    _write_json(output / "branch_verdict.json", branch)
    (output / "branch_verdict.md").write_text(_render_branch(branch), encoding="utf-8")
    _write_json(output / "s20_s30_rank_stability.json", ranks)
    (output / "s20_s30_rank_stability.md").write_text(_render_rank(ranks), encoding="utf-8")
    _write_json(output / "campaign_crosscheck.json", crosscheck)
    (output / "README.md").write_text(_render_readme(branch, ranks), encoding="utf-8")
    artifacts = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts), encoding="utf-8"
    )
    return {"output_dir": str(output), "verdict": branch["verdict"], "artifacts": len(artifacts)}


def parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--five-arm-root", type=Path, required=True)
    result.add_argument("--five-arm-commit", required=True)
    result.add_argument("--five-arm-job-state", default="FAILED")
    result.add_argument("--ppo-root", type=Path, required=True)
    result.add_argument("--ppo-commit", required=True)
    result.add_argument("--job13376-analysis-json", type=Path)
    result.add_argument("--job13378-analysis-json", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--expected-scenarios", type=int, default=48)
    result.add_argument("--expected-seeds", type=int, default=30)
    result.add_argument("--expected-horizon", type=int, default=600)
    result.add_argument("--seed-start", type=int, default=111)
    result.add_argument("--bootstrap-samples", type=int, default=30_000)
    result.add_argument("--bootstrap-seed", type=int, default=123)
    return result


def main() -> int:
    """Run the packet builder CLI."""

    args = parser().parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
