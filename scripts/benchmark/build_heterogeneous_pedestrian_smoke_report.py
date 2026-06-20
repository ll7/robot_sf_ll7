#!/usr/bin/env python3
"""Build a diagnostic report for issue #3206 heterogeneous-pedestrian smoke rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "issue_3206_heterogeneous_pedestrian_smoke_report.v1"
DEFAULT_METRICS = (
    "success",
    "collisions",
    "min_distance",
    "mean_distance",
    "robot_ped_within_5m_frac",
)
CLAIM_BOUNDARY = (
    "diagnostic_smoke_not_benchmark_evidence: summarizes a tiny homogeneous-vs-mixed "
    "pedestrian composition smoke. It records metric deltas and distributional-metric "
    "readiness limits; it does not establish pedestrian realism, real-world fairness, "
    "planner ranking, or a limitation-replacement decision."
)


def _number(value: Any) -> float | None:
    """Return a finite float for numeric-like scalars."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load episode records from a JSONL file."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a non-object JSONL row")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no episode rows")
    return rows


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    scenario_params = row.get("scenario_params")
    if not isinstance(scenario_params, dict):
        return {}
    metadata = scenario_params.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _simulation_config(row: Mapping[str, Any]) -> dict[str, Any]:
    scenario_params = row.get("scenario_params")
    if not isinstance(scenario_params, dict):
        return {}
    sim_config = scenario_params.get("simulation_config")
    return sim_config if isinstance(sim_config, dict) else {}


def _condition(row: Mapping[str, Any]) -> str:
    metadata = _metadata(row)
    return str(metadata.get("archetype_condition") or row.get("scenario_id") or "unknown")


def _mean_metrics(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    out: dict[str, dict[str, float | int | None]] = {}
    for metric in metrics:
        values: list[float] = []
        for row in rows:
            row_metrics = row.get("metrics")
            if not isinstance(row_metrics, dict):
                continue
            value = _number(row_metrics.get(metric))
            if value is not None:
                values.append(value)
        out[metric] = {
            "mean": sum(values) / len(values) if values else None,
            "count": len(values),
        }
    return out


def _distributional_status(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support_counts: dict[str, int] = {}
    missing_reasons: dict[str, dict[str, int]] = {}
    block_count = 0
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        block = metrics.get("distributional_disruption")
        if not isinstance(block, dict):
            continue
        block_count += 1
        for cohort, count in (block.get("support_counts") or {}).items():
            if isinstance(count, int):
                support_counts[str(cohort)] = support_counts.get(str(cohort), 0) + count
        for cohort, missing in (block.get("missing_data") or {}).items():
            if not isinstance(missing, dict):
                continue
            reason = str(missing.get("reason") or "unspecified")
            bucket = missing_reasons.setdefault(str(cohort), {})
            bucket[reason] = bucket.get(reason, 0) + 1

    supported_total = sum(support_counts.values())
    if block_count == 0:
        status = "absent"
        limitation = "No distributional_disruption block was present in the episode rows."
    elif supported_total == 0:
        status = "not_computable"
        limitation = (
            "Distributional-disruption blocks are present, but all support counts are zero. "
            "The current smoke did not provide the control trace required for per-cohort "
            "displacement or delay support."
        )
    else:
        status = "computable"
        limitation = ""
    return {
        "status": status,
        "block_count": block_count,
        "support_counts": support_counts,
        "missing_reasons": missing_reasons,
        "limitation": limitation,
    }


def _planned_composition(row: Mapping[str, Any]) -> dict[str, Any]:
    sim_config = _simulation_config(row)
    composition = sim_config.get("archetype_composition")
    speed_factors = sim_config.get("archetype_speed_factors")
    return {
        "composition": composition if isinstance(composition, dict) else {},
        "speed_factors": speed_factors if isinstance(speed_factors, dict) else {},
        "archetype_seed": sim_config.get("archetype_seed"),
        "realized_counts_status": "not_recorded_in_episode_rows",
    }


def _metric_deltas(
    baseline: Mapping[str, Mapping[str, float | int | None]],
    variant: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    for metric in sorted(set(baseline) & set(variant)):
        baseline_mean = baseline[metric].get("mean")
        variant_mean = variant[metric].get("mean")
        if isinstance(baseline_mean, (int, float)) and isinstance(variant_mean, (int, float)):
            absolute = float(variant_mean) - float(baseline_mean)
            relative = None if float(baseline_mean) == 0.0 else absolute / float(baseline_mean)
        else:
            absolute = None
            relative = None
        deltas[metric] = {
            "baseline": float(baseline_mean) if isinstance(baseline_mean, (int, float)) else None,
            "variant": float(variant_mean) if isinstance(variant_mean, (int, float)) else None,
            "absolute_delta": absolute,
            "relative_delta": relative,
        }
    return deltas


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_condition: str = "homogeneous_standard",
    variant_condition: str = "mixed_balanced",
    metrics: Sequence[str] = DEFAULT_METRICS,
    input_ref: str = "",
) -> dict[str, Any]:
    """Build a compact heterogeneous-pedestrian smoke report."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_condition(row), []).append(row)
    if baseline_condition not in grouped:
        raise ValueError(f"baseline condition {baseline_condition!r} is missing")
    if variant_condition not in grouped:
        raise ValueError(f"variant condition {variant_condition!r} is missing")

    conditions: dict[str, Any] = {}
    for condition, condition_rows in sorted(grouped.items()):
        first = condition_rows[0]
        conditions[condition] = {
            "episode_count": len(condition_rows),
            "seeds": sorted({row.get("seed") for row in condition_rows}),
            "scenario_ids": sorted({str(row.get("scenario_id", "")) for row in condition_rows}),
            "planned_archetype_population": _planned_composition(first),
            "metrics": _mean_metrics(condition_rows, metrics),
            "distributional_disruption": _distributional_status(condition_rows),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3206,
        "status": "diagnostic_smoke_report",
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "episodes_jsonl": input_ref,
            "raw_episodes_committed": False,
        },
        "source_episode_git_hashes": sorted(
            {str(row.get("git_hash")) for row in rows if row.get("git_hash")}
        ),
        "baseline_condition": baseline_condition,
        "variant_condition": variant_condition,
        "conditions": conditions,
        "delta_variant_minus_baseline": _metric_deltas(
            conditions[baseline_condition]["metrics"],
            conditions[variant_condition]["metrics"],
        ),
        "per_archetype_distributional_status": (
            "not_computable_from_current_smoke"
            if any(
                condition["distributional_disruption"]["status"] != "computable"
                for condition in conditions.values()
            )
            else "computable"
        ),
    }


def format_markdown(report: Mapping[str, Any]) -> str:
    """Render the smoke report as Markdown."""
    source_hashes = ", ".join(report["source_episode_git_hashes"]) or "unknown"
    lines = [
        "# Issue #3206 Heterogeneous Pedestrian Smoke Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Source episode git hash(es): `{source_hashes}`",
        f"- Input rows: {report['inputs']['episodes_jsonl']}",
        f"- Claim boundary: {report['claim_boundary']}",
        f"- Per-archetype distributional status: `{report['per_archetype_distributional_status']}`",
        "",
        "## Provenance",
        "",
        "- Branch: `issue-3206-archetype-reporting`",
        "- Matrix: `configs/scenarios/sets/issue_3206_heterogeneous_pedestrian_smoke.yaml`",
        "- Planner: `simple_policy`",
        "- Horizon: `80`",
        "- Time step: `0.1`",
        "- Seeds: scenario seeds `101`, `102`, `103`; route spawn seed `3206`; archetype seed `3206`",
        "- Raw episodes: generated under local `output/` or `/tmp`; intentionally not committed.",
        "",
        "## Commands",
        "",
        "```bash",
        "scripts/dev/run_worktree_shared_venv.sh -- robot_sf_bench --quiet run \\",
        "  --matrix configs/scenarios/sets/issue_3206_heterogeneous_pedestrian_smoke.yaml \\",
        "  --out output/benchmarks/issue_3206_heterogeneous_pedestrian_smoke/episodes.jsonl \\",
        "  --algo simple_policy --workers 1 --horizon 80 --dt 0.1 --no-video --video-renderer none \\",
        "  --no-resume --benchmark-profile baseline-safe --structured-output json",
        "",
        "scripts/dev/run_worktree_shared_venv.sh -- python scripts/benchmark/build_heterogeneous_pedestrian_smoke_report.py \\",
        "  --episodes output/benchmarks/issue_3206_heterogeneous_pedestrian_smoke/episodes.jsonl \\",
        "  --output-dir docs/context/evidence/issue_3206_heterogeneous_pedestrian_smoke_2026-06-20",
        "```",
        "",
        "## Condition Metrics",
        "",
        "| Condition | Episodes | Success | Collisions | Min distance mean | Mean distance mean | Robot within 5 m frac | Distributional status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for condition, summary in report["conditions"].items():
        metrics = summary["metrics"]
        lines.append(
            f"| `{condition}` | {summary['episode_count']} | "
            f"{_fmt(metrics['success']['mean'])} | {_fmt(metrics['collisions']['mean'])} | "
            f"{_fmt(metrics['min_distance']['mean'])} | "
            f"{_fmt(metrics['mean_distance']['mean'])} | "
            f"{_fmt(metrics['robot_ped_within_5m_frac']['mean'])} | "
            f"`{summary['distributional_disruption']['status']}` |"
        )

    lines.extend(["", "## Planned Composition", ""])
    for condition, summary in report["conditions"].items():
        composition = summary["planned_archetype_population"]["composition"]
        lines.append(f"- `{condition}`: `{json.dumps(composition, sort_keys=True)}`")

    lines.extend(
        [
            "",
            "## Distributional/Fairness Boundary",
            "",
            "The current smoke carries `distributional_disruption` blocks, but support counts are zero",
            "because no control trace was provided. That means per-archetype displacement or delay",
            "fairness-style metrics are not computable from this smoke. The result remains useful as",
            "a runtime and metric-delta smoke only.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "`null`" if number is None else f"{number:.3f}"


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    """Write JSON and Markdown report files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(format_markdown(report), encoding="utf-8")


def _input_ref(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"worktree-local ignored artifact summarized in this report ({path.name})"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, help="Episode JSONL from the smoke run.")
    parser.add_argument("--baseline-condition", default="homogeneous_standard")
    parser.add_argument("--variant-condition", default="mixed_balanced")
    parser.add_argument(
        "--output-dir",
        default="docs/context/evidence/issue_3206_heterogeneous_pedestrian_smoke_2026-06-20",
    )
    return parser


def main() -> int:
    """Run the smoke-report builder CLI."""
    args = _build_arg_parser().parse_args()
    episodes = Path(args.episodes)
    report = build_report(
        _load_jsonl(episodes),
        baseline_condition=args.baseline_condition,
        variant_condition=args.variant_condition,
        input_ref=_input_ref(episodes),
    )
    write_report(report, REPO_ROOT / args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
