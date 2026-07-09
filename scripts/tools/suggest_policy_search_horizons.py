#!/usr/bin/env python3
"""Suggest scenario horizons from policy-search episode JSONL evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json
from robot_sf.benchmark.identity.hash_utils import read_jsonl as _load_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", nargs="+", type=Path)
    parser.add_argument("--success-rate-min", type=float, default=0.75)
    parser.add_argument("--collision-rate-max", type=float, default=0.03)
    parser.add_argument("--p95-multiplier", type=float, default=1.2)
    parser.add_argument("--buffer-steps", type=int, default=20)
    parser.add_argument("--floor-steps", type=int, default=80)
    parser.add_argument("--cap-steps", type=int, default=600)
    parser.add_argument("--near-cap-margin", type=int, default=20)
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=Path("output/policy_search/horizon_recommendations.yaml"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("output/policy_search/horizon_recommendations.md"),
    )
    return parser.parse_args()


def _resolve_jsonl_path(summary_path: Path, payload: dict[str, Any]) -> Path:
    """Resolve the episode JSONL path referenced by a summary payload.

    Returns:
        Absolute or best-effort relative JSONL path.
    """
    raw = payload.get("jsonl_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Summary is missing jsonl_path: {summary_path}")
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, summary_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _portable_summary_reference(summary_path: Path) -> tuple[str | None, str]:
    """Return a durable summary reference, or mark worktree output as non-durable provenance."""
    path = summary_path.as_posix()
    if path.startswith("output/"):
        return None, "worktree_output_not_promoted"
    return path, "tracked"


def _summary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Extract selection metrics from a policy-search summary.

    Returns:
        Rounded success, collision, and near-miss rates.
    """
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "success_rate": _round_float(float(summary.get("success_rate", 0.0))),
        "collision_rate": _round_float(float(summary.get("collision_rate", 1.0))),
        "near_miss_rate": _round_float(float(summary.get("near_miss_rate", 0.0))),
    }


def _quantile(values: list[float], q: float) -> float | None:
    """Compute a linearly interpolated quantile.

    Returns:
        Quantile value, or ``None`` when no values are available.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return float(ordered[low] * (1.0 - fraction) + ordered[high] * fraction)


def _round_float(value: float | None, digits: int = 6) -> float | None:
    """Round generated YAML floats without changing integer horizon decisions."""
    if value is None:
        return None
    return round(float(value), digits)


def _is_success(record: dict[str, Any]) -> bool:
    """Determine whether one episode record is a success.

    Returns:
        True when ``metrics.success`` is explicitly true.
    """
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return bool(metrics.get("success") is True)


def _is_timeout(record: dict[str, Any]) -> bool:
    """Determine whether one failed episode ended by timeout or max steps.

    Returns:
        True when outcome or termination metadata indicates a timeout.
    """
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    termination = str(record.get("termination_reason", "")).strip().lower()
    return bool(outcome.get("timeout_event")) or termination in {"max_steps", "timeout"}


def _scenario_id(record: dict[str, Any]) -> str:
    """Read the scenario id from one episode record.

    Returns:
        Scenario identifier string, or ``unknown`` when absent.
    """
    return str(record.get("scenario_id") or "unknown")


def _recommend_horizon(
    success_steps: list[float],
    *,
    buffer_steps: int,
    p95_multiplier: float,
    floor_steps: int,
    cap_steps: int,
    near_cap_margin: int,
) -> tuple[int, str]:
    """Recommend a horizon from successful episode step counts.

    Returns:
        Recommended horizon and status label.
    """
    p95 = _quantile(success_steps, 0.95)
    if p95 is None:
        return int(cap_steps), "planner_blocked"
    recommended = int(
        min(cap_steps, max(floor_steps, math.ceil((p95 * p95_multiplier) + buffer_steps)))
    )
    if recommended >= cap_steps - near_cap_margin:
        return recommended, "needs_longer_probe"
    return recommended, "recommended"


def _bucket(recommended: int, status: str) -> str:
    """Place a recommended horizon into a coarse runtime bucket.

    Returns:
        Bucket label for reporting.
    """
    if status == "planner_blocked":
        return "planner_blocked"
    if recommended <= 150:
        return "short"
    if recommended <= 300:
        return "medium"
    if recommended <= 500:
        return "long"
    return "extended"


def main() -> int:
    """Build horizon recommendations from safe incumbent summaries."""
    args = parse_args()

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    records_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for summary_path in args.summary_json:
        payload = _load_json(summary_path)
        candidate = str(payload.get("candidate", "unknown"))
        stage = str(payload.get("stage", "unknown"))
        metrics = _summary_metrics(payload)
        portable_summary, artifact_status = _portable_summary_reference(summary_path)
        summary_record = {
            "candidate": candidate,
            "stage": stage,
            "summary_json": portable_summary,
            "source_artifact_status": artifact_status,
            **metrics,
        }
        if metrics["success_rate"] >= float(args.success_rate_min) and metrics[
            "collision_rate"
        ] <= float(args.collision_rate_max):
            selected.append(summary_record)
            jsonl_path = _resolve_jsonl_path(summary_path, payload)
            for record in _load_jsonl(jsonl_path):
                records_by_scenario[_scenario_id(record)].append(record)
        else:
            rejected.append(summary_record)

    scenarios: dict[str, dict[str, Any]] = {}
    for scenario, records in sorted(records_by_scenario.items()):
        success_steps = [
            float(record.get("steps", 0.0)) for record in records if _is_success(record)
        ]
        failure_records = [record for record in records if not _is_success(record)]
        timeout_count = sum(1 for record in failure_records if _is_timeout(record))
        recommended, status = _recommend_horizon(
            success_steps,
            buffer_steps=int(args.buffer_steps),
            p95_multiplier=float(args.p95_multiplier),
            floor_steps=int(args.floor_steps),
            cap_steps=int(args.cap_steps),
            near_cap_margin=int(args.near_cap_margin),
        )
        scenarios[scenario] = {
            "recommended_horizon_steps": recommended,
            "status": status,
            "bucket": _bucket(recommended, status),
            "safe_incumbent_episode_count": len(records),
            "success_episode_count": len(success_steps),
            "failure_episode_count": len(failure_records),
            "timeout_failure_count": timeout_count,
            "success_steps_p50": _round_float(_quantile(success_steps, 0.50)),
            "success_steps_p90": _round_float(_quantile(success_steps, 0.90)),
            "success_steps_p95": _round_float(_quantile(success_steps, 0.95)),
            "success_steps_max": _round_float(max(success_steps) if success_steps else None),
        }

    output = {
        "version": 1,
        "source": "policy_search_h500_safe_incumbents",
        "selection": {
            "success_rate_min": _round_float(float(args.success_rate_min)),
            "collision_rate_max": _round_float(float(args.collision_rate_max)),
            "p95_multiplier": _round_float(float(args.p95_multiplier)),
            "buffer_steps": int(args.buffer_steps),
            "floor_steps": int(args.floor_steps),
            "cap_steps": int(args.cap_steps),
            "near_cap_margin": int(args.near_cap_margin),
        },
        "selected_summaries": selected,
        "rejected_summaries": rejected,
        "scenarios": scenarios,
    }

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")

    lines = [
        "# Policy Search Horizon Recommendations",
        "",
        "Generated from safe incumbent policy-search summaries. Scenario horizons use "
        f"`ceil(p95_success_steps * {float(args.p95_multiplier):.2f} "
        f"+ {int(args.buffer_steps)})`, floor "
        f"`{int(args.floor_steps)}`, and cap `{int(args.cap_steps)}`; scenarios with no "
        "safe-incumbent successes are marked `planner_blocked`.",
        "",
        "## Selected Summaries",
        "",
        "| Candidate | Stage | Success | Collision | Near Miss |",
        "|---|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['candidate']}` | `{row['stage']}` | {row['success_rate']:.4f} | "
            f"{row['collision_rate']:.4f} | {row['near_miss_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Scenario Recommendations",
            "",
            "| Scenario | Horizon | Status | Bucket | Success Episodes | P95 Steps | Timeouts/Failures |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for scenario, row in scenarios.items():
        p95 = row["success_steps_p95"]
        p95_text = "n/a" if p95 is None else f"{float(p95):.1f}"
        lines.append(
            f"| `{scenario}` | {row['recommended_horizon_steps']} | `{row['status']}` | "
            f"`{row['bucket']}` | {row['success_episode_count']} | {p95_text} | "
            f"{row['timeout_failure_count']}/{row['failure_episode_count']} |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"yaml": str(args.output_yaml), "markdown": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
