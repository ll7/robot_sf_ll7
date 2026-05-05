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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", nargs="+", type=Path)
    parser.add_argument("--success-rate-min", type=float, default=0.75)
    parser.add_argument("--collision-rate-max", type=float, default=0.03)
    parser.add_argument("--buffer-steps", type=int, default=20)
    parser.add_argument("--floor-steps", type=int, default=80)
    parser.add_argument("--cap-steps", type=int, default=500)
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _resolve_jsonl_path(summary_path: Path, payload: dict[str, Any]) -> Path:
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


def _summary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "success_rate": float(summary.get("success_rate", 0.0)),
        "collision_rate": float(summary.get("collision_rate", 1.0)),
        "near_miss_rate": float(summary.get("near_miss_rate", 0.0)),
    }


def _quantile(values: list[float], q: float) -> float | None:
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


def _is_success(record: dict[str, Any]) -> bool:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return bool(metrics.get("success") is True)


def _is_timeout(record: dict[str, Any]) -> bool:
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    termination = str(record.get("termination_reason", "")).strip().lower()
    return bool(outcome.get("timeout_event")) or termination in {"max_steps", "timeout"}


def _scenario_id(record: dict[str, Any]) -> str:
    return str(record.get("scenario_id") or "unknown")


def _recommend_horizon(
    success_steps: list[float],
    *,
    buffer_steps: int,
    floor_steps: int,
    cap_steps: int,
    near_cap_margin: int,
) -> tuple[int, str]:
    p95 = _quantile(success_steps, 0.95)
    if p95 is None:
        return int(cap_steps), "planner_blocked"
    recommended = int(min(cap_steps, max(floor_steps, math.ceil(p95 + buffer_steps))))
    if recommended >= cap_steps - near_cap_margin:
        return recommended, "needs_longer_probe"
    return recommended, "recommended"


def _bucket(recommended: int, status: str) -> str:
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
        summary_record = {
            "candidate": candidate,
            "stage": stage,
            "summary_json": summary_path.as_posix(),
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
            "success_steps_p50": _quantile(success_steps, 0.50),
            "success_steps_p90": _quantile(success_steps, 0.90),
            "success_steps_p95": _quantile(success_steps, 0.95),
            "success_steps_max": max(success_steps) if success_steps else None,
        }

    output = {
        "version": 1,
        "source": "policy_search_h500_safe_incumbents",
        "selection": {
            "success_rate_min": float(args.success_rate_min),
            "collision_rate_max": float(args.collision_rate_max),
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
        "Generated from safe incumbent policy-search summaries.",
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
