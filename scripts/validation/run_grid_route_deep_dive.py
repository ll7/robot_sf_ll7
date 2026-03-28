#!/usr/bin/env python3
"""Run the experimental grid-route planner across all shipped scenario-set manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from loguru import logger

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_ALGO_CONFIG = ROOT / "configs/algos/grid_route_camera_ready.yaml"
DEFAULT_SCENARIO_DIR = ROOT / "configs/scenarios/sets"
DEFAULT_OUT_DIR = ROOT / "output/validation/grid_route_deep_dive/latest"
DEFAULT_SET_NAMES = (
    "atomic_navigation_minimal_full_v1.yaml",
    "atomic_navigation_validation_fixtures_v1.yaml",
    "classic_crossing_subset.yaml",
    "safety_barrier_static_slice_v1.yaml",
    "verified_simple_subset_v1.yaml",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the grid-route deep-dive runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument(
        "--scenario-set",
        action="append",
        dest="scenario_sets",
        help="Optional explicit scenario-set path. Repeat to run a subset.",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--algo-config", type=Path, default=DEFAULT_ALGO_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--algo", default="grid_route")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=320)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _set_paths(args: argparse.Namespace) -> list[Path]:
    if args.scenario_sets:
        return [Path(item).resolve() for item in args.scenario_sets]
    return [(args.scenario_dir / name).resolve() for name in DEFAULT_SET_NAMES]


def _scenario_name(row: dict[str, Any]) -> str:
    scenario_payload = row.get("scenario")
    if isinstance(scenario_payload, dict) and scenario_payload.get("name"):
        return str(scenario_payload["name"])
    return str(row.get("scenario_id", "unknown"))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reasons = Counter()
    successes = 0
    collisions = 0
    for row in rows:
        scenario_name = _scenario_name(row)
        by_scenario[scenario_name].append(row)
        reason = str(row.get("termination_reason", "unknown"))
        reasons[reason] += 1
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and float(metrics.get("success", 0.0) or 0.0) >= 0.5:
            successes += 1
        if reason == "collision":
            collisions += 1

    per_scenario: list[dict[str, Any]] = []
    for name, scenario_rows in sorted(by_scenario.items()):
        scenario_successes = sum(
            1
            for row in scenario_rows
            if isinstance(row.get("metrics"), dict)
            and float(row["metrics"].get("success", 0.0) or 0.0) >= 0.5
        )
        per_scenario.append(
            {
                "name": name,
                "episodes": len(scenario_rows),
                "success_rate": scenario_successes / max(len(scenario_rows), 1),
                "termination_counts": dict(
                    Counter(str(row.get("termination_reason", "unknown")) for row in scenario_rows)
                ),
            }
        )

    return {
        "episodes": len(rows),
        "success_rate": successes / max(len(rows), 1),
        "collision_rate": collisions / max(len(rows), 1),
        "termination_counts": dict(reasons),
        "per_scenario": per_scenario,
    }


def _overall_summary(set_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_episodes = 0
    total_successes = 0.0
    total_collisions = 0.0
    combined_reasons: Counter[str] = Counter()
    successful_sets = 0
    failed_sets = 0
    for item in set_summaries:
        if item.get("status") != "ok":
            failed_sets += 1
            continue
        successful_sets += 1
        summary = item["summary"]
        total_episodes += int(summary["episodes"])
        total_successes += float(summary["success_rate"]) * int(summary["episodes"])
        total_collisions += float(summary["collision_rate"]) * int(summary["episodes"])
        combined_reasons.update(summary["termination_counts"])
    return {
        "sets_attempted": len(set_summaries),
        "sets_succeeded": successful_sets,
        "sets_failed": failed_sets,
        "episodes": total_episodes,
        "success_rate": total_successes / max(total_episodes, 1),
        "collision_rate": total_collisions / max(total_episodes, 1),
        "termination_counts": dict(combined_reasons),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Grid Route Deep Dive",
        "",
        "## Overall",
        "",
        f"- Sets attempted: `{payload['overall']['sets_attempted']}`",
        f"- Sets succeeded: `{payload['overall']['sets_succeeded']}`",
        f"- Sets failed: `{payload['overall']['sets_failed']}`",
        f"- Episodes: `{payload['overall']['episodes']}`",
        f"- Success rate: `{payload['overall']['success_rate']:.4f}`",
        f"- Collision rate: `{payload['overall']['collision_rate']:.4f}`",
        f"- Termination counts: `{payload['overall']['termination_counts']}`",
        "",
        "## Scenario Sets",
        "",
    ]

    for item in payload["sets"]:
        lines.append(f"### {item['set_name']}")
        lines.append("")
        lines.append(f"- Manifest: `{item['manifest']}`")
        lines.append(f"- Expanded scenarios: `{item['scenario_count']}`")
        lines.append(f"- Status: `{item['status']}`")
        if item["status"] != "ok":
            lines.append(f"- Error: `{item['error']}`")
            lines.append("")
            continue
        summary = item["summary"]
        lines.append(f"- Episodes: `{summary['episodes']}`")
        lines.append(f"- Success rate: `{summary['success_rate']:.4f}`")
        lines.append(f"- Collision rate: `{summary['collision_rate']:.4f}`")
        lines.append(f"- Termination counts: `{summary['termination_counts']}`")
        lines.append("")
        lines.append("| scenario | episodes | success_rate | termination_counts |")
        lines.append("| --- | ---: | ---: | --- |")
        for row in summary["per_scenario"]:
            lines.append(
                f"| {row['name']} | {row['episodes']} | {row['success_rate']:.4f} | {row['termination_counts']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run grid-route on every configured scenario-set manifest and write a deep-dive summary."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_summaries: list[dict[str, Any]] = []
    for set_path in _set_paths(args):
        set_name = set_path.stem
        episodes_path = out_dir / set_name / "episodes.jsonl"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        if episodes_path.exists():
            episodes_path.unlink()

        scenarios: list[dict[str, Any]] = []
        try:
            scenarios = list(load_scenarios(set_path, base_dir=set_path))
            batch_summary = run_map_batch(
                set_path,
                episodes_path,
                schema_path=args.schema,
                algo=str(args.algo),
                algo_config_path=str(args.algo_config),
                horizon=int(args.horizon),
                dt=float(args.dt),
                workers=int(args.workers),
                resume=False,
                benchmark_profile="experimental",
            )
            if not episodes_path.exists():
                benchmark_availability = batch_summary.get("benchmark_availability", {})
                availability_status = str(
                    benchmark_availability.get("availability_status", "unknown")
                )
                availability_reason = str(
                    benchmark_availability.get("availability_reason", "episodes output not written")
                )
                set_summaries.append(
                    {
                        "set_name": set_name,
                        "manifest": str(set_path.relative_to(ROOT)),
                        "scenario_count": len(scenarios),
                        "status": "error",
                        "error": f"{availability_status}: {availability_reason}",
                        "batch_summary": batch_summary,
                    }
                )
                continue
            rows = _load_rows(episodes_path)
            set_summaries.append(
                {
                    "set_name": set_name,
                    "manifest": str(set_path.relative_to(ROOT)),
                    "scenario_count": len(scenarios),
                    "status": "ok",
                    "batch_summary": batch_summary,
                    "summary": _aggregate_rows(rows),
                }
            )
        except Exception as exc:
            set_summaries.append(
                {
                    "set_name": set_name,
                    "manifest": str(set_path.relative_to(ROOT)),
                    "scenario_count": len(scenarios) if "scenarios" in locals() else 0,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = {"overall": _overall_summary(set_summaries), "sets": set_summaries}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, out_dir / "summary.md")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
