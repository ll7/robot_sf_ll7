#!/usr/bin/env python3
"""Run the testing-only safety-barrier planner on a narrow static scenario slice."""

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

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_SET = ROOT / "configs/scenarios/sets/safety_barrier_static_slice_v1.yaml"
DEFAULT_ALGO_CONFIG = ROOT / "configs/algos/safety_barrier_camera_ready.yaml"
DEFAULT_SCHEMA = ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_OUT_DIR = ROOT / "output/validation/safety_barrier_static_slice/latest"


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the static safety-barrier slice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-set", type=Path, default=DEFAULT_SCENARIO_SET)
    parser.add_argument("--algo-config", type=Path, default=DEFAULT_ALGO_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--algo", default="safety_barrier")
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


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reasons = Counter()
    successes = 0
    collisions = 0
    for row in rows:
        scenario_payload = row.get("scenario")
        scenario_name = (
            str(scenario_payload.get("name"))
            if isinstance(scenario_payload, dict) and scenario_payload.get("name")
            else str(row.get("scenario_id", "unknown"))
        )
        by_scenario[scenario_name].append(row)
        reason = str(row.get("termination_reason", "unknown"))
        reasons[reason] += 1
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and float(metrics.get("success", 0.0) or 0.0) >= 0.5:
            successes += 1
        if reason == "collision":
            collisions += 1

    per_scenario = []
    for name, scenario_rows in sorted(by_scenario.items()):
        scenario_success = sum(
            1
            for row in scenario_rows
            if isinstance(row.get("metrics"), dict)
            and float(row["metrics"].get("success", 0.0)) >= 0.5
        )
        per_scenario.append(
            {
                "name": name,
                "episodes": len(scenario_rows),
                "success_rate": scenario_success / max(len(scenario_rows), 1),
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


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Safety Barrier Static Slice",
        "",
        f"- Episodes: `{summary['episodes']}`",
        f"- Success rate: `{summary['success_rate']:.4f}`",
        f"- Collision rate: `{summary['collision_rate']:.4f}`",
        f"- Termination counts: `{summary['termination_counts']}`",
        "",
        "## Per Scenario",
        "",
        "| scenario | episodes | success_rate | termination_counts |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in summary["per_scenario"]:
        lines.append(
            f"| {row['name']} | {row['episodes']} | {row['success_rate']:.4f} | {row['termination_counts']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the static slice and write JSON/Markdown summaries."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "episodes.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()

    run_map_batch(
        args.scenario_set,
        out_jsonl,
        schema_path=args.schema,
        algo=str(args.algo),
        algo_config_path=str(args.algo_config),
        horizon=int(args.horizon),
        dt=float(args.dt),
        workers=int(args.workers),
        resume=False,
        benchmark_profile="experimental",
    )

    rows = _load_rows(out_jsonl)
    summary = _aggregate(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / "summary.md")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
