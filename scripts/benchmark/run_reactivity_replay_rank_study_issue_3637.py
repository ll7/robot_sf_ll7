#!/usr/bin/env python3
"""Run the packet-backed issue #3637 reactivity-vs-replay rank study.

This is the execution bridge between the frozen launch packet and the existing
#3573 paired campaign runner. It does not choose planners, seeds, scenario set,
or evidence labels on the command line; those come from the predeclared packet
after the same fail-closed preflight used by the planning checker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# Keep this script runnable directly from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.reactivity_replay_preflight import (  # noqa: E402
    build_preflight_manifest,
    run_plan_from_packet,
)

_RUNNER_PATH = (
    _REPO_ROOT / "scripts" / "benchmark" / "run_reactivity_ablation_campaign_issue_3573.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location("_issue_3573_reactivity_runner", _RUNNER_PATH)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    raise ImportError(f"cannot load campaign runner from {_RUNNER_PATH}")
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)
run_campaign = _RUNNER.run_campaign

DEFAULT_PACKET = Path(
    "configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml"
)
DEFAULT_OUTPUT_DIR = Path("output/issue_3637_reactivity_rank_study")
DEFAULT_REPORT_JSON = DEFAULT_OUTPUT_DIR / "report.json"
ISSUE = 3637
EVIDENCE_TIER = "seed_sufficient_candidate"


def _load_packet(path: Path) -> dict[str, Any]:
    """Load the issue #3637 YAML launch packet."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected YAML mapping top level")
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the packet-backed campaign launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_PACKET,
        help="Frozen issue #3637 launch-packet YAML.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for campaign JSONL files and report.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help="Path for campaign report JSON.",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable campaign report on stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the preflighted #3637 packet through the existing paired runner."""
    args = _parse_args(argv)
    packet = _load_packet(args.packet)
    plan = run_plan_from_packet(packet)
    preflight = build_preflight_manifest(plan)
    if preflight["status"] != "ready":
        print(json.dumps(preflight, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    report = run_campaign(
        Path(plan.scenario_set),
        list(plan.arm_seeds["reactive"]),
        tuple(plan.planners),
        args.out_dir,
        horizon=plan.horizon,
        dt=args.dt,
        workers=args.workers,
        study_issue=ISSUE,
        evidence_tier=EVIDENCE_TIER,
    )
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    report["launch_packet"] = str(args.packet)
    report["preflight_status"] = preflight["status"]
    report["post_run_analyzer"] = (
        "uv run python scripts/benchmark/"
        "analyze_reactivity_replay_rank_study_issue_3637.py "
        f"--packet {args.packet} --campaign-dir {args.out_dir} "
        f"--campaign-report {args.report_json} --output-dir {args.out_dir / 'analysis'}"
    )

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
