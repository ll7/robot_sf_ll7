#!/usr/bin/env python3
"""CPU-only preflight for the paper-grade reactivity-vs-replay rank study (issue #3637).

Reads a launch-packet YAML describing the *proposed* paper-grade run (planners, paired seeds,
scenario set, horizon, replay-limitation metadata) and checks the **plan-level preconditions**
before any compute is spent:

* >= 3 planners;
* exactly the reactive/replay arms with **paired** (identical) seeds — common random numbers;
* a seed budget at/above the rank-stability floor and above the #3573 diagnostic matrix;
* the replay limitation is stated and ``replay`` is force-off, not trajectory playback.

It is a thin orchestrator over the canonical pure checker
``robot_sf.benchmark.reactivity_replay_preflight``; it never re-implements the checks. It does
**not** run the benchmark, measure or interpret rank stability, submit Slurm/GPU jobs, or make any
paper-facing claim. Exit 0 = plan is ready to launch; exit 1 = plan is blocked (fix the packet);
exit 2 = usage/IO error.

Usage::

    preflight_reactivity_replay_rank_study_issue_3637.py \
        [--packet configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml] \
        [--output-json output/issue_3637/preflight.json] [--json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# Make the repo root importable when run as a bare script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.reactivity_replay_preflight import (  # noqa: E402
    build_preflight_manifest,
    run_plan_from_packet,
)

DEFAULT_PACKET = Path(
    "configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml"
)


def _git_head() -> str | None:
    """Read the current git HEAD SHA for provenance, or ``None`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_packet(path: Path) -> dict[str, Any]:
    """Load and minimally validate the launch-packet YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return payload


def _render_human(manifest: dict[str, Any]) -> str:
    """Render a human-readable summary of the preflight manifest."""
    lines = [
        f"reactivity-vs-replay preflight (issue #{manifest['issue']})",
        f"git HEAD: {manifest.get('provenance', {}).get('git_head') or 'unknown'}",
        f"packet: {manifest.get('provenance', {}).get('packet') or 'unknown'}",
        f"status: {manifest['status'].upper()}",
        "checks:",
    ]
    for check in manifest["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  [{mark}] {check['name']}: {check['detail']}")
    lines.append(f"replay limitation: {manifest['replay_limitation']['note']}")
    if manifest["blocking_issues"]:
        lines.append("blocking issues:")
        lines.extend(f"  - {issue}" for issue in manifest["blocking_issues"])
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_PACKET,
        help="Launch-packet YAML describing the proposed run plan.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the preflight manifest JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable JSON manifest to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the preflight. Returns 0 (ready), 1 (blocked), or 2 (usage/IO error)."""
    args = _parse_args(argv)
    try:
        packet = _load_packet(args.packet)
        plan = run_plan_from_packet(packet)
        manifest = build_preflight_manifest(plan)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        message = f"preflight failed for {args.packet}: {exc}"
        if args.json:
            print(json.dumps({"error": message}))
        else:
            print(f"error: {message}", file=sys.stderr)
        return 2

    manifest["provenance"] = {
        "git_head": _git_head(),
        "packet": str(args.packet),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(_render_human(manifest))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not args.json:
            print(f"\nwrote {args.output_json}")

    return 0 if manifest["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
