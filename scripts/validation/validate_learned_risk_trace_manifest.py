"""Validate durable learned-risk trace manifest before #1472 training.

Exit codes distinct so callers (and #1472 readiness) branch mechanically:

- ``0`` -- manifest is contract-complete: ``ready_for_training_handoff``.
- ``2`` -- manifest is structurally invalid and cannot be evaluated.
- ``3`` -- manifest is well-formed but durable inputs are unresolved:
  ``artifact_retrieval_blocked`` (fail-closed; never treated as training-ready).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.learned_risk_trace_manifest import (
    DECISION_BLOCKED,
    LearnedRiskTraceManifestError,
    build_trace_manifest_status_packet,
    validate_trace_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate durable learned-risk trace manifest (fail-closed)."
    )
    parser.add_argument("--config", required=True, type=Path, help="Trace-manifest YAML path.")
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON validation report.")
    parser.add_argument(
        "--status-json",
        action="store_true",
        help="Emit compact #1472 handoff status JSON instead of the full validation report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate trace manifest and return decision-coded exit status."""
    args = build_arg_parser().parse_args(argv)

    try:
        report = validate_trace_manifest(args.config, repo_root=args.repo_root)
    except LearnedRiskTraceManifestError as exc:
        if args.json or args.status_json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 2

    if args.status_json:
        status_packet = build_trace_manifest_status_packet(report, manifest_path=args.config)
        print(json.dumps(status_packet, indent=2, sort_keys=True))
    elif args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        decision = report["training_readiness_decision"]
        print(f"learned-risk trace manifest decision: {decision}")
        for blocker in report["blockers"]:
            print(f" - blocker: {blocker}")

    return 3 if report["training_readiness_decision"] == DECISION_BLOCKED else 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
