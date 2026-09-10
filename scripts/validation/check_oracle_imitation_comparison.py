"""Build and check the outcome-free #1496 BC-versus-RL comparison packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.training.oracle_imitation_comparison import (
    ComparisonArtifactBlockedError,
    ComparisonPacketError,
    build_comparison_packet,
    run_startup_canary,
)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the #1496 BC-warm-start versus RL-only packet without nominal "
            "dataset access or training."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Comparison packet YAML.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve relative references.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Authorized root for private/local shard verification.",
    )
    parser.add_argument(
        "--check-artifacts",
        action="store_true",
        help="Require every local/private shard and verify its size and SHA-256.",
    )
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Run the one-update synthetic/disjoint startup canary after validation.",
    )
    parser.add_argument(
        "--canary-output",
        type=Path,
        default=Path("output/smoke/issue_1496/comparison_canary"),
        help="Disposable output directory for the startup canary.",
    )
    parser.add_argument("--check", action="store_true", help="Validate the packet (default).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def _emit(payload: dict[str, Any], *, as_json: bool) -> None:
    """Emit one stable result envelope."""
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    status = payload.get("status", "unknown")
    print(f"oracle-imitation comparison packet: {status}")
    if payload.get("manifest_digest"):
        print(f"  manifest: {payload['manifest_digest']}")
    if payload.get("startup_canary"):
        print(f"  startup canary: {payload['startup_canary'].get('status', 'unknown')}")
    if payload.get("error"):
        print(f"  error: {payload['error']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Validate a comparison packet and optionally run its bounded canary."""
    args = _parser().parse_args(argv)
    try:
        packet = build_comparison_packet(
            args.config,
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
            require_artifacts=args.check_artifacts or args.canary,
        )
        if args.canary:
            dataset_manifest = packet["source"]["dataset_consumer_manifest"]
            canary = run_startup_canary(
                Path(dataset_manifest),
                args.canary_output,
                repo_root=args.repo_root,
                artifact_root=args.artifact_root,
            )
            packet["startup_canary"] = canary
        _emit(packet, as_json=args.json)
        return 0
    except ComparisonArtifactBlockedError as exc:
        payload = {"status": "blocked", "error": str(exc), "claim_boundary": "no training"}
        _emit(payload, as_json=args.json)
        return 1
    except ComparisonPacketError as exc:
        payload = {"status": "invalid", "error": str(exc), "claim_boundary": "no training"}
        _emit(payload, as_json=args.json)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
