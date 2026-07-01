"""Validate oracle-imitation dataset launch packets before Slurm collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.training.oracle_imitation_launch_packet import (
    LaunchPacketError,
    validate_launch_packet,
)

_DECISION_SCHEMA = "oracle-imitation-collection-readiness-decision.v1"
_ISSUE = 1496


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate pre-Slurm oracle-imitation dataset launch packet."
    )
    parser.add_argument("--config", required=True, type=Path, help="Launch-packet YAML path.")
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used resolve relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON validation report.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path emitted collection-readiness decision manifest.",
    )
    parser.add_argument(
        "--require-training-ready",
        action="store_true",
        help=(
            "Fail closed unless packet concrete durable train/validation/evaluation "
            "trace artifact URIs downstream imitation training."
        ),
    )
    return parser


def _first_launch_packet_error(exc: LaunchPacketError) -> str:
    """Return first actionable launch-packet error line."""
    for line in str(exc).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("oracle-imitation") and stripped.endswith("failed validation:"):
            continue
        return stripped.removeprefix("- ").strip()
    return str(exc)


def _decision_manifest_payload(
    *,
    config: Path,
    report: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build stable collection-readiness decision packet."""
    status = "ready" if report is not None else "blocked"
    blockers = [] if error is None else [error]
    return {
        "schema": _DECISION_SCHEMA,
        "issue": _ISSUE,
        "config": str(config),
        "status": status,
        "report": report if report is not None else None,
        "blockers": blockers,
        "forbidden_actions_confirmed": {
            "data_collection": False,
            "compute_submit": False,
            "training": False,
        },
    }


def _write_decision_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write decision packet JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Validate launch packet and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        report = validate_launch_packet(
            args.config,
            repo_root=args.repo_root,
            require_training_ready=args.require_training_ready,
        )
    except LaunchPacketError as exc:
        decision = _decision_manifest_payload(
            config=args.config,
            error=_first_launch_packet_error(exc),
        )
        if args.output:
            _write_decision_manifest(args.output, decision)
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 2

    if args.output:
        _write_decision_manifest(
            args.output,
            _decision_manifest_payload(config=args.config, report=report),
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "oracle-imitation launch packet valid: "
            f"{report['dataset_id']} ({report['episode_count']} planned episodes, "
            f"training_ready={report['training_ready']})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
