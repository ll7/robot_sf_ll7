"""Check oracle-imitation warm-start training prerequisites (issue #1496).

Read-only preflight: validates durable dataset launch packet (via canonical
validator), training-side config/contract files named in readiness manifest, and
prints compact readiness report plus explicit blockers.
It trains nothing, collects no data, submits no compute.

Examples:
    uv run python scripts/validation/check_oracle_imitation_warm_start_readiness.py \
        --manifest configs/training/ppo_imitation/oracle_warm_start_readiness_issue_1496.yaml \
        --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.training.oracle_imitation_warm_start_readiness import (
    PrerequisitesNotReadyError,
    WarmStartReadinessError,
    check_warm_start_readiness,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Preflight oracle-imitation warm-start training prerequisites.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Warm-start readiness manifest YAML path.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for emitted readiness decision manifest.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON readiness report.")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero when any prerequisite blocker remains (fail-closed gate).",
    )
    return parser


def _write_output_manifest(output_path: Path, report: dict[str, Any]) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": 1496,
        "schema": "oracle-imitation-warm-start-readiness-decision.v1",
        "report": report,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run readiness check and return shell-friendly exit code.

    Returns:
        * 0: ready
        * 1: blocked or ``--require-ready`` rejects blocked manifest
        * 2: manifest itself malformed
    """
    args = build_arg_parser().parse_args(argv)

    try:
        report = check_warm_start_readiness(
            args.manifest,
            repo_root=args.repo_root,
            require_ready=False,
        )
        if args.output:
            _write_output_manifest(args.output, report)

        if args.require_ready and report["status"] != "ready":
            raise PrerequisitesNotReadyError(
                "prerequisites not ready\n- " + "\n- ".join(report["blockers"])
            )

        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(
                f"oracle-imitation warm-start readiness: {report['status']} "
                f"({report['experiment_id']}, {len(report['blockers'])} blockers)"
            )
            for blocker in report["blockers"]:
                print(f" - {blocker}")

        return 0 if report["status"] == "ready" else 1

    except PrerequisitesNotReadyError as exc:
        # Keep a stable, explicit blocker manifest for machine review.
        if args.output:
            report = {
                "status": "blocked",
                "schema_version": "unknown",
                "experiment_id": "unknown",
                "prerequisites": {},
                "blockers": [str(exc)],
            }
            _write_output_manifest(args.output, report)

        if args.json:
            print(json.dumps({"status": "blocked", "error": str(exc)} , indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 1
    except WarmStartReadinessError as exc:
        if args.output:
            payload = {
                "status": "invalid",
                "error": str(exc),
            }
            _write_output_manifest(args.output, payload)

        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point guard
    raise SystemExit(main())
