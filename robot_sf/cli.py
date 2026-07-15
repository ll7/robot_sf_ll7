"""Top-level ``robot-sf`` command line interface.

Thin, user-facing entry point for everyday Robot SF workflows. The
``doctor`` subcommand wraps the existing runtime diagnostics in
:mod:`robot_sf.benchmark.doctor` behind one obvious command, adding
friendly, remedy-bearing output for beginners.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.doctor import collect_doctor_report, doctor_exit_code

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with its subcommands.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="robot-sf",
        description="Robot SF top-level command line interface.",
    )
    sub = parser.add_subparsers(dest="cmd")
    doc = sub.add_parser(
        "doctor",
        help="Environment/readiness check with friendly remedies (uv run robot-sf doctor)",
    )
    doc.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
    doc.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("output"),
        help="Artifact root to probe for temporary write access (default: output)",
    )
    doc.add_argument(
        "--skip-env-smoke",
        action="store_true",
        default=False,
        help="Skip the minimal reset/step environment smoke check",
    )
    doc.add_argument(
        "--skip-quickstart-smoke",
        action="store_true",
        default=False,
        help="Skip executing the manifest-declared quickstart examples",
    )
    return parser


def _handle_doctor(args: argparse.Namespace) -> int:
    """Run the doctor check and print the report.

    Returns:
        int: Doctor command exit code.
    """
    report = collect_doctor_report(
        artifact_root=args.artifact_root,
        run_env_smoke=not args.skip_env_smoke,
        run_quickstart_smoke=not args.skip_quickstart_smoke,
    )
    if args.format == "json":
        import json  # noqa: PLC0415

        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        from robot_sf.benchmark.doctor import _format_human  # noqa: PLC0415

        sys.stdout.write(_format_human(report))
    return doctor_exit_code(report)


_HANDLERS = {
    "doctor": _handle_doctor,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Top-level ``robot-sf`` entry point.

    Returns:
        int: Process-style exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 1
    handler = _HANDLERS.get(args.cmd)
    if handler is None:  # pragma: no cover - defensive
        parser.error(f"unknown command: {args.cmd}")
        return 2
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    raise SystemExit(main())
