"""Top-level ``robot-sf`` command line interface.

The command exposes the existing runtime doctor and the one-command visual
demo from the adoption/UX epic. More everyday workflows can be added here
without creating another top-level entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.doctor import collect_doctor_report, doctor_exit_code

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``robot-sf`` argument parser.

    Returns:
        argparse.ArgumentParser: Parser with the registered subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="robot-sf",
        description="Robot SF top-level command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser(
        "doctor",
        help="Environment/readiness check with friendly remedies.",
    )
    doctor.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
    doctor.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("output"),
        help="Artifact root to probe for temporary write access (default: output).",
    )
    doctor.add_argument(
        "--skip-env-smoke",
        action="store_true",
        help="Skip the minimal reset/step environment smoke check.",
    )

    demo = subparsers.add_parser(
        "demo",
        help="Run the one-command visual demo (tiny deterministic episode + viewer).",
    )
    demo.add_argument("--output-root", type=Path, default=None)
    demo.add_argument("--scenario", type=Path, default=None)
    demo.add_argument("--seed", type=int, default=None)
    demo.add_argument("--verbose", action="store_true")

    return parser


def _handle_doctor(args: argparse.Namespace) -> int:
    """Run the doctor check and print the report."""
    report = collect_doctor_report(
        artifact_root=args.artifact_root,
        run_env_smoke=not args.skip_env_smoke,
    )
    if args.format == "json":
        import json  # noqa: PLC0415

        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        from robot_sf.benchmark.doctor import _format_human  # noqa: PLC0415

        sys.stdout.write(_format_human(report))
    return doctor_exit_code(report)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to the requested ``robot-sf`` subcommand.

    Returns:
        int: Process exit status code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _handle_doctor(args)

    if args.command == "demo":
        from scripts.demo.quickstart_demo import main as demo_main  # noqa: PLC0415

        demo_argv = []
        if args.output_root is not None:
            demo_argv += ["--output-root", str(args.output_root)]
        if args.scenario is not None:
            demo_argv += ["--scenario", str(args.scenario)]
        if args.seed is not None:
            demo_argv += ["--seed", str(args.seed)]
        if args.verbose:
            demo_argv.append("--verbose")
        return demo_main(demo_argv)

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
