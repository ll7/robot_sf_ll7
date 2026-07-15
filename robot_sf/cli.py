"""Top-level ``robot-sf`` command line interface.

Thin, user-facing entry point for everyday Robot SF workflows. Subcommands
wrap existing tooling behind one obvious command:

* ``robot-sf doctor`` — runtime diagnostics from :mod:`robot_sf.benchmark.doctor`
  with friendly, remedy-bearing output for beginners.
* ``robot-sf examples`` — discover and run examples from
  ``examples/examples_manifest.yaml`` (issue #5794).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.doctor import collect_doctor_report, doctor_exit_code
from robot_sf.examples_cli import examples_cli_main

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
    # The ``examples`` subcommand owns its own sub-subcommand parser
    # (``list``/``run``); it is registered here only so the top-level parser
    # recognises the token. Remaining args are forwarded by the handler.
    sub.add_parser(
        "examples",
        add_help=False,
        help="List and run examples from examples_manifest.yaml (issue #5794)",
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
    )
    if args.format == "json":
        import json  # noqa: PLC0415

        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        from robot_sf.benchmark.doctor import _format_human  # noqa: PLC0415

        sys.stdout.write(_format_human(report))
    return doctor_exit_code(report)


def _handle_examples(extra_args: Sequence[str]) -> int:
    """Forward to the examples discovery CLI.

    Args:
        extra_args: The arguments following the ``examples`` token.

    Returns:
        int: Process-style exit code from the examples CLI.
    """
    return examples_cli_main(list(extra_args))


_HANDLERS = {
    "doctor": _handle_doctor,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Top-level ``robot-sf`` entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        int: Process-style exit code.
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    # The ``examples`` subcommand owns its own sub-parser (``list``/``run``), so
    # forward everything after the token verbatim and avoid letting the
    # top-level parser consume example-specific options.
    if args_list and args_list[0] == "examples":
        return _handle_examples(args_list[1:])

    parser = _build_parser()
    args = parser.parse_args(args_list)
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
