"""Command-line helper for the run-tracking telemetry subsystem.

The full command surface will be implemented incrementally as tasks from
`specs/001-performance-tracking/tasks.md` progress. For now, the CLI exposes the
shape of the interface and provides actionable guidance so contributors can test
parsers before wiring business logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from robot_sf.telemetry.config import RunTrackerConfig


@dataclass(slots=True)
class CommandContext:
    """Shared context passed into command handlers."""

    config: RunTrackerConfig
    run_id: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot SF run tracker CLI")
    parser.add_argument(
        "--artifact-root", type=Path, help="Override artifact root for tracker output"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_status_parser(subparsers)
    add_list_parser(subparsers)
    add_summary_parser(subparsers)
    add_watch_parser(subparsers)
    add_perf_parser(subparsers)
    add_tensorboard_parser(subparsers)
    return parser


def add_status_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("status", help="Show the latest state of a run")
    parser.add_argument("run_id", help="Run identifier (directory name under output/run-tracker)")


def add_list_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("list", help="List historical runs")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--status", choices=["pending", "running", "completed", "failed", "cancelled"]
    )
    parser.add_argument("--since", type=str, help="ISO timestamp filter (UTC)")


def add_summary_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "summary", help="Print summary of a run, including recommendations"
    )
    parser.add_argument("run_id")


def add_watch_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("watch", help="Tail manifest updates for a run")
    parser.add_argument("run_id")


def add_perf_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("perf-tests", help="Execute the telemetry performance wrapper")
    parser.add_argument("--scenario", help="Optional scenario config override")


def add_tensorboard_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "enable-tensorboard", help="Mirror metrics to TensorBoard logdir"
    )
    parser.add_argument("run_id")
    parser.add_argument("--logdir", type=Path, required=True)


def dispatch(context: CommandContext, args: argparse.Namespace) -> int:
    handlers = {
        "status": handle_placeholder,
        "list": handle_placeholder,
        "summary": handle_placeholder,
        "watch": handle_placeholder,
        "perf-tests": handle_placeholder,
        "enable-tensorboard": handle_placeholder,
    }
    handler = handlers[args.command]
    handler(context, args)
    return 0


def handle_placeholder(context: CommandContext, args: argparse.Namespace) -> None:
    run_hint = getattr(args, "run_id", context.run_id)
    message = (
        "Command not implemented yet. Refer to specs/001-performance-tracking/tasks.md "
        f"for the remaining implementation steps (run={run_hint!r})."
    )
    print(message)


def build_context(args: argparse.Namespace) -> CommandContext:
    base_root = args.artifact_root if args.artifact_root else None
    config = RunTrackerConfig(artifact_root=base_root)
    return CommandContext(config=config)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    context = build_context(args)
    return dispatch(context, args)


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    raise SystemExit(main())
