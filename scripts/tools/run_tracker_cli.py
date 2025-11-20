"""Command-line helper for the run-tracking telemetry subsystem."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.telemetry import RunHistoryEntry, TensorBoardAdapter, list_runs, load_run
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
    add_export_parser(subparsers)
    return parser


def add_status_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("status", help="Show the latest state of a run")
    parser.add_argument(
        "run_id",
        help="Run identifier or path (directory name under output/run-tracker)",
    )


def add_list_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("list", help="List historical runs")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--status", choices=["pending", "running", "completed", "failed", "cancelled"]
    )
    parser.add_argument("--since", type=str, help="ISO timestamp filter (UTC)")
    parser.add_argument("--scenario", type=str, help="Filter by scenario identifier")
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format for the run list (default: table)",
    )


def add_summary_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "summary",
        aliases=["show"],
        help="Print summary of a run, including recommendations",
    )
    parser.add_argument("run_id")
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Choose between text, JSON, or Markdown output",
    )


def add_watch_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("watch", help="Tail manifest updates for a run")
    parser.add_argument("run_id")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)",
    )


def add_perf_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("perf-tests", help="Execute the telemetry performance wrapper")
    parser.add_argument("--scenario", help="Optional scenario config override")
    parser.add_argument(
        "--output",
        type=str,
        help="Optional run identifier or path for perf-test tracker artifacts",
    )
    parser.add_argument(
        "--num-resets",
        type=int,
        default=5,
        help="Number of environment resets to benchmark (default: 5)",
    )


def add_tensorboard_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "enable-tensorboard", help="Mirror metrics to TensorBoard logdir"
    )
    parser.add_argument("run_id")
    parser.add_argument("--logdir", type=Path, required=True)


def add_export_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("export", help="Export a run summary to a file")
    parser.add_argument("run_id")
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Export format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination file for the exported summary",
    )


def dispatch(context: CommandContext, args: argparse.Namespace) -> int:
    handlers = {
        "status": handle_status,
        "list": handle_list,
        "summary": handle_summary,
        "show": handle_summary,
        "watch": handle_watch,
        "perf-tests": handle_perf_tests,
        "enable-tensorboard": handle_enable_tensorboard,
        "export": handle_export,
    }
    handler = handlers[args.command]
    return handler(context, args)


def handle_placeholder(context: CommandContext, args: argparse.Namespace) -> int:
    run_hint = getattr(args, "run_id", context.run_id)
    message = (
        "Command not implemented yet. Refer to specs/001-performance-tracking/tasks.md "
        f"for the remaining implementation steps (run={run_hint!r})."
    )
    print(message)
    return 0


def handle_perf_tests(context: CommandContext, args: argparse.Namespace) -> int:
    try:
        from scripts.telemetry.run_perf_tests import run_perf_tests
    except ModuleNotFoundError as exc:  # pragma: no cover - dev only
        print(f"Performance wrapper unavailable: {exc}")
        return 2

    try:
        exit_code, run_dir = run_perf_tests(
            scenario=args.scenario,
            output_hint=args.output,
            num_resets=args.num_resets,
            artifact_root=context.config.artifact_root,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Performance test failed: {exc}")
        return 2
    if run_dir is not None:
        print(f"Perf test artifacts written to {run_dir}")
    return exit_code


def handle_enable_tensorboard(context: CommandContext, args: argparse.Namespace) -> int:
    try:
        run_dir = _resolve_run_directory(context.config, args.run_id)
    except FileNotFoundError as exc:
        print(f"Tracker assets not found: {exc}")
        return 1
    telemetry_path = run_dir / context.config.telemetry_filename
    if not telemetry_path.is_file():
        print(f"Telemetry log not found: {telemetry_path}")
        return 1
    adapter = TensorBoardAdapter(log_dir=args.logdir)
    if not adapter.is_available:
        print(
            "TensorBoard SummaryWriter is unavailable. Install torch or tensorboardX to enable this command.",
        )
        return 2
    try:
        count = adapter.mirror_file(telemetry_path)
    except RuntimeError as exc:
        print(f"Unable to mirror telemetry to TensorBoard: {exc}")
        return 2
    print(f"Mirrored {count} telemetry samples to {adapter.log_dir}")
    return 0


def handle_status(context: CommandContext, args: argparse.Namespace) -> int:
    try:
        run_dir = _resolve_run_directory(context.config, args.run_id)
        steps = _load_step_entries(context.config, run_dir)
    except FileNotFoundError as exc:
        print(f"Tracker assets not found: {exc}")
        return 1
    summary = _summarize_steps(steps)
    _print_status(run_dir, summary)
    return 0


def handle_list(context: CommandContext, args: argparse.Namespace) -> int:
    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(str(exc))
        return 2
    entries = list_runs(
        context.config,
        limit=args.limit,
        status=args.status,
        since=since,
        scenario=getattr(args, "scenario", None),
    )
    if args.format == "json":
        payload = [entry.to_dict() for entry in entries]
        print(json.dumps(payload, indent=2))
        return 0
    _print_run_table(entries)
    return 0


def handle_summary(context: CommandContext, args: argparse.Namespace) -> int:
    try:
        entry = load_run(context.config, args.run_id)
    except FileNotFoundError as exc:
        print(f"Tracker assets not found: {exc}")
        return 1
    if args.format == "json":
        print(json.dumps(entry.to_dict(), indent=2))
        return 0
    if args.format == "markdown":
        print(_render_markdown(entry))
        return 0
    _print_run_summary(entry)
    return 0


def handle_export(context: CommandContext, args: argparse.Namespace) -> int:
    try:
        entry = load_run(context.config, args.run_id)
    except FileNotFoundError as exc:
        print(f"Tracker assets not found: {exc}")
        return 1
    if args.format == "json":
        content = json.dumps(entry.to_dict(), indent=2)
    else:
        content = _render_markdown(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Exported run summary to {args.output}")
    return 0


def handle_watch(context: CommandContext, args: argparse.Namespace) -> int:
    interval = max(args.interval, 0.5)
    try:
        while True:
            result = handle_status(context, args)
            if result != 0:
                return result
            print("-" * 60)
            time.sleep(interval)
    except KeyboardInterrupt:  # pragma: no cover - interactive command
        return 0


def build_context(args: argparse.Namespace) -> CommandContext:
    base_root = args.artifact_root if args.artifact_root else None
    config = RunTrackerConfig(artifact_root=base_root)
    return CommandContext(config=config)


def _resolve_run_directory(config: RunTrackerConfig, run_hint: str) -> Path:
    candidate = Path(run_hint).expanduser()
    if candidate.is_dir():
        return candidate
    tracker_root = config.run_tracker_root
    run_dir = tracker_root / run_hint
    if run_dir.is_dir():
        return run_dir
    raise FileNotFoundError(run_dir)


def _load_step_entries(config: RunTrackerConfig, run_dir: Path) -> list[dict[str, Any]]:
    steps_path = run_dir / config.steps_filename
    if not steps_path.is_file():
        raise FileNotFoundError(steps_path)
    data = json.loads(steps_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):  # pragma: no cover - defensive guard
        raise ValueError(f"Unexpected step index format in {steps_path}")
    return data


def _summarize_steps(entries: list[dict[str, Any]]) -> dict[str, Any]:
    completed = 0
    current = None
    last_completed = None
    for entry in entries:
        status = entry.get("status")
        if status == "running" and current is None:
            current = entry
        if status == "completed":
            completed += 1
            last_completed = entry
    return {
        "total": len(entries),
        "completed": completed,
        "current": current,
        "last_completed": last_completed,
        "entries": entries,
    }


def _print_status(run_dir: Path, summary: dict[str, Any]) -> None:
    total = summary["total"]
    completed = summary["completed"]
    current = summary["current"]
    print(f"Run directory: {run_dir}")
    print(f"Steps: {completed}/{total}")
    if current:
        order = current.get("order")
        name = current.get("display_name", current.get("step_id"))
        elapsed = _format_seconds(_current_elapsed_seconds(current))
        eta = _format_seconds(current.get("eta_snapshot_seconds"))
        print(f"Current: Step {order}/{total} – {name} (elapsed={elapsed}, eta={eta})")
    elif summary["last_completed"]:
        last = summary["last_completed"]
        name = last.get("display_name", last.get("step_id"))
        duration = _format_seconds(last.get("duration_seconds"))
        print(f"Last completed: Step {last.get('order')} – {name} (duration={duration})")
    else:
        print("No steps started yet.")

    print("Step breakdown:")
    for entry in summary["entries"]:
        status = entry.get("status", "unknown")
        name = entry.get("display_name", entry.get("step_id"))
        duration = _format_seconds(entry.get("duration_seconds"))
        if status == "running":
            duration = _format_seconds(_current_elapsed_seconds(entry))
        print(
            f"  - [{status.upper():9}] Step {entry.get('order'):>2}: {name} (duration={duration})"
        )


def _print_run_table(entries: list[RunHistoryEntry]) -> None:
    if not entries:
        print("No run tracker entries found.")
        return
    header = f"{'Run ID':<24} {'Status':<10} {'Started':<20} {'Completed':<20} Steps"
    print(header)
    print("-" * len(header))
    for entry in entries:
        created = entry.created_at.isoformat(timespec="seconds") if entry.created_at else "--"
        completed = entry.completed_at.isoformat(timespec="seconds") if entry.completed_at else "--"
        steps_total = len(entry.steps)
        steps_done = sum(1 for step in entry.steps if step.get("status") == "completed")
        print(
            f"{entry.run_id:<24} {entry.status.value:<10} {created:<20} "
            f"{completed:<20} {steps_done}/{steps_total}"
        )


def _print_run_summary(entry: RunHistoryEntry) -> None:
    for line in _iter_run_summary_lines(entry):
        print(line)


def _iter_run_summary_lines(entry: RunHistoryEntry) -> list[str]:
    header = _format_run_header(entry)
    summary_section = _format_summary_section(entry.summary)
    steps_section = _format_steps_section(entry.steps)
    rec_section = _format_recommendation_section(entry.recommendations)
    perf_section = _format_perf_section(entry.perf_tests)
    return header + summary_section + steps_section + rec_section + perf_section


def _format_run_header(entry: RunHistoryEntry) -> list[str]:
    created = entry.created_at.isoformat(timespec="seconds") if entry.created_at else "--"
    completed = entry.completed_at.isoformat(timespec="seconds") if entry.completed_at else "--"
    return [
        f"Run: {entry.run_id}",
        f"Status: {entry.status.value}",
        f"Started: {created}",
        f"Completed: {completed}",
        f"Artifact dir: {entry.artifact_dir}",
    ]


def _format_summary_section(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["Summary: (none)"]
    lines = ["Summary:"]
    if isinstance(summary, dict):
        for key, value in summary.items():
            if key == "telemetry" and isinstance(value, dict):
                lines.extend(_format_telemetry_lines(value))
                continue
            lines.append(f"  - {key}: {value}")
    return lines


def _format_telemetry_lines(values: dict[str, Any]) -> list[str]:
    lines = ["  - Telemetry:"]
    for key, value in values.items():
        lines.append(f"      * {key}: {_format_summary_value(value)}")
    return lines


def _format_steps_section(steps: list[dict[str, Any]]) -> list[str]:
    lines = ["Steps:"]
    for step in steps:
        status = step.get("status", "unknown").upper()
        name = step.get("display_name", step.get("step_id"))
        order = step.get("order")
        duration = _format_seconds(step.get("duration_seconds"))
        lines.append(f"  - [{status:9}] Step {order}: {name} (duration={duration})")
    return lines


def _format_recommendation_section(recommendations: tuple[dict[str, Any], ...]) -> list[str]:
    if not recommendations:
        return []
    lines = ["Recommendations:"]
    for rec in recommendations:
        severity = str(rec.get("severity", "")).upper() or "INFO"
        message = rec.get("message", "(no message)")
        lines.append(f"  - [{severity}] {message}")
        for action in rec.get("suggested_actions", ()):
            lines.append(f"      * {action}")
    return lines


def _format_perf_section(perf_tests: tuple[dict[str, Any], ...]) -> list[str]:
    if not perf_tests:
        return []
    lines = ["Performance tests:"]
    for test in perf_tests:
        test_id = test.get("test_id", "unknown")
        status = test.get("status", "n/a")
        throughput = _format_numeric(test.get("throughput_measured"))
        baseline = _format_numeric(test.get("throughput_baseline"))
        lines.append(f"  - {test_id} ({status}) throughput={throughput} baseline={baseline}")
    return lines


def _format_numeric(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:.2f}"
    return str(value)


def _render_markdown(entry: RunHistoryEntry) -> str:
    lines: list[str] = []
    lines.extend(_markdown_header_lines(entry))
    lines.extend(_markdown_summary_sections(entry))
    lines.extend(_markdown_steps_section(entry.steps))
    lines.extend(_markdown_recommendations_section(entry.recommendations))
    lines.extend(_markdown_perf_section(entry.perf_tests))
    return "\n".join(lines)


def _markdown_header_lines(entry: RunHistoryEntry) -> list[str]:
    created = entry.created_at.isoformat(timespec="seconds") if entry.created_at else "--"
    completed = entry.completed_at.isoformat(timespec="seconds") if entry.completed_at else "--"
    return [
        f"# Run {entry.run_id}",
        "",
        f"- **Status:** {entry.status.value}",
        f"- **Started:** {created}",
        f"- **Completed:** {completed}",
        f"- **Artifact Dir:** `{entry.artifact_dir}`",
    ]


def _markdown_summary_sections(entry: RunHistoryEntry) -> list[str]:
    lines = ["", "## Summary"]
    summary_lines = _markdown_summary_lines(entry.summary)
    lines.extend(summary_lines or ["- _(none)_"])
    telemetry = entry.summary.get("telemetry") if isinstance(entry.summary, dict) else None
    if isinstance(telemetry, dict) and telemetry:
        lines.append("")
        lines.append("## Telemetry")
        lines.extend(_markdown_telemetry_lines(telemetry))
    return lines


def _markdown_summary_lines(summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(summary, dict):
        return []
    lines: list[str] = []
    for key, value in summary.items():
        if key == "telemetry":
            continue
        lines.append(f"- **{key}:** {value}")
    return lines


def _markdown_telemetry_lines(values: dict[str, Any]) -> list[str]:
    return [f"- **{key}:** {_format_summary_value(value)}" for key, value in values.items()]


def _markdown_steps_section(steps: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Steps"]
    for step in steps:
        status = step.get("status", "unknown").upper()
        name = step.get("display_name", step.get("step_id"))
        duration = _format_seconds(step.get("duration_seconds"))
        lines.append(f"- [{status}] Step {step.get('order')}: {name} (duration={duration})")
    return lines


def _markdown_recommendations_section(recommendations: tuple[dict[str, Any], ...]) -> list[str]:
    if not recommendations:
        return []
    lines = ["", "## Recommendations"]
    for rec in recommendations:
        severity = str(rec.get("severity", "")).upper() or "INFO"
        message = rec.get("message", "(no message)")
        lines.append(f"- **{severity}:** {message}")
        actions = rec.get("suggested_actions") or []
        for action in actions:
            lines.append(f"  - {action}")
    return lines


def _markdown_perf_section(perf_tests: tuple[dict[str, Any], ...]) -> list[str]:
    if not perf_tests:
        return []
    lines = ["", "## Performance Tests"]
    for test in perf_tests:
        test_id = test.get("test_id", "unknown")
        status = test.get("status", "n/a")
        throughput = _format_summary_value(test.get("throughput_measured"))
        baseline = _format_summary_value(test.get("throughput_baseline"))
        lines.append(
            f"- **{test_id}:** status={status}, throughput={throughput}, baseline={baseline}"
        )
    return lines


def _current_elapsed_seconds(entry: dict[str, Any]) -> float | None:
    started_at = entry.get("started_at")
    if not started_at:
        return None
    start_dt = _parse_timestamp(started_at)
    if start_dt is None:
        return None
    return max(0.0, (datetime.now(UTC) - start_dt).total_seconds())


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:  # pragma: no cover - defensive
        return None


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "--"
    seconds = int(max(value, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _format_summary_value(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            "Invalid ISO timestamp for --since; expected ISO 8601 string with optional timezone"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    context = build_context(args)
    return dispatch(context, args)


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    raise SystemExit(main())
