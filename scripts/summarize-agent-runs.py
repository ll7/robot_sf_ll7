#!/usr/bin/env python3
"""Summarize compact delegated-agent artifacts from the common Git directory."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    """Compact, log-free summary of one delegated run."""

    run_id: str
    provider: str
    model: str
    task_class: str
    status: str
    validation: str
    changed_files: int | None
    artifact_complete: bool
    path: Path


REQUIRED_ARTIFACTS = ("result.json", "RESULT.md", "diffstat.txt", "validation.json")
LEGACY_ARTIFACTS = ("result.json", "status.txt", "diffstat.txt", "changed_files.txt")


def common_git_dir() -> Path:
    """Resolve the repository common Git directory from any linked worktree."""
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def run_root() -> Path:
    """Return the common compact-artifact root."""
    return common_git_dir() / "codex-agent-runs"


def nonblank_count(path: Path) -> int | None:
    """Count nonblank lines without reading worker logs."""
    if not path.is_file():
        return None
    return sum(
        1
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    )


def read_json_object(path: Path) -> tuple[dict[str, Any], bool]:
    """Read one compact JSON object and report whether it is well-formed."""
    if not path.is_file():
        return {}, False
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    return (value, True) if isinstance(value, dict) else ({}, False)


def normalized_result_status(data: dict[str, Any]) -> str:
    """Normalize current and legacy worker completion fields."""
    value = data.get("status", data.get("worker_status", "unknown"))
    if value == 0 or str(value).lower() in {"success", "passed", "complete", "completed"}:
        return "complete"
    if str(value).lower() in {"failed", "failure", "error", "nonzero"}:
        return "failed"
    return str(value)


def validation_status(run_dir: Path, metrics: dict[str, Any], validation: dict[str, Any]) -> str:
    """Return explicit validation state or detect compact evidence markers."""
    value = metrics.get("validation_status")
    if isinstance(value, str) and value:
        return value
    value = validation.get("status")
    if isinstance(value, str) and value:
        lowered = value.lower()
        if "pass" in lowered:
            return "passed"
        if "fail" in lowered:
            return "failed"
        return value
    commands = validation.get("commands")
    if isinstance(commands, list):
        statuses = {
            str(item.get("result", item.get("status", ""))).lower()
            for item in commands
            if isinstance(item, dict)
        }
        if statuses & {"fail", "failed"}:
            return "failed"
        if statuses & {"pass", "passed"}:
            return "passed"
    pattern = re.compile(
        r"validation run|validated|pytest|unittest|ruff|bash -n|git diff --check", re.I
    )
    for name in ("RESULT.md", "status.txt"):
        path = run_dir / name
        if path.is_file() and pattern.search(path.read_text(encoding="utf-8", errors="replace")):
            return "evidence_present"
    return "not_run"


def changed_file_count(run_dir: Path, data: dict[str, Any]) -> int | None:
    """Read changed-file counts from canonical files or result metadata."""
    count = nonblank_count(run_dir / "changed_files.txt")
    if count is not None:
        return count
    for key in ("new_changed_files", "fix_forward_changed_files", "changed_files"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    for key in ("changed_file_count", "files_changed"):
        value = data.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def read_run(run_dir: Path) -> RunSummary:
    """Read canonical compact artifacts, retaining incomplete status."""
    data, result_valid = read_json_object(run_dir / "result.json")
    validation, validation_valid = read_json_object(run_dir / "validation.json")
    metrics: dict[str, Any] = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics, _ = read_json_object(metrics_path)
    canonical_complete = all((run_dir / name).is_file() for name in REQUIRED_ARTIFACTS)
    legacy_complete = all((run_dir / name).is_file() for name in LEGACY_ARTIFACTS)
    artifact_complete = result_valid and (
        (canonical_complete and validation_valid) or legacy_complete
    )
    status = normalized_result_status(data) if result_valid else "malformed_json"
    if not artifact_complete:
        status = "incomplete_artifact" if result_valid else "malformed_json"
    return RunSummary(
        run_id=str(metrics.get("run_id") or data.get("run_id") or run_dir.name),
        provider=str(metrics.get("provider") or data.get("provider") or "unknown"),
        model=str(metrics.get("model") or data.get("model") or "unknown"),
        task_class=str(
            metrics.get("task_class") or data.get("task_class") or data.get("mode") or "unknown"
        ),
        status=status,
        validation=validation_status(run_dir, metrics, validation),
        changed_files=changed_file_count(run_dir, data),
        artifact_complete=artifact_complete,
        path=run_dir,
    )


def collect_runs(root: Path, limit: int) -> list[RunSummary]:
    """Collect the newest compact runs, excluding control directories."""
    if not root.is_dir():
        return []
    candidates = [
        path
        for path in sorted(root.iterdir(), reverse=True)
        if path.is_dir()
        and path.name not in {"active", "notes"}
        and any((path / name).is_file() for name in (*REQUIRED_ARTIFACTS, *LEGACY_ARTIFACTS))
    ]
    return [read_run(path) for path in candidates[:limit]]


def note_fields(path: Path) -> dict[str, str]:
    """Read a small set of front-matter/body fields from one private note."""
    text = path.read_text(encoding="utf-8", errors="replace")
    fields: dict[str, str] = {}
    for field in ("observation_class", "confidence", "routing_signal", "candidate_lesson"):
        match = re.search(rf"(?im)^\s*-?\s*{re.escape(field)}\s*:\s*`?([^`\n]+)", text)
        if match:
            fields[field] = match.group(1).strip()
    return fields


def collect_notes(root: Path) -> list[tuple[Path, dict[str, str]]]:
    """Collect private inbox notes without traversing worker logs."""
    note_root = root / "notes" / "inbox"
    if not note_root.is_dir():
        return []
    return [(path, note_fields(path)) for path in sorted(note_root.glob("*.md"))]


def print_runs(runs: list[RunSummary]) -> None:
    """Print a compact run table."""
    print(f"Delegated runs: {len(runs)}")
    if not runs:
        return
    print("run_id\tstatus\tvalidation\tprovider\ttask\tchanged_files")
    for run in runs:
        print(
            f"{run.run_id}\t{run.status}\t{run.validation}\t{run.provider}\t"
            f"{run.task_class}\t{run.changed_files}"
        )


def print_notes(notes: list[tuple[Path, dict[str, str]]]) -> None:
    """Print a compact note summary."""
    print(f"Workflow notes: {len(notes)}")
    classes = Counter(fields.get("observation_class", "unknown") for _, fields in notes)
    for name, count in sorted(classes.items()):
        print(f"observation_class={name}\tcount={count}")


def print_routing_feedback(notes: list[tuple[Path, dict[str, str]]]) -> None:
    """Summarize routing signals without changing routing configuration."""
    signals = Counter(fields.get("routing_signal", "unknown") for _, fields in notes)
    print(f"Routing feedback notes: {len(notes)}")
    for name, count in sorted(signals.items()):
        print(f"routing_signal={name}\tcount={count}")


def print_reliability(runs: list[RunSummary]) -> None:
    """Print conservative route counts from metrics-backed compact runs."""
    route_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for run in runs:
        if not run.artifact_complete:
            continue
        route_stats[f"{run.provider}/{run.model}"]["total"] += 1
        if run.validation in {"passed", "evidence_present"} and run.status == "complete":
            route_stats[f"{run.provider}/{run.model}"]["passed"] += 1
    print(f"Route reliability from {len(runs)} recent runs:")
    for route, counts in sorted(route_stats.items()):
        print(f"route={route}\ttotal={counts['total']}\tvalidated={counts['passed']}")


def parse_args() -> argparse.Namespace:
    """Parse the documented summary modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--include-notes", action="store_true")
    parser.add_argument("--notes-only", action="store_true")
    parser.add_argument("--json-metrics", action="store_true")
    parser.add_argument("--route-reliability", action="store_true")
    parser.add_argument("--routing-feedback", action="store_true")
    parser.add_argument("--reliability-suggestions", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the compact summary command."""
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    try:
        root = run_root()
    except subprocess.CalledProcessError as exc:
        print(f"Unable to resolve repository common Git directory: {exc}")
        return 1
    runs = collect_runs(root, args.limit)
    notes = (
        collect_notes(root)
        if (
            args.include_notes
            or args.notes_only
            or args.routing_feedback
            or args.reliability_suggestions
        )
        else []
    )
    if args.json_metrics:
        print(
            json.dumps(
                [run.__dict__ | {"path": str(run.path)} for run in runs],
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    elif not args.notes_only:
        print_runs(runs)
    if args.include_notes or args.notes_only:
        print_notes(notes)
    if args.routing_feedback:
        print_routing_feedback(notes)
    if args.route_reliability or args.reliability_suggestions:
        print_reliability(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
