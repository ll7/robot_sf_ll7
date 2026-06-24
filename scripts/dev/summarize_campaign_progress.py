#!/usr/bin/env python3
"""Emit a bounded progress snapshot for long benchmark campaign output dirs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = "campaign_progress_summary.v1"
DEFAULT_FAILURE_LIMIT = 5
DEFAULT_TOP_ARTIFACTS = 8
SUMMARY_NAMES = ("summary.json", "campaign_summary.json")
REPORT_NAMES = ("report.md", "campaign_report.md")
STATE_FILE_NAME = ".campaign_progress_state.json"


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    """Compact file metadata for one campaign artifact."""

    path: str
    size_bytes: int
    mtime_epoch: float
    mtime_iso: str
    line_count: int | None
    size_delta_bytes: int | None
    line_delta: int | None


@dataclass(frozen=True, slots=True)
class CampaignProgress:
    """Bounded campaign progress snapshot."""

    schema: str
    output_dir: str
    exists: bool
    newest_artifact: ArtifactSummary | None
    summary_exists: bool
    summary_path: str | None
    report_exists: bool
    report_path: str | None
    completed_variants: list[str]
    completed_variant_count: int
    jsonl_artifact_count: int
    jsonl_line_count: int
    active_variant: str | None
    active_suite: str | None
    failure_markers: list[str]
    failure_markers_truncated: bool
    recommended_next_poll_seconds: int
    tracked_artifacts: list[ArtifactSummary]
    state_file: str | None


def _mtime_iso(mtime: float) -> str:
    """Format an mtime as a compact local timestamp."""

    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(mtime))


def _line_count(path: Path) -> int:
    """Count lines in a text artifact without loading it all at once."""

    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def _load_state(path: Path | None) -> dict[str, dict[str, int]]:
    """Load previous artifact sizes/line counts."""

    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for key, value in artifacts.items():
        if isinstance(value, dict):
            out[str(key)] = {
                "size_bytes": int(value.get("size_bytes", 0)),
                "line_count": int(value.get("line_count", 0)),
            }
    return out


def _write_state(path: Path | None, artifacts: list[ArtifactSummary]) -> None:
    """Persist tiny polling state for next-run deltas."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "artifacts": {
            item.path: {
                "size_bytes": item.size_bytes,
                "line_count": item.line_count if item.line_count is not None else 0,
            }
            for item in artifacts
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _variant_and_suite(path: Path) -> tuple[str | None, str | None]:
    """Infer suite and variant from campaign filenames like hard__baseline__ckpt.jsonl."""

    parts = path.stem.split("__")
    if len(parts) < 3:
        return None, None
    suite, variant = parts[0], parts[1]
    if not suite or not variant:
        return None, None
    return variant, suite


def _existing_named_file(output_dir: Path, names: tuple[str, ...]) -> Path | None:
    """Return first existing named file in *output_dir*."""

    for name in names:
        path = output_dir / name
        if path.exists():
            return path
    return None


def summarize_campaign_progress(  # noqa: C901
    output_dir: Path,
    *,
    state_file: Path | None = None,
    write_state: bool = True,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
    top_artifacts: int = DEFAULT_TOP_ARTIFACTS,
) -> CampaignProgress:
    """Build a compact progress summary for one campaign output directory."""

    output_dir = output_dir.resolve()
    if not output_dir.exists():
        return CampaignProgress(
            schema=SCHEMA_VERSION,
            output_dir=str(output_dir),
            exists=False,
            newest_artifact=None,
            summary_exists=False,
            summary_path=None,
            report_exists=False,
            report_path=None,
            completed_variants=[],
            completed_variant_count=0,
            jsonl_artifact_count=0,
            jsonl_line_count=0,
            active_variant=None,
            active_suite=None,
            failure_markers=[],
            failure_markers_truncated=False,
            recommended_next_poll_seconds=600,
            tracked_artifacts=[],
            state_file=str(state_file) if state_file is not None else None,
        )

    if state_file is None and write_state:
        state_file = output_dir / STATE_FILE_NAME
    previous = _load_state(state_file)

    files = sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.is_file() and (state_file is None or path.resolve() != state_file.resolve())
        ),
        key=lambda p: p.name,
    )
    artifacts: list[ArtifactSummary] = []
    jsonl_line_count = 0
    for path in files:
        stat = path.stat()
        line_count = _line_count(path) if path.suffix in {".jsonl", ".log"} else None
        if path.suffix == ".jsonl" and line_count is not None:
            jsonl_line_count += line_count
        old = previous.get(str(path))
        size_delta = stat.st_size - old["size_bytes"] if old else None
        line_delta = line_count - old["line_count"] if old and line_count is not None else None
        artifacts.append(
            ArtifactSummary(
                path=str(path),
                size_bytes=stat.st_size,
                mtime_epoch=stat.st_mtime,
                mtime_iso=_mtime_iso(stat.st_mtime),
                line_count=line_count,
                size_delta_bytes=size_delta,
                line_delta=line_delta,
            )
        )

    newest = max(artifacts, key=lambda item: item.mtime_epoch, default=None)
    newest_path = Path(newest.path) if newest is not None else None
    active_variant, active_suite = (
        _variant_and_suite(newest_path)
        if newest_path is not None and newest_path.suffix == ".jsonl"
        else (None, None)
    )

    jsonl_by_variant: dict[str, set[str]] = {}
    for path in files:
        if path.suffix != ".jsonl":
            continue
        variant, suite = _variant_and_suite(path)
        if variant is None or suite is None:
            continue
        jsonl_by_variant.setdefault(variant, set()).add(suite)
    completed_variants = sorted(
        variant for variant, suites in jsonl_by_variant.items() if {"hard", "global"} <= suites
    )

    summary_path = _existing_named_file(output_dir, SUMMARY_NAMES)
    report_path = _existing_named_file(output_dir, REPORT_NAMES)
    marker_paths = [
        path for path in files if "failure" in path.name.lower() or "error" in path.name.lower()
    ]
    failure_markers = [str(path) for path in marker_paths[:failure_limit]]

    recent_delta = newest.size_delta_bytes if newest is not None else None
    final_exists = summary_path is not None or report_path is not None
    if final_exists:
        next_poll = 0
    elif recent_delta is not None and recent_delta > 0:
        next_poll = 300
    else:
        next_poll = 600

    tracked = sorted(artifacts, key=lambda item: item.mtime_epoch, reverse=True)[:top_artifacts]
    if write_state:
        _write_state(state_file, artifacts)

    return CampaignProgress(
        schema=SCHEMA_VERSION,
        output_dir=str(output_dir),
        exists=True,
        newest_artifact=newest,
        summary_exists=summary_path is not None,
        summary_path=str(summary_path) if summary_path is not None else None,
        report_exists=report_path is not None,
        report_path=str(report_path) if report_path is not None else None,
        completed_variants=completed_variants,
        completed_variant_count=len(completed_variants),
        jsonl_artifact_count=sum(1 for path in files if path.suffix == ".jsonl"),
        jsonl_line_count=jsonl_line_count,
        active_variant=active_variant,
        active_suite=active_suite,
        failure_markers=failure_markers,
        failure_markers_truncated=len(marker_paths) > failure_limit,
        recommended_next_poll_seconds=next_poll,
        tracked_artifacts=tracked,
        state_file=str(state_file) if state_file is not None else None,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Campaign output directory.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional polling state path. Defaults to .campaign_progress_state.json in output dir.",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not read or write polling state; deltas will be null.",
    )
    parser.add_argument(
        "--failure-limit",
        type=int,
        default=DEFAULT_FAILURE_LIMIT,
        help="Maximum failure/error marker paths to print.",
    )
    parser.add_argument(
        "--top-artifacts",
        type=int,
        default=DEFAULT_TOP_ARTIFACTS,
        help="Maximum newest artifact metadata rows to include.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    if args.failure_limit < 0:
        raise SystemExit("--failure-limit must be non-negative")
    if args.top_artifacts < 0:
        raise SystemExit("--top-artifacts must be non-negative")
    summary = summarize_campaign_progress(
        args.output_dir,
        state_file=None if args.no_state else args.state_file,
        write_state=not args.no_state,
        failure_limit=args.failure_limit,
        top_artifacts=args.top_artifacts,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
