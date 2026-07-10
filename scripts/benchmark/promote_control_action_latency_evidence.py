#!/usr/bin/env python3
"""Promote raw control-action-latency sweep episode rows into durable evidence (#5034).

Reads raw fidelity-campaign episode rows (the JSONL the runner writes under
``output/``), isolates the ``control_action_latency`` axis, and promotes a
compact durable evidence bundle under ``docs/context/evidence/`` that reports the
action-latency metadata plus success / collision / minimum-clearance metrics for
each completed native latency cell and classifies every fallback / degraded /
non-native row as an exclusion rather than a result.

This command runs **no episode** and makes **no benchmark / simulator-realism /
sim-to-real / paper-facing claim**. It fails closed when the latency preflight is
not ready or when the native result rows do not cover the required action-latency
step set (0, 1, 3), so a partial or non-latency run cannot be promoted as the
latency sweep.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml

from robot_sf.benchmark.control_action_latency_evidence import (
    PROMOTION_SCHEMA_VERSION,
    LatencyEvidenceError,
    build_latency_evidence,
    load_latency_rows,
    write_latency_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/research/fidelity_sensitivity_v1.yaml"
DEFAULT_RAW_ROWS = "output/fidelity_latency_raw/episode_rows.jsonl"
DEFAULT_EVIDENCE_DIR = "docs/context/evidence/issue_5034_control_action_latency_sweep"


def _git_head() -> str:
    """Return the current git head, or ``unknown`` when unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _repo_rel(path: Path) -> str:
    """Return a repo-relative path string when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Fidelity-sensitivity study config (default: %(default)s).",
    )
    parser.add_argument(
        "--raw-rows",
        default=DEFAULT_RAW_ROWS,
        help="Raw episode rows JSONL emitted by the fidelity campaign runner.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=DEFAULT_EVIDENCE_DIR,
        help="Durable evidence output directory under docs/context/evidence/.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Classify and validate the raw rows without writing the evidence bundle; "
            "prints a compact JSON status and exits non-zero when promotion would fail."
        ),
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero (fail closed) when the rows cannot be promoted as the latency sweep.",
    )
    parser.add_argument("--date", default=dt.datetime.now(tz=dt.UTC).date().isoformat())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    raw_rows_path = Path(args.raw_rows)
    if not raw_rows_path.is_absolute():
        raw_rows_path = REPO_ROOT / raw_rows_path

    try:
        rows = load_latency_rows(raw_rows_path)
        packet = build_latency_evidence(
            rows,
            config=config,
            config_path=_repo_rel(config_path),
            git_head=_git_head(),
            date=str(args.date),
            raw_rows_path=_repo_rel(raw_rows_path),
        )
    except LatencyEvidenceError as exc:
        status = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "status": "blocked",
            "issue": 5034,
            "reason": str(exc),
        }
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1 if args.require_ready else 0

    if args.check_only:
        print(
            json.dumps(
                {
                    "schema_version": PROMOTION_SCHEMA_VERSION,
                    "status": "promotable",
                    "issue": 5034,
                    "result_row_count": packet["scope"]["result_row_count"],
                    "excluded_row_count": packet["scope"]["excluded_row_count"],
                    "latency_coverage": packet["latency_coverage"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.is_absolute():
        evidence_dir = REPO_ROOT / evidence_dir
    written = write_latency_evidence(packet, evidence_dir)
    result = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "promoted",
        "issue": 5034,
        "evidence_dir": _repo_rel(evidence_dir),
        "written_files": [_repo_rel(path) for path in written],
        "result_row_count": packet["scope"]["result_row_count"],
        "excluded_row_count": packet["scope"]["excluded_row_count"],
        "latency_coverage": packet["latency_coverage"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
