#!/usr/bin/env python3
"""Derive the control-action-latency SNQI analysis from a durable input (#5912).

Reads a **durable sufficient input** (the promoted ``snqi_latency_inputs.csv``
carrying exactly the per-episode SNQI-v0 terms) or, when re-deriving from the
private artifact, the raw job 13516 ``episode_rows.jsonl`` (validated by its
registered SHA-256). Validates input checksum / fixed-scope coverage / execution
modes / fallback-degraded exclusions, then computes SNQI-v0 per episode, the
per-unit ordinary-least-squares latency slope, and the paired cluster-bootstrap
uncertainty, and emits ``snqi_analysis.json`` + ``snqi_by_latency.csv`` under the
``control-action-latency-snqi-analysis.v1`` schema.

This command runs **no episode** and makes **no benchmark / simulator-realism /
sim-to-real / paper-facing claim**. It fails closed when the input cannot be
analyzed as the registered latency-sweep SNQI packet.

Modes
-----
- (default) analyze: read ``--input`` (default the committed durable input) and
  write the SNQI analysis packet.
- ``--promote-input``: read ``--raw-rows`` (validated by SHA-256) and write the
  durable sufficient-input CSV plus its provenance sidecar; do not analyze.
- ``--verify-against <reference.json>``: after analyzing, compare the generated
  packet to the registered reference under the reproducibility tolerance contract
  and fail closed on any violation.
- ``--check-only``: classify and validate the input without writing artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from robot_sf.benchmark.control_action_latency_snqi import (
    ANALYSIS_SCHEMA_VERSION,
    SnqiLatencyAnalysisError,
    build_snqi_analysis,
    derive_inputs_from_raw_rows,
    load_input_provenance,
    load_input_rows,
    load_raw_rows,
    validate_input_checksum,
    verify_against_reference,
    write_input_provenance,
    write_input_rows,
    write_snqi_analysis,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_5034_control_action_latency_sweep"
DEFAULT_INPUT = EVIDENCE_DIR / "snqi_latency_inputs.csv"
DEFAULT_INPUT_PROVENANCE = EVIDENCE_DIR / "snqi_latency_inputs.csv.provenance.json"
DEFAULT_WEIGHTS = REPO_ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
DEFAULT_BASELINE = REPO_ROOT / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"
DEFAULT_REFERENCE = EVIDENCE_DIR / "snqi_analysis.json"


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


def _load_json(path: Path) -> dict:
    """Load a JSON object, failing closed on read/parse errors."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnqiLatencyAnalysisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SnqiLatencyAnalysisError(f"{path} must contain a JSON object")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Durable sufficient-input CSV (default: the committed job 13516 input).",
    )
    parser.add_argument(
        "--input-provenance",
        default=str(DEFAULT_INPUT_PROVENANCE),
        help="Provenance sidecar anchoring the durable input to the raw rows.",
    )
    parser.add_argument(
        "--raw-rows",
        default=None,
        help=(
            "Raw campaign episode rows JSONL. Used with --promote-input to (re)derive the "
            "durable input; its SHA-256 must match the registered job 13516 raw-row checksum."
        ),
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="SNQI-v0 weights config (default: the camera-ready v3 weights).",
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="SNQI-v0 normalization baseline config (default: the camera-ready v3 baseline).",
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(EVIDENCE_DIR),
        help="Output directory for the SNQI analysis artifacts.",
    )
    parser.add_argument(
        "--verify-against",
        default=None,
        help=(
            "Reference snqi_analysis.json to compare the generated packet against under the "
            "reproducibility tolerance contract. Defaults to the registered packet when present."
        ),
    )
    parser.add_argument(
        "--promote-input",
        action="store_true",
        help=(
            "Read --raw-rows (validated by SHA-256) and write the durable sufficient-input CSV "
            "plus its provenance sidecar; do not analyze."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Classify and validate the input without writing the analysis artifacts.",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Skip durable-input checksum verification (diagnostic use only).",
    )
    parser.add_argument(
        "--date",
        default=dt.datetime.now(tz=dt.UTC).date().isoformat(),
        help="ISO date string recorded in the generated packet provenance.",
    )
    return parser.parse_args(argv)


def _run_promote_input(args: argparse.Namespace) -> int:
    """Read raw rows, validate the SHA-256, and write the durable input + provenance."""
    if not args.raw_rows:
        raise SnqiLatencyAnalysisError("--promote-input requires --raw-rows")
    raw_rows_path = Path(args.raw_rows)
    if not raw_rows_path.is_absolute():
        raw_rows_path = REPO_ROOT / raw_rows_path
    rows = load_raw_rows(raw_rows_path)
    inputs = derive_inputs_from_raw_rows(rows)
    if not inputs:
        raise SnqiLatencyAnalysisError(
            f"raw rows {raw_rows_path} contain no '{_axis_label()}' axis rows"
        )
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    provenance_path = Path(args.input_provenance)
    if not provenance_path.is_absolute():
        provenance_path = REPO_ROOT / provenance_path
    write_input_rows(inputs, input_path)
    write_input_provenance(
        input_path,
        provenance_path,
        raw_rows_path=_repo_rel(raw_rows_path),
        promoter_git_head=_git_head(),
        date=str(args.date),
    )
    result = {
        "schema_version": "control-action-latency-snqi-inputs-promotion.v1",
        "status": "promoted",
        "input_path": _repo_rel(input_path),
        "input_provenance_path": _repo_rel(provenance_path),
        "input_row_count": len(inputs),
        "raw_rows_path": _repo_rel(raw_rows_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _axis_label() -> str:
    """Return the control-action-latency axis key (kept out of the hot import path)."""
    from robot_sf.benchmark.control_action_latency_preflight import AXIS_KEY

    return AXIS_KEY


def _load_inputs(args: argparse.Namespace) -> list:
    """Load and validate the durable sufficient input (or re-derive from raw rows)."""
    if args.raw_rows and not args.promote_input:
        raw_rows_path = Path(args.raw_rows)
        if not raw_rows_path.is_absolute():
            raw_rows_path = REPO_ROOT / raw_rows_path
        rows = load_raw_rows(raw_rows_path)
        return derive_inputs_from_raw_rows(rows)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not args.no_checksum:
        provenance_path = Path(args.input_provenance)
        if not provenance_path.is_absolute():
            provenance_path = REPO_ROOT / provenance_path
        provenance = load_input_provenance(provenance_path)
        validate_input_checksum(input_path, provenance)
    return load_input_rows(input_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        if args.promote_input:
            return _run_promote_input(args)

        inputs = _load_inputs(args)
        weights = _load_json(Path(args.weights))
        baseline = _load_json(Path(args.baseline))
        packet = build_snqi_analysis(
            inputs, weights=weights, baseline_stats=baseline, date=str(args.date)
        )
    except SnqiLatencyAnalysisError as exc:
        print(
            json.dumps(
                {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "status": "blocked",
                    "issue": 5912,
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    report: dict | None = None
    if args.verify_against:
        reference_path = Path(args.verify_against)
        if not reference_path.is_absolute():
            reference_path = REPO_ROOT / reference_path
    elif DEFAULT_REFERENCE.exists():
        reference_path = DEFAULT_REFERENCE
    else:
        reference_path = None
    if reference_path is not None and reference_path.exists():
        reference = _load_json(reference_path)
        try:
            report = verify_against_reference(packet, reference)
        except SnqiLatencyAnalysisError as exc:
            print(
                json.dumps(
                    {
                        "schema_version": ANALYSIS_SCHEMA_VERSION,
                        "status": "verification_failed",
                        "issue": 5912,
                        "reason": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

    if args.check_only:
        print(
            json.dumps(
                {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "status": "analyzable",
                    "issue": 5912,
                    "latency_row_count": packet["scope_verification"]["latency_episode_row_count"],
                    "scope_verification": packet["scope_verification"],
                    "verification": report,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.is_absolute():
        evidence_dir = REPO_ROOT / evidence_dir
    written = write_snqi_analysis(packet, evidence_dir)
    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "analyzed",
        "issue": 5912,
        "evidence_dir": _repo_rel(evidence_dir),
        "written_files": [_repo_rel(path) for path in written],
        "verification": report,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
