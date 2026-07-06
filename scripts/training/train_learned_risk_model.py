"""Learned-risk model v1 trainer entrypoint (issue #4617, parent #1472).

This is the trainer entrypoint that draft PR #4552's launch-packet
``slurm_execution`` block cites as a must-exist path. It composes the core
training logic in :mod:`robot_sf.training.learned_risk_trainer`.

Two modes:

- ``--smoke`` (CPU smoke): fabricates a tiny synthetic trace fixture (or reads a
  ``--trace-fixture`` JSONL), fits one dependency-light logistic head per label,
  writes a status artifact with smoke diagnostics, and exits ``0``. No private
  artifacts, no network, no Slurm.
- real mode (default, no ``--smoke``): fail-closed. It validates the launch
  packet and the campaign-readiness gate and, while the durable trace manifest
  (issue #2312 / #4586) stays unresolved, refuses to train, writes a
  ``blocked_trace_manifest`` status, and exits ``3``. It never submits Slurm,
  fetches artifacts, publishes checkpoints, or promotes a learned-risk claim.

Exit codes:

- ``0`` -- smoke completed (or, in principle, a real launch-ready handoff).
- ``2`` -- config/contract error (nothing to train on).
- ``3`` -- real mode blocked by the fail-closed campaign gate.

Usage::

    uv run python scripts/training/train_learned_risk_model.py \\
        --config configs/training/learned_risk_model_v1.yaml \\
        --smoke \\
        --output-root output/issue_4617_learned_risk_smoke \\
        --status-out output/issue_4617_learned_risk_smoke/status.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.learned_risk_trainer import (
    STATE_BLOCKED_TRACE_MANIFEST,
    STATE_SMOKE_COMPLETED,
    LearnedRiskTrainerError,
    build_status,
    evaluate_real_mode_readiness,
    is_launch_ready,
    load_jsonl_records,
    load_training_config,
    synthesize_smoke_records,
    train_smoke,
    validate_launch_packet_or_raise,
    write_status,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the learned-risk model v1 (CPU smoke) or fail closed in real "
            "mode until durable trace inputs resolve. Never submits Slurm."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Training YAML config path.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a CPU smoke on synthetic (or --trace-fixture) traces; no Slurm, no real data.",
    )
    parser.add_argument(
        "--trace-fixture",
        type=Path,
        default=None,
        help="Optional JSONL trace fixture for smoke mode (defaults to synthetic rows).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the config output root (e.g. a transient smoke output/ dir).",
    )
    parser.add_argument(
        "--status-out",
        type=Path,
        default=None,
        help="Override the config status-artifact path.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used to resolve relative contract paths.",
    )
    return parser


def _resolve_paths(args: argparse.Namespace, config: dict) -> tuple[str, Path]:
    """Resolve the effective output root and status-out path (CLI over config).

    Returns:
        ``(output_root_text, status_out_path)``. ``output_root_text`` is recorded
        verbatim in the status artifact; ``status_out_path`` is where the status
        JSON is written.
    """
    output_root = str(args.output_root) if args.output_root else str(config["output_root"])
    if args.status_out is not None:
        status_out = args.status_out
    elif args.output_root is not None:
        status_out = args.output_root / "status.json"
    else:
        status_out = Path(config["status_artifact_path"])
    return output_root, status_out


def main(argv: list[str] | None = None) -> int:
    """Run the trainer entrypoint and return a decision-coded exit status."""
    args = build_arg_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()

    try:
        config = load_training_config(args.config)
        validate_launch_packet_or_raise(config, repo_root)
    except LearnedRiskTrainerError as exc:
        print(f"learned-risk trainer config error: {exc}")
        return 2

    output_root, status_out = _resolve_paths(args, config)

    if args.smoke:
        return _run_smoke(args, config, output_root, status_out)
    return _run_real(config, args.config, repo_root, output_root, status_out)


def _run_smoke(
    args: argparse.Namespace,
    config: dict,
    output_root: str,
    status_out: Path,
) -> int:
    """Execute the CPU smoke path and write a ``smoke_completed`` status."""
    try:
        if args.trace_fixture is not None:
            records = load_jsonl_records(args.trace_fixture)
        else:
            records = synthesize_smoke_records(config)
        result = train_smoke(config, records)
    except LearnedRiskTrainerError as exc:
        print(f"learned-risk trainer smoke error: {exc}")
        return 2

    status = build_status(
        config=config,
        config_path=args.config,
        mode="smoke",
        training_state=STATE_SMOKE_COMPLETED,
        output_root=output_root,
        status_artifact_path=str(status_out),
        row_count=result["row_count"],
        diagnostics=result["diagnostics"],
    )
    write_status(status, status_out)
    print(f"learned-risk smoke completed: {result['row_count']} rows -> {status_out}")
    return 0


def _run_real(
    config: dict,
    config_path: Path,
    repo_root: Path,
    output_root: str,
    status_out: Path,
) -> int:
    """Execute the fail-closed real-mode path.

    Real mode never trains while the campaign gate is blocked. It writes a
    ``blocked_trace_manifest`` status carrying the campaign blockers and exits
    ``3``. A launch-ready gate (not reachable while the trace manifest is pending)
    still stops short of full training here -- durable-trace training belongs to
    the gated Slurm run, not this CPU entrypoint.
    """
    readiness = evaluate_real_mode_readiness(config, repo_root)
    blockers = list(readiness.get("blockers", []))
    if not is_launch_ready(readiness):
        status = build_status(
            config=config,
            config_path=config_path,
            mode="real",
            training_state=STATE_BLOCKED_TRACE_MANIFEST,
            output_root=output_root,
            status_artifact_path=str(status_out),
            blockers=blockers,
        )
        write_status(status, status_out)
        print("learned-risk real mode blocked: campaign not launch ready (fail-closed).")
        for blocker in blockers:
            print(f"  blocker: {blocker}")
        print(f"  status -> {status_out}")
        return 3

    # Launch-ready but durable-trace training is intentionally out of scope for
    # this CPU entrypoint; keep fail-closed rather than imply a completed run.
    status = build_status(
        config=config,
        config_path=config_path,
        mode="real",
        training_state=STATE_BLOCKED_TRACE_MANIFEST,
        output_root=output_root,
        status_artifact_path=str(status_out),
        blockers=[
            "campaign launch-ready but durable-trace training runs on Slurm, not this entrypoint"
        ],
    )
    write_status(status, status_out)
    print("learned-risk real mode: launch-ready; durable-trace training deferred to Slurm.")
    print(json.dumps(readiness, indent=2, sort_keys=True))
    return 3


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
