#!/usr/bin/env python3
"""Backfill a training-run manifest from a completed run's retained artefacts.

This recovers runs that trained and evaluated successfully but lost their
``benchmarks/ppo_imitation/runs/<run_id>.json`` manifest because the manifest
write raised on a cross-worktree path (see ``imitation_manifest`` and the
``manifest_path_outside_allowed_roots_after_training`` failure mode). The
training compute is intact on disk; only the manifest JSON is missing, so it is
reconstructed from the artefacts the pipeline already wrote:

* ``benchmarks/expert_policies/<policy>.json``        -> metrics, seeds
* ``benchmarks/ppo_imitation/perf/<run>.json``        -> run_id, wall-clock
* ``benchmarks/ppo_imitation/eval_by_scenario/<run>.json`` -> scenario coverage
* ``benchmarks/ppo_imitation/{episodes,eval_timeline,perf}/*`` -> path fields

No training or evaluation is re-run. The reconstructed manifest is marked as
backfilled in its notes so downstream consumers can tell it apart from a manifest
written inline by the training pipeline.

Usage::

    python scripts/tools/backfill_training_run_manifest.py --run-dir <dir> [--dry-run]

``--run-dir`` is the directory that contains ``benchmarks/`` (the per-job output
root). Pass ``--benchmarks-dir`` to point directly at a ``benchmarks`` tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from robot_sf.benchmark.imitation_manifest import (
    write_training_run_manifest,
)
from robot_sf.common import (
    MetricAggregate,
    TrainingRunArtifact,
    TrainingRunStatus,
    TrainingRunType,
)


class BackfillError(RuntimeError):
    """Raised when the retained artefacts are insufficient to rebuild a manifest."""


def _load_json(path: Path) -> Any:
    """Load a JSON file, raising :class:`BackfillError` with context on failure."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        raise BackfillError(f"Could not read {path}: {exc}") from exc


def _single(paths: list[Path], label: str, *, required: bool = True) -> Path | None:
    """Return the lone path in ``paths`` or raise/return ``None`` per ``required``."""
    real = [p for p in paths if "checkpoints" not in p.parts]
    if len(real) == 1:
        return real[0]
    if not real:
        if required:
            raise BackfillError(f"No {label} artefact found")
        return None
    raise BackfillError(f"Expected exactly one {label} artefact, found {len(real)}: {real}")


def _metric_from_payload(payload: dict[str, Any]) -> MetricAggregate:
    """Rebuild a :class:`MetricAggregate` from a serialized metric mapping."""
    ci95 = payload.get("ci95")
    return MetricAggregate(
        mean=payload["mean"],
        median=payload["median"],
        p95=payload["p95"],
        ci95=(ci95[0], ci95[1]) if ci95 else None,
    )


def _scenario_coverage(eval_by_scenario: list[dict[str, Any]]) -> dict[str, int]:
    """Sum evaluated episodes per scenario at the final evaluation checkpoint."""
    if not eval_by_scenario:
        return {}
    final_step = max(int(row.get("eval_step", 0)) for row in eval_by_scenario)
    coverage: dict[str, int] = defaultdict(int)
    for row in eval_by_scenario:
        if int(row.get("eval_step", 0)) != final_step:
            continue
        scenario = str(row.get("scenario_id", "unknown"))
        coverage[scenario] += int(row.get("episodes", 0))
    return dict(sorted(coverage.items()))


def build_artifact(
    benchmarks_dir: Path,
    *,
    run_type: TrainingRunType = TrainingRunType.EXPERT_TRAINING,
    evaluation_scenario_config: Path | None = None,
    input_artefacts: tuple[str, ...] = (),
) -> tuple[TrainingRunArtifact, Path]:
    """Reconstruct a :class:`TrainingRunArtifact` from retained artefacts.

    Returns:
        The reconstructed artifact and the ``ppo_imitation`` directory it was
        rebuilt from (used to locate the manifest destination).
    """
    imitation_dir = benchmarks_dir / "ppo_imitation"
    if not imitation_dir.is_dir():
        raise BackfillError(f"No ppo_imitation directory under {benchmarks_dir}")

    # Record artefact paths relative to the run's output root (the parent of benchmarks/), so
    # the manifest stores portable ``benchmarks/...`` paths exactly like an inline-written one.
    # This keeps serialization independent of which checkout/worktree runs the backfill.
    output_root = benchmarks_dir.parent

    def _rel(path: Path) -> Path:
        try:
            return path.resolve(strict=False).relative_to(output_root.resolve(strict=False))
        except ValueError:  # pragma: no cover - artefacts always live under output_root
            return path

    perf_path = _single(list((imitation_dir / "perf").glob("*.json")), "perf summary")
    episode_path = _single(list((imitation_dir / "episodes").glob("*.jsonl")), "episode log")
    eval_scenario_path = _single(
        list((imitation_dir / "eval_by_scenario").glob("*.json")), "eval-by-scenario"
    )
    eval_timeline_path = _single(
        list((imitation_dir / "eval_timeline").glob("*.json")), "eval timeline", required=False
    )
    expert_path = _single(
        list((benchmarks_dir / "expert_policies").glob("*.json")), "expert policy manifest"
    )

    assert perf_path is not None and episode_path is not None
    assert eval_scenario_path is not None and expert_path is not None

    perf = _load_json(perf_path)
    expert = _load_json(expert_path)
    eval_by_scenario = _load_json(eval_scenario_path)

    run_id = perf.get("run_id") or episode_path.stem
    wall_sec = perf.get("total_wall_clock_sec")
    wall_clock_hours = round(float(wall_sec) / 3600.0, 4) if wall_sec is not None else 0.0

    metrics = {
        name: _metric_from_payload(payload) for name, payload in expert.get("metrics", {}).items()
    }
    seeds = tuple(int(seed) for seed in expert.get("seeds", ()))

    artifact = TrainingRunArtifact(
        run_id=run_id,
        run_type=run_type,
        input_artefacts=input_artefacts,
        seeds=seeds,
        metrics=metrics,
        episode_log_path=_rel(episode_path),
        wall_clock_hours=wall_clock_hours,
        status=TrainingRunStatus.COMPLETED,
        eval_timeline_path=_rel(eval_timeline_path) if eval_timeline_path else None,
        eval_per_scenario_path=_rel(eval_scenario_path),
        perf_summary_path=_rel(perf_path),
        evaluation_scenario_config=evaluation_scenario_config,
        scenario_coverage=_scenario_coverage(eval_by_scenario),
        notes=[
            "Backfilled by scripts/tools/backfill_training_run_manifest.py from retained "
            "artefacts after the inline manifest write failed (cross-worktree path).",
        ],
    )
    return artifact, imitation_dir


def backfill(
    run_dir: Path,
    *,
    benchmarks_dir: Path | None = None,
    evaluation_scenario_config: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> Path:
    """Rebuild and write the training-run manifest for a single run directory.

    Returns:
        The manifest path that was (or would be) written.
    """
    bench = benchmarks_dir or (run_dir / "benchmarks")
    if not bench.is_dir():
        raise BackfillError(f"No benchmarks directory at {bench}")

    artifact, imitation_dir = build_artifact(
        bench, evaluation_scenario_config=evaluation_scenario_config
    )
    target = imitation_dir / "runs" / f"{artifact.run_id}.json"

    if target.exists() and not force:
        raise BackfillError(f"Manifest already exists at {target}; pass --force to overwrite")

    if dry_run:
        print(f"[dry-run] would write manifest for run_id={artifact.run_id}")
        print(f"[dry-run]   metrics: {sorted(artifact.metrics)}")
        print(f"[dry-run]   seeds={artifact.seeds} wall_clock_hours={artifact.wall_clock_hours}")
        print(f"[dry-run]   scenarios covered: {len(artifact.scenario_coverage)}")
        print(f"[dry-run]   target: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    written = write_training_run_manifest(artifact, manifest_path=target)
    print(f"Wrote training-run manifest: {written}")
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Per-job output root containing benchmarks/")
    src.add_argument("--benchmarks-dir", type=Path, help="Path to the benchmarks/ tree directly")
    parser.add_argument(
        "--evaluation-scenario-config",
        type=Path,
        default=None,
        help="Optional path to the evaluation scenario config to record (degrades to basename).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code (0 on success, 1 on a recoverable backfill error).
    """
    args = _parse_args(argv)
    try:
        backfill(
            run_dir=args.run_dir if args.run_dir else args.benchmarks_dir.parent,
            benchmarks_dir=args.benchmarks_dir,
            evaluation_scenario_config=args.evaluation_scenario_config,
            dry_run=args.dry_run,
            force=args.force,
        )
    except BackfillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
