"""Immutable setup context for the Full Classic benchmark run.

The public benchmark entrypoint still accepts the existing configuration object.  This
module makes the derived paths, matrix hash, planned scenarios, and initial jobs explicit
before execution starts, so later phases consume one stable plan instead of rebuilding
pieces of it independently.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .planning import expand_episode_jobs, load_scenario_matrix, plan_scenarios


@dataclass
class BenchmarkManifest:
    """Manifest metadata for a benchmark run."""

    output_root: Path
    git_hash: str
    scenario_matrix_hash: str
    config: object
    episodes_path: str
    created_at: float = field(default_factory=time.time)
    executed_jobs: int = 0
    skipped_jobs: int = 0
    notes: str = "skeleton_t029"
    runtime_sec: float = 0.0
    episodes_per_second: float = 0.0
    workers: int = 1
    scaling_efficiency: dict = field(default_factory=dict)
    freeze_validation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FullClassicRunContext:
    """Resolved, immutable inputs shared by all benchmark execution phases.

    The context owns the initial plan and its provenance.  Jobs and scenarios are exposed
    as tuples so a phase cannot accidentally append, reorder, or replace the run plan.
    Individual planning descriptors retain their historical shape for compatibility with
    the existing episode and aggregation contracts.
    """

    cfg: object
    root: Path
    episodes_dir: Path
    aggregates_dir: Path
    reports_dir: Path
    plots_dir: Path
    episodes_path: Path
    raw_scenarios: tuple[dict[str, Any], ...]
    scenario_matrix_hash: str
    scenarios: tuple[object, ...]
    jobs: tuple[object, ...]


def _prepare_output_dirs(cfg) -> tuple[Path, Path, Path, Path, Path]:
    """Create and return directories used by the benchmark artifact contract.

    Returns:
        Tuple containing the root, episodes, aggregates, reports, and plots directories.
    """
    root = Path(cfg.output_root)
    episodes_dir = root / "episodes"
    aggregates_dir = root / "aggregates"
    reports_dir = root / "reports"
    plots_dir = root / "plots"
    for directory in (episodes_dir, aggregates_dir, reports_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return root, episodes_dir, aggregates_dir, reports_dir, plots_dir


def _scenario_matrix_hash(raw_scenarios: list[dict[str, Any]]) -> str:
    """Return the canonical short hash used by manifests and freeze validation."""
    matrix_bytes = json.dumps(
        raw_scenarios,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(matrix_bytes).hexdigest()[:12]


def _apply_smoke_bounds(scenarios: list[object], jobs: list[object], cfg):
    """Apply smoke limits while preserving immutable context values.

    Returns:
        Tuple of bounded scenario and job sequences.
    """
    scenarios_list = list(scenarios)
    jobs_list = list(jobs)
    smoke_limit = bool(getattr(cfg, "smoke_limit_jobs", False))
    if getattr(cfg, "smoke", False) and scenarios_list and smoke_limit:
        scenarios_list = scenarios_list[:1]
        allowed = {scenario.scenario_id for scenario in scenarios_list}
        jobs_list = [job for job in jobs_list if job.scenario_id in allowed]
        jobs_list = jobs_list[: max(1, int(getattr(cfg, "smoke_episodes", 1) or 1))]

    if getattr(cfg, "smoke", False):
        horizon_cap = int(getattr(cfg, "smoke_horizon_cap", 40) or 40)
        bounded_jobs = []
        for job in jobs_list:
            try:
                horizon = min(int(getattr(job, "horizon", horizon_cap)), horizon_cap)
            except (ValueError, TypeError, AttributeError):
                horizon = horizon_cap
            bounded_jobs.append(replace(job, horizon=horizon))
        jobs_list = bounded_jobs
    return tuple(scenarios_list), tuple(jobs_list)


def build_run_context(cfg) -> FullClassicRunContext:
    """Resolve output paths, matrix provenance, scenarios, and initial jobs once.

    Returns:
        Immutable context consumed by the benchmark phases.
    """
    root, episodes_dir, aggregates_dir, reports_dir, plots_dir = _prepare_output_dirs(cfg)
    raw_scenarios = load_scenario_matrix(cfg.scenario_matrix_path)
    matrix_hash = _scenario_matrix_hash(raw_scenarios)
    rng = random.Random(int(getattr(cfg, "master_seed", 123)))
    scenarios = plan_scenarios(raw_scenarios, cfg, rng=rng)
    jobs = expand_episode_jobs(scenarios, cfg)
    bounded_scenarios, bounded_jobs = _apply_smoke_bounds(scenarios, jobs, cfg)
    return FullClassicRunContext(
        cfg=cfg,
        root=root,
        episodes_dir=episodes_dir,
        aggregates_dir=aggregates_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        episodes_path=episodes_dir / "episodes.jsonl",
        raw_scenarios=tuple(raw_scenarios),
        scenario_matrix_hash=matrix_hash,
        scenarios=bounded_scenarios,
        jobs=bounded_jobs,
    )


__all__ = [
    "BenchmarkManifest",
    "FullClassicRunContext",
    "build_run_context",
]
