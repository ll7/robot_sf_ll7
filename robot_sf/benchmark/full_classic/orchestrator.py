"""Orchestration logic for executing episode jobs and adaptive sampling.

Implemented incrementally in tasks T026-T029, T027 (parallel), T028 (adaptive iteration),
T029 (full run orchestration skeleton).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Set

from loguru import logger

from .io_utils import append_episode_record


def _episode_id_from_job(job) -> str:
    """Deterministically derive an episode_id from a job.

    Contract (early phase): scenario_id + '-' + seed. Horizon intentionally excluded
    to keep reproducibility with initial tests; may evolve later when multi‑horizon
    episodes are introduced.
    """
    return f"{job.scenario_id}-{job.seed}"


def _scan_existing_episode_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Lightweight JSON parse without importing json if possible? Simplicity first.
                import json  # local import keeps module import surface minimal

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed episode record line in {}", path)
                    continue
                ep_id = rec.get("episode_id")
                if isinstance(ep_id, str):
                    ids.add(ep_id)
    except OSError as exc:  # pragma: no cover - unlikely on normal FS
        logger.warning("Failed reading existing episodes file {}: {}", path, exc)
    return ids


def _make_episode_record(job, cfg) -> Dict[str, Any]:  # minimal synthetic execution
    """Produce a placeholder EpisodeRecord structure matching contract fixtures.

    Real implementation (post T026) will execute simulation & compute metrics. For now
    we populate required structural fields with synthetic values to allow downstream
    aggregation/effects tasks to proceed under TDD.
    """
    episode_id = _episode_id_from_job(job)
    record: Dict[str, Any] = {
        "episode_id": episode_id,
        "scenario_id": job.scenario_id,
        "seed": job.seed,
        "archetype": job.archetype,
        "density": job.density,
        "status": "success",
        "metrics": {
            # Placeholder metric values (kept deterministic)
            "collision_rate": 0.0,
            "success_rate": 1.0,
            "time_to_goal": 10.0,
            "path_efficiency": 0.9,
            "average_speed": 1.0,
            "snqi": 0.75,
        },
        "steps": min(job.horizon, 120),  # bounded placeholder
        "wall_time_sec": 0.0,  # will be populated with timing later (T041)
        "algo": getattr(cfg, "algo", "unknown"),
        "created_at": 0.0,  # real timestamp added later
    }
    return record


def run_episode_jobs(jobs: Iterable[object], cfg, manifest) -> Iterator[dict]:  # T026
    """Execute episode jobs sequentially with basic resume support.

    Responsibilities (T026 scope):
      - Read existing episodes file (if any) and collect existing episode_ids.
      - For each job, derive deterministic episode_id; skip if already present.
      - For new jobs, create a synthetic placeholder EpisodeRecord and append it.
      - Yield each newly created record (iterator semantics allow future streaming / parallelism).

    Notes:
      - Parallel execution & timing metrics deferred to T027/T041.
      - Adaptive sampling integration deferred to T028/T034.
    """
    episodes_path = Path(getattr(manifest, "episodes_path"))
    existing_ids = _scan_existing_episode_ids(episodes_path)
    logger.debug("Found {} existing episode records (resume)", len(existing_ids))

    for job in jobs:
        ep_id = _episode_id_from_job(job)
        if ep_id in existing_ids:
            logger.debug(
                "Skipping existing episode {} (job {})", ep_id, getattr(job, "job_id", "?")
            )
            continue
        record = _make_episode_record(job, cfg)
        append_episode_record(episodes_path, record)
        existing_ids.add(ep_id)
        yield record


def adaptive_sampling_iteration(current_records, cfg, scenarios, manifest):  # T028
    raise NotImplementedError("Implemented in task T028")


def run_full_benchmark(cfg):  # T029
    raise NotImplementedError("Implemented in task T029")
