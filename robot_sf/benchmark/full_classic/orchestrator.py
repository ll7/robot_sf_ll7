"""Orchestration logic for executing episode jobs and adaptive sampling.

Implemented incrementally in tasks T026-T029, T027 (parallel), T028 (adaptive iteration),
T029 (full run orchestration skeleton).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set

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


def _partition_jobs(existing_ids: Set[str], job_iter: Iterable[object]) -> tuple[List[object], int]:
    run_list: List[object] = []
    skip_count = 0
    for jb in job_iter:
        if _episode_id_from_job(jb) in existing_ids:
            skip_count += 1
        else:
            run_list.append(jb)
    return run_list, skip_count


def _execute_seq(
    job_list: List[object], existing_ids: Set[str], episodes_path: Path, cfg, manifest
) -> Iterator[dict]:
    for jb in job_list:
        rec = _make_episode_record(jb, cfg)
        append_episode_record(episodes_path, rec)
        existing_ids.add(rec["episode_id"])
        if hasattr(manifest, "executed_jobs"):
            manifest.executed_jobs += 1
        yield rec


def _worker_job_wrapper(job, algo):  # top-level for pickling on spawn
    class _TempCfg:  # noqa: D401
        def __init__(self, a):
            self.algo = a

    return _make_episode_record(job, _TempCfg(algo))


def _execute_parallel(
    job_list: List[object], existing_ids: Set[str], episodes_path: Path, cfg, manifest, workers: int
) -> Iterator[dict]:
    logger.debug("Executing {} jobs in parallel with {} workers", len(job_list), workers)
    algo = getattr(cfg, "algo", "unknown")
    results_map: Dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_worker_job_wrapper, j, algo): j for j in job_list}
        for fut in as_completed(future_map):
            rec = fut.result()
            results_map[rec["episode_id"]] = rec
    # Deterministic ordering for append
    for jb in job_list:
        ep_id = _episode_id_from_job(jb)
        rec = results_map[ep_id]
        append_episode_record(episodes_path, rec)
        existing_ids.add(ep_id)
        if hasattr(manifest, "executed_jobs"):
            manifest.executed_jobs += 1
        yield rec


def run_episode_jobs(jobs: Iterable[object], cfg, manifest) -> Iterator[dict]:  # T026/T027
    """Execute episode jobs with resume + optional parallel workers.

    T026 (completed earlier): sequential execution + resume scan.
    T027 extension:
      - If cfg.workers > 1 use a process pool to compute episode records in parallel.
      - Parent process performs file appends (avoids concurrent writes).
      - Update manifest counters: executed_jobs, skipped_jobs.
    """
    episodes_path = Path(getattr(manifest, "episodes_path"))
    existing_ids = _scan_existing_episode_ids(episodes_path)
    logger.debug("Found {} existing episode records (resume)", len(existing_ids))
    to_run, skipped = _partition_jobs(existing_ids, list(jobs))
    if hasattr(manifest, "skipped_jobs"):
        manifest.skipped_jobs += skipped
    workers = int(getattr(cfg, "workers", 1) or 1)
    if workers <= 1 or len(to_run) <= 1:
        yield from _execute_seq(to_run, existing_ids, episodes_path, cfg, manifest)
    else:
        yield from _execute_parallel(to_run, existing_ids, episodes_path, cfg, manifest, workers)


def adaptive_sampling_iteration(current_records, cfg, scenarios, manifest):  # T028
    raise NotImplementedError("Implemented in task T028")


def run_full_benchmark(cfg):  # T029
    raise NotImplementedError("Implemented in task T029")
