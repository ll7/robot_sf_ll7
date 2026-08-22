"""Resume-aware sequential and process scheduling for Full Classic episodes."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from .io_utils import append_episode_record

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


def episode_id_from_job(job) -> str:
    """Return the stable episode identity used by resume and append ordering."""
    return f"{job.scenario_id}-{job.seed}"


def scan_existing_episode_ids(path: Path) -> set[str]:
    """Return episode IDs already present in an episodes JSONL file."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed episode record line in {}", path)
                    continue
                episode_id = record.get("episode_id")
                if isinstance(episode_id, str):
                    ids.add(episode_id)
    except OSError as exc:  # pragma: no cover - unlikely on normal FS
        logger.warning("Failed reading existing episodes file {}: {}", path, exc)
    return ids


def partition_jobs(existing_ids: set[str], job_iter: Iterable[object]) -> tuple[list[object], int]:
    """Split incoming jobs into runnable jobs and skipped count.

    Returns:
        Tuple of runnable jobs and the number skipped by resume.
    """
    run_list: list[object] = []
    skip_count = 0
    for job in job_iter:
        if episode_id_from_job(job) in existing_ids:
            skip_count += 1
        else:
            run_list.append(job)
    return run_list, skip_count


def execute_sequential(
    job_list: list[object],
    existing_ids: set[str],
    episodes_path: Path,
    cfg,
    manifest,
    record_builder: Callable[[object, object], dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Execute jobs sequentially and append records in completion order."""
    for job in job_list:
        start = time.time()
        record = record_builder(job, cfg)
        record["wall_time_sec"] = time.time() - start
        append_episode_record(episodes_path, record)
        existing_ids.add(record["episode_id"])
        if hasattr(manifest, "executed_jobs"):
            manifest.executed_jobs += 1
        yield record


def _worker_job_wrapper(job, cfg_payload, record_builder):
    """Run one job in a process with a serializable configuration payload.

    Returns:
        Episode record produced by the worker callback.
    """

    class _TempCfg:
        """Namespace-like wrapper for passing config values into worker jobs."""

        def __init__(self, payload):
            for key, value in payload.items():
                setattr(self, key, value)

    start = time.time()
    record = record_builder(job, _TempCfg(cfg_payload))
    record["wall_time_sec"] = time.time() - start
    return record


def execute_parallel(
    job_list: list[object],
    existing_ids: set[str],
    episodes_path: Path,
    cfg,
    manifest,
    workers: int,
    record_builder: Callable[[object, object], dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Execute jobs in processes, then append results in planned order."""
    logger.debug("Executing {} jobs in parallel with {} workers", len(job_list), workers)
    cfg_payload = vars(cfg).copy() if hasattr(cfg, "__dict__") else {}
    if "disable_videos" not in cfg_payload:
        cfg_payload["disable_videos"] = True
    results_map: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_worker_job_wrapper, job, cfg_payload, record_builder): job
            for job in job_list
        }
        for future in as_completed(future_map):
            record = future.result()
            results_map[record["episode_id"]] = record
    for job in job_list:
        episode_id = episode_id_from_job(job)
        record = results_map[episode_id]
        append_episode_record(episodes_path, record)
        existing_ids.add(episode_id)
        if hasattr(manifest, "executed_jobs"):
            manifest.executed_jobs += 1
        yield record


def execute_episode_jobs(
    jobs: Iterable[object],
    cfg,
    manifest,
    *,
    record_builder: Callable[[object, object], dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Execute jobs with resume accounting and deterministic append ordering."""
    episodes_path = Path(manifest.episodes_path)
    existing_ids = scan_existing_episode_ids(episodes_path)
    logger.debug("Found {} existing episode records (resume)", len(existing_ids))
    to_run, skipped = partition_jobs(existing_ids, list(jobs))
    if hasattr(manifest, "skipped_jobs"):
        manifest.skipped_jobs += skipped
    workers = int(getattr(cfg, "workers", 1) or 1)
    if workers <= 1 or len(to_run) <= 1:
        yield from execute_sequential(
            to_run,
            existing_ids,
            episodes_path,
            cfg,
            manifest,
            record_builder,
        )
    else:
        yield from execute_parallel(
            to_run,
            existing_ids,
            episodes_path,
            cfg,
            manifest,
            workers,
            record_builder,
        )


# Private aliases retain the old inspection/test names for callers that imported them.
_episode_id_from_job = episode_id_from_job
_scan_existing_episode_ids = scan_existing_episode_ids
_partition_jobs = partition_jobs
_execute_seq = execute_sequential
_execute_parallel = execute_parallel


__all__ = [
    "episode_id_from_job",
    "execute_episode_jobs",
    "execute_parallel",
    "execute_sequential",
    "partition_jobs",
    "scan_existing_episode_ids",
]
