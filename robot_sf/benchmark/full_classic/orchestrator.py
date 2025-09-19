"""Orchestration logic for executing episode jobs and adaptive sampling.

Implemented incrementally in tasks T026-T029, T027 (parallel), T028 (adaptive iteration),
T029 (full run orchestration skeleton).
"""

from __future__ import annotations

from typing import Iterator


def run_episode_jobs(jobs, cfg, manifest) -> Iterator[object]:  # T026
    """Placeholder: yields nothing until implemented."""
    raise NotImplementedError("Implemented in task T026")


def adaptive_sampling_iteration(current_records, cfg, scenarios, manifest):  # T028
    raise NotImplementedError("Implemented in task T028")


def run_full_benchmark(cfg):  # T029
    raise NotImplementedError("Implemented in task T029")
