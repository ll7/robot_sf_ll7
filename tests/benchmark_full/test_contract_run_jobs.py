"""Contract test T008 for `run_episode_jobs` resume behavior.

Expectations (contract):
  - Jobs whose episode_id already exists in the episodes file are skipped.
  - Manifest.skip count should reflect skipped jobs (later implementation detail).
  - Function yields only new EpisodeRecord objects.

Current state: run_episode_jobs not implemented, so this test will FAIL
with NotImplementedError until T026 provides implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_episode_jobs


@dataclass
class _Job:
    job_id: str
    scenario_id: str
    seed: int
    archetype: str
    density: str
    horizon: int


@dataclass
class _Manifest:
    episodes_path: str


def _episode_id(job: _Job) -> str:  # simplistic deterministic id for test
    return f"{job.scenario_id}-{job.seed}"


def test_run_episode_jobs_resume(temp_results_dir, synthetic_episode_record):
    episodes_dir = Path(temp_results_dir) / "episodes"
    episodes_dir.mkdir()
    episodes_file = episodes_dir / "episodes.jsonl"

    # Pre-populate with one existing episode (seed=1)
    existing_job = _Job("j1", "scenario_a", 1, "crossing", "low", 200)
    with episodes_file.open("w", encoding="utf-8") as f:
        rec = synthetic_episode_record(
            episode_id=_episode_id(existing_job),
            scenario_id=existing_job.scenario_id,
            seed=existing_job.seed,
        )
        f.write(json.dumps(rec) + "\n")

    new_job = _Job("j2", "scenario_a", 2, "crossing", "low", 200)
    manifest = _Manifest(episodes_path=str(episodes_file))

    cfg = type("Cfg", (), {})()  # simple dynamic object
    cfg.smoke = True
    cfg.workers = 1
    cfg.algo = "ppo"

    jobs = [existing_job, new_job]
    # Execute run_episode_jobs; should skip existing (seed=1) and yield only new (seed=2)
    new_records = list(run_episode_jobs(jobs, cfg, manifest))
    assert len(new_records) == 1
    rec = new_records[0]
    assert rec["seed"] == 2
    assert rec["scenario_id"] == new_job.scenario_id
    # File should now contain 2 lines (existing + new)
    with episodes_file.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 2, "Episodes file should contain both existing and newly appended record"
