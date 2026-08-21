"""Focused contracts for the Full Classic orchestration phase boundaries."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.full_classic.context import BenchmarkManifest, build_run_context
from robot_sf.benchmark.full_classic.execution import (
    EpisodeExecutionHooks,
    execute_episode_record,
)
from robot_sf.benchmark.full_classic.finalizer import finalize_run
from robot_sf.benchmark.full_classic.scheduler import (
    execute_episode_jobs,
    partition_jobs,
    scan_existing_episode_ids,
)


class _Job:
    """Small job object sufficient for the execution-phase contract."""

    scenario_id = "scenario_a"
    seed = 7
    archetype = "crossing"
    density = "low"
    horizon = 20


def _build_scheduler_record(job, cfg):
    """Build a picklable record for scheduler process-boundary tests."""
    return {
        "episode_id": f"{job.scenario_id}-{job.seed}",
        "seed": job.seed,
        "disable_videos": getattr(cfg, "disable_videos", False),
    }


def test_run_context_is_frozen_and_reproducible(config_factory) -> None:
    """Setup derives one immutable plan with stable IDs and matrix provenance."""
    cfg = config_factory(smoke=True, smoke_horizon_cap=8, master_seed=91)

    first = build_run_context(cfg)
    second = build_run_context(cfg)

    assert isinstance(first.raw_scenarios, tuple)
    assert isinstance(first.scenarios, tuple)
    assert isinstance(first.jobs, tuple)
    assert first.scenario_matrix_hash == second.scenario_matrix_hash
    assert [job.job_id for job in first.jobs] == [job.job_id for job in second.jobs]
    assert all(job.horizon <= 8 for job in first.jobs)
    with pytest.raises(FrozenInstanceError):
        first.root = Path("/tmp/must-not-change")  # type: ignore[misc]


def test_execution_boundary_preserves_stub_record_contract() -> None:
    """The isolated record phase keeps metadata injection and threshold enrichment."""
    job = _Job()
    cfg = SimpleNamespace(fast_stub=True, algo="ppo", capture_replay=False)

    def make_stub_episode_record(job, _cfg, *, episode_id: str, horizon: int):
        return {
            "version": "v1",
            "episode_id": episode_id,
            "scenario_id": job.scenario_id,
            "seed": job.seed,
            "archetype": job.archetype,
            "density": job.density,
            "metrics": {"success_rate": 1.0, "collision_rate": 0.0},
            "scenario_params": {},
            "horizon": horizon,
        }

    hooks = EpisodeExecutionHooks(
        episode_id_from_job=lambda item: f"{item.scenario_id}-{item.seed}",
        resolve_horizon=lambda item, _cfg: item.horizon,
        make_stub_episode_record=make_stub_episode_record,
        require_job_scenario=lambda *_args, **_kwargs: pytest.fail(
            "stub must not resolve a scenario"
        ),
        orchestrate_real_episode=lambda *_args, **_kwargs: pytest.fail("stub must not roll out"),
        attach_replay_payload=lambda *_args, **_kwargs: pytest.fail("replay is disabled"),
        termination_payload_from_metrics=lambda _metrics: (
            "success",
            {"route_complete": True, "collision": False, "timeout": False},
            [],
        ),
        ensure_algo_metadata=lambda record, *, algo, episode_id: (
            record.update(
                {"algo": algo, "scenario_params": {"algo": algo, "episode_id": episode_id}}
            )
            or record
        ),
    )

    record = execute_episode_record(job, cfg, hooks)

    assert record["episode_id"] == "scenario_a-7"
    assert record["algo"] == "ppo"
    assert record["scenario_params"]["algo"] == "ppo"
    assert record["metric_parameters"]["threshold_profile"]["profile_id"]


def test_scheduler_partition_keeps_input_order_and_resume_count() -> None:
    """Resume filtering skips existing IDs without reordering runnable jobs."""
    jobs = [SimpleNamespace(scenario_id="s", seed=seed) for seed in (1, 2, 3)]

    runnable, skipped = partition_jobs({"s-2"}, jobs)

    assert [job.seed for job in runnable] == [1, 3]
    assert skipped == 1


def test_scheduler_scan_ignores_blank_and_malformed_records(tmp_path: Path) -> None:
    """Resume scanning keeps valid IDs while tolerating bad JSONL lines."""
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        '\nnot-json\n{"episode_id": "known"}\n{"episode_id": 7}\n',
        encoding="utf-8",
    )

    assert scan_existing_episode_ids(episodes_path) == {"known"}


def test_scheduler_parallel_branch_preserves_order_and_resume(tmp_path: Path) -> None:
    """The process path appends in plan order and injects its safe config default."""
    jobs = [SimpleNamespace(scenario_id="s", seed=seed) for seed in (1, 2, 3)]
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text('{"episode_id":"s-2"}\n', encoding="utf-8")
    manifest = SimpleNamespace(
        episodes_path=str(episodes_path),
        executed_jobs=0,
        skipped_jobs=0,
    )
    cfg = SimpleNamespace(workers=2)

    records = list(
        execute_episode_jobs(
            jobs,
            cfg,
            manifest,
            record_builder=_build_scheduler_record,
        )
    )

    assert [record["episode_id"] for record in records] == ["s-1", "s-3"]
    assert all(record["disable_videos"] for record in records)
    assert manifest.executed_jobs == 2
    assert manifest.skipped_jobs == 1
    assert [json.loads(line)["episode_id"] for line in episodes_path.read_text().splitlines()] == [
        "s-2",
        "s-1",
        "s-3",
    ]


def test_finalizer_closes_manifest_and_invokes_publication_phase(tmp_path: Path) -> None:
    """Finalization persists the manifest before invoking the visual publication hook."""
    cfg = SimpleNamespace(workers=1, smoke=True)
    manifest = BenchmarkManifest(
        output_root=tmp_path,
        git_hash="test",
        scenario_matrix_hash="matrix",
        config=cfg,
        episodes_path=str(tmp_path / "episodes.jsonl"),
    )
    observed: list[str] = []

    def write_run_meta(_root, _cfg, _manifest):
        observed.append("run_meta")

    def publish_visuals(_root, _cfg, _groups, _records):
        observed.append("visuals")

    finalize_run(
        tmp_path,
        cfg,
        manifest,
        groups=[],
        all_records=[],
        write_run_meta_files_fn=write_run_meta,
        visual_generator=publish_visuals,
        visualization_available=False,
    )

    assert (tmp_path / "manifest.json").exists()
    assert manifest.scaling_efficiency["finalized"] is True
    assert observed == ["run_meta", "visuals"]
