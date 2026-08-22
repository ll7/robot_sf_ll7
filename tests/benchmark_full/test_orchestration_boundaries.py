"""Focused contracts for the Full Classic orchestration phase boundaries."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import robot_sf.benchmark.full_classic.orchestrator as orch
from robot_sf.benchmark.full_classic.aggregation import AggregateMetric, AggregateMetricsGroup
from robot_sf.benchmark.full_classic.context import BenchmarkManifest, build_run_context
from robot_sf.benchmark.full_classic.effects import EffectSizeEntry, EffectSizeReport
from robot_sf.benchmark.full_classic.execution import (
    EpisodeExecutionHooks,
    execute_episode_record,
)
from robot_sf.benchmark.full_classic.finalizer import (
    finalize_run,
    publish_visual_artifacts,
    serialize_effects,
    serialize_groups,
    serialize_precision,
    write_iteration_artifacts,
    write_json,
)
from robot_sf.benchmark.full_classic.precision import (
    PrecisionEntry,
    ScenarioPrecisionStatus,
    StatisticalSufficiencyReport,
)
from robot_sf.benchmark.full_classic.replay import ReplayCapture
from robot_sf.benchmark.full_classic.scheduler import (
    _worker_job_wrapper,
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


def test_execution_boundary_builds_real_record_and_replay_payload() -> None:
    """The real path assembles runtime metadata and attaches an optional replay."""
    job = _Job()
    cfg = SimpleNamespace(fast_stub=False, algo="ppo", capture_replay=True)
    scenario = SimpleNamespace(
        map_path="maps/test.svg",
        hash_fragment="abc123",
        raw={"simulation_config": {"max_episode_steps": 20}, "metadata": {"tag": "test"}},
    )
    replay_capture = ReplayCapture(episode_id="scenario_a-7", scenario_id="scenario_a", dt=0.1)
    replay_capture.record(0.0, 1.0, 2.0, 0.5, action=(0.1, 0.2))
    replay_capture.record(0.1, 1.1, 2.1, 0.6, action=(0.2, 0.3))
    attached: list[dict] = []

    def attach_replay(record, **kwargs):
        attached.append(kwargs)
        record["replay_steps"] = [
            (step.t, step.x) for step in kwargs["replay_capture"].finalize().steps
        ]

    hooks = EpisodeExecutionHooks(
        episode_id_from_job=lambda item: f"{item.scenario_id}-{item.seed}",
        resolve_horizon=lambda item, _cfg: item.horizon,
        make_stub_episode_record=lambda *_args, **_kwargs: pytest.fail("real path used stub"),
        require_job_scenario=lambda item, **_kwargs: scenario,
        orchestrate_real_episode=lambda *_args, **_kwargs: {
            "metrics": {"success_rate": 1.0, "collision_rate": 0.0},
            "steps_taken": 2,
            "wall_time": 0.5,
            "start_time": 123.0,
            "replay_capture": replay_capture,
            "ped_forces": [[[0.0, 0.0]]],
        },
        attach_replay_payload=attach_replay,
        termination_payload_from_metrics=lambda _metrics: (
            "success",
            {"route_complete": True, "collision": False, "timeout": False},
            [],
        ),
        ensure_algo_metadata=lambda record, *, algo, episode_id: (
            record.update({"algo": algo, "algo_episode_id": episode_id}) or record
        ),
    )

    record = execute_episode_record(job, cfg, hooks)

    assert record["status"] == "success"
    assert record["scenario_params"]["map_file"] == "maps/test.svg"
    assert record["timing"]["steps_per_second"] == 4.0
    assert record["replay_steps"] == [(0.0, 1.0), (0.1, 1.1)]
    assert attached and attached[0]["scenario"] is scenario
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


def test_scheduler_sequential_and_worker_boundaries(tmp_path: Path) -> None:
    """Sequential execution, missing-file resume, and the worker adapter stay compatible."""
    missing_path = tmp_path / "does-not-exist.jsonl"
    assert scan_existing_episode_ids(missing_path) == set()

    jobs = [SimpleNamespace(scenario_id="s", seed=seed) for seed in (4, 5)]
    episodes_path = tmp_path / "episodes.jsonl"
    manifest = SimpleNamespace(
        episodes_path=str(episodes_path),
        executed_jobs=0,
        skipped_jobs=0,
    )
    cfg = SimpleNamespace(workers=1)

    records = list(
        execute_episode_jobs(
            jobs,
            cfg,
            manifest,
            record_builder=_build_scheduler_record,
        )
    )
    worker_record = _worker_job_wrapper(
        jobs[0],
        {"disable_videos": True},
        _build_scheduler_record,
    )

    assert [record["episode_id"] for record in records] == ["s-4", "s-5"]
    assert all(record["wall_time_sec"] >= 0.0 for record in records)
    assert worker_record["disable_videos"] is True
    assert worker_record["wall_time_sec"] >= 0.0
    assert manifest.executed_jobs == 2


def test_orchestrator_compatibility_facades_delegate(monkeypatch, tmp_path: Path) -> None:
    """Historical orchestrator helpers delegate to the extracted phase owners."""
    job = _Job()
    cfg = SimpleNamespace(fast_stub=True)
    observed: dict[str, object] = {}

    def fake_execute(job_arg, cfg_arg, hooks):
        observed["hooks"] = hooks
        return {"episode_id": f"{job_arg.scenario_id}-{job_arg.seed}", "cfg": cfg_arg}

    monkeypatch.setattr(orch, "execute_episode_record", fake_execute)
    assert orch._make_episode_record(job, cfg)["episode_id"] == "scenario_a-7"
    assert isinstance(observed["hooks"], EpisodeExecutionHooks)

    monkeypatch.setattr(
        orch,
        "execute_episode_jobs",
        lambda jobs, cfg, manifest, *, record_builder: iter(
            [{"count": len(list(jobs)), "builder": record_builder}]
        ),
    )
    facade_records = list(orch.run_episode_jobs([job], cfg, SimpleNamespace()))
    assert facade_records[0]["count"] == 1
    assert facade_records[0]["builder"] is orch._make_episode_record

    def fake_sequential(*args):
        observed["sequential_builder"] = args[-1]
        yield {"path": "sequential"}

    def fake_parallel(*args):
        observed["parallel_builder"] = args[-1]
        yield {"path": "parallel"}

    monkeypatch.setattr(orch._scheduler, "execute_sequential", fake_sequential)
    monkeypatch.setattr(orch._scheduler, "execute_parallel", fake_parallel)
    assert list(
        orch._execute_seq([], set(), tmp_path / "episodes.jsonl", cfg, SimpleNamespace())
    ) == [{"path": "sequential"}]
    assert list(
        orch._execute_parallel([], set(), tmp_path / "episodes.jsonl", cfg, SimpleNamespace(), 2)
    ) == [{"path": "parallel"}]
    assert observed["sequential_builder"] is orch._make_episode_record
    assert observed["parallel_builder"] is orch._make_episode_record


def test_orchestrator_run_coordinator_resolves_context_and_finalizes(
    monkeypatch, tmp_path: Path
) -> None:
    """The public coordinator passes one resolved context through to finalization."""
    cfg = SimpleNamespace(freeze_manifest_path=None, max_episodes=0, smoke=False)
    context = SimpleNamespace(
        root=tmp_path,
        episodes_path=tmp_path / "episodes" / "episodes.jsonl",
        raw_scenarios=({"name": "raw"},),
        scenario_matrix_hash="matrix-hash",
        scenarios=(SimpleNamespace(scenario_id="scenario_a"),),
        jobs=(_Job(),),
    )
    manifest = BenchmarkManifest(
        output_root=tmp_path,
        git_hash="test",
        scenario_matrix_hash="matrix-hash",
        config=cfg,
        episodes_path=str(context.episodes_path),
    )
    precision_report = SimpleNamespace(final_pass=True)
    observed: dict[str, object] = {}

    monkeypatch.setattr(orch, "build_run_context", lambda _cfg: context)
    monkeypatch.setattr(orch, "_init_manifest", lambda *args: manifest)
    monkeypatch.setattr(orch, "run_episode_jobs", lambda *_args: iter([]))
    monkeypatch.setattr(orch, "aggregate_metrics", lambda *_args: [])
    monkeypatch.setattr(orch, "compute_effect_sizes", lambda *_args: [])
    monkeypatch.setattr(orch, "evaluate_precision", lambda *_args: precision_report)
    monkeypatch.setattr(orch, "_update_scaling_efficiency", lambda *_args: {"workers": 1})
    monkeypatch.setattr(
        orch, "_write_iteration_artifacts", lambda *args: observed.update(artifacts=args)
    )

    def fake_finalize(*args, **kwargs):
        observed["finalize_args"] = args
        observed["finalize_kwargs"] = kwargs

    monkeypatch.setattr(orch, "finalize_run", fake_finalize)

    result = orch.run_full_benchmark(cfg)

    assert result is manifest
    assert observed["artifacts"][:2] == (tmp_path, [])
    assert observed["finalize_args"][:3] == (tmp_path, cfg, manifest)
    assert observed["finalize_kwargs"]["groups"] == []
    assert observed["finalize_kwargs"]["all_records"] == []


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


def test_finalizer_serializes_reports_and_handles_optional_visuals(tmp_path: Path) -> None:
    """Report serializers preserve nested schemas and optional real visuals stay non-fatal."""
    metric = AggregateMetric(
        name="success_rate",
        mean=0.8,
        median=0.8,
        p95=0.9,
        mean_ci=(0.7, 0.9),
        median_ci=(0.75, 0.85),
    )
    group = AggregateMetricsGroup(
        archetype="crossing",
        density="low",
        count=2,
        metrics={"success_rate": metric},
    )
    effects = [
        EffectSizeReport(
            archetype="crossing",
            comparisons=[
                EffectSizeEntry(
                    metric="success_rate",
                    density_low="low",
                    density_high="high",
                    diff=0.1,
                    standardized=0.2,
                )
            ],
        )
    ]
    precision = StatisticalSufficiencyReport(
        evaluations=[
            ScenarioPrecisionStatus(
                scenario_id="crossing-low",
                archetype="crossing",
                density="low",
                episodes=2,
                metric_status=[PrecisionEntry("success_rate", 0.1, 0.2, True)],
                all_pass=True,
            )
        ],
        final_pass=True,
        scaling_efficiency={},
    )

    assert serialize_groups([group])[0]["metrics"]["success_rate"]["mean"] == 0.8
    assert serialize_effects(effects)[0]["comparisons"][0]["diff"] == 0.1
    assert serialize_precision(precision)["evaluations"][0]["metric_status"][0]["passed"]
    (tmp_path / "aggregates").mkdir()
    (tmp_path / "reports").mkdir()
    write_iteration_artifacts(tmp_path, [group], effects, precision)
    assert json.loads((tmp_path / "aggregates" / "summary.json").read_text())[0]["count"] == 2
    assert json.loads((tmp_path / "reports" / "effect_sizes.json").read_text())[0]["archetype"] == (
        "crossing"
    )

    write_json(tmp_path / "missing" / "artifact.json", {"value": object()})

    calls: list[str] = []

    def visual_generator(_root, _cfg, _groups, _records):
        calls.append("visual")

    def plot_generator(_records, _root):
        calls.append("plots")
        return ["plot-artifact"]

    validation = SimpleNamespace(passed=True, failed_artifacts=[])
    publish_visual_artifacts(
        tmp_path,
        SimpleNamespace(smoke=False),
        [group],
        [{"episode_id": "scenario_a-7"}],
        visual_generator=visual_generator,
        visualization_available=True,
        plot_generator=plot_generator,
        validation_fn=lambda artifacts: calls.append(f"validate:{len(artifacts)}") or validation,
    )

    assert calls == ["visual", "plots", "validate:1"]
    assert (tmp_path / "plots").is_dir()
    assert (tmp_path / "videos").is_dir()
