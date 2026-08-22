"""Episode record execution boundary for the Full Classic benchmark.

Record construction depends on the environment-specific helpers in the legacy
orchestrator.  The hooks below make that dependency explicit while keeping the record
schema and runtime classification in one independently testable phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.termination_reason import status_from_termination_reason
from robot_sf.benchmark.thresholds import ensure_metric_parameters

from .replay import ReplayCapture

if TYPE_CHECKING:
    # Imported only for annotations; postponed evaluation keeps worker imports lightweight.
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class EpisodeExecutionHooks:
    """Environment and metadata operations required by record assembly."""

    episode_id_from_job: Callable[[Any], str]
    resolve_horizon: Callable[[Any, Any], int]
    make_stub_episode_record: Callable[..., dict[str, Any]]
    require_job_scenario: Callable[..., Any]
    orchestrate_real_episode: Callable[..., dict[str, Any]]
    attach_replay_payload: Callable[..., None]
    termination_payload_from_metrics: Callable[
        [dict[str, float]], tuple[str, dict[str, bool], list[str]]
    ]
    ensure_algo_metadata: Callable[..., dict[str, Any]]


def execute_episode_record(job, cfg, hooks: EpisodeExecutionHooks) -> dict[str, Any]:
    """Build one canonical episode record using the supplied environment hooks.

    Returns:
        Episode record with the established v1 schema and metadata contract.
    """
    episode_id = hooks.episode_id_from_job(job)
    horizon = hooks.resolve_horizon(job, cfg)
    if bool(getattr(cfg, "fast_stub", False)):
        record = hooks.make_stub_episode_record(
            job,
            cfg,
            episode_id=episode_id,
            horizon=horizon,
        )
        hooks.ensure_algo_metadata(
            record,
            algo=getattr(cfg, "algo", None),
            episode_id=episode_id,
        )
        ensure_metric_parameters(record)
        return record

    scenario = hooks.require_job_scenario(job, episode_id=episode_id)
    runtime = hooks.orchestrate_real_episode(
        job,
        cfg,
        episode_id=episode_id,
        scenario=scenario,
        horizon=horizon,
    )
    metrics = runtime["metrics"]
    steps_taken = int(runtime["steps_taken"])
    wall_time = float(runtime["wall_time"])
    start_time = float(runtime["start_time"])
    termination_reason, outcome, contradictions = hooks.termination_payload_from_metrics(metrics)
    record: dict[str, Any] = {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": job.scenario_id,
        "seed": job.seed,
        "archetype": job.archetype,
        "density": job.density,
        "status": status_from_termination_reason(termination_reason),
        "termination_reason": termination_reason,
        "outcome": outcome,
        "integrity": {"contradictions": contradictions},
        "metrics": metrics,
        "steps": steps_taken,
        "horizon": horizon,
        "wall_time_sec": wall_time,
        "created_at": start_time,
        "timing": {
            "steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0,
        },
        "scenario_params": {
            "archetype": job.archetype,
            "density": job.density,
            "max_episode_steps": horizon,
            "scenario_id": job.scenario_id,
            "map_file": getattr(scenario, "map_path", ""),
            "simulation_config": getattr(scenario, "raw", {}).get("simulation_config", {}),
            "metadata": getattr(scenario, "raw", {}).get("metadata", {}),
            "hash_fragment": getattr(scenario, "hash_fragment", ""),
        },
    }
    replay_capture = runtime["replay_capture"]
    if bool(getattr(cfg, "capture_replay", False)) and isinstance(replay_capture, ReplayCapture):
        hooks.attach_replay_payload(
            record,
            scenario=scenario,
            replay_capture=replay_capture,
            ped_forces=runtime["ped_forces"],
        )
    hooks.ensure_algo_metadata(
        record,
        algo=getattr(cfg, "algo", None),
        episode_id=episode_id,
    )
    ensure_metric_parameters(record)
    return record


__all__ = ["EpisodeExecutionHooks", "execute_episode_record"]
