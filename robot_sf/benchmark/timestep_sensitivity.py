"""Expand or witness the frozen #10516 development comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from robot_sf.benchmark.camera_ready._config import (
    _apply_fixed_campaign_horizon,
    _load_campaign_scenarios,
    _scenario_with_kinematics,
    load_campaign_config,
)
from robot_sf.benchmark.camera_ready.campaign import DEFAULT_EPISODE_SCHEMA_PATH
from robot_sf.benchmark.constants import NEAR_MISS_DIST
from robot_sf.benchmark.fallback_policy import summarize_benchmark_availability
from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.benchmark.map_runner.map_runner_batch_plan import build_seed_jobs
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    _build_socnav_config,
    resolve_episode_policy_runtime,
)
from robot_sf.benchmark.utils import _git_hash_fallback
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.errors import RobotSfError
from robot_sf.planner.guarded_ppo import build_guarded_ppo_config
from robot_sf.planner.hybrid_rule_local_planner import build_hybrid_rule_local_planner_config
from robot_sf.planner.predictive_mppi import build_predictive_mppi_config
from robot_sf.planner.risk_dwa import build_risk_dwa_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.benchmark.algorithm_readiness import BenchmarkProfile
    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

CONFIG_PATTERN = "configs/benchmarks/issue_10516_timestep_dev_v1_{ms:03d}ms.yaml"
STEP_SIZES_MS = (50, 100, 200)
WITNESS_SCENARIO = "classic_urban_crossing_medium"
WITNESS_SECONDS = 0.4


def _grid(interval_s: float, steps: int) -> dict[str, float | int]:
    """Return the interval, number of samples and duration of a time grid.

    Returns:
        Time grid with the interval and duration in seconds.
    """
    return {"interval_s": interval_s, "steps": steps, "horizon_s": interval_s * steps}


def _dwa_timing(raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve the two time bases used by RiskDWA.

    Returns:
        Sampled control grid and dynamic-window interval.
    """
    cfg = build_risk_dwa_config(raw)
    return {
        "sampled_controls": _grid(cfg.rollout_dt, cfg.rollout_steps),
        "dynamic_window_interval_s": cfg.dynamic_window_dt,
    }


def _planner_timing(algo: str, raw: dict[str, Any], dt: float) -> dict[str, Any]:
    """Describe the timing used by each resolved planner in the reference roster.

    Returns:
        Planner-specific timing and model bindings.
    """
    if algo in {"prediction_planner", "predictive_mppi"}:
        mppi = build_predictive_mppi_config(raw) if algo == "predictive_mppi" else None
        cfg = mppi.socnav if mppi is not None else _build_socnav_config(raw)
        forecast = _grid(cfg.predictive_rollout_dt, cfg.predictive_horizon_steps)
        timing = {
            "predictor": cfg.predictive_model_id,
            "forecast": forecast,
            "sampled_controls": (
                _grid(mppi.rollout_dt, mppi.horizon_steps) if mppi is not None else forecast
            ),
        }
        if cfg.predictive_sequence_search_enabled:
            segments = cfg.predictive_sequence_segments
            timing["full_sequence_segment_s"] = (
                math.ceil(cfg.predictive_horizon_steps / segments) * cfg.predictive_rollout_dt
            )
        return timing
    if algo == "risk_dwa":
        return _dwa_timing(raw)
    if algo == "guarded_ppo":
        cfg = build_guarded_ppo_config(raw)
        return {
            "guard_rollout": _grid(cfg.rollout_dt, cfg.rollout_steps),
            "guard_alternate_controller": _dwa_timing(raw.get("fallback_risk_dwa", {})),
        }
    if algo == "hybrid_rule_local_planner":
        hybrid = build_hybrid_rule_local_planner_config(raw)
        return {
            "control_segment_s": hybrid.control_period,
            "sampled_controls": _grid(
                hybrid.rollout_dt, math.ceil(hybrid.rollout_horizon / hybrid.rollout_dt)
            ),
            "plant_clearance_sample_interval_s": dt,
            "hard_collision_horizon_s": hybrid.hard_collision_horizon,
            "soft_prediction_horizon_s": hybrid.soft_prediction_horizon,
        }
    if algo == "socnav_sampling":
        horizon = _build_socnav_config(raw).sampling_horizon_s
        return {"sampled_controls": _grid(dt, math.ceil(horizon / dt))}
    if algo == "sacadrl":
        return {
            "value_model": _build_socnav_config(raw).sacadrl_model_id,
            "heading_increment_interval_s": dt,
        }
    if algo == "orca":
        cfg = _build_socnav_config(raw)
        return {
            "solve_interval_s": dt,
            "pedestrian_horizon_s": cfg.orca_time_horizon,
            "obstacle_horizon_s": cfg.orca_time_horizon_obst,
        }
    return {}


def _step_size(cfg: CampaignConfig, planner: PlannerSpec) -> float:
    """Read the step size from the declared campaign and planner settings.

    Returns:
        The effective step size in seconds.
    """
    if cfg.dt is None:
        raise ValueError("The time-step study requires an explicit campaign dt")
    return cfg.dt if planner.dt_override is None else planner.dt_override


def _resolved_arm(
    planner: PlannerSpec,
    scenario: dict[str, Any],
    cfg: CampaignConfig,
    kinematics: str,
) -> dict[str, Any]:
    """Record the binding for one planner, scenario and robot kinematics.

    Returns:
        Resolved planner, scenario, seeds, and timing metadata.
    """
    path = planner.algo_config_path
    algo, raw = resolve_episode_policy_runtime(
        default_algo=planner.algo,
        algo_config_path=str(path) if path is not None else None,
        scenario=scenario,
        seed=scenario["seeds"][0],
    )
    dt = _step_size(cfg, planner)
    steps = scenario["simulation_config"]["max_episode_steps"]
    if planner.horizon_override is not None:
        steps = planner.horizon_override
    return {
        "planner": planner.key,
        "algorithm": algo,
        "planner_config": path.relative_to(get_repository_root()).as_posix() if path else None,
        "model_id": raw.get("model_id"),
        "scenario": scenario["name"],
        "kinematics": kinematics,
        "seeds": scenario["seeds"],
        "simulation": _grid(dt, steps),
        "controller_interval_s": dt,
        "close_clearance_sample_interval_s": dt,
        "planner_timing": _planner_timing(algo, raw, dt),
    }


def expand_timestep_campaign(
    cfg: CampaignConfig, scenarios: list[dict[str, Any]]
) -> dict[str, Any]:
    """Expand one development config using the campaign and map-runner bindings.

    Returns:
        Campaign identity, episode count and expanded arm bindings.
    """
    jobs = build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")
    arms = [
        _resolved_arm(planner, scenario, cfg, kinematics)
        for planner in cfg.planners
        if planner.enabled
        for kinematics in cfg.kinematics_matrix
        for scenario in scenarios
    ]
    return {
        "campaign": cfg.name,
        "config_sha256": cfg.source_config_sha256,
        "analysis_only": not cfg.paper_facing,
        "episode_count": len(jobs)
        * len(cfg.kinematics_matrix)
        * sum(p.enabled for p in cfg.planners),
        "arms": arms,
    }


def _witness_result(summary: dict[str, Any], row: dict[str, Any], dt: float) -> dict[str, Any]:
    """Describe the witnessed runtime and retain any refusal or degraded status.

    Returns:
        Runtime timing and availability for one witnessed episode.
    """
    availability = summarize_benchmark_availability(summary)
    result = {
        "execution_mode": availability.execution_mode,
        "status": "qualified",
        "reason": None,
    }
    if not availability.benchmark_success or not row:
        return {
            **result,
            "status": "refused",
            "reason": availability.availability_reason or "No episode was written",
        }
    integrity = row["integrity"]
    algo_metadata = row["algorithm_metadata"]
    trace = algo_metadata["simulation_step_trace"]
    timestamps = [step["time_s"] for step in trace["steps"]]
    foresight = algo_metadata.get("foresight_prediction") or {}
    checkpoint = algo_metadata.get("planner_runtime", {}).get("checkpoint_provenance") or {}
    if foresight or checkpoint:
        result["model"] = {
            "id": foresight.get("requested_model_id") or checkpoint.get("model_id"),
            "load_status": foresight.get("load_status") or checkpoint.get("load_status"),
            "sha256": foresight.get("observed_checkpoint_sha256")
            or checkpoint.get("checkpoint_sha256"),
        }
    result.update(
        steps=row["steps"],
        timestamps_s=timestamps,
        dt_s=row["effective_physics"]["integration"]["dt_s"],
        outcome=row["outcome"],
        close_clearance_s=row["metrics"]["near_misses"] * dt,
    )
    if integrity["effective_view"]["degraded"] or integrity.get("contradictions"):
        return {**result, "status": "degraded", "reason": "Episode integrity"}
    if row["spawn_validity"]["invalid_run"]:
        return {**result, "status": "invalid_spawn", "reason": "Invalid spawn"}
    if not (
        result["dt_s"] == trace["dt"] == dt
        and len(timestamps) == row["steps"] == round(WITNESS_SECONDS / dt)
        and all(math.isclose(t, index * dt) for index, t in enumerate(timestamps, start=1))
    ):
        return {
            **result,
            "status": "timing_mismatch",
            "reason": "Resolved step or time grid differs",
        }
    return result


def _run_witness(
    cfg: CampaignConfig, scenarios: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    """Run the real map batch for one matched episode per planner.

    Returns:
        One result per enabled planner, including refusals.
    """
    root = get_repository_root()
    scenario = next(s for s in scenarios if s["name"] == WITNESS_SCENARIO)
    seed = scenario["seeds"][0]
    results = []
    for planner in (p for p in cfg.planners if p.enabled):
        path = planner.algo_config_path
        output = output_dir / cfg.name / planner.key / "episodes.jsonl"
        dt = _step_size(cfg, planner)
        steps = round(WITNESS_SECONDS / dt)
        scoped = _scenario_with_kinematics(
            scenario,
            kinematics=cfg.kinematics_matrix[0],
            holonomic_command_mode=cfg.holonomic_command_mode,
        )
        scoped["seeds"] = [seed]
        try:
            summary = run_map_batch(
                _apply_fixed_campaign_horizon(
                    [scoped], horizon=steps, protocol_version=cfg.protocol_version
                ),
                output,
                root / DEFAULT_EPISODE_SCHEMA_PATH,
                scenario_path=root / "scoped_scenarios.json",
                provenance_scenario_path=cfg.scenario_matrix_path,
                horizon=steps,
                dt=dt,
                record_forces=cfg.record_forces,
                algo=planner.algo,
                algo_config_path=str(path) if path is not None else None,
                benchmark_profile=cast("BenchmarkProfile", planner.benchmark_profile),
                socnav_missing_prereq_policy="fail-fast",
                adapter_impact_eval=planner.adapter_impact_eval,
                record_simulation_step_trace=True,
                workers=1,
                resume=False,
            )
            rows = output.read_text(encoding="utf-8").splitlines() if output.is_file() else []
            row = json.loads(rows[0]) if len(rows) == 1 else {}
            result = _witness_result(summary, row, dt)
        except (
            ImportError,
            LookupError,
            OSError,
            RobotSfError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            result = {"status": "refused", "reason": f"{type(exc).__name__}: {exc}"}
        results.append(
            {"planner": planner.key, "scenario": WITNESS_SCENARIO, "seed": seed, **result}
        )
    return results


def main(argv: Sequence[str] | None = None) -> None:
    """Write the expansion and optional native planning witness as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--witness", action="store_true", help="Run short planning episodes.")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "timestep-sensitivity.v1",
        "source_commit": _git_hash_fallback(),
        "metrics": {
            "success": "outcome.route_complete and no outcome.collision_event",
            "collision_episode": "outcome.collision_event",
            "close_clearance_s": "metrics.near_misses * simulation.interval_s",
            "close_clearance_range_m": {"inclusive_min": 0.0, "exclusive_max": NEAR_MISS_DIST},
        },
        "campaigns": [],
    }
    for ms in STEP_SIZES_MS:
        cfg = load_campaign_config(get_repository_root() / CONFIG_PATTERN.format(ms=ms))
        scenarios = _load_campaign_scenarios(cfg)
        campaign = expand_timestep_campaign(cfg, scenarios)
        if args.witness:
            campaign["witness"] = _run_witness(cfg, scenarios, args.output_dir)
        manifest["campaigns"].append(campaign)
    path = args.output_dir / "timestep_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
