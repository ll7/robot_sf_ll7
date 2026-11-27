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
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_episode import _bind_episode_horizon
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    _build_socnav_config,
    resolve_episode_policy_runtime,
)
from robot_sf.benchmark.utils import _git_hash_fallback
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.errors import RobotSfError
from robot_sf.planner.guarded_ppo import build_guarded_ppo_config
from robot_sf.planner.hybrid_rule_local_planner import (
    PROGRESS_HISTORY_SAMPLE_CAPACITY,
    build_hybrid_rule_local_planner_config,
)
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
CAMPAIGN_SECONDS = 60.0


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


def _predictive_timing(algo: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Report the forecast and control grids for the two learned predictors.

    Returns:
        Forecast model and planned grids.
    """
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
        segment = _grid(
            cfg.predictive_rollout_dt, math.ceil(cfg.predictive_horizon_steps / segments)
        )
        timing["full_sequence_segment_s"] = segment["horizon_s"]
        timing["full_sequence_segment"] = segment
    if mppi is not None:
        timing["first_step_clearance_check"] = _grid(mppi.rollout_dt, 1)
    return timing


def _planner_timing(algo: str, raw: dict[str, Any], dt: float) -> dict[str, Any]:
    """Describe the timing used by each resolved planner in the reference roster.

    Returns:
        Planner-specific timing and model bindings.
    """
    if algo in {"prediction_planner", "predictive_mppi"}:
        return _predictive_timing(algo, raw)
    if algo == "risk_dwa":
        return _dwa_timing(raw)
    if algo == "guarded_ppo":
        cfg = build_guarded_ppo_config(raw)
        return {
            "guard_rollout": _grid(cfg.rollout_dt, cfg.rollout_steps),
            "first_step_clearance_check": _grid(cfg.rollout_dt, 1),
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
            "plant_clearance_checks_enabled": hybrid.physical_static_exclusion_enabled,
            "hard_collision_horizon_s": hybrid.hard_collision_horizon,
            "soft_prediction_horizon_s": hybrid.soft_prediction_horizon,
            "step_timers": {
                "oscillation_history": {
                    **_grid(dt, max(hybrid.oscillation_window, 2)),
                    "weight": hybrid.oscillation_weight,
                },
                "heading_smoothness": {
                    **_grid(dt, 1),
                    "weight": hybrid.heading_smoothness_weight,
                },
                "braking_reaction": {
                    **_grid(dt, 1),
                    "speed_cap_reaction_s": dt
                    if hybrid.v4_reaction_time < 0
                    else hybrid.v4_reaction_time,
                },
                "progress_history_capacity": {
                    **_grid(dt, PROGRESS_HISTORY_SAMPLE_CAPACITY - 1),
                    "capacity_samples": PROGRESS_HISTORY_SAMPLE_CAPACITY,
                    "progress_windows_s": [1, 3, 5],
                },
                "route_trace_recovery_hold": {
                    **_grid(dt, max(hybrid.route_trace_recovery_hold_steps, 0)),
                    "enabled": hybrid.route_trace_recovery_enabled,
                },
            },
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
            "step_timers": {
                "stall_before_commit": _grid(dt, cfg.orca_stall_cycles_before_commit),
                "commit_persistence": _grid(dt, max(cfg.orca_commit_persistence_steps, 1)),
                "stall_progress_sample": {
                    **_grid(dt, 1),
                    "progress_threshold_m": cfg.orca_stall_progress_epsilon,
                },
            },
        }
    if algo == "social_force":
        return {"steering_filter_time_constant_s": _build_socnav_config(raw).social_force_tau}
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
    timing = _planner_timing(algo, raw, dt)
    timing.setdefault("step_timers", {})
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
        "planner_timing": timing,
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


def _witness_result(
    summary: dict[str, Any], row: dict[str, Any], dt: float, expected: dict[str, Any]
) -> dict[str, Any]:
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
    model: dict[str, Any] = {}
    if foresight or checkpoint:
        model = {
            "id": foresight.get("requested_model_id") or checkpoint.get("model_id"),
            "load_status": foresight.get("load_status") or checkpoint.get("load_status"),
            "sha256": foresight.get("observed_checkpoint_sha256")
            or checkpoint.get("checkpoint_sha256"),
        }
        result["model"] = model
    result.update(
        algorithm=algo_metadata.get("canonical_algorithm"),
        budget_steps=row.get("effective_budget_steps"),
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
    timing = expected["planner_timing"]
    expected_model = timing.get("predictor") or timing.get("value_model") or expected["model_id"]
    if (
        result["algorithm"] != expected["algorithm"]
        or model.get("id") != expected_model
        or (expected_model and model.get("load_status") != "loaded")
    ):
        return {
            **result,
            "status": "binding_mismatch",
            "reason": f"Expected algorithm {expected['algorithm']} and model {expected_model}",
        }
    if "forecast" in timing:
        result["forecast"] = foresight.get("forecast_grid")
        if result["forecast"] != {key: timing["forecast"][key] for key in ("interval_s", "steps")}:
            return {**result, "status": "timing_mismatch", "reason": "Forecast grid differs"}
    if not (
        result["dt_s"] == trace["dt"] == dt
        and row["horizon"] == result["budget_steps"] == round(WITNESS_SECONDS / dt)
        and len(timestamps) == row["steps"] == result["budget_steps"]
        and all(math.isclose(t, index * dt) for index, t in enumerate(timestamps, start=1))
    ):
        return {
            **result,
            "status": "timing_mismatch",
            "reason": "Resolved step or time grid differs",
        }
    return result


def _campaign_timing_mismatch(
    cfg: CampaignConfig, scenarios: list[dict[str, Any]], scenario_path: Path
) -> str | None:
    """Check the full runner and simulator limits before shortening the witness.

    Returns:
        The first inconsistent scenario binding, or None if all match.
    """
    dt, horizon = cfg.dt, cfg.horizon
    if dt is None or dt <= 0 or horizon is None or not math.isclose(dt * horizon, CAMPAIGN_SECONDS):
        return "Campaign cap differs from the expected 60 s"

    for kinematics in cfg.kinematics_matrix:
        for scenario in scenarios:
            cap = scenario.get("simulation_config", {}).get("max_episode_steps")
            if cap != horizon:
                return f"Scenario {scenario['name']} uses {cap} steps, expected {horizon}"
            scoped = _scenario_with_kinematics(
                scenario,
                kinematics=kinematics,
                holonomic_command_mode=cfg.holonomic_command_mode,
            )
            try:
                env = build_env_config(scoped, scenario_path=scenario_path)
                sim = env.sim_config
                # The runner's explicit dt takes precedence over the scenario default.
                sim.time_per_step_in_secs = dt
                runner_cap = _bind_episode_horizon(scoped, env, horizon, horizon, horizon)
            except ValueError as exc:
                return f"Scenario {scenario['name']} could not bind the expected limit: {exc}"
            if not (
                sim.time_per_step_in_secs == cfg.dt
                and sim.episode_step_limit == sim.max_sim_steps == runner_cap == cfg.horizon
                and math.isclose(sim.sim_time_in_secs, CAMPAIGN_SECONDS)
            ):
                return (
                    f"Scenario {scenario['name']} has an inconsistent runner or simulator limit, "
                    f"expected {horizon} steps at {cfg.dt} s"
                )
    return None


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
    scenario_path = root / "scoped_scenarios.json"
    campaign_mismatch = _campaign_timing_mismatch(cfg, scenarios, scenario_path)
    results = []
    for planner in (p for p in cfg.planners if p.enabled):
        path = planner.algo_config_path
        output = output_dir / cfg.name / planner.key / "episodes.jsonl"
        identity: dict[str, Any] = {
            "planner": planner.key,
            "scenario": WITNESS_SCENARIO,
            "seed": seed,
        }
        if campaign_mismatch is not None:
            results.append({**identity, "status": "timing_mismatch", "reason": campaign_mismatch})
            continue
        dt = _step_size(cfg, planner)
        horizon = cfg.horizon if planner.horizon_override is None else planner.horizon_override
        if dt != cfg.dt or horizon is None or horizon != cfg.horizon:
            results.append(
                {
                    **identity,
                    "status": "timing_mismatch",
                    "reason": f"Planner uses {dt} s and {horizon} steps, expected {cfg.dt} s and {cfg.horizon} steps",
                }
            )
            continue
        identity["campaign_grid"] = _grid(dt, horizon)
        steps = round(WITNESS_SECONDS / dt)
        scoped = _scenario_with_kinematics(
            scenario,
            kinematics=cfg.kinematics_matrix[0],
            holonomic_command_mode=cfg.holonomic_command_mode,
        )
        scoped["seeds"] = [seed]
        expected = None
        try:
            expected = _resolved_arm(planner, scoped, cfg, cfg.kinematics_matrix[0])
            summary = run_map_batch(
                _apply_fixed_campaign_horizon(
                    [scoped], horizon=steps, protocol_version=cfg.protocol_version
                ),
                output,
                root / DEFAULT_EPISODE_SCHEMA_PATH,
                scenario_path=scenario_path,
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
            result = _witness_result(summary, row, dt, expected)
        except (
            LookupError,
            OSError,
            RobotSfError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            result = {"status": "refused", "reason": f"{type(exc).__name__}: {exc}"}
        if expected is not None:
            result["planner_timing"] = expected["planner_timing"]
        results.append({**identity, **result})
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
