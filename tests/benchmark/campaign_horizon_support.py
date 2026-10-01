"""Shared immutable inputs and a non-stepping production context resolver."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


TEMPLATE = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)


AUTHORED_BUDGETS = {
    "classic_bottleneck_high": 500,
    "classic_bottleneck_low": 500,
    "classic_bottleneck_medium": 500,
    "classic_cross_trap_high": 600,
    "classic_cross_trap_low": 600,
    "classic_cross_trap_medium": 600,
    "classic_doorway_high": 500,
    "classic_doorway_low": 500,
    "classic_doorway_medium": 500,
    "classic_group_crossing_high": 500,
    "classic_group_crossing_low": 500,
    "classic_group_crossing_medium": 500,
    "classic_head_on_corridor_low": 500,
    "classic_head_on_corridor_medium": 500,
    "classic_merging_low": 600,
    "classic_merging_medium": 600,
    "classic_overtaking_low": 600,
    "classic_overtaking_medium": 600,
    "classic_realworld_double_bottleneck_high": 700,
    "classic_station_platform_medium": 650,
    "classic_t_intersection_low": 500,
    "classic_t_intersection_medium": 500,
    "classic_urban_crossing_medium": 600,
    "francis2023_accompanying_peer": 400,
    "francis2023_blind_corner": 400,
    "francis2023_circular_crossing": 400,
    "francis2023_crowd_navigation": 400,
    "francis2023_down_path": 400,
    "francis2023_entering_elevator": 400,
    "francis2023_entering_room": 400,
    "francis2023_exiting_elevator": 400,
    "francis2023_exiting_room": 400,
    "francis2023_following_human": 400,
    "francis2023_frontal_approach": 400,
    "francis2023_intersection_no_gesture": 400,
    "francis2023_intersection_proceed": 400,
    "francis2023_intersection_wait": 400,
    "francis2023_join_group": 400,
    "francis2023_leading_human": 400,
    "francis2023_leave_group": 400,
    "francis2023_narrow_doorway": 400,
    "francis2023_narrow_hallway": 400,
    "francis2023_parallel_traffic": 400,
    "francis2023_pedestrian_obstruction": 400,
    "francis2023_pedestrian_overtaking": 400,
    "francis2023_perpendicular_traffic": 400,
    "francis2023_robot_crowding": 400,
    "francis2023_robot_overtaking": 400,
}


def _scheduled_context(scenario, dt):
    """Resolve the production runner context without executing an episode."""
    from robot_sf.benchmark.map_runner.map_runner_episode import _resolve_episode_run_context

    return _resolve_episode_run_context(
        scenario=scenario,
        seed=1001,
        horizon=0,
        dt=dt,
        algo="goal",
        scenario_path=ROOT / "scoped_scenarios.json",
        algo_config=None,
        algo_config_path=None,
        experimental_ped_impact=False,
        ped_impact_radius_m=2.0,
        ped_impact_window_steps=5,
        observation_mode=None,
        observation_level=None,
        benchmark_track=None,
        track_schema_version=None,
        observation_noise=None,
        tracking_precision=None,
        synthetic_actuation_profile=None,
        latency_stress_profile=None,
        safety_wrapper=None,
        cbf_safety_filter=None,
    )
