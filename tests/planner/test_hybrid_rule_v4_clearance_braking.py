"""Tests for hybrid v4 clearance- and braking-consistent speed limits (issue #9726)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.planner.hybrid_rule_local_planner import (
    HYBRID_RULE_V4_CLEARANCE_BRAKING_VARIANT,
    HybridRuleCandidate,
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    braking_speed_limit,
    build_hybrid_rule_local_planner_config,
    stopping_distance,
)
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from tests.planner.hybrid_rule_closed_loop_helpers import (
    ENCOUNTERS,
    PED_RADIUS,
    ROBOT_RADIUS,
    load_planner_config,
    repo_root,
    run_closed_loop,
)

CANDIDATES = "configs/policy_search/candidates/"
V4_BASE = "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
V3_TO_V4 = {
    "hybrid_rule_v3_fast_progress_static_escape_s30_h600_release.yaml": (
        "hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml"
    ),
    "hybrid_rule_v3_fast_progress_static_escape_continuous_s30_h600_release.yaml": (
        "hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release.yaml"
    ),
    "scenario_adaptive_hybrid_orca_v2_collision_guard_s30_h600_release.yaml": (
        "scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release.yaml"
    ),
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_s30_h600_release.yaml": (
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4_s30_h600_release.yaml"
    ),
}
V3_IDENTITY_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hybrid_rule_v3_identity_issue_9726.json"
)
BENCHMARK_DT = 0.1


def _v4_planner(**overrides) -> HybridRuleLocalPlannerAdapter:
    cfg = load_planner_config(V4_BASE, "classic_doorway_high")
    cfg.update(overrides)
    return HybridRuleLocalPlannerAdapter(build_hybrid_rule_local_planner_config(cfg))


def _state(
    *,
    clearance: float,
    speed: float = 0.0,
    ped_velocity: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """Planner state with one pedestrian straight ahead at a given surface clearance."""
    centre = clearance + ROBOT_RADIUS + PED_RADIUS
    return {
        "robot_pos": np.array([0.0, 0.0]),
        "heading": 0.0,
        "current_speed": float(speed),
        "goal": np.array([20.0, 0.0]),
        "ped_pos": np.array([[centre, 0.0]]),
        "ped_vel": np.array([ped_velocity], dtype=float),
        "robot_radius": ROBOT_RADIUS,
        "ped_radius": PED_RADIUS,
        "dt": BENCHMARK_DT,
        "observation": {},
    }


def _bind(planner: HybridRuleLocalPlannerAdapter, robot_config) -> None:
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[SimpleNamespace(config=robot_config)]))
    planner.bind_env(env)


# ---------------------------------------------------------------------------
# Stopping-distance model and threshold derivation
# ---------------------------------------------------------------------------


def test_braking_speed_limit_inverts_stopping_distance() -> None:
    """The braking cap is the exact inverse of the stopping-distance model."""
    for decel in (0.5, 1.0, 2.5):
        for reaction in (0.0, 0.1, 0.3):
            for speed in (0.0, 0.15, 0.6, 1.3, 2.0):
                distance = stopping_distance(speed, decel, reaction)
                assert braking_speed_limit(distance, decel, reaction) == pytest.approx(speed)
    assert braking_speed_limit(0.0, 1.0, 0.1) == 0.0
    assert braking_speed_limit(-1.0, 1.0, 0.1) == 0.0


def test_default_v4_levels_are_derived_from_stopping_distance() -> None:
    """Each default level lies beyond the stopping distance of the speed allowed above it."""
    cfg = HybridRuleLocalPlannerConfig(planner_variant=HYBRID_RULE_V4_CLEARANCE_BRAKING_VARIANT)
    drive = DifferentialDriveSettings()
    decel = float(drive.max_linear_decel)
    margin = cfg.v4_braking_margin
    # Values quoted in the config comments.
    assert stopping_distance(cfg.very_slow_speed, decel, BENCHMARK_DT) + margin == pytest.approx(
        0.12625
    )
    assert stopping_distance(cfg.moderate_speed, decel, BENCHMARK_DT) + margin == pytest.approx(
        0.34
    )
    assert stopping_distance(drive.max_linear_speed, decel, BENCHMARK_DT) + margin == (
        pytest.approx(2.3)
    )
    assert (
        stopping_distance(cfg.very_slow_speed, decel, BENCHMARK_DT) + margin
        <= cfg.v4_stop_clearance_human
    )
    assert (
        stopping_distance(cfg.moderate_speed, decel, BENCHMARK_DT) + margin
        <= cfg.v4_slow_clearance_human
    )
    assert cfg.v4_stop_clearance_human < cfg.v4_slow_clearance_human
    assert cfg.v4_slow_clearance_human < cfg.v4_moderate_clearance_human


@pytest.mark.parametrize(
    ("accel", "decel"),
    [(1.0, 1.0), (1.0, 0.5), (2.0, 3.0)],
)
def test_v4_speed_cap_is_stopping_distance_consistent(accel: float, decel: float) -> None:
    """At every surface clearance the cap can be braked to zero before contact.

    This is the core #9726 property: v3 capped speed on centre distance, so with
    a 1.0 m robot and 0.4 m pedestrian its stop and slow levels (0.5/1.0 m
    centre distance) could only fire after contact at 1.4 m.
    """
    planner = _v4_planner()
    _bind(
        planner,
        DifferentialDriveSettings(max_linear_accel=accel, max_linear_decel=decel),
    )
    margin = planner.config.v4_braking_margin
    for clearance in np.linspace(0.0, 6.0, 241):
        cap = planner._v4_human_speed_cap(_state(clearance=float(clearance)))
        needed = stopping_distance(cap, decel, BENCHMARK_DT)
        if cap > 0.0:
            assert needed + margin <= clearance + 1e-9, (clearance, cap, needed)
        if clearance < planner.config.v4_stop_clearance_human:
            assert cap == 0.0
    # Far from pedestrians the drive maximum is reachable.
    assert planner._v4_human_speed_cap(_state(clearance=50.0)) == pytest.approx(2.0)


def test_v4_levels_trigger_before_contact_for_benchmark_radii() -> None:
    """Stop, slow and moderate levels all fire at positive surface clearance."""
    planner = _v4_planner()
    _bind(planner, DifferentialDriveSettings())
    levels = {}
    for clearance in (0.05, 0.2, 0.5, 1.0, 3.0):
        planner._v4_human_speed_cap(_state(clearance=clearance))
        levels[clearance] = planner._last_v4_speed_safety["level"]
    assert levels == {
        0.05: "stop",
        0.2: "slow",
        0.5: "moderate",
        1.0: "braking",
        3.0: "braking",
    }
    # v3's stop/slow levels (0.5/1.0 m centre distance) cannot fire before
    # contact at 1.4 m: 0.05 m from contact it still allows moderate_speed.
    v3 = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(
            load_planner_config("configs/algos/hybrid_rule_v3_teb_like_rollout.yaml", "x")
        )
    )
    assert v3._human_speed_cap(0.05 + ROBOT_RADIUS + PED_RADIUS) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Drive limits, rollout and braking check
# ---------------------------------------------------------------------------


def test_v4_reads_drive_limits_from_bound_environment() -> None:
    """Rollout limits come from the robot drive config, not the planner YAML."""
    planner = _v4_planner(max_linear_accel=3.0, max_linear_decel=2.5, max_linear_speed=3.0)
    assert planner._v4_drive_limits()["source"] == "differential_drive_settings_defaults"
    _bind(
        planner,
        DifferentialDriveSettings(max_linear_accel=0.8, max_linear_decel=0.6, max_linear_speed=1.5),
    )
    limits = planner._v4_drive_limits()
    assert limits["max_linear_accel"] == pytest.approx(0.8)
    assert limits["max_linear_decel"] == pytest.approx(0.6)
    assert limits["source"] == "env_robot_config:DifferentialDriveSettings"
    assert planner._v4_effective_max_speed() == pytest.approx(1.5)
    # Planner-config accelerations are ignored by the v4 dynamic window.
    v_min, v_max, _w_min, _w_max = planner._dynamic_window(1.0, 3.0)
    period = planner.config.control_period
    assert v_min == pytest.approx(1.0 - 0.6 * period)
    assert v_max == pytest.approx(1.0 + 0.8 * period)

    _bind(planner, BicycleDriveSettings(max_accel=0.7, max_decel=0.4, max_velocity=2.5))
    limits = planner._v4_drive_limits()
    assert (limits["max_linear_accel"], limits["max_linear_decel"]) == pytest.approx((0.7, 0.4))
    assert limits["source"] == "env_robot_config:BicycleDriveSettings+default_angular_limits"


def test_v4_rollout_follows_drive_braking_limit() -> None:
    """A stop candidate at 2.0 m/s is predicted to decelerate at the drive limit."""
    planner = _v4_planner()
    _bind(planner, DifferentialDriveSettings())
    commands = [(0.0, 0.0)] * 8
    realized = planner._v4_realized_rollout_commands(commands, current_speed=2.0, dt=0.2)
    assert [round(v, 6) for v, _ in realized] == [1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]
    speeding = planner._v4_realized_rollout_commands([(3.0, 0.0)] * 3, current_speed=0.0, dt=0.2)
    assert [round(v, 6) for v, _ in speeding] == [0.2, 0.4, 0.6]


def test_v4_braking_check_rejects_unstoppable_candidate() -> None:
    """Doorway-high-111-like state: 1.45 m/s toward a pedestrian closing at 0.63 m/s.

    Braking from 1.45 m/s at 1.0 m/s^2 needs about 1.2 m, and the pedestrian
    closes about 1.0 m meanwhile; with 2.05 m surface clearance full speed must
    be rejected while braking is still feasible.
    """
    planner = _v4_planner()
    _bind(planner, DifferentialDriveSettings())
    state = _state(clearance=2.05, speed=1.45, ped_velocity=(-0.63, 0.0))
    radius = ROBOT_RADIUS + PED_RADIUS + planner.config.hard_safety_margin
    keep_speed = planner._v4_braking_rejection(
        candidate=HybridRuleCandidate(2.0, 0.0, "dynamic_window"),
        state=state,
        collision_radius=radius,
    )
    assert keep_speed is not None
    assert keep_speed["reason"] == "braking_infeasible"
    brake = planner._v4_braking_rejection(
        candidate=HybridRuleCandidate(0.0, 0.0, "stop"),
        state=state,
        collision_radius=radius,
    )
    assert brake is None


def test_v4_braking_check_ignores_contact_after_robot_stopped() -> None:
    """A pedestrian walking into an already stopped robot does not block standing still."""
    planner = _v4_planner()
    _bind(planner, DifferentialDriveSettings())
    state = _state(clearance=0.5, speed=0.0, ped_velocity=(-1.0, 0.0))
    radius = ROBOT_RADIUS + PED_RADIUS + planner.config.hard_safety_margin
    assert (
        planner._v4_braking_rejection(
            candidate=HybridRuleCandidate(0.0, 0.3, "rotate_left"),
            state=state,
            collision_radius=radius,
        )
        is None
    )


def test_v4_decision_and_episode_metadata_record_speed_safety() -> None:
    """v4 records its speed-safety state; v3 payloads carry no new key."""
    planner = _v4_planner()
    _bind(planner, DifferentialDriveSettings())
    planner.plan(
        {
            "robot": {
                "position": np.array([0.0, 0.0]),
                "heading": np.array([0.0]),
                "speed": np.array([0.5]),
                "radius": np.array([ROBOT_RADIUS]),
            },
            "goal": {"current": np.array([10.0, 0.0]), "next": np.array([10.0, 0.0])},
            "pedestrians": {
                "positions": np.array([[3.0, 0.0]]),
                "velocities": np.array([[0.0, 0.0]]),
                "count": np.array([1.0]),
                "radius": PED_RADIUS,
            },
            "sim": {"timestep": BENCHMARK_DT},
        }
    )
    decision = planner.last_decision()
    assert decision["planner_variant"] == HYBRID_RULE_V4_CLEARANCE_BRAKING_VARIANT
    assert decision["speed_safety"]["min_surface_clearance"] == pytest.approx(1.6)
    diagnostics = planner.diagnostics()
    assert diagnostics["speed_safety"]["clearance_reference"] == "surface"
    assert diagnostics["speed_safety"]["drive_limits"]["max_linear_decel"] == pytest.approx(1.0)

    v3 = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(
            load_planner_config("configs/algos/hybrid_rule_v3_teb_like_rollout.yaml", "x")
        )
    )
    assert "speed_safety" not in v3.diagnostics()


# ---------------------------------------------------------------------------
# Flat map-runner observations (review of PR #9747)
# ---------------------------------------------------------------------------


def _flat_obs(
    *,
    heading: float,
    speed: float,
    ped_world_velocity: tuple[float, float],
    include_timestep: bool = True,
) -> dict:
    """Flat map-runner observation: robot at the origin, one pedestrian ahead.

    Pedestrian velocities are robot-ego-frame, as the SocNav sensor emits them.
    """
    ahead = np.array([np.cos(heading), np.sin(heading)])
    ped = ahead * (2.0 + ROBOT_RADIUS + PED_RADIUS)
    vx, vy = ped_world_velocity
    ego = (
        np.cos(heading) * vx + np.sin(heading) * vy,
        -np.sin(heading) * vx + np.cos(heading) * vy,
    )
    obs = {
        "robot_position": np.array([0.0, 0.0], dtype=np.float32),
        "robot_heading": np.array([heading], dtype=np.float32),
        "robot_speed": np.array([speed], dtype=np.float32),
        "robot_radius": np.array([ROBOT_RADIUS], dtype=np.float32),
        "goal_current": (ahead * 20.0).astype(np.float32),
        "goal_next": (ahead * 20.0).astype(np.float32),
        "pedestrians_positions": np.array([ped], dtype=np.float32),
        "pedestrians_velocities": np.array([ego], dtype=np.float32),
        "pedestrians_count": np.array([1.0], dtype=np.float32),
        "pedestrians_radius": np.array([PED_RADIUS], dtype=np.float32),
    }
    if include_timestep:
        obs["sim_timestep"] = np.array([BENCHMARK_DT], dtype=np.float32)
    return obs


def test_v4_rotates_flat_observation_velocities_to_world_frame() -> None:
    """At heading pi an approaching pedestrian must not look like it walks away."""
    planner = _v4_planner()
    _bind(planner, DifferentialDriveSettings())
    obs = _flat_obs(heading=np.pi, speed=1.5, ped_world_velocity=(1.0, 0.0))
    state = planner._extract_state(obs)
    np.testing.assert_allclose(state["ped_vel"], [[1.0, 0.0]], atol=1e-5)
    planner.plan(obs)
    rejections = planner.last_decision()["rejection_counts"]
    assert rejections.get("braking_infeasible", 0) > 0, rejections


def test_v3_flat_observation_frame_is_unchanged() -> None:
    """v3 keeps its historical flat-observation behaviour (tracked in #9752)."""
    v3 = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(
            load_planner_config("configs/algos/hybrid_rule_v3_teb_like_rollout.yaml", "x")
        )
    )
    obs = _flat_obs(heading=np.pi, speed=1.5, ped_world_velocity=(1.0, 0.0))
    np.testing.assert_allclose(v3._extract_state(obs)["ped_vel"], [[-1.0, 0.0]], atol=1e-5)


def test_v4_reads_flat_timestep_and_records_sources() -> None:
    """v4 uses sim_timestep from flat observations, else the bound simulator config."""
    planner = _v4_planner()
    env = SimpleNamespace(
        simulator=SimpleNamespace(
            robots=[SimpleNamespace(config=DifferentialDriveSettings())],
            config=SimpleNamespace(time_per_step_in_secs=0.05),
        )
    )
    planner.bind_env(env)
    state = planner._extract_state(_flat_obs(heading=0.0, speed=0.0, ped_world_velocity=(0, 0)))
    assert state["dt"] == pytest.approx(BENCHMARK_DT)
    assert planner._v4_state_sources == {
        "dt": "observation.sim_timestep",
        "robot_radius": "observation.robot_radius",
        "ped_radius": "observation.pedestrians_radius",
    }
    state = planner._extract_state(
        _flat_obs(heading=0.0, speed=0.0, ped_world_velocity=(0, 0), include_timestep=False)
    )
    assert state["dt"] == pytest.approx(0.05)
    assert planner._v4_state_sources["dt"] == "env.sim_config"
    planner.plan(_flat_obs(heading=0.0, speed=0.0, ped_world_velocity=(0, 0)))
    assert planner.last_decision()["state_sources"]["dt"] == "observation.sim_timestep"
    assert planner.last_decision()["speed_safety"]["reaction_time"] == pytest.approx(BENCHMARK_DT)


def test_v4_binds_angular_limits_to_the_drive() -> None:
    """Turn rate and angular acceleration come from DifferentialDriveSettings."""
    planner = _v4_planner()
    _bind(planner, DifferentialDriveSettings())
    assert planner._max_angular_speed() == pytest.approx(1.0)
    assert planner._max_angular_accel() == pytest.approx(1.0)
    _v_min, _v_max, w_min, w_max = planner._dynamic_window(0.0, 1.0)
    period = planner.config.control_period
    assert (w_min, w_max) == pytest.approx((-1.0 * period, 1.0 * period))
    realized = planner._v4_realized_rollout_commands([(0.0, 1.2)] * 8, current_speed=0.0, dt=0.2)
    assert [round(w, 6) for _v, w in realized] == [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
    v3 = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(
            load_planner_config("configs/algos/hybrid_rule_v3_teb_like_rollout.yaml", "x")
        )
    )
    assert v3._max_angular_speed() == pytest.approx(1.2)
    assert v3._max_angular_accel() == pytest.approx(4.0)


def test_v4_braking_check_turns_at_the_drive_angular_limit() -> None:
    """A hard swerve cannot escape a head-on pedestrian within one step at 1 rad/s^2."""
    planner = _v4_planner()
    _bind(planner, DifferentialDriveSettings())
    state = _state(clearance=1.6, speed=1.5, ped_velocity=(-0.8, 0.0))
    radius = ROBOT_RADIUS + PED_RADIUS + planner.config.hard_safety_margin
    swerve = planner._v4_braking_rejection(
        candidate=HybridRuleCandidate(1.5, 1.0, "dynamic_window"),
        state=state,
        collision_radius=radius,
    )
    assert swerve is not None and swerve["reason"] == "braking_infeasible"


def test_v4_pedestrian_gates_use_surface_clearance() -> None:
    """Corridor-subgoal and static-recovery gates compare surface clearance in v4."""
    planner = _v4_planner(recovery_enabled=True)
    _bind(planner, DifferentialDriveSettings())
    counts = Counter({"static_clearance": 3})
    # 1.5 m surface clearance is beyond the 1.0 m gate; 0.5 m is inside it.
    assert planner._static_recovery_allowed(counts, 1.5) is True
    assert planner._static_recovery_allowed(counts, 0.5) is False
    obs = _flat_obs(heading=0.0, speed=0.0, ped_world_velocity=(0, 0))
    captured = {}
    original = planner._corridor_subgoal_activation

    def _spy(**kwargs):
        captured["nearest_ped"] = kwargs["nearest_ped"]
        return original(**kwargs)

    planner._corridor_subgoal_activation = _spy
    planner.plan(obs)
    assert captured["nearest_ped"] == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Closed-loop behaviour and v3 identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encounter", sorted(ENCOUNTERS))
def test_v4_never_contacts_a_pedestrian_while_moving(encounter: str) -> None:
    """With non-reactive pedestrians, v4 contact (if any) only happens after it stopped."""
    for v4_name in V3_TO_V4.values():
        cfg = load_planner_config(CANDIDATES + v4_name, "classic_doorway_high")
        result = run_closed_loop(cfg, encounter, steps=120)
        assert result["moving_contact"] is False, (v4_name, encounter, result["min_clearance"])


def test_v3_release_configs_are_behaviourally_unchanged() -> None:
    """v3 command sequences match goldens recorded on the pre-v4 commit.

    The fixture was generated at origin/main cf33e1b51 (before the v4 code path
    existed) with ``tests/planner/hybrid_rule_closed_loop_helpers.py``.
    """
    golden = json.loads(V3_IDENTITY_FIXTURE.read_text(encoding="utf-8"))
    assert golden, "empty v3 identity fixture"
    for key, expected in golden["runs"].items():
        config_path, scenario, encounter = key.split("|")
        cfg = load_planner_config(config_path, scenario)
        result = run_closed_loop(cfg, encounter, steps=golden["steps"])
        assert result["sources"] == expected["sources"], key
        np.testing.assert_allclose(
            np.asarray(result["commands"]),
            np.asarray(expected["commands"]),
            rtol=0.0,
            atol=1e-9,
            err_msg=key,
        )


# ---------------------------------------------------------------------------
# Config and registry contracts
# ---------------------------------------------------------------------------


def _read_yaml(path: str) -> dict:
    return yaml.safe_load((repo_root() / path).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("v3_name", "v4_name"), sorted(V3_TO_V4.items()))
def test_v4_twin_configs_mirror_their_v3_release_config(v3_name: str, v4_name: str) -> None:
    """v4 twins change only the base config, speed ceiling and yielding units."""
    v3 = _read_yaml(CANDIDATES + v3_name)
    v4 = _read_yaml(CANDIDATES + v4_name)
    assert v4["name"] == v4_name.removesuffix(".yaml")
    assert v4["algo"] == v3["algo"] == "hybrid_rule_local_planner"
    assert v3["base_config_path"] == "configs/algos/hybrid_rule_v3_teb_like_rollout.yaml"
    assert v4["base_config_path"] == V4_BASE
    v3_params = dict(v3["params"])
    v4_params = dict(v4["params"])
    assert v3_params.pop("max_linear_speed") == 3.0
    assert v3_params.pop("max_linear_accel") == 3.0
    assert v4_params.pop("max_linear_speed") == 2.0
    assert v4_params == v3_params
    assert v4.get("scenario_algo_overrides") == v3.get("scenario_algo_overrides")
    assert set(v4.get("scenario_overrides", {})) == set(v3.get("scenario_overrides", {}))
    for scenario, v3_override in v3.get("scenario_overrides", {}).items():
        v4_override = dict(v4["scenario_overrides"][scenario])
        v3_override = dict(v3_override)
        if "slow_distance_human" in v3_override:
            slow = v3_override.pop("slow_distance_human")
            moderate = v3_override.pop("moderate_distance_human")
            assert v4_override.pop("v4_slow_clearance_human") == pytest.approx(
                slow - ROBOT_RADIUS - PED_RADIUS
            )
            assert v4_override.pop("v4_moderate_clearance_human") == pytest.approx(
                moderate - ROBOT_RADIUS - PED_RADIUS
            )
        assert v4_override == v3_override, scenario


@pytest.mark.parametrize("v4_name", sorted(V3_TO_V4.values()))
def test_v4_configs_resolve_to_the_v4_variant(v4_name: str) -> None:
    """Every v4 twin builds a v4 planner for every overridden scenario."""
    manifest = _read_yaml(CANDIDATES + v4_name)
    scenarios = ["classic_doorway_high", *manifest.get("scenario_overrides", {})]
    for scenario in scenarios:
        _algo, effective = resolve_candidate_manifest_runtime(
            default_algo="hybrid_rule_local_planner",
            manifest=manifest,
            scenario={"name": scenario},
            load_config=lambda path: _read_yaml(str(path)) if path else {},
        )
        cfg = build_hybrid_rule_local_planner_config(effective)
        assert cfg.planner_variant == HYBRID_RULE_V4_CLEARANCE_BRAKING_VARIANT
        assert cfg.max_linear_speed == pytest.approx(2.0)
        assert cfg.v4_braking_check_enabled is True
        assert cfg.continuous_static_clearance_enabled is True
        HybridRuleLocalPlannerAdapter(cfg)


def test_v3_configs_keep_the_v3_variant() -> None:
    """The frozen v3 release configs never opt into the v4 code path."""
    for v3_name in V3_TO_V4:
        effective = load_planner_config(CANDIDATES + v3_name, "classic_doorway_high")
        planner = HybridRuleLocalPlannerAdapter(build_hybrid_rule_local_planner_config(effective))
        assert planner.config.planner_variant == "hybrid_rule_v3_teb_like_rollout"
        assert planner._v4_clearance_braking is False
        assert planner._drive_limits is None


def test_v4_candidates_are_registered_in_policy_search_registry() -> None:
    """The policy-search candidate registry lists each v4 twin with its config path."""
    registry = _read_yaml("docs/context/policy_search/candidate_registry.yaml")["candidates"]
    for v4_name in V3_TO_V4.values():
        key = v4_name.removesuffix(".yaml")
        assert key in registry, key
        assert registry[key]["candidate_config_path"] == CANDIDATES + v4_name
        assert (repo_root() / registry[key]["candidate_config_path"]).is_file()
