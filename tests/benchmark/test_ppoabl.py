"""Diagnostic plant isolation and guarded-arm identity regression evidence."""

import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.map_runner.map_runner import _ppo_planner_config
from robot_sf.benchmark.map_runner.map_runner_episode import _resolve_episode_run_context
from robot_sf.robot.differential_drive import DifferentialDriveMotion
from scripts.analysis.compare_ppoeval import compare

# Independently specified (speed cap, linear accel cap, reverse, angular accel cap).
V3 = (3.0, None, True, None)
V2 = (2.0, 1.0, False, 1.0)
FACTORS = {
    "a1": (2.0, None, True, None),
    "a2": (3.0, 1.0, True, None),
    "a3": (3.0, None, False, None),
    "a4": (3.0, None, True, 1.0),
    "a5": (3.0, 1.0, False, 1.0),
}
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def campaign():
    """Load the actual diagnostic contract without running any episodes."""
    cfg = load_campaign_config(ROOT / "configs/benchmarks/diagnostics/ppoeval.yaml")
    return cfg, _load_campaign_scenarios(cfg)


def context(cfg, scenario, arm, policy):
    """Resolve the real map-runner plant wiring; no fake environment or plant."""
    return _resolve_episode_run_context(
        scenario=scenario,
        seed=1001,
        horizon=600,
        dt=0.1,
        algo=arm,
        scenario_path=ROOT / "scoped_scenarios.json",
        algo_config=policy,
        algo_config_path=None,
        experimental_ped_impact=False,
        ped_impact_radius_m=1.0,
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


def assert_transitions(settings, factors):
    """Check forward/reverse saturation, ramping, braking and turn reversals."""
    cap, accel, reverse, angular = factors
    motion = DifferentialDriveMotion(settings)
    for v, w, target_v, target_w, dt in product(
        [0.0, 1.5, 2.0],
        [-0.75, 0.0, 0.75],
        [-4.0, -0.5, 0.5, 4.0],
        [-2.0, 0.5, 2.0],
        [0.1, 0.2],
    ):
        dv, dw = target_v - v, target_w - w
        if accel is not None:
            dv = max(-accel * dt, min(accel * dt, dv))
        if angular is not None:
            dw = max(-angular * dt, min(angular * dt, dw))
        expected = (max(-cap if reverse else 0.0, min(cap, v + dv)), max(-1.0, min(1.0, w + dw)))
        actual = motion._robot_velocity((v, w), ((target_v - v) / dt, (target_w - w) / dt), dt)
        assert actual == pytest.approx(expected), (factors, v, w, target_v, target_w, dt)


@pytest.mark.parametrize("variant", FACTORS)
@pytest.mark.parametrize("arm", ["ppo", "guarded_ppo"])
def test_one_factor_plant_through_episode_context(campaign, variant, arm):
    """Every selected scenario/arm resolves exactly the specified plant factor."""
    cfg, scenarios = campaign
    assert len(scenarios) == 16
    template = yaml.safe_load(
        next(p for p in cfg.planners if p.key == arm).algo_config_path.read_text()
    )
    # V3 proposals keep signed policy bounds; A5 uses the V2 policy clipping.
    policy = dict(
        template, diagnostic_training_plant=variant != "a5", diagnostic_plant_variant=variant
    )
    baseline_policy = dict(template, diagnostic_training_plant=variant != "a5")
    baseline_policy.pop("diagnostic_plant_variant", None)
    expected = FACTORS[variant]
    baseline = V2 if variant == "a5" else V3
    assert sum(a != b for a, b in zip(expected, baseline, strict=True)) == 1
    for scenario in scenarios:
        resolved = context(cfg, scenario, arm, policy)
        original = context(cfg, scenario, arm, baseline_policy)
        current = resolved.config.robot_config
        assert current.max_linear_speed == expected[0]
        assert current.allow_backwards == expected[2]
        assert_transitions(current, expected)
        # All other physical fields must remain byte-for-byte equivalent.
        changes = {
            k
            for k, value in asdict(current).items()
            if value != asdict(original.config.robot_config)[k]
        }
        permitted = {"diagnostic_plant_variant"}
        if variant in {"a1", "a5"}:
            permitted.add("max_linear_speed")
        if variant == "a3":
            permitted.add("allow_backwards")
        assert changes == permitted
        assert resolved.policy_cfg == policy
        assert_transitions(original.config.robot_config, baseline)


def test_branch_selected_plant(campaign):
    """Both checked-in arm profiles select this branch's same plant."""
    cfg, scenarios = campaign
    variants = set()
    for planner in cfg.planners:
        policy = yaml.safe_load(planner.algo_config_path.read_text())
        variant = policy["diagnostic_plant_variant"]
        variants.add(variant)
        assert policy["v_max"] == (2.0 if variant == "a5" else 3.0)
        assert policy["omega_max"] == 1.0
        assert policy["diagnostic_training_plant"] == (variant != "a5")
        assert_transitions(
            context(cfg, scenarios[0], planner.key, policy).config.robot_config, FACTORS[variant]
        )
    assert len(variants) == 1
    assert list(cfg.seed_policy.seeds) == list(range(1001, 1011))
    assert not cfg.paper_facing


def test_compare_uses_top_level_arm(tmp_path):
    """Two arms with identical base algorithm metadata stay distinct."""
    (tmp_path / "ppoeval_receipt.json").write_text(
        json.dumps({"head_sha": "fixture", "expected_episodes": 2})
    )
    rows = []
    for arm in ["ppo", "guarded_ppo"]:
        rows.append(
            {
                "algo": arm,
                "scenario_id": "classic_head_on_corridor_medium",
                "seed": 1001,
                "metrics": {},
                "steps": 1,
                "outcome": {
                    "route_complete": True,
                    "collision_event": False,
                    "timeout_event": False,
                },
                "algorithm_metadata": {
                    "algorithm": "ppo",
                    "simulation_step_trace": {
                        "reset": {"robot": {"position": [0, 0]}, "pedestrians": []},
                        "steps": [
                            {
                                "time_s": 0.1,
                                "robot": {"position": [0.1, 0]},
                                "pedestrians": [],
                                "planner": {
                                    "ppoeval_proposal": {"requested_command": [1, 0]},
                                    "selected_action": {
                                        "linear_velocity": 1,
                                        "angular_velocity": 0,
                                    },
                                },
                            }
                        ],
                    },
                },
            }
        )
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    report = compare({"a1": tmp_path})
    assert {r["arm"] for r in report["rows"]} == {"ppo", "guarded_ppo"}
    assert all(r["episodes"] == 1 for r in report["rows"])


@pytest.mark.parametrize("arm", ["ppo", "guarded_ppo"])
def test_branch_keeps_policy_bounds_and_delta_semantics(campaign, arm):
    """The plant selector never becomes a policy constructor field or changes delta semantics."""
    cfg, _ = campaign
    policy = yaml.safe_load(
        next(p for p in cfg.planners if p.key == arm).algo_config_path.read_text()
    )
    typed = _ppo_planner_config(policy)
    assert "diagnostic_plant_variant" not in typed
    planner = PPOPlanner.__new__(PPOPlanner)
    planner.config = PPOPlannerConfig(**typed)
    planner._action_semantics = planner._resolve_action_semantics(planner.config)
    assert planner._action_semantics == "velocity_delta"
    reverse = policy["diagnostic_plant_variant"] != "a5"
    for raw, current, expected in [
        ([-1.0, 0.5], [0.5, 0.2], [-0.5 if reverse else 0.0, 0.7]),
        ([2.0, 1.0], [2.0, 0.5], [3.0 if reverse else 2.0, 1.0]),
    ]:
        command = planner._action_vec_to_dict_from_array(np.array(raw), np.array(current))
        assert [command["v"], command["omega"]] == pytest.approx(expected)
        assert planner._ppoeval_proposal["raw_policy_output"] == raw
        assert planner._ppoeval_proposal["requested_command"] == pytest.approx(
            np.array(raw) + current
        )
