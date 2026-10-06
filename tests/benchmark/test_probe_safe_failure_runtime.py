"""Release-roster admission with canonical outcomes from the unchanged canary config."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.infeasible_probe_safe_failure import (
    SAFE_FAILURE,
    UNRESOLVED,
    classify_probe_row,
)
from robot_sf.benchmark.termination_reason import (
    build_outcome_payload,
    collision_event,
    resolve_termination_reason,
    route_complete_success,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.analysis.compare_release_0_0_7_to_0_0_8 import _probe_gate


@pytest.fixture(scope="module")
def runtime_timeout_row():
    path = Path("configs/scenarios/canary_corridor.yaml")
    scenario = load_scenarios(path)[0]
    config = build_robot_config_from_scenario(scenario, scenario_path=path)
    env = make_robot_env(config=config)
    try:
        env.reset(seed=1001)
        # Advance only the episode clock to the final interval. Config/map bytes stay unchanged.
        env.state.timestep = env.state.max_sim_steps - 1
        env.state.sim_time_elapsed = env.state.timestep * env.state.d_t
        _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        success, collision = route_complete_success(info), collision_event(info)
        reason = resolve_termination_reason(
            terminated=terminated, truncated=truncated, success=success, collision=collision
        )
        assert reason == "terminated"
        assert info["meta"]["is_timesteps_exceeded"]
        assert not success and not collision
        return {
            "termination_reason": reason,
            "outcome": build_outcome_payload(
                route_complete=success,
                collision=collision,
                timeout=info["meta"]["is_timesteps_exceeded"],
            ),
            "metrics": {"collisions": 0.0, "success": False},
            "integrity": {"contradictions": []},
        }
    finally:
        env.close()


def test_canonical_environment_timeout_on_real_release_roster(runtime_timeout_row):
    manifest = yaml.safe_load(
        Path("configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.yaml").read_text()
    )
    slots = {
        (planner, kin, "francis2023_narrow_doorway", seed, "")
        for planner in manifest["planners"]["keys"]
        for kin in manifest["kinematics"]["matrix"]
        for seed in range(1001, 1031)
    }
    assert len(slots) == 420
    report = _probe_gate(dict.fromkeys(slots, runtime_timeout_row), slots)
    assert report["class_counts"] == {SAFE_FAILURE: 420}
    assert report["safe_failure_rate"] == 1.0


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "termination_reason", "garbage"),
        ("metrics", "collisions", -3),
        ("metrics", "collisions", None),
        ("metrics", "collisions", float("nan")),
        ("metrics", "collisions", False),
        (None, "integrity", "bad"),
        (None, "integrity", {"contradictions": ""}),
        ("metrics", "collisions", 1),
        ("metrics", "success", 1),
        (None, "termination_reason", "success"),
        (None, "termination_reason", "collision"),
        ("outcome", "route_complete", True),
    ],
    ids=[
        "unknown_reason",
        "negative_count",
        "missing_count",
        "nan_count",
        "bool_count",
        "malformed_integrity",
        "malformed_contradictions",
        "collision_disagreement",
        "success_disagreement",
        "false_success_reason",
        "false_collision_reason",
        "success_with_timeout",
    ],
)
def test_invalid_runtime_rows_fail_admission(runtime_timeout_row, section, key, value):
    row = deepcopy(runtime_timeout_row)
    row["termination_reason"] = "max_steps"
    target = row[section] if section else row
    if value is None:
        del target[key]
    else:
        target[key] = value
    assert classify_probe_row(row) == UNRESOLVED
