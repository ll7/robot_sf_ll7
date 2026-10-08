"""Opt-in diagnostic profile contract; no released plant changes."""

import json
from hashlib import sha256
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/robots/assumed_sidewalk_delivery_v1.yaml"


def test_profile_loads_through_scenario_robot_config(tmp_path):
    """A real scenario must retain all diagnostic plant values, including latency."""
    profile = yaml.safe_load(PROFILE.read_text())
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump({"scenarios": [{"name": "diagnostic", **profile}]}))
    scenario = load_scenarios(path)[0]
    config = build_robot_config_from_scenario(scenario, scenario_path=path)
    robot = config.robot_config
    assert robot.radius == 0.31
    assert robot.max_linear_speed == 1.2
    assert robot.max_linear_accel == 0.5
    assert robot.max_linear_decel == 0.8
    assert robot.max_angular_speed == 1.0
    assert robot.min_linear_speed == 0.0
    assert robot.allow_backwards is False
    assert robot.limited_reverse is False
    assert config.sim_config.action_latency_ms == 200.0
    assert config.sim_config.action_latency_metadata()["effective_ms"] == 200.0


def test_unselected_profile_preserves_default_bytes(tmp_path):
    """Default robot settings and latency keep their pre-profile serialization."""
    config = build_robot_config_from_scenario({}, scenario_path=tmp_path / "scenario.yaml")
    # Exact default robot + simulation bytes captured on origin/main dada635b4.
    default_bytes = json.dumps(
        {
            "robot_config": asdict(config.robot_config),
            "simulation_config": config.sim_config.to_dict(),
        },
        sort_keys=True,
    ).encode()
    assert sha256(default_bytes).hexdigest() == (
        "4f89830e00963476c5cd056d75b210d6d8f235de7a10ad4c2eb77d47d3283117"
    )
    assert json.dumps(asdict(config.robot_config), sort_keys=True) == (
        '{"allow_backwards": false, "interaxis_length": 0.3, "max_angular_accel": 1.0, '
        '"max_angular_speed": 1.0, "max_linear_accel": 1.0, "max_linear_decel": 1.0, '
        '"max_linear_speed": 2.0, "radius": 1.0, "wheel_radius": 0.05}'
    )
    assert json.dumps(config.sim_config.action_latency_metadata(), sort_keys=True) == (
        '{"configured_ms": null, "configured_steps": 0, "effective_ms": 0.0, "effective_steps": 0}'
    )


def test_profile_marks_every_value_as_assumption():
    """Each selectable value carries an inline assumption qualification."""
    values = yaml.safe_load(PROFILE.read_text())["robot_config"]
    lines = PROFILE.read_text().splitlines()
    for key in values:
        line = next(line for line in lines if line.startswith(f"  {key}:"))
        assert "# assumption:" in line, key


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True])
def test_robot_latency_rejects_invalid_values(tmp_path, value):
    """Invalid opt-in latencies cannot be silently ignored or enter the action queue."""
    with pytest.raises(ValueError, match="control_latency_s"):
        build_robot_config_from_scenario(
            {"robot_config": {"control_latency_s": value}},
            scenario_path=tmp_path / "scenario.yaml",
        )
