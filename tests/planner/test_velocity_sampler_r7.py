"""Development checks for turning out of a clear-space stall."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner import _build_socnav_config
from robot_sf.planner.velocity_sampler_r4 import PassingWeights
from robot_sf.planner.velocity_sampler_r6 import ClearanceVelocitySamplerAdapter
from robot_sf.planner.velocity_sampler_r7 import RecoveryTurnVelocitySamplerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings

CONFIGS = Path(__file__).resolve().parents[2] / "configs/algos"


def _resume_from_stall(adapter_type, *, pedestrian_nearby: bool = False):
    config = yaml.safe_load((CONFIGS / "velocity_sampler_dev_r9.yaml").read_bytes())
    config["sampling_path_distance"] = False
    weights = dict(config.pop("velocity_sampler"))
    weights.pop("variant")
    adapter = adapter_type(config=_build_socnav_config(config), weights=PassingWeights(**weights))
    drive = DifferentialDriveSettings()
    walls = [
        ((1.28, 0.35), (2.78, 0.35)),
        ((1.28, 0.35), (1.28, 2.18)),
        ((-10.0, 2.18), (12.0, 2.18)),
    ]
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=drive),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: walls),
        )
    )
    adapter.reset(seed=1163)
    robot = DifferentialDriveRobot(drive)
    robot.state.pose = ((0.0, 0.0), 1.74)
    positions = [[5.75, -1.1], [6.4, -1.8], [7.8, -2.4]]
    if pedestrian_nearby:
        positions[0] = [-2.0, -0.6]
    dt = 0.1
    observation = {
        "robot": {
            "position": robot.pos,
            "heading": [robot.pose[1]],
            "speed": robot.current_speed,
            "radius": [1.0],
        },
        "goal": {"current": [10.0, -8.0]},
        "pedestrians": {
            "positions": positions,
            "velocities": [[0.0, 0.0]] * len(positions),
            "count": [len(positions)],
            "radius": [0.4],
        },
        "sim": {"timestep": [dt]},
    }
    # Hold the pose through the four-second stall threshold, then execute the commands.
    for _ in range(41):
        adapter.plan(observation)
    commands = []
    for _ in range(35):
        speed, turn = adapter.plan(observation)
        commands.append((speed, turn))
        current_speed, current_turn = robot.current_speed
        robot.apply_action(((speed - current_speed) / dt, (turn - current_turn) / dt), dt)
        observation["robot"].update(
            position=robot.pos, heading=[robot.pose[1]], speed=robot.current_speed
        )
    return robot, commands


def test_stalled_robot_turns_and_resumes_progress_past_fixture() -> None:
    """A robot stopped facing the fixture turns toward the aisle and advances."""
    robot, commands = _resume_from_stall(RecoveryTurnVelocitySamplerAdapter)
    assert commands[0][0] == pytest.approx(0.0)
    assert robot.pos[0] > 0.2
    assert robot.pose[1] < 0.0


def test_nearby_pedestrian_keeps_existing_stall_response() -> None:
    """A close pedestrian leaves the passing sampler in control."""
    _, previous = _resume_from_stall(ClearanceVelocitySamplerAdapter, pedestrian_nearby=True)
    _, current = _resume_from_stall(RecoveryTurnVelocitySamplerAdapter, pedestrian_nearby=True)
    assert current == previous
