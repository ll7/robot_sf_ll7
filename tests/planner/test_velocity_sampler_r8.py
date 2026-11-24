"""Check the ORCA handoff through the development sampler's policy factory."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner import _build_socnav_config
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.planner.velocity_sampler_r4 import PassingWeights
from robot_sf.planner.velocity_sampler_r8 import ProgressVelocitySamplerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveSettings

CONFIGS = Path(__file__).resolve().parents[2] / "configs/algos"


def _stalled_sampler(walls, pedestrian, velocity=(0.0, 0.0)):
    config = yaml.safe_load((CONFIGS / "velocity_sampler_dev_r9.yaml").read_bytes())
    config["sampling_path_distance"] = False
    weights = dict(config.pop("velocity_sampler"))
    weights.pop("variant")
    adapter = ProgressVelocitySamplerAdapter(
        config=_build_socnav_config(config), weights=PassingWeights(**weights)
    )
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=DifferentialDriveSettings()),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: walls),
        )
    )
    adapter.reset(seed=1163)
    observation = {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [0.0, 0.0],
            "radius": [1.0],
        },
        "goal": {"current": [10.0, 0.0]},
        "pedestrians": {
            "positions": [pedestrian],
            "velocities": [velocity],
            "count": [1],
            "radius": [0.4],
        },
        "sim": {"timestep": [0.1]},
    }
    return adapter, observation


def _stall_in_place(adapter, observation):
    for _ in range(41):
        adapter.plan(observation)
    return adapter.plan(observation)


def test_stalled_narrow_pass_uses_reachable_orca_command(monkeypatch) -> None:
    """A stalled sampler hands off when its clear forecasts advance too little."""
    calls = []

    def orca_plan(self, observation):
        calls.append(observation)
        return 2.0, 0.0

    monkeypatch.setattr(ORCAPlannerAdapter, "plan", orca_plan)
    walls = [((-5.0, y), (15.0, y)) for y in (-1.2, 1.2)]
    adapter, observation = _stalled_sampler(walls, (2.0, 0.0))
    adapter.plan(observation)
    assert not calls

    speed, _ = _stall_in_place(adapter, observation)
    assert calls, "the stalled sampler should consult ORCA"
    assert speed == pytest.approx(0.1)
    assert adapter.diagnostics()["orca_recovery_steps"] > 0

    activations = len(calls)
    adapter.reset(seed=1163)
    adapter.plan(observation)
    assert len(calls) == activations
    assert adapter.diagnostics()["orca_recovery_steps"] == 0


@pytest.mark.parametrize("contact", ["wall", "pedestrian"])
def test_orca_command_requires_clear_braking_path(monkeypatch, contact) -> None:
    """Decline a reachable ORCA command if applying it and braking predicts contact."""
    called = []

    def orca_plan(self, observation):
        called.append(True)
        return 2.0, 0.0

    monkeypatch.setattr(ORCAPlannerAdapter, "plan", orca_plan)
    walls = [((-5.0, y), (15.0, y)) for y in (-1.2, 1.2)]
    pedestrian = (2.0, 0.0)
    if contact == "wall":
        walls.append(((1.003, -1.2), (1.003, 1.2)))
    else:
        pedestrian = (1.403, 0.0)
    adapter, observation = _stalled_sampler(walls, pedestrian)

    speed, _ = _stall_in_place(adapter, observation)
    assert called, "the scene must reach the braking check"
    assert speed == pytest.approx(0.0)
    assert adapter.diagnostics()["orca_recovery_steps"] == 0


def test_clear_progress_keeps_sampler_in_control(monkeypatch) -> None:
    """A nearby walker to the side leaves a clear, advancing sample available."""

    def orca_plan(self, observation):
        pytest.fail("a clear advancing sample should keep the sampler in control")

    monkeypatch.setattr(ORCAPlannerAdapter, "plan", orca_plan)
    walls = [((-5.0, y), (15.0, y)) for y in (-3.6, 3.6)]
    adapter, observation = _stalled_sampler(walls, (0.0, 2.1))

    speed, _ = _stall_in_place(adapter, observation)
    assert speed > 0.0
    assert adapter.diagnostics()["orca_recovery_steps"] == 0
