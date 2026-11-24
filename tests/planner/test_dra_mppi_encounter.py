"""Exercise encounter selection through the benchmark policy factory."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner import build_map_policy
from robot_sf.robot.differential_drive import DifferentialDriveSettings

_CONFIGS = Path(__file__).resolve().parents[2] / "configs/algos"
_DEV_SEED = 1031


def _profile(name):
    return yaml.safe_load((_CONFIGS / name).read_bytes())


def _policy(*, opening=True, combo=True):
    config = _profile("dra_mppi_combo.yaml" if combo else "dra_mppi_r3.yaml")
    if combo:
        config["encounter_sampler"]["clear_hold_s"] = 0.25
    policy, _ = build_map_policy("dra_mppi", config, robot_kinematics="differential_drive")
    extents = [(-3.0, 1.0), (6.2, 15.0)] if opening else [(-3.0, 15.0)]
    walls = [[(start, y), (end, y)] for start, end in extents for y in (-2.2, 2.2)]
    drive = DifferentialDriveSettings()
    simulator = SimpleNamespace(
        robots=[SimpleNamespace(config=drive)], iter_obstacle_segments=lambda: walls
    )
    vars(policy)["_planner_bind_env"](
        SimpleNamespace(
            simulator=simulator,
            env_config=SimpleNamespace(robot_config=drive),
            _get_static_grid_obstacles=lambda: (walls, []),
        )
    )
    vars(policy)["_planner_reset"](seed=_DEV_SEED)
    return policy


def _observation(positions=(), velocities=()):
    return {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0, 0.0]},
        "goal": {"current": [12.0, 0.0]},
        "pedestrians": {
            "positions": positions,
            "velocities": velocities,
            "count": [len(positions)],
            "radius": [0.4],
        },
        "sim": {"timestep": [0.1]},
    }


def _modes(policy):
    return vars(policy)["_planner_adapter"].diagnostics().get("mode_steps")


def test_crossing_holds_the_sampler_and_reset_restores_dra_mppi() -> None:
    """A crossing starts a hold that survives missing motion and clears after a reset."""
    policy = _policy()
    clear = _observation()
    default_command = policy(clear)
    crossing = _observation([(3.0, -2.5)], [(0.5, 0.9)])
    policy(crossing)
    assert _modes(policy) == {"dra_mppi": 1, "velocity_sampler": 1, "orca_handoff": 0}
    assert default_command == _policy(combo=False)(clear)

    for _ in range(3):
        policy(clear)
    assert _modes(policy) == {"dra_mppi": 2, "velocity_sampler": 3, "orca_handoff": 0}

    policy(crossing)
    vars(policy)["_planner_reset"](seed=_DEV_SEED)
    assert policy(clear) == default_command
    assert _modes(policy) == {"dra_mppi": 1, "velocity_sampler": 0, "orca_handoff": 0}


def test_walkers_staged_on_both_sides_trigger_before_they_move() -> None:
    """A staged pair triggers early when its opening and exit have room."""
    policy = _policy()
    waiting = [(3.0, -2.5), (3.2, 2.5)]
    policy(_observation(waiting, [(0.0, 0.0)] * 2))
    assert _modes(policy) == {"dra_mppi": 0, "velocity_sampler": 1, "orca_handoff": 0}

    vars(policy)["_planner_reset"](seed=_DEV_SEED)
    policy(_observation([*waiting, (8.0, 0.4)], [(0.0, 0.0)] * 3))
    assert _modes(policy) == {"dra_mppi": 1, "velocity_sampler": 0, "orca_handoff": 0}

    policy(_observation(waiting, [(0.0, 0.0)] * 2))
    assert _modes(policy) == {"dra_mppi": 2, "velocity_sampler": 0, "orca_handoff": 0}

    corridor = _policy(opening=False)
    corridor(_observation(waiting, [(0.0, 0.0)] * 2))
    assert _modes(corridor) == {"dra_mppi": 1, "velocity_sampler": 0, "orca_handoff": 0}


def test_combo_requires_explicit_experimental_opt_in() -> None:
    """Encounter settings alone do not admit the combination planner."""
    config = _profile("dra_mppi_combo.yaml")
    del config["allow_testing_algorithms"]
    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        build_map_policy("dra_mppi", config, robot_kinematics="differential_drive")
