import numpy as np
import pytest

from pysocialforce.config import SimulatorConfig
from pysocialforce.ped_grouping import PedestrianStates, PedestrianGroupings
from pysocialforce.map_config import MapDefinition
from pysocialforce import forces
from pysocialforce import Simulator_v2 as Simulator


@pytest.fixture()
def generate_scene():
    raw_states = np.zeros((5, 7))
    raw_states[:, :4] = np.array([
        [1, 1, 1, 0],
        [1, 1.1, 0, 1],
        [3, 3, 1, 1],
        [3, 3.01, 1, 2],
        [3, 4, 3, 1]
        ])

    def populate(sim_config: SimulatorConfig, map_def: MapDefinition):
        states = PedestrianStates(raw_states)
        groupings = PedestrianGroupings(states, {})
        return states, groupings, []

    scene = Simulator(populate=populate)
    return scene


@pytest.fixture()
def generate_scene_with_groups():
    groups = {0: {1, 0}, 1: {3, 2}}
    raw_states = np.zeros((5, 7))
    raw_states[:, :4] = np.array([
        [1, 1, 1, 0],
        [1, 1.1, 0, 1],
        [3, 3, 1, 1],
        [3, 3.01, 1, 2],
        [3, 4, 3, 1]
        ])

    def populate(sim_config: SimulatorConfig, map_def: MapDefinition):
        states = PedestrianStates(raw_states)
        groupings = PedestrianGroupings(states, groups)
        return states, groupings, []

    scene = Simulator(populate=populate)
    return scene


def test_desired_force(generate_scene: Simulator):
    scene = generate_scene
    config = scene.config
    f = forces.DebuggableForce(forces.DesiredForce(config.desired_force_config, scene.peds))
    config.desired_force_config.factor = 1.0
    assert f(debug=True) == pytest.approx(
        np.array(
            [
                [-3.83847763, -1.83847763],
                [-1.74894926, -3.92384419],
                [-4.6, -4.6],
                [-6.10411508, -8.11779546],
                [-10.93315315, -8.57753753],
                ]
            )
        )


def test_social_force(generate_scene: Simulator):
    scene = generate_scene
    config = scene.config
    f = forces.DebuggableForce(forces.SocialForce(config.social_force_config, scene.peds))
    config.social_force_config.factor = 1.0
    assert f(debug=True) == pytest.approx(
        np.array(
            [
                [3.18320152e-12, 1.74095049e-12],
                [-3.64726290e-05, -6.76204532e-05],
                [7.86014187e-03, 1.66840389e-04],
                [7.70524167e-03, -3.03788477e-05],
                [9.12767677e-06, 1.23117582e-05],
                ]
            )
        )


def test_group_rep_force(generate_scene_with_groups: Simulator):
    scene = generate_scene_with_groups
    print(scene)
    config = scene.config
    scene.peds.groups = [[1, 0], [3, 2]]
    f = forces.DebuggableForce(forces.GroupRepulsiveForce(
        config.group_repulsive_force_config, scene.peds))
    config.group_repulsive_force_config.factor = 1.0
    assert f(debug=True) == pytest.approx(
        np.array([[0.0, -0.1], [0.0, 0.1], [0.0, -0.01], [0.0, 0.01], [0.0, 0.0]])
        )
