"""Synthetic pedestrian behavior with social groups
simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from warnings import warn

import numpy as np

from pysocialforce import forces
from pysocialforce.config import SimulatorConfig
from pysocialforce.map_config import MapDefinition
from pysocialforce.ped_behavior import PedestrianBehavior
from pysocialforce.ped_grouping import PedestrianGroupings, PedestrianStates
from pysocialforce.ped_population import populate_simulation
from pysocialforce.scene import EnvState, PedState

Line2D = tuple[float, float, float, float]
SimState = tuple[np.ndarray, list[list[int]]]
EMPTY_MAP = MapDefinition([], [], [])
SimPopulator = Callable[
    [SimulatorConfig, MapDefinition],
    tuple[PedestrianStates, PedestrianGroupings, list[PedestrianBehavior]],
]


class ForceContext(Protocol):
    """Protocol for simulators compatible with force construction."""

    peds: PedState

    def get_obstacles(self) -> list[np.ndarray]:
        """Return the obstacles visible to the simulator."""
        ...

    def get_raw_obstacles(self) -> np.ndarray:
        """Return the raw obstacle array."""
        ...


ForceFactory = Callable[[ForceContext, SimulatorConfig], list[forces.Force]]


def make_forces(sim: ForceContext, config: SimulatorConfig) -> list[forces.Force]:
    """Initialize forces required for simulation."""
    enable_group = config.scene_config.enable_group
    force_list = [
        forces.DesiredForce(config.desired_force_config, sim.peds),
        forces.SocialForce(config.social_force_config, sim.peds),
        forces.ObstacleForce(config.obstacle_force_config, sim),
    ]
    group_forces = [
        forces.GroupCoherenceForceAlt(config.group_coherence_force_config, sim.peds),
        forces.GroupRepulsiveForce(config.group_repulsive_force_config, sim.peds),
        forces.GroupGazeForceAlt(config.group_gaze_force_config, sim.peds),
    ]
    return force_list + group_forces if enable_group else force_list


class Simulator_v2:
    """Simulator v2 class."""

    def __init__(
        self,
        map_definition: MapDefinition = EMPTY_MAP,
        config: SimulatorConfig = SimulatorConfig(),
        make_forces: ForceFactory = make_forces,
        populate: SimPopulator = lambda s, m: populate_simulation(
            s.scene_config.tau, s.ped_spawn_config, m.routes, m.crowded_zones
        ),
        on_step: Callable[[int, SimState], None] = lambda t, s: None,
    ):
        """
        Initializes a Simulator_v2 object.

        Args:
            map_definition (MapDefinition, optional): The definition of the map. Defaults to EMPTY_MAP.
            config (SimulatorConfig, optional): The configuration for the simulator. Defaults to SimulatorConfig().
            make_forces (ForceFactory, optional): A function that creates a list of forces. Defaults to make_forces.
            populate (SimPopulator, optional): A function that populates the simulation with initial states, groupings, and behaviors. Defaults to a lambda function.
            on_step (Callable[[SimState], None], optional): A function that is called after each step. Defaults to a lambda function.
        """
        self.config = config
        self.on_step = on_step
        self.states, self.groupings, self.behaviors = populate(config, map_definition)
        obstacles = (
            [line for o in map_definition.obstacles for line in o.lines]
            if map_definition.obstacles
            else []
        )
        self.env = EnvState(obstacles, self.config.scene_config.resolution)
        self.peds: PedState = PedState(
            self.states.raw_states, self.groupings.groups_as_lists, self.config.scene_config
        )
        self.forces = make_forces(self, config)
        self.t = 0

    @property
    def current_state(self) -> SimState:
        """
        Returns the current state of the simulation.

        Returns:
            SimState: The current state of the simulation.
        """
        return self.peds.state, self.peds.groups

    @property
    def obstacles(self) -> list[np.ndarray]:
        """
        Returns the obstacles in the environment.

        Returns:
            List[Line]: The obstacles in the environment.
        """
        return self.env.obstacles

    @property
    def raw_obstacles(self) -> np.ndarray:
        """
        Returns the raw obstacles in the environment.

        Returns:
            List[Line]: The raw obstacles in the environment.
        """
        return self.env.obstacles_raw

    def get_obstacles(self) -> list[np.ndarray]:
        """
        Returns the obstacles in the environment.

        Returns:
            List[Line]: The obstacles in the environment.
        """
        return self.env.obstacles

    def get_raw_obstacles(self) -> np.ndarray:
        """
        Returns the raw obstacles in the environment.

        Returns:
            List[Line]: The raw obstacles in the environment.
        """
        return self.env.obstacles_raw

    def _step_once(self) -> None:
        """
        Performs a single step in the simulation.
        """
        forces = sum(force() for force in self.forces)
        self.peds.step(forces)
        for behavior in self.behaviors:
            behavior.step()

    def step(self, n: int = 1) -> None:
        """
        Performs n steps in the simulation.

        Args:
            n (int, optional): The number of steps to perform. Defaults to 1.
        """
        for _ in range(n):
            self._step_once()
            self.on_step(self.t, self.current_state)
            self.t += 1


class Simulator:
    """Simulator class."""

    def __init__(
        self,
        state: np.ndarray,
        groups: list[list[int]] | None = None,
        obstacles: list[Line2D] | None = None,
        config: SimulatorConfig = SimulatorConfig(),
        make_forces: ForceFactory = make_forces,
        on_step: Callable[[int, SimState], None] = lambda t, s: None,
    ):
        """Init.

        Args:
            state: Auto-generated placeholder description.
            groups: Auto-generated placeholder description.
            obstacles: Auto-generated placeholder description.
            config: Auto-generated placeholder description.
            make_forces: Auto-generated placeholder description.
            on_step: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        self.config = config
        self.on_step = on_step
        resolution = self.config.scene_config.resolution
        self.env = EnvState(obstacles or [], resolution)
        self.peds: PedState = PedState(state, groups or [], self.config.scene_config)
        self.forces = make_forces(self, config)
        self.t = 0

    def compute_forces(self):
        """compute forces"""
        return sum(force() for force in self.forces)

    @property
    def current_state(self) -> SimState:
        """Current state.

        Returns:
            SimState: Auto-generated placeholder description.
        """
        return self.peds.state, self.peds.groups

    def get_states(self):
        """Get states.

        Returns:
            Any: Auto-generated placeholder description.
        """
        warn(
            "For performance reasons This function does not retrieve the whole \
              state history (it used to facilitate video recordings). \
              Please use the on_step callback for recording purposes instead!",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self) -> list[np.ndarray]:
        """Get obstacles.

        Returns:
            list[np.ndarray]: Auto-generated placeholder description.
        """
        return self.env.obstacles

    def get_raw_obstacles(self) -> np.ndarray:
        """Get raw obstacles.

        Returns:
            np.ndarray: Auto-generated placeholder description.
        """
        return self.env.obstacles_raw

    def step_once(self) -> None:
        """step once"""
        self.peds.step(self.compute_forces())

    def step(self, n: int = 1) -> None:
        """Step n time"""
        for _ in range(n):
            self.step_once()
            self.on_step(self.t, self.current_state)
            self.t += 1
