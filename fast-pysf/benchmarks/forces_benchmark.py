"""Performance benchmark for forces computation in pedestrian simulation.

This module provides functionality to benchmark the social and obstacle forces
calculation using the Scalene profiler to measure performance characteristics.
"""

from dataclasses import dataclass

import numpy as np
import pysocialforce as psf
from scalene import scalene_profiler


@dataclass
class SimConfig:
    """Configuration for a simulation instance.

    Attributes:
        sim_steps: Number of simulation steps to run.
        initial_state: Initial pedestrian state array.
        groups: List of group memberships.
        obstacles: List of obstacle line segments.
    """

    sim_steps: int
    initial_state: np.ndarray
    groups: list[list[int]]
    obstacles: list[tuple[float, float, float, float]]
    # config: psf.utils.config.DefaultConfig = psf.utils.config.DefaultConfig()


@dataclass
class SimSettings:
    """Settings for generating simulation scenarios.

    Attributes:
        map_width: Width of the simulation map.
        map_height: Height of the simulation map.
        sim_steps: Number of simulation steps.
        num_peds: Number of pedestrians.
        num_groups: Number of pedestrian groups.
        group_cov: Covariance matrix for group position sampling.
        num_obstacles: Number of obstacles to generate.
    """

    map_width: float
    map_height: float
    sim_steps: int
    num_peds: int
    num_groups: int
    group_cov: list[list[float]]  # shape (2, 2)
    num_obstacles: int

    def sample(self) -> SimConfig:
        """Generate a random simulation configuration.

        Returns:
            SimConfig: Randomly generated simulation configuration.
        """
        spawned_peds = 0
        group_centroids = self.rand_2d_coords(self.num_groups)
        group_goals = self.rand_2d_coords(self.num_groups)
        individual_dynamics = np.random.normal(size=(self.num_peds, 2))
        avg_peds_on_group = self.num_peds / self.num_groups

        individuals = np.zeros((self.num_peds, 6))
        groups = []

        for gid in range(self.num_groups):
            remaining_peds = self.num_peds - spawned_peds
            remaining_groups = self.num_groups - gid

            if remaining_peds <= remaining_groups:
                num_peds_on_group = 1
            elif gid == self.num_groups - 1:
                num_peds_on_group = remaining_peds
            else:
                # Sample candidate, ensure at least 1, then clamp to leave at least 1 per remaining group
                candidate = max(1, round(np.random.normal(avg_peds_on_group)))
                max_allowed = remaining_peds - (remaining_groups - 1)
                num_peds_on_group = min(candidate, max_allowed)

            peds_pos_of_group = np.random.multivariate_normal(
                group_centroids[gid], self.group_cov, num_peds_on_group
            )
            groups.append(list(range(spawned_peds, spawned_peds + num_peds_on_group)))

            for i, ped_pos in enumerate(peds_pos_of_group):
                ped_id = spawned_peds + i
                ped_dyn = individual_dynamics[ped_id]
                goal = group_goals[gid]
                ped_features = [
                    ped_pos[0],
                    ped_pos[1],
                    ped_dyn[0],
                    ped_dyn[1],
                    goal[0],
                    goal[1],
                ]
                individuals[ped_id] = ped_features
            spawned_peds += num_peds_on_group

        obs_start = self.rand_2d_coords(self.num_obstacles)
        obs_end = self.rand_2d_coords(self.num_obstacles)
        obstacles = [
            [s_x, s_y, e_x, e_y]
            for ((s_x, s_y), (e_x, e_y)) in zip(obs_start, obs_end, strict=False)
        ]

        config = SimConfig(self.sim_steps, individuals, groups, obstacles)
        return config

    def rand_2d_coords(self, num_coords: int) -> list[tuple[float, float]]:
        """Generate random 2D coordinates.

        Args:
            num_coords: Number of coordinates to generate.

        Returns:
            List of (x, y) coordinate tuples.
        """
        x = np.random.uniform(0, self.map_width, num_coords)
        y = np.random.uniform(0, self.map_height, num_coords)
        return list(zip(x, y, strict=False))


def simulate(config: SimConfig):
    """Run a simulation with the given configuration.

    Args:
        config: Simulation configuration parameters.
    """
    s = psf.Simulator(config.initial_state, config.groups, config.obstacles)
    s.step(config.sim_steps)


def benchmark():
    """Run the forces performance benchmark.

    Executes 100 simulations with 25 pedestrians, 3 groups, and 10 obstacles
    while profiling with Scalene to analyze performance characteristics.
    """
    num_simulations = 100
    settings = SimSettings(
        map_width=80,
        map_height=80,
        sim_steps=100,
        num_peds=25,
        num_groups=3,
        group_cov=[[1, 0], [0, 1]],
        num_obstacles=10,
    )
    configs = [settings.sample() for _ in range(num_simulations)]

    scalene_profiler.start()
    for sim_id, config in enumerate(configs):
        print(f"simulating {sim_id + 1} / {num_simulations}")
        simulate(config)
    scalene_profiler.stop()


if __name__ == "__main__":
    benchmark()
