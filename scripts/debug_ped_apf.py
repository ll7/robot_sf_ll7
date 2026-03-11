"""Debug script for pedestrian adversarial force visualization.

This module provides utilities to visualize and analyze adversarial pedestrian
forces in the robot simulation environment. It includes force plotting functions
and a main debug loop that loads a trained PPO model and renders force vectors
over simulation timesteps.
"""

import logging
import sys
from time import sleep

import loguru
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_npc.adversial_ped_force import AdversialPedForceConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = loguru.logger

logger.remove()
logger.add(sys.stderr, level="WARNING")


def make_env(svg_map_path):
    """Create a robot simulation environment with adversarial pedestrian forces.

    Parameters
    ----------
    svg_map_path : str
        Path to the SVG map file to load.

    Returns
    -------
    gym.Env
        A configured robot simulation environment with adversarial pedestrian
        forces enabled.
    """
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 1

    map_definition = convert_map(svg_map_path)

    apf_config = AdversialPedForceConfig(is_active=True, offset=10.0)

    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=difficulty,
            ped_density_by_difficulty=ped_densities,
            peds_reset_follow_route_at_start=True,
            apf_config=apf_config,
            debug_without_robot_movement=True,  # Enable debug mode to prevent robot movement
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )
    env = make_robot_env(
        config=config,
        debug=True,
        recording_enabled=False,
    )

    return env


def run(svg_map_path: str):
    """Run the debug visualization for adversarial pedestrian forces.

    Loads a trained PPO model, runs a simulation with adversarial pedestrian
    forces enabled, renders the environment, and plots force vectors over time.
    """
    env = make_env(svg_map_path)
    model = PPO.load("./model/run_043", env=env)
    logger.info("Loading robot model from ./model/run_043")

    obs, _ = env.reset()
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        env.render()
        sleep(0.1)

        if done:
            obs, _ = env.reset()
            env.render()
    # forces = env.simulator.force_history

    # plot_forces_over_time2(forces)
    # plot_forces_quiver(forces)

    env.exit()


def plot_forces_over_time2(forces_over_time):
    """
    Plot the force over time for a given pedestrian and force component.

    forces_over_time: array of shape (T, N, M, 2)
    ped_idx: index of the pedestrian
    component_idx: which force component (0 = desired force)
    """
    forces_over_time = np.array(forces_over_time)
    print(forces_over_time.shape)
    timesteps = np.arange(forces_over_time.shape[0])

    # F[time,force,ped,:]=(Fx​,Fy​)
    Fx = forces_over_time[:, 0, 2, 0]  # x-component
    Fy = forces_over_time[:, 0, 2, 1]  # y-component

    plt.plot(timesteps, Fx, label="Fx (desired)")
    plt.plot(timesteps, Fy, label="Fy (desired)")

    Fx = forces_over_time[:, 6, 2, 0]  # x-component
    Fy = forces_over_time[:, 6, 2, 1]  # y-component

    plt.plot(timesteps, Fx, label="Fx (adversial)")
    plt.plot(timesteps, Fy, label="Fy (adversial)")

    plt.xlabel("Timestep")
    plt.ylabel("Force")
    plt.title("force on Pedestrian over Time")
    plt.legend()
    plt.show()


def plot_forces_quiver(forces_over_time, ped_idx=0, force_indices=None, scale=1.0):
    """
    Plot forces as arrows over time for a given pedestrian.

    forces_over_time: (T, M, N, 2)
    ped_idx: pedestrian index
    force_indices: list of force component indices to plot (e.g., [0]=desired, [6]=adversarial)
    scale: scaling factor for arrows
    """
    if force_indices is None:
        force_indices = [0, 6]
    forces_over_time = np.array(forces_over_time)
    timesteps = np.arange(forces_over_time.shape[0])

    plt.figure()

    # Draw arrows for each selected force type
    for idx in force_indices:
        Fx = forces_over_time[:, idx, ped_idx, 0]
        Fy = forces_over_time[:, idx, ped_idx, 1]

        # y position = 0 (just spread arrows along horizontal axis)
        y = np.zeros_like(Fx) + idx  # offset so multiple force types don’t overlap

        plt.quiver(
            timesteps, y, Fx, Fy, angles="xy", scale_units="xy", scale=scale, label=f"Force {idx}"
        )

    plt.xlabel("Timestep")
    plt.ylabel("Force type offset")
    plt.title(f"Force vectors over time for pedestrian {ped_idx}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    SVG_MAP = "maps/svg_maps/debug_05.svg"
    SVG_MAP = "maps/svg_maps/narrow_corridor2.svg"
    run(SVG_MAP)
