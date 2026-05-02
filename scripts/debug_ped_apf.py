"""Debug script for pedestrian adversarial force visualization.

This module provides utilities to visualize and analyze adversarial pedestrian
forces in the robot simulation environment. It includes force plotting functions
and a main debug loop that loads a trained PPO model and renders force vectors
over simulation timesteps.
"""

import sys
from pathlib import Path

import loguru
import matplotlib.pyplot as plt
import numpy as np

from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_npc.adversial_ped_force import AdversarialPedForceConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger

logger.remove()
logger.add(sys.stderr, level="WARNING")


# Model Configuration: Easy to swap different robot models here
MODEL_NAME = "run_043"


def get_model_profile(model_name: str) -> dict:
    """Return environment/runtime profile settings for a given model.

    Profiles keep robot dynamics and sim defaults aligned with the model's
    training assumptions.
    """
    if model_name == "run_023":
        # Defensive model profile from examples/advanced/09_defensive_policy.py
        return {
            "stack_steps": 1,
            "difficulty": 0,
            "ped_density_by_difficulty": [0.06],
            "robot_config": DifferentialDriveSettings(radius=1.0, max_angular_speed=0.5),
        }

    # Default profile for run_043
    return {
        "stack_steps": 3,
        "difficulty": 1,
        "ped_density_by_difficulty": [0.01, 0.02, 0.04, 0.08],
        "robot_config": BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    }


def load_model(model_name: str):
    """Load a trained robot policy model.

    Supports both built-in model names and custom paths.

    Parameters
    ----------
    model_name : str
        Model identifier. Can be:
        - "run_023": defensive policy model (model/run_023.zip)
        - "run_043": current baseline model (model/run_043.zip)
        - Full path to custom model file (e.g., "./model/custom_model.zip")

    Returns
    -------
    stable_baselines3.PPO or similar
        Loaded trained policy model.

    Raises:
        FileNotFoundError: If model file cannot be found.
    """
    # Check if it's a built-in model name
    if model_name in ["run_023", "run_043"]:
        model_path = Path(__file__).resolve().parents[1] / "model" / f"{model_name}.zip"
        logger.info(f"Loading built-in model: {model_name} from {model_path}")
    else:
        # Assume it's a custom path
        model_path = Path(model_name)
        logger.info(f"Loading custom model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return load_trained_policy(str(model_path))


def get_obs_adapter(model_name: str):
    """Get the observation adapter function for a specific model.

    Different models were trained with different observation formats. This function
    returns an adapter that converts the current environment's dict observations
    to the format expected by each model.

    Parameters
    ----------
    model_name : str
        Model identifier ("run_023", "run_043").

    Returns
    -------
    callable or None
        Adapter function that transforms observations, or None if no adaptation needed.
    """
    if model_name == "run_023":
        # Defensive policy was trained with old observation format
        # Expects: [ray_state, drive_state]
        def adapt_obs_run_023(obs_dict):
            """Convert dict observation to flat format expected by run_023."""
            if isinstance(obs_dict, dict):
                # Keep key usage consistent with legacy defensive policy adapter.
                drive_state = np.asarray(obs_dict[OBS_DRIVE_STATE])
                ray_state = np.asarray(obs_dict[OBS_RAYS])

                # Legacy adapter behavior
                drive_state = drive_state[:, :-1]
                drive_state[:, 2] *= 10

                # Ensure both arrays are 1-D before concatenation.
                drive_state = np.squeeze(drive_state).reshape(-1)
                ray_state = np.squeeze(ray_state).reshape(-1)
                return np.concatenate((ray_state, drive_state), axis=0)
            return obs_dict

        return adapt_obs_run_023

    # run_043 use dict observations directly
    return None


def make_env(svg_map_path: str, model_name: str):
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
    profile = get_model_profile(model_name)

    map_definition = convert_map(svg_map_path)

    apf_config = AdversarialPedForceConfig(is_active=True, offset=5.0)

    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            stack_steps=profile["stack_steps"],
            difficulty=profile["difficulty"],
            ped_density_by_difficulty=profile["ped_density_by_difficulty"],
            peds_reset_follow_route_at_start=True,
            apf_config=apf_config,
            debug_without_robot_movement=False,  # Keep robot movement enabled during APF debug
        ),
        robot_config=profile["robot_config"],
    )
    env = make_robot_env(
        config=config,
        debug=True,
        recording_enabled=False,
    )

    return env


def run(svg_map_path: str, model_name: str | None = None):
    """Run the debug visualization for adversarial pedestrian forces.

    Loads a trained PPO model, runs a simulation with adversarial pedestrian
    forces enabled, renders the environment, and plots force vectors over time.

    Parameters
    ----------
    svg_map_path : str
        Path to the SVG map file to load.
    model_name : str, optional
        Robot model to load. If None, uses the MODEL_NAME constant.
        Can be "run_023", "run_043", or a custom path.
    """
    # Use provided model or fall back to global configuration
    if model_name is None:
        model_name = MODEL_NAME

    env = make_env(svg_map_path, model_name)
    try:
        model = load_model(model_name)
        obs_adapter = get_obs_adapter(model_name)

        obs, _ = env.reset()
        for _ in range(10000):
            # Adapt observation if needed for the model
            if obs_adapter is not None:
                adapted_obs = obs_adapter(obs)
            else:
                adapted_obs = obs

            action, _ = model.predict(adapted_obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            env.render()
            # sleep(0.1)

            if done:
                obs, _ = env.reset()
                env.render()
        # forces = env.simulator.force_history

        # plot_forces_over_time2(forces)
        # plot_forces_quiver(forces)
    finally:
        env.exit()


def plot_forces_over_time2(
    forces_over_time: np.ndarray,
    *,
    ped_idx: int = 2,
    force_indices: tuple[int, int] = (0, 6),
) -> None:
    """Plot desired and adversarial force components for one pedestrian over time.

    Args:
        forces_over_time: Force-history array with shape ``(T, F, P, 2)``.
        ped_idx: Pedestrian index to visualize.
        force_indices: Pair ``(desired_force_idx, adversarial_force_idx)``.
    """
    forces_over_time = np.array(forces_over_time)
    if forces_over_time.ndim != 4 or forces_over_time.shape[-1] != 2:
        raise ValueError(
            f"Expected forces_over_time with shape (T, F, P, 2); got {forces_over_time.shape}."
        )
    if not 0 <= ped_idx < forces_over_time.shape[2]:
        raise IndexError(
            f"ped_idx {ped_idx} out of range for {forces_over_time.shape[2]} pedestrians"
        )

    desired_force_idx, adversarial_force_idx = force_indices
    for force_idx in force_indices:
        if not 0 <= force_idx < forces_over_time.shape[1]:
            raise IndexError(
                f"force index {force_idx} out of range for {forces_over_time.shape[1]} force terms"
            )

    timesteps = np.arange(forces_over_time.shape[0])

    # F[time, force, pedestrian, :] = (Fx, Fy)
    Fx = forces_over_time[:, desired_force_idx, ped_idx, 0]
    Fy = forces_over_time[:, desired_force_idx, ped_idx, 1]

    plt.plot(timesteps, Fx, label="Fx (desired)")
    plt.plot(timesteps, Fy, label="Fy (desired)")

    Fx = forces_over_time[:, adversarial_force_idx, ped_idx, 0]
    Fy = forces_over_time[:, adversarial_force_idx, ped_idx, 1]

    plt.plot(timesteps, Fx, label="Fx (adversarial)")
    plt.plot(timesteps, Fy, label="Fy (adversarial)")

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
    # SVG_MAP = "maps/svg_maps/debug_05.svg"
    # SVG_MAP = "maps/svg_maps/narrow_corridor2.svg"
    SVG_MAP = "maps/svg_maps/masterthesis/headon.svg"

    run(SVG_MAP, model_name="run_043")
