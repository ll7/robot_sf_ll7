"""Debug runner for pedestrian policy models in the SocialForce simulator."""

import os
from pathlib import Path

import loguru
import numpy as np
from stable_baselines3 import PPO

from robot_sf.common.types import Line2D
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.nav_types import SvgRectangle
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def dummy_map():
    """Create a minimal map definition for deterministic pedestrian routing.

    This keeps environment setup stable across tests to isolate behavior under test.
    """
    width = 40.0
    height = 40.0
    bounds: list[Line2D] = [
        (0, width, 0, 0),  # bottom
        (0, width, height, height),  # top
        (0, 0, 0, height),  # left
        (width, width, 0, height),  # right
    ]
    r_spawn = SvgRectangle(
        x=20, y=30, width=1, height=1, label="robot_spawn", id_="robot_spawn"
    ).get_zone()

    r_goal = SvgRectangle(
        x=30, y=30, width=1, height=1, label="robot_goal", id_="robot_goal"
    ).get_zone()

    r_path = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[[22, 30], [29, 30]],
        spawn_zone=r_spawn,
        goal_zone=r_goal,
    )

    ped_spawn = SvgRectangle(
        x=5, y=10, width=1, height=1, label="ped_spawn", id_="ped_spawn"
    ).get_zone()

    ped_goal = SvgRectangle(
        x=35, y=10, width=1, height=1, label="ped_goal", id_="ped_goal"
    ).get_zone()

    # ped_path = GlobalRoute(
    #     spawn_id=0,
    #     goal_id=0,
    #     waypoints=[[7, 10], [34, 10]],
    #     spawn_zone=ped_spawn,
    #     goal_zone=ped_goal,
    # )

    single_pedestrians = [
        SinglePedestrianDefinition(
            id="npc_0",
            start=(1.0, 10.0),
            goal=(34.0, 10.0),
        ),
    ]

    map_def = MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[r_spawn],
        ped_spawn_zones=[ped_spawn],
        robot_goal_zones=[r_goal],
        bounds=bounds,
        robot_routes=[r_path],
        ped_goal_zones=[ped_goal],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=single_pedestrians,
    )
    return map_def


def make_env(svg_map_path):
    """Create a pedestrian simulation environment for debugging.

    Parameters
    ----------
    svg_map_path : str
        Path to the SVG map file.

    Returns
    -------
    gym.Env
        Pedestrian simulation environment with loaded robot model.
    """
    # map_definition = convert_map(svg_map_path)
    map_definition = dummy_map()
    robot_model = PPO.load("./model/run_043", env=None)

    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            debug_without_robot_movement=True,
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
        spawn_near_robot=False,
    )
    env = make_pedestrian_env(
        config=config,
        robot_model=robot_model,
        debug=True,
        recording_enabled=False,
    )

    return env


def get_file():
    """Get the latest model file."""

    filename = max(
        os.listdir("model/pedestrian"),
        key=lambda x: os.path.getctime(os.path.join("model/pedestrian", x)),
    )
    return Path("model/pedestrian", filename)


def run():
    """Run the debug environment to analyze pedestrian social forces."""
    env = make_env("maps/svg_maps/debug_06.svg")
    filename = get_file()
    # filename = "./model_ped/ppo_2024-09-06_23-52-17.zip"
    logger.info(f"Loading pedestrian model from {filename}")

    # model = PPO.load(filename, env=env)

    _ = env.reset()
    env.simulator.ego_ped.state.pose = ((12.0, 10.0), 0)

    # Teleport NPC
    # def teleport_npc(x, y):
    #     env.simulator.pysf_sim.peds.state[0, 0:2] = [x, y]  # position
    #     env.simulator.pysf_sim.peds.state[0, 2:4] = [1.0, 0.0]  # velocity

    # teleport_npc(1.0, 10.0)

    social_forces = []
    all_forces = []
    total_forces = []
    social_force_threshold = 0.5
    # force_names = [type(force).__name__ for force in env.simulator.pysf_sim.forces]
    for i in range(400):
        action = np.array([0, 0])  # No action for the ego pedestrian
        _, _, done, _, _ = env.step(action)
        # Get the social force for the NPC (index 0)
        sf = env.simulator.pysf_sim.forces[1]()  # index 1 is social force
        force_components = [force() for force in env.simulator.pysf_sim.forces]
        npc_social_force = sf[0].copy()
        social_forces.append(npc_social_force)
        social_force_norm = float(np.linalg.norm(npc_social_force))
        if social_force_norm > social_force_threshold:
            npc_pos = env.simulator.pysf_sim.peds.state[0, 0:2].copy()
            logger.info(
                f"step={i} social_force={npc_social_force.tolist()} "
                f"|f|={social_force_norm:.6f} "
                f"npc_pos=({float(npc_pos[0]):.6f}, {float(npc_pos[1]):.6f})"
            )
        npc_force_components = np.stack(
            [component[0].copy() for component in force_components], axis=0
        )
        all_forces.append(npc_force_components)
        total_forces.append(npc_force_components.sum(axis=0))
        # teleport_npc(11, 10.0)  # Keep teleporting to isolate social force response
        env.render()
        # time.sleep(0.5)

        if done:
            break
    # logger.error(f"sf: {social_forces}")
    # logger.error(f"force names: {force_names}")
    # logger.error(f"all forces (per step, per component for NPC 0): {all_forces}")
    # logger.error(f"total summed force (per step for NPC 0): {total_forces}")
    env.exit()


def extract_info(meta: dict, reward: float) -> str:
    """Extract and format episode statistics from metadata.

    Parameters
    ----------
    meta : dict
        Metadata dictionary containing episode information.
    reward : float
        Cumulative reward for the episode.

    Returns
    -------
    str
        Formatted string containing episode number, steps, done conditions,
        reward, and distance to robot.
    """
    meta = meta["meta"]
    eps_num = meta["episode"]
    steps = meta["step_of_episode"]
    done = [key for key, value in meta.items() if value is True]
    dis = meta["distance_to_robot"]
    return f"Episode: {eps_num}, Steps: {steps}, Done: {done}, Reward: {reward}, Distance: {dis}"


if __name__ == "__main__":
    run()
