"""Debug runner for pedestrian policy models in the SocialForce simulator."""

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


def dummy_map() -> MapDefinition:
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


def make_env():
    """Create a deterministic pedestrian simulation environment for debugging."""
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


def _find_social_force(env):
    """Return the SocialForce callable from the configured PySF force list."""
    for force in env.simulator.pysf_sim.forces:
        if type(force).__name__ == "SocialForce":
            return force
    raise RuntimeError("SocialForce not found in env.simulator.pysf_sim.forces")


def run():
    """Run the debug environment to analyze pedestrian social forces."""
    env = make_env()
    try:
        _ = env.reset()
        env.simulator.ego_ped.state.pose = ((12.0, 10.0), 0)
        social_force_fn = _find_social_force(env)

        social_forces = []
        all_forces = []
        total_forces = []
        social_force_threshold = 0.5
        for i in range(400):
            action = np.array([0, 0])  # No action for the ego pedestrian
            _, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            sf = social_force_fn()
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
                [component[0].copy() for component in force_components],
                axis=0,
            )
            all_forces.append(npc_force_components)
            total_forces.append(npc_force_components.sum(axis=0))
            env.render()

            if done:
                break
    finally:
        env.exit()


if __name__ == "__main__":
    run()
