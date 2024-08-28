import os
from pathlib import Path

from stable_baselines3 import PPO

from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.robot.bicycle_drive import BicycleDriveSettings



def make_env():
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    map_definition = convert_map("maps/svg_maps/debug_01.svg")
    robot_model = PPO.load("./model/run_043", env=None)

    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True)
    )
    env_config.sim_config.ped_density_by_difficulty = ped_densities
    env_config.sim_config.difficulty = difficulty
    return PedestrianEnv(env_config, robot_model=robot_model, debug=True,)

def get_file():
    """Get the latest model file."""

    filename = max(
        os.listdir('model_ped'), key=lambda x: os.path.getctime(os.path.join('model_ped', x)))
    return Path('model_ped', filename)


def training():
    env = make_env()
    filename = get_file()
    model = PPO.load(filename, env=env)

    obs = env.reset()
    for _ in range(10000):
        if isinstance(obs, tuple):
            action, _ = model.predict(obs[0], deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ , _= env.step(action)
        env.render()

        if done:
            obs = env.reset()
            env.render()
    env.exit()


if __name__ == '__main__':
    training()
