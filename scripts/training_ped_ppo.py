import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.tb_logging import AdversialPedestrianMetricsCallback
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.robot.bicycle_drive import BicycleDriveSettings


def training():
    n_envs = 10
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2


    def make_env():
        map_definition = convert_map("maps/svg_maps/debug_02.svg")
        robot_model = PPO.load("./model/run_043", env=None)

        env_config = PedEnvSettings(
            map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
            sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
            robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True)
        )
        env_config.sim_config.ped_density_by_difficulty = ped_densities
        env_config.sim_config.difficulty = difficulty
        return PedestrianEnv(env_config, robot_model=robot_model)

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(features_extractor_class=DynamicsExtractor)
    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log="./logs/ppo_logs/",
        policy_kwargs=policy_kwargs
        )
    save_model_callback = CheckpointCallback(
        500_000 // n_envs,
        "./model/backup",
        "ppo_model"
        )
    collect_metrics_callback = AdversialPedestrianMetricsCallback(n_envs)
    combined_callback = CallbackList(
        [save_model_callback, collect_metrics_callback]
        )

    model.learn(
        total_timesteps=600_000,
        progress_bar=True,
        callback=combined_callback
        )
    now = datetime.datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S")
    model.save(f"./model_ped/ppo_{filename}")


if __name__ == '__main__':
    training()
