import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from robot_sf.robot_env import RobotEnv, OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.sim_config import EnvSettings
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings

def training():
    env_config = EnvSettings(
        sim_config=SimulationSettings(
            stack_steps=1,
            difficulty=0,
            ped_density_by_difficulty=[0.06]
            ),
        robot_config=DifferentialDriveSettings(radius=1.0)
        )
    env = RobotEnv(env_config, debug=True)
    env.observation_space, env.action_space = prepare_gym_spaces()
    model = PPO.load("./model/run_023", env=env)

    def obs_adapter(orig_obs):
        drive_state = orig_obs[OBS_DRIVE_STATE]
        ray_state = orig_obs[OBS_RAYS]
        drive_state = drive_state[:, :-1]
        drive_state[:, 2] *= 10
        drive_state = np.squeeze(drive_state)
        ray_state = np.squeeze(ray_state)
        return np.concatenate((ray_state, drive_state), axis=0)

    obs = env.reset()
    for _ in range(10000):
        obs = obs_adapter(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        env.render()

        if done:
            obs = env.reset()
            env.render()
    env.exit()


def prepare_gym_spaces():
    obs_low = np.array([ 0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,          0. ,         0.,          0.,
                         0.,          0.,          0.,         -0.5,         0.,         -3.14159265])


    obs_high = np.array([1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
                         2.00000000e+00, 5.00000000e-01, 1.07480231e+03, 3.14159265e+00])

    action_low = np.array([-2.0, -0.5])
    action_high = np.array([2.0, 0.5])

    obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
    action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)
    return obs_space, action_space


if __name__ == '__main__':
    training()
