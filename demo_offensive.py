from stable_baselines3 import PPO
from robot_sf.robot_env import RobotEnv
from robot_sf.sim_config import EnvSettings
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.robot.bicycle_drive import BicycleDriveSettings


def training():
    env_config = EnvSettings(
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True))
    env = RobotEnv(env_config, debug=True)
    model = PPO.load("./model/run_043", env=env)

    obs = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        env.render()

        if done:
            obs = env.reset()
            env.render()
    env.exit()


if __name__ == '__main__':
    training()
