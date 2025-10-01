from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings


def demo_offensive_policy():
    env_config = EnvSettings(
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )
    env = RobotEnv(env_config, debug=True, recording_enabled=False)
    model = load_trained_policy("./model/run_043")

    obs, _ = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        env.render()

        if done:
            obs, _ = env.reset()
            env.render()
    env.exit()


if __name__ == "__main__":
    demo_offensive_policy()
