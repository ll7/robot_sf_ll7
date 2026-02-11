"""TODO docstring. Document this module."""

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.gym_env.robot_env import RobotEnv


def _make_robot_env() -> RobotEnv:
    """Create a default robot environment instance for vectorized training."""

    return RobotEnv()


def training():
    """Train an A2C policy on vectorized RobotEnv instances and save the model."""
    logs_dir = get_artifact_category_path("tmp") / "tensorboard" / "a2c_logs"
    model_dir = get_artifact_category_path("tmp") / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(_make_robot_env, n_envs=50, vec_env_cls=SubprocVecEnv)
    model = A2C("MlpPolicy", env, tensorboard_log=str(logs_dir))
    model.learn(total_timesteps=50_000_000, progress_bar=True)
    model.save(str(model_dir / "a2c_model"))


if __name__ == "__main__":
    training()
