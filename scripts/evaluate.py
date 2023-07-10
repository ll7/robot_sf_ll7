from typing import Callable
from tqdm import tqdm
from stable_baselines3 import PPO

from robot_sf.robot_env import RobotEnv
from robot_sf.eval import EnvMetrics


def evaluate(model_path: str):
    env = RobotEnv()
    model = PPO.load(model_path, env=env)
    num_episodes = 100
    eval_metrics = EnvMetrics(cache_size=num_episodes)

    for _ in tqdm(range(num_episodes)):
        is_end_of_route = False
        obs = env.reset()
        while not is_end_of_route:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, meta = env.step(action)
            meta = meta["meta"]
            eval_metrics.update(meta)
            if done:
                obs = env.reset()
                is_end_of_route = meta["is_pedestrian_collision"] or meta["is_obstacle_collision"] \
                    or meta["is_route_complete"] or meta["is_timesteps_exceeded"]

    metrics = {
        "route_completion_rate": eval_metrics.route_completion_rate,
        "obstacle_collision_rate": eval_metrics.obstacle_collision_rate,
        "pedestrian_collision_rate": eval_metrics.pedestrian_collision_rate,
        "timeout_rate": eval_metrics.timeout_rate,
    }
    print(metrics)



if __name__ == '__main__':
    evaluate("./model/ppo_model")
