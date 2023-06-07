from tqdm import tqdm
from stable_baselines3 import PPO

from robot_sf.robot_env import RobotEnv
from robot_sf.sim_eval import EpisodeLoggingCallback, EvaluationPlotting


def evaluate():
    env = RobotEnv()
    model = PPO.load("./model/ppo_model", env=env)
    eval_logger = EpisodeLoggingCallback()

    print('collect evaluation data ...')
    obs = env.reset()
    for _ in tqdm(range(10000)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, meta = env.step(action)
        eval_logger(reward, done, meta)
        if done:
            obs = env.reset()

    plotting = EvaluationPlotting(eval_logger.results)
    plotting.plot_episode_completion()
    plotting.plot_episode_mean_rewards()
    plotting.plot_episode_mean_steps()


if __name__ == '__main__':
    evaluate()
