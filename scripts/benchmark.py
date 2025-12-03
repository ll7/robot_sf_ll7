"""TODO docstring. Document this module."""

import time

from scalene import scalene_profiler

from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.gym_env.robot_env import RobotEnv


def benchmark():
    """TODO docstring. Document this function."""
    total_steps = 10000
    env = RobotEnv()
    model = load_trained_policy("./model/ppo_model")
    # Gymnasium-style reset returns (obs, info)
    obs, _info = env.reset()

    # NOTE: RobotEnv exposes the underlying simulator as `simulator`; previous code
    # referenced a non-existent `sim_env` attribute which triggered lint warnings.
    _peds_sim = env.simulator

    env.step(env.action_space.sample())  # warm-up step
    obs, _info = env.reset()
    print("start of simulation")

    start_time = time.perf_counter()
    scalene_profiler.start()

    episode = 0
    ep_rewards = 0
    for step in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info_step = env.step(action)
        done = terminated or truncated
        ep_rewards += reward
        # print(f'step {step}, reward {reward} (peds: {peds_sim.peds.size()})')

        if done:
            episode += 1
            print(
                f"end of episode {episode}, total rewards {ep_rewards:.3f}, remaining steps {total_steps - step}",
            )
            ep_rewards = 0
            obs, _info = env.reset()

    print("end of simulation")
    scalene_profiler.stop()

    end_time = time.perf_counter()
    print(f"benchmark took {end_time - start_time} seconds")


if __name__ == "__main__":
    benchmark()
