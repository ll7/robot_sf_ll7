"""Test that environments gracefully fallback to a default reward function when None is passed.

This guards against regressions where a user (or integration script) explicitly sets
reward_func=None via the factory which previously could lead to a TypeError when stepping.
"""

from robot_sf.gym_env.environment_factory import make_robot_env


def test_single_robot_env_reward_fallback():
    env = make_robot_env(reward_func=None)
    # Internally we should have substituted a callable reward (simple_reward)
    # Access concrete attribute (RobotEnv defines reward_func)
    assert hasattr(env, "reward_func") and callable(env.reward_func)
    env.reset()
    action = env.action_space.sample()
    _obs, _rew, _term, _trunc, _info = env.step(action)
    env.close()
