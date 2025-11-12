"""Switch Robot SF backends using unified config.

Usage:
    uv run python examples/advanced/01_backend_selection.py

Prerequisites:
    - None

Expected Output:
    - Console logs comparing fast-pysf and dummy backend resets.

Limitations:
    - Headless run only; no rendering or recording occurs.

References:
    - docs/dev_guide.md#environment-factory
"""

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig


def demo_fast_pysf_backend():
    """Standard fast-pysf backend (default)."""
    print("\n=== Demo: fast-pysf backend ===")
    config = RobotSimulationConfig()
    config.backend = "fast-pysf"
    config.peds_have_obstacle_forces = True

    env = make_robot_env(config=config, debug=False)
    obs, _info = env.reset(seed=123)
    obs_size = len(obs) if isinstance(obs, dict) else obs.shape
    print(f"Reset complete. Observation: {obs_size}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, _info = env.step(action)
        print(f"  Step {step + 1}: reward={reward:.3f} term={term} trunc={trunc}")
        if term or trunc:
            break

    env.close()
    print("Fast-pysf backend demo complete.\n")


def demo_dummy_backend():
    """Minimal dummy backend for testing."""
    print("\n=== Demo: dummy backend ===")
    config = RobotSimulationConfig()
    config.backend = "dummy"

    env = make_robot_env(config=config, debug=False)
    obs, _info = env.reset(seed=456)
    obs_size = len(obs) if isinstance(obs, dict) else getattr(obs, "shape", len(obs))
    print(f"Reset complete. Observation: {obs_size}")

    for step in range(3):
        action = env.action_space.sample()
        obs, reward, term, trunc, _info = env.step(action)
        print(f"  Step {step + 1}: reward={reward:.3f} term={term} trunc={trunc}")
        if term or trunc:
            break

    env.close()
    print("Dummy backend demo complete.\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Backend Selection Demo")
    print("=" * 60)
    print("\nDemonstrates swapping simulation backends via config")
    print("without changing environment code.\n")

    demo_fast_pysf_backend()
    demo_dummy_backend()

    print("=" * 60)
    print("All backend demos completed successfully!")
    print("=" * 60 + "\n")
