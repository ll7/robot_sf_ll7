#!/usr/bin/env python3
"""Explore Social Force planner scenarios and options.

Usage:
    uv run python examples/advanced/12_social_force_planner_demo.py [--scenario SCENARIO] [--visualize]

Prerequisites:
    - fast-pysf subtree (included in repo)
    - `uv sync --all-extras` to install numpy, numba, matplotlib

Expected Output:
    - Console summaries (and optional plots) for each Social Force scenario.

Limitations:
    - Visualization requires a GUI backend; use `--mock` for dependency-light runs.

References:
    - docs/baselines/social_force.md
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

# Try to import required packages, fall back to mock mode if missing
try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

    # Mock numpy for basic functionality
    class MockNumpy:
        """TODO docstring. Document this class."""

        def array(self, data, dtype=None):
            """TODO docstring. Document this function.

            Args:
                data: TODO docstring.
                dtype: TODO docstring.
            """
            return data

        # Mock ndarray attribute for type hints
        ndarray = list

        def linalg(self):
            """TODO docstring. Document this function."""
            return self

        def norm(self, vec):
            """TODO docstring. Document this function.

            Args:
                vec: TODO docstring.
            """
            return math.sqrt(sum(x * x for x in vec))

        def mean(self, data):
            """TODO docstring. Document this function.

            Args:
                data: TODO docstring.
            """
            return sum(data) / len(data)

        def max(self, data):
            """TODO docstring. Document this function.

            Args:
                data: TODO docstring.
            """
            return max(data)

    np = MockNumpy()  # type: ignore[assignment]

# Add the robot_sf package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import Social Force planner, fall back to mock if dependencies missing
try:
    from robot_sf.baselines import get_baseline
    from robot_sf.baselines.social_force import Observation, SFPlannerConfig

    SOCIAL_FORCE_AVAILABLE = True
except ImportError as e:
    SOCIAL_FORCE_AVAILABLE = False
    print(f"⚠️  Social Force implementation not available: {e}")
    print("   This may be due to missing dependencies (numba, fast-pysf subtree).")
    print("   Running in mock mode to demonstrate interface...")

    # Mock implementations for demo purposes
    class Observation:
        """TODO docstring. Document this class."""

        def __init__(self, dt, robot, agents, obstacles=None):
            """TODO docstring. Document this function.

            Args:
                dt: TODO docstring.
                robot: TODO docstring.
                agents: TODO docstring.
                obstacles: TODO docstring.
            """
            self.dt = dt
            self.robot = robot
            self.agents = agents or []
            self.obstacles = obstacles or []

    class SFPlannerConfig:
        """TODO docstring. Document this class."""

        def __init__(self, **kwargs):
            # Default configuration values
            """TODO docstring. Document this function.

            Args:
                kwargs: TODO docstring.
            """
            self.action_space = kwargs.get("action_space", "velocity")
            self.v_max = kwargs.get("v_max", 2.0)
            self.omega_max = kwargs.get("omega_max", 1.0)
            self.desired_speed = kwargs.get("desired_speed", 1.0)
            self.A = kwargs.get("A", 5.1)
            self.B = kwargs.get("B", 0.35)
            self.lambda_anisotropy = kwargs.get("lambda_anisotropy", 2.0)
            self.noise_std = kwargs.get("noise_std", 0.0)
            self.safety_clamp = kwargs.get("safety_clamp", True)
            # Apply all other kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MockSocialForcePlanner:
        """Mock Social Force Planner for demonstration purposes."""

        def __init__(self, config, seed=None):
            """TODO docstring. Document this function.

            Args:
                config: TODO docstring.
                seed: TODO docstring.
            """
            if isinstance(config, dict):
                self.config = SFPlannerConfig(**config)
            else:
                self.config = config
            self.seed = seed

        def step(self, obs) -> dict[str, float]:
            """Mock implementation that demonstrates basic Social Force behavior."""
            robot_pos = obs.robot["position"]
            goal_pos = obs.robot["goal"]

            # Goal attraction force (simplified)
            goal_dir = [goal_pos[0] - robot_pos[0], goal_pos[1] - robot_pos[1]]
            goal_dist = math.sqrt(goal_dir[0] ** 2 + goal_dir[1] ** 2)

            if goal_dist > 0.1:
                goal_dir = [goal_dir[0] / goal_dist, goal_dir[1] / goal_dist]
                desired_vel = [
                    goal_dir[0] * self.config.desired_speed,
                    goal_dir[1] * self.config.desired_speed,
                ]
            else:
                desired_vel = [0.0, 0.0]

            # Agent repulsion forces (simplified)
            repulsion = [0.0, 0.0]
            for agent in obs.agents:
                agent_pos = agent["position"]
                dx = robot_pos[0] - agent_pos[0]
                dy = robot_pos[1] - agent_pos[1]
                dist = math.sqrt(dx**2 + dy**2)

                if dist > 0.1:  # Avoid division by zero
                    # Simple exponential repulsion
                    force_mag = self.config.A * math.exp(-dist / self.config.B)
                    repulsion[0] += force_mag * dx / dist
                    repulsion[1] += force_mag * dy / dist

            # Combine forces to get desired velocity
            total_vel = [desired_vel[0] + repulsion[0], desired_vel[1] + repulsion[1]]

            # Apply velocity limits if safety_clamp is enabled
            if self.config.safety_clamp:
                vel_mag = math.sqrt(total_vel[0] ** 2 + total_vel[1] ** 2)
                if vel_mag > self.config.v_max:
                    scale = self.config.v_max / vel_mag
                    total_vel = [total_vel[0] * scale, total_vel[1] * scale]

            # Return action based on action space
            if self.config.action_space == "unicycle":
                # Convert to unicycle commands
                v = math.sqrt(total_vel[0] ** 2 + total_vel[1] ** 2)
                current_vel = obs.robot["velocity"]
                current_heading = math.atan2(current_vel[1], current_vel[0])
                desired_heading = math.atan2(total_vel[1], total_vel[0])

                omega = desired_heading - current_heading
                # Normalize angle to [-pi, pi]
                while omega > math.pi:
                    omega -= 2 * math.pi
                while omega < -math.pi:
                    omega += 2 * math.pi

                # Apply limits
                if abs(omega) > self.config.omega_max:
                    omega = self.config.omega_max * (1 if omega > 0 else -1)

                return {"v": min(v, self.config.v_max), "omega": omega}
            return {"vx": total_vel[0], "vy": total_vel[1]}

        def reset(self, seed=None):
            """TODO docstring. Document this function.

            Args:
                seed: TODO docstring.
            """
            if seed is not None:
                self.seed = seed

        def configure(self, new_config):
            """TODO docstring. Document this function.

            Args:
                new_config: TODO docstring.
            """
            if isinstance(new_config, dict):
                for key, value in new_config.items():
                    setattr(self.config, key, value)

        def close(self):
            """TODO docstring. Document this function."""
            pass

    def get_baseline(name):
        """TODO docstring. Document this function.

        Args:
            name: TODO docstring.
        """
        if name == "baseline_sf":
            return MockSocialForcePlanner
        else:
            raise KeyError(f"Unknown baseline: {name}")


# Ensure a mock planner is always available if user forces mock mode via --mock
if "MockSocialForcePlanner" not in globals():

    class MockSocialForcePlanner:  # type: ignore[redefinition]
        """Lightweight mock Social Force Planner (fallback when real deps present)."""

        def __init__(self, config, seed=None):
            """TODO docstring. Document this function.

            Args:
                config: TODO docstring.
                seed: TODO docstring.
            """
            self.config = config
            self.seed = seed

        def step(self, obs) -> dict[str, float]:
            # Trivial straight-line goal seeker with zero avoidance
            """TODO docstring. Document this function.

            Args:
                obs: TODO docstring.

            Returns:
                TODO docstring.
            """
            goal = obs.robot["goal"]
            pos = obs.robot["position"]
            dx = goal[0] - pos[0]
            dy = goal[1] - pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-6:
                ux, uy = dx / dist, dy / dist
            else:
                ux, uy = 0.0, 0.0
            speed = min(
                getattr(self.config, "desired_speed", 1.0),
                getattr(self.config, "v_max", 1.0),
            )
            if getattr(self.config, "action_space", "velocity") == "unicycle":
                return {"v": 0.0 if dist < 1e-3 else min(speed, dist), "omega": 0.0}
            return {"vx": ux * speed, "vy": uy * speed}

        def reset(self, seed=None):
            """TODO docstring. Document this function.

            Args:
                seed: TODO docstring.
            """
            if seed is not None:
                self.seed = seed

        def configure(self, new_config):
            """TODO docstring. Document this function.

            Args:
                new_config: TODO docstring.
            """
            pass

        def close(self):
            """TODO docstring. Document this function."""
            pass


class SFPDemo:
    """Social Force Planner demonstration class."""

    def __init__(self, visualize: bool = False, dt: float = 0.1, mock_mode: bool = False):
        """Initialize the demo.

        Args:
            visualize: Whether to show real-time visualization
            dt: Simulation timestep
            mock_mode: Use mock planner if dependencies are missing
        """
        self.visualize = visualize and MATPLOTLIB_AVAILABLE
        self.dt = dt
        self.mock_mode = mock_mode or not SOCIAL_FORCE_AVAILABLE

        if self.mock_mode:
            # Always use mock planner when mock_mode requested
            self.SocialForcePlanner = MockSocialForcePlanner
        else:
            self.SocialForcePlanner = get_baseline("baseline_sf")

        # Visualization setup
        if self.visualize:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 8))

        if self.mock_mode:
            print("🎭 Running in mock mode (dependencies not available)")
            print("    This demo shows the interface and basic planning concepts")
            print("    For full Social Force simulation, install: numba, fast-pysf submodule")

    def run_scenario(self, scenario_name: str, max_steps: int = 200) -> dict[str, Any]:
        """Run a specific scenario and return results.

        Args:
            scenario_name: Name of the scenario to run
            max_steps: Maximum number of simulation steps

        Returns:
            Dictionary with scenario results including trajectory and metrics
        """
        print(f"\n🎯 Running scenario: {scenario_name}")
        print("=" * 50)

        if scenario_name == "basic":
            return self._run_basic_scenario(max_steps)
        elif scenario_name == "single_ped":
            return self._run_single_pedestrian_scenario(max_steps)
        elif scenario_name == "multiple":
            return self._run_multiple_pedestrians_scenario(max_steps)
        elif scenario_name == "crossing":
            return self._run_crossing_scenario(max_steps)
        elif scenario_name == "corridor":
            return self._run_corridor_scenario(max_steps)
        elif scenario_name == "unicycle":
            return self._run_unicycle_demo(max_steps)
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")

    def _run_basic_scenario(self, max_steps: int) -> dict[str, Any]:
        """Basic goal navigation without obstacles."""
        print("📍 Basic Goal Navigation")
        print("Robot starts at (0,0) and navigates to (10,0) with no obstacles")

        # Configuration
        config = SFPlannerConfig(
            action_space="velocity",
            v_max=2.0,
            desired_speed=1.5,
            noise_std=0.0,
            safety_clamp=True,
        )

        planner = self.SocialForcePlanner(config, seed=42)

        # Initial conditions
        robot_pos = [0.0, 0.0]
        robot_vel = [0.0, 0.0]
        goal_pos = [10.0, 0.0]

        return self._simulate_scenario(
            planner=planner,
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            goal_pos=goal_pos,
            agents=[],
            obstacles=[],
            max_steps=max_steps,
            scenario_name="Basic Navigation",
        )

    def _run_single_pedestrian_scenario(self, max_steps: int) -> dict[str, Any]:
        """Navigation around a single pedestrian."""
        print("🚶 Single Pedestrian Avoidance")
        print("Robot navigates around a stationary pedestrian directly in its path")

        config = SFPlannerConfig(
            action_space="velocity",
            v_max=1.5,
            desired_speed=1.0,
            A=8.0,  # Strong social force for clear avoidance
            B=0.4,
            lambda_anisotropy=2.5,  # Stronger forward avoidance
            noise_std=0.0,
        )

        planner = self.SocialForcePlanner(config, seed=42)

        # Initial conditions
        robot_pos = [0.0, 0.0]
        robot_vel = [0.0, 0.0]
        goal_pos = [10.0, 0.0]

        # Pedestrian directly in path
        agents = [{"position": [5.0, 0.0], "velocity": [0.0, 0.0], "radius": 0.35}]

        return self._simulate_scenario(
            planner=planner,
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            goal_pos=goal_pos,
            agents=agents,
            obstacles=[],
            max_steps=max_steps,
            scenario_name="Single Pedestrian Avoidance",
        )

    def _run_multiple_pedestrians_scenario(self, max_steps: int) -> dict[str, Any]:
        """Navigation through multiple pedestrians."""
        print("👥 Multiple Pedestrian Navigation")
        print("Robot navigates through a crowd of pedestrians")

        config = SFPlannerConfig(
            action_space="velocity",
            v_max=1.5,
            desired_speed=1.0,
            A=5.1,
            B=0.35,
            lambda_anisotropy=2.0,
            cutoff_radius=5.0,  # Larger interaction radius
            noise_std=0.0,
        )

        planner = self.SocialForcePlanner(config, seed=42)

        # Initial conditions
        robot_pos = [0.0, 2.0]
        robot_vel = [0.0, 0.0]
        goal_pos = [10.0, 2.0]

        # Multiple pedestrians with different positions and velocities
        agents = [
            {"position": [3.0, 2.5], "velocity": [-0.2, 0.1], "radius": 0.35},
            {"position": [4.5, 1.5], "velocity": [0.3, 0.0], "radius": 0.35},
            {"position": [6.0, 3.0], "velocity": [0.1, -0.3], "radius": 0.35},
            {"position": [7.5, 1.0], "velocity": [-0.1, 0.2], "radius": 0.35},
            {"position": [8.0, 2.8], "velocity": [0.0, -0.1], "radius": 0.35},
        ]

        return self._simulate_scenario(
            planner=planner,
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            goal_pos=goal_pos,
            agents=agents,
            obstacles=[],
            max_steps=max_steps,
            scenario_name="Multiple Pedestrian Navigation",
        )

    def _run_crossing_scenario(self, max_steps: int) -> dict[str, Any]:
        """Pedestrians crossing robot's path."""
        print("↔️ Crossing Pedestrians")
        print("Robot navigates while pedestrians cross its path")

        config = SFPlannerConfig(
            action_space="velocity",
            v_max=1.5,
            desired_speed=1.2,
            A=6.0,  # Stronger repulsion for dynamic avoidance
            B=0.3,
            lambda_anisotropy=3.0,
            noise_std=0.0,
        )

        planner = self.SocialForcePlanner(config, seed=42)

        # Initial conditions
        robot_pos = [0.0, 2.0]
        robot_vel = [0.0, 0.0]
        goal_pos = [10.0, 2.0]

        # Pedestrians crossing from different directions
        agents = [
            {"position": [3.0, 0.0], "velocity": [0.0, 0.8], "radius": 0.35},  # Crossing up
            {"position": [6.0, 4.0], "velocity": [0.0, -0.6], "radius": 0.35},  # Crossing down
            {"position": [8.5, 0.5], "velocity": [0.0, 0.7], "radius": 0.35},  # Crossing up
        ]

        return self._simulate_scenario(
            planner=planner,
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            goal_pos=goal_pos,
            agents=agents,
            obstacles=[],
            max_steps=max_steps,
            scenario_name="Crossing Pedestrians",
            dynamic_agents=True,
        )

    def _run_corridor_scenario(self, max_steps: int) -> dict[str, Any]:
        """Navigation through a narrow corridor."""
        print("🏢 Corridor Navigation")
        print("Robot navigates through a narrow corridor with walls")

        config = SFPlannerConfig(
            action_space="velocity",
            v_max=1.2,
            desired_speed=0.8,  # Slower for careful navigation
            A=5.1,
            B=0.35,
            A_obs=15.0,  # Strong obstacle repulsion
            B_obs=0.1,  # Sharp wall forces
            noise_std=0.0,
        )

        planner = self.SocialForcePlanner(config, seed=42)

        # Initial conditions
        robot_pos = [0.0, 2.0]
        robot_vel = [0.0, 0.0]
        goal_pos = [10.0, 2.0]

        # Corridor walls (simplified as static obstacles)
        agents = [
            {
                "position": [5.0, 3.2],
                "velocity": [0.0, 0.0],
                "radius": 0.35,
            },  # Pedestrian in corridor
        ]

        # Note: Wall obstacles would need obstacle force implementation
        # For this demo, we simulate narrow corridor with strategic pedestrians
        obstacles = []  # Obstacle forces would be implemented here

        return self._simulate_scenario(
            planner=planner,
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            goal_pos=goal_pos,
            agents=agents,
            obstacles=obstacles,
            max_steps=max_steps,
            scenario_name="Corridor Navigation",
        )

    def _run_unicycle_demo(self, max_steps: int) -> dict[str, Any]:
        """Demonstration of unicycle action space."""
        print("🤖 Unicycle Control Demo")
        print("Robot uses unicycle controls (v, omega) instead of velocity (vx, vy)")

        config = SFPlannerConfig(
            action_space="unicycle",  # Key difference!
            v_max=1.5,
            omega_max=1.2,
            desired_speed=1.0,
            A=5.1,
            B=0.35,
            noise_std=0.0,
        )

        planner = self.SocialForcePlanner(config, seed=42)

        # Initial conditions - robot faces different direction than goal
        robot_pos = [0.0, 0.0]
        robot_vel = [1.0, 0.0]  # Initially moving in +x
        goal_pos = [5.0, 5.0]  # Goal requires turning

        agents = [
            {"position": [2.0, 1.0], "velocity": [0.0, 0.0], "radius": 0.35},
        ]

        return self._simulate_scenario(
            planner=planner,
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            goal_pos=goal_pos,
            agents=agents,
            obstacles=[],
            max_steps=max_steps,
            scenario_name="Unicycle Control",
            unicycle_mode=True,
        )

    def _simulate_scenario(  # noqa: C901,PLR0913,PLR0915
        self,
        planner,
        robot_pos: list[float],
        robot_vel: list[float],
        goal_pos: list[float],
        agents: list[dict],
        obstacles: list[dict],
        max_steps: int,
        scenario_name: str,
        dynamic_agents: bool = False,
        unicycle_mode: bool = False,
    ) -> dict[str, Any]:
        """Run the simulation for a scenario."""

        trajectory = []
        goal_tolerance = 0.3

        # Convert to arrays for easier manipulation (handle both real numpy and mock)
        if hasattr(np, "array"):
            robot_pos = np.array(robot_pos, dtype=float)
            robot_vel = np.array(robot_vel, dtype=float)
            goal_pos = np.array(goal_pos, dtype=float)
        else:
            # Mock numpy - just use lists
            robot_pos = [float(x) for x in robot_pos]
            robot_vel = [float(x) for x in robot_vel]
            goal_pos = [float(x) for x in goal_pos]

        # For unicycle mode, we need to track orientation
        if unicycle_mode:
            robot_orientation = math.atan2(robot_vel[1], robot_vel[0])

        print(f"🚀 Starting simulation: {scenario_name}")
        print(f"   Start: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")
        print(f"   Goal:  ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
        print(f"   Agents: {len(agents)}")

        if self.visualize:
            self._setup_visualization(robot_pos, goal_pos, agents, scenario_name)

        for step in range(max_steps):
            # Update dynamic agents if needed
            if dynamic_agents:
                for agent in agents:
                    agent["position"][0] += agent["velocity"][0] * self.dt
                    agent["position"][1] += agent["velocity"][1] * self.dt

            # Create observation
            # Convert container types safely
            def _to_list(x):
                """TODO docstring. Document this function.

                Args:
                    x: TODO docstring.
                """
                if hasattr(x, "tolist"):
                    return x.tolist()
                if isinstance(x, list | tuple):
                    return list(x)
                return [x]

            obs = Observation(
                dt=self.dt,
                robot={
                    "position": _to_list(robot_pos),
                    "velocity": _to_list(robot_vel),
                    "goal": _to_list(goal_pos),
                    "radius": 0.3,
                },
                agents=agents.copy(),
                obstacles=obstacles.copy(),
            )

            # Get action from planner
            action = planner.step(obs)

            # Update robot state based on action type
            if unicycle_mode:
                # Unicycle dynamics
                v = action["v"]
                omega = action["omega"]

                # Update orientation
                robot_orientation += omega * self.dt

                # Update velocity based on current orientation
                robot_vel[0] = v * math.cos(robot_orientation)
                robot_vel[1] = v * math.sin(robot_orientation)

                # Update position
                robot_pos[0] += robot_vel[0] * self.dt
                robot_pos[1] += robot_vel[1] * self.dt

            else:
                # Velocity control
                robot_vel[0] = action["vx"]
                robot_vel[1] = action["vy"]
                robot_pos[0] += robot_vel[0] * self.dt
                robot_pos[1] += robot_vel[1] * self.dt

            # Record trajectory
            trajectory_point = {
                "step": step,
                "time": step * self.dt,
                "position": robot_pos.copy(),
                "velocity": robot_vel.copy(),
                "action": action.copy(),
            }

            if unicycle_mode:
                trajectory_point["orientation"] = robot_orientation

            trajectory.append(trajectory_point)

            # Update visualization
            if self.visualize:
                self._update_visualization(robot_pos, robot_vel, agents, trajectory, step)
                plt.pause(0.05)

            # Check if goal reached
            goal_distance = math.sqrt(
                (robot_pos[0] - goal_pos[0]) ** 2 + (robot_pos[1] - goal_pos[1]) ** 2,
            )
            if goal_distance < goal_tolerance:
                print(f"✅ Goal reached in {step} steps ({step * self.dt:.1f}s)")
                print(f"   Final position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
                print(f"   Goal distance: {goal_distance:.3f}m")
                break

            # Progress updates
            if step > 0 and step % 50 == 0:
                print(f"   Step {step}: distance to goal = {goal_distance:.2f}m")

        else:
            goal_distance = math.sqrt(
                (robot_pos[0] - goal_pos[0]) ** 2 + (robot_pos[1] - goal_pos[1]) ** 2,
            )
            print(f"⏰ Simulation timeout after {max_steps} steps")
            print(f"   Final distance to goal: {goal_distance:.2f}m")

        planner.close()

        # Calculate metrics
        metrics = self._calculate_metrics(trajectory, goal_pos, goal_tolerance)

        return {
            "scenario": scenario_name,
            "trajectory": trajectory,
            "metrics": metrics,
            "config": planner.config,
            "success": goal_distance < goal_tolerance,
        }

    def _setup_visualization(self, robot_pos, goal_pos, agents, scenario_name):
        """Set up the visualization plot."""
        if not self.visualize or not MATPLOTLIB_AVAILABLE:
            return

        self.ax.clear()
        self.ax.set_xlim(-1, 12)
        self.ax.set_ylim(-1, 6)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f"Social Force Planner Demo: {scenario_name}")
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")

        # Plot goal
        goal_circle = plt.Circle(goal_pos, 0.3, color="green", alpha=0.5, label="Goal")
        self.ax.add_patch(goal_circle)

        # Plot initial robot position
        robot_circle = plt.Circle(robot_pos, 0.3, color="blue", alpha=0.7, label="Robot")
        self.ax.add_patch(robot_circle)

        # Plot agents
        for i, agent in enumerate(agents):
            agent_circle = plt.Circle(
                agent["position"],
                agent["radius"],
                color="red",
                alpha=0.6,
                label="Pedestrian" if i == 0 else "",
            )
            self.ax.add_patch(agent_circle)

        self.ax.legend()
        plt.tight_layout()

    def _update_visualization(self, robot_pos, robot_vel, agents, trajectory, step):
        """Update the visualization with current state."""
        if not self.visualize or not MATPLOTLIB_AVAILABLE:
            return

        if step % 10 != 0:  # Update every 10 steps for performance
            return

        # Clear previous robot and agent positions
        self.ax.patches = [
            patch for patch in self.ax.patches if patch.get_facecolor()[2] == 0.5
        ]  # Keep only goal (green)

        # Plot trajectory
        if len(trajectory) > 1:
            traj_x = [p["position"][0] for p in trajectory[::5]]  # Every 5th point
            traj_y = [p["position"][1] for p in trajectory[::5]]
            self.ax.plot(traj_x, traj_y, "b--", alpha=0.5, linewidth=1)

        # Plot current robot position
        robot_circle = plt.Circle(robot_pos, 0.3, color="blue", alpha=0.8)
        self.ax.add_patch(robot_circle)

        # Plot velocity vector
        vel_mag = math.sqrt(robot_vel[0] ** 2 + robot_vel[1] ** 2)
        if vel_mag > 0.1:
            self.ax.arrow(
                robot_pos[0],
                robot_pos[1],
                robot_vel[0] * 0.5,
                robot_vel[1] * 0.5,
                head_width=0.1,
                head_length=0.1,
                fc="blue",
                ec="blue",
                alpha=0.7,
            )

        # Plot current agent positions
        for agent in agents:
            agent_circle = plt.Circle(agent["position"], agent["radius"], color="red", alpha=0.6)
            self.ax.add_patch(agent_circle)

            # Plot agent velocity vector
            agent_vel = agent["velocity"]
            if abs(agent_vel[0]) > 0.05 or abs(agent_vel[1]) > 0.05:
                self.ax.arrow(
                    agent["position"][0],
                    agent["position"][1],
                    agent_vel[0] * 0.5,
                    agent_vel[1] * 0.5,
                    head_width=0.05,
                    head_length=0.05,
                    fc="red",
                    ec="red",
                    alpha=0.5,
                )

    def _calculate_metrics(
        self,
        trajectory: list[dict],
        goal_pos: np.ndarray,
        goal_tolerance: float,
    ) -> dict[str, float]:
        """Calculate performance metrics for the trajectory."""
        if not trajectory:
            return {}

        # Path length
        path_length = 0.0
        for i in range(1, len(trajectory)):
            pos1 = trajectory[i - 1]["position"]
            pos2 = trajectory[i]["position"]
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            path_length += math.sqrt(dx * dx + dy * dy)

        # Direct distance (optimal path length)
        start_pos = trajectory[0]["position"]
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        direct_distance = math.sqrt(dx * dx + dy * dy)

        # Final distance to goal
        final_pos = trajectory[-1]["position"]
        dx = goal_pos[0] - final_pos[0]
        dy = goal_pos[1] - final_pos[1]
        final_distance = math.sqrt(dx * dx + dy * dy)

        # Average and maximum speed
        speeds = []
        for point in trajectory:
            vel = point["velocity"]
            speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
            speeds.append(speed)

        if hasattr(np, "mean"):
            avg_speed = np.mean(speeds) if speeds else 0.0
            max_speed = np.max(speeds) if speeds else 0.0
        else:
            # Mock numpy
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
            max_speed = max(speeds) if speeds else 0.0

        # Path efficiency (lower is better)
        path_efficiency = path_length / direct_distance if direct_distance > 0 else float("inf")

        # Success
        success = final_distance < goal_tolerance

        return {
            "path_length": path_length,
            "direct_distance": direct_distance,
            "path_efficiency": path_efficiency,
            "final_distance": final_distance,
            "total_time": trajectory[-1]["time"],
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "success": success,
            "num_steps": len(trajectory),
        }

    def print_results(self, results: dict[str, Any]):
        """Print scenario results in a formatted way."""
        print(f"\n📊 Results for {results['scenario']}:")
        print("-" * 40)

        metrics = results["metrics"]

        if metrics.get("success", False):
            print("✅ SUCCESS - Goal reached!")
        else:
            print("❌ TIMEOUT - Goal not reached")

        print(f"Total time:        {metrics.get('total_time', 0):.1f}s")
        print(f"Total steps:       {metrics.get('num_steps', 0)}")
        print(f"Path length:       {metrics.get('path_length', 0):.2f}m")
        print(f"Direct distance:   {metrics.get('direct_distance', 0):.2f}m")
        print(f"Path efficiency:   {metrics.get('path_efficiency', 0):.2f}x")
        print(f"Final distance:    {metrics.get('final_distance', 0):.3f}m")
        print(f"Average speed:     {metrics.get('avg_speed', 0):.2f}m/s")
        print(f"Maximum speed:     {metrics.get('max_speed', 0):.2f}m/s")

        # Action space info
        config = results["config"]
        print(f"Action space:      {config.action_space}")
        print(f"Max velocity:      {config.v_max:.1f}m/s")
        if config.action_space == "unicycle":
            print(f"Max ang. velocity: {config.omega_max:.1f}rad/s")

    def demo_configuration_effects(self):
        """Demonstrate the effects of different configuration parameters."""
        print("\n🔧 Configuration Effects Demo")
        print("=" * 50)
        print("Comparing different Social Force parameters on the same scenario")

        # Base scenario: robot at origin, goal at (8,0), pedestrian at (4,0)
        robot_pos = [0.0, 0.0]
        robot_vel = [0.0, 0.0]
        goal_pos = [8.0, 0.0]
        agents = [{"position": [4.0, 0.0], "velocity": [0.0, 0.0], "radius": 0.35}]

        # Different configurations to test
        configs = [
            ("Low Social Force", {"A": 2.0, "B": 0.5}),
            ("High Social Force", {"A": 10.0, "B": 0.2}),
            ("Anisotropic", {"A": 5.1, "B": 0.35, "lambda_anisotropy": 4.0}),
            ("Conservative", {"desired_speed": 0.5, "v_max": 1.0}),
            ("Aggressive", {"desired_speed": 2.0, "v_max": 2.5}),
        ]

        results = []

        for config_name, config_params in configs:
            print(f"\n🧪 Testing {config_name} configuration")
            # Merge base defaults with overrides to avoid duplicate keyword errors
            base_params = {
                "action_space": "velocity",
                "v_max": 1.5,
                "desired_speed": 1.0,
                "A": 5.1,
                "B": 0.35,
                "noise_std": 0.0,
            }
            base_params.update(config_params)
            config = SFPlannerConfig(**base_params)

            planner = self.SocialForcePlanner(config, seed=42)

            result = self._simulate_scenario(
                planner=planner,
                robot_pos=robot_pos.copy(),
                robot_vel=robot_vel.copy(),
                goal_pos=goal_pos,
                agents=agents.copy(),
                obstacles=[],
                max_steps=150,
                scenario_name=config_name,
            )

            results.append(result)

        # Compare results
        print("\n🔍 Configuration Comparison:")
        print("-" * 60)
        print(f"{'Config':<15} {'Success':<8} {'Time':<6} {'Path':<6} {'Efficiency':<10}")
        print("-" * 60)

        for result in results:
            metrics = result["metrics"]
            success = "✅ Yes" if metrics.get("success", False) else "❌ No"
            time_str = f"{metrics.get('total_time', 0):.1f}s"
            path_str = f"{metrics.get('path_length', 0):.1f}m"
            eff_str = f"{metrics.get('path_efficiency', 0):.2f}x"

            print(
                f"{result['scenario']:<15} {success:<8} {time_str:<6} {path_str:<6} {eff_str:<10}",
            )

    def close(self):
        """Clean up visualization resources."""
        if self.visualize and MATPLOTLIB_AVAILABLE:
            plt.ioff()
            plt.close("all")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Social Force Planner Demo")
    parser.add_argument(
        "--scenario",
        choices=[
            "basic",
            "single_ped",
            "multiple",
            "crossing",
            "corridor",
            "unicycle",
            "config",
            "all",
        ],
        default="all",
        help="Scenario to run",
    )
    parser.add_argument("--visualize", action="store_true", help="Show real-time visualization")
    parser.add_argument("--steps", type=int, default=200, help="Maximum simulation steps")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode (no full dependencies)",
    )

    args = parser.parse_args()

    print("🤖 Social Force Planner (SFP) Demo")
    print("=" * 50)
    print("This demo showcases the Social Force baseline algorithm for navigation planning.")
    print("The Social Force model treats navigation as a physics system with attractive")
    print("forces toward goals and repulsive forces from obstacles and other agents.")
    print()

    if not SOCIAL_FORCE_AVAILABLE or args.mock:
        print("🎭 Running in mock mode due to missing dependencies")
        print("   To run with full implementation, ensure:")
        print("   1. git submodule update --init --recursive")
        print("   2. pip install numba numpy matplotlib")
        print("   3. Add fast-pysf to PYTHONPATH")
        print()

    if args.visualize and not MATPLOTLIB_AVAILABLE:
        print("⚠️  Visualization disabled (matplotlib not available)")
        args.visualize = False
    elif args.visualize:
        print("📺 Real-time visualization enabled")
        print("   Close the plot window to continue to the next scenario")

    demo = SFPDemo(visualize=args.visualize, mock_mode=args.mock)

    try:
        if args.scenario == "all":
            # Run all scenarios
            scenarios = ["basic", "single_ped", "multiple", "crossing", "corridor", "unicycle"]

            for scenario in scenarios:
                result = demo.run_scenario(scenario, args.steps)
                demo.print_results(result)

                if args.visualize:
                    input("\nPress Enter to continue to next scenario...")

            # Configuration effects demo
            demo.demo_configuration_effects()

        elif args.scenario == "config":
            # Just run configuration effects demo
            demo.demo_configuration_effects()

        else:
            # Run specific scenario
            result = demo.run_scenario(args.scenario, args.steps)
            demo.print_results(result)

            if args.visualize:
                input("\nPress Enter to close...")

    finally:
        demo.close()

    print("\n✨ Demo complete!")
    print("\nKey takeaways:")
    print("• Social Force planners excel at natural, human-like navigation")
    print("• Force parameters (A, B) control avoidance strength and range")
    print("• Anisotropy creates stronger forward vs backward avoidance")
    print("• Both velocity and unicycle control modes are supported")
    print("• Deterministic behavior enables reproducible benchmarking")
    print("\nFor more information, see:")
    print("• docs/baselines/social_force.md - User guide and configuration")
    print("• robot_sf/baselines/social_force.py - Implementation details")


if __name__ == "__main__":
    main()
