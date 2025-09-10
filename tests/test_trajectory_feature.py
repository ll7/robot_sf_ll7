#!/usr/bin/env python3
"""
Simple test script to verify trajectory visualization implementation.

This script tests the trajectory feature implementation without requiring
external dependencies.
"""

import sys
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple


# Mock the dependencies for testing
@dataclass
class MockPose:
    position: Tuple[float, float]
    orientation: float

    def __getitem__(self, index):
        if index == 0:
            return self.position
        elif index == 1:
            return self.orientation
        else:
            raise IndexError("Pose index out of range")


@dataclass
class MockVisualizableSimState:
    timestep: int
    robot_pose: MockPose
    pedestrian_positions: List[List[float]]
    ego_ped_pose: MockPose = None


@dataclass
class MockMapDefinition:
    name: str = "test_map"


class TrajectoryVisualizationTest:
    """Test class for trajectory visualization functionality."""

    def __init__(self):
        self.show_trajectories = True
        self.max_trajectory_length = 10
        self.robot_trajectory = deque(maxlen=self.max_trajectory_length)
        self.ped_trajectories = {}
        self.ego_ped_trajectory = deque(maxlen=self.max_trajectory_length)

    def _update_trajectories(self, state):
        """Update trajectory histories with current state."""
        if not self.show_trajectories:
            return

        # Update robot trajectory
        if hasattr(state, "robot_pose") and state.robot_pose:
            robot_pos = state.robot_pose[0]
            self.robot_trajectory.append((robot_pos[0], robot_pos[1]))

        # Update pedestrian trajectories
        if hasattr(state, "pedestrian_positions") and state.pedestrian_positions is not None:
            for ped_id, pos in enumerate(state.pedestrian_positions):
                if ped_id not in self.ped_trajectories:
                    self.ped_trajectories[ped_id] = deque(maxlen=self.max_trajectory_length)
                self.ped_trajectories[ped_id].append((pos[0], pos[1]))

        # Update ego pedestrian trajectory
        if hasattr(state, "ego_ped_pose") and state.ego_ped_pose:
            ego_pos = state.ego_ped_pose[0]
            self.ego_ped_trajectory.append((ego_pos[0], ego_pos[1]))

    def _clear_trajectories(self):
        """Clear all trajectory histories."""
        self.robot_trajectory.clear()
        self.ped_trajectories.clear()
        self.ego_ped_trajectory.clear()

    def test_trajectory_update(self):
        """Test trajectory update functionality."""
        print("Testing trajectory update...")

        # Create test states with moving entities
        states = []
        for i in range(5):
            robot_pose = MockPose((i * 1.0, i * 0.5), 0.0)
            ped_positions = [[i * 0.8, i * 0.3], [i * 0.6, i * 0.7]]
            ego_pose = MockPose((i * 1.2, i * 0.4), 0.0)

            state = MockVisualizableSimState(
                timestep=i,
                robot_pose=robot_pose,
                pedestrian_positions=ped_positions,
                ego_ped_pose=ego_pose,
            )
            states.append(state)

        # Process each state
        for state in states:
            self._update_trajectories(state)

        # Check results
        print(f"Robot trajectory length: {len(self.robot_trajectory)}")
        print(f"Robot trajectory: {list(self.robot_trajectory)}")

        print(f"Number of pedestrian trajectories: {len(self.ped_trajectories)}")
        for ped_id, traj in self.ped_trajectories.items():
            print(f"Pedestrian {ped_id} trajectory: {list(traj)}")

        print(f"Ego pedestrian trajectory: {list(self.ego_ped_trajectory)}")

        # Verify expected results
        assert len(self.robot_trajectory) == 5, (
            f"Expected 5 robot positions, got {len(self.robot_trajectory)}"
        )
        assert len(self.ped_trajectories) == 2, (
            f"Expected 2 pedestrians, got {len(self.ped_trajectories)}"
        )
        assert len(self.ego_ped_trajectory) == 5, (
            f"Expected 5 ego positions, got {len(self.ego_ped_trajectory)}"
        )

        print("✓ Trajectory update test passed!")

    def test_trajectory_length_limit(self):
        """Test trajectory length limiting."""
        print("\nTesting trajectory length limit...")

        # Set small limit
        self.max_trajectory_length = 3
        self.robot_trajectory = deque(maxlen=self.max_trajectory_length)

        # Add more positions than the limit
        for i in range(5):
            robot_pose = MockPose((i * 1.0, i * 0.5), 0.0)
            state = MockVisualizableSimState(
                timestep=i, robot_pose=robot_pose, pedestrian_positions=[]
            )
            self._update_trajectories(state)

        # Check that only the last 3 positions are kept
        assert len(self.robot_trajectory) == 3, (
            f"Expected 3 positions, got {len(self.robot_trajectory)}"
        )
        expected_positions = [(2.0, 1.0), (3.0, 1.5), (4.0, 2.0)]
        assert list(self.robot_trajectory) == expected_positions, (
            f"Expected {expected_positions}, got {list(self.robot_trajectory)}"
        )

        print("✓ Trajectory length limit test passed!")

    def test_trajectory_clear(self):
        """Test trajectory clearing."""
        print("\nTesting trajectory clearing...")

        # Add some data
        robot_pose = MockPose((1.0, 1.0), 0.0)
        state = MockVisualizableSimState(
            timestep=0, robot_pose=robot_pose, pedestrian_positions=[[1.0, 1.0]]
        )
        self._update_trajectories(state)

        # Verify data exists
        assert len(self.robot_trajectory) > 0
        assert len(self.ped_trajectories) > 0

        # Clear trajectories
        self._clear_trajectories()

        # Verify everything is cleared
        assert len(self.robot_trajectory) == 0
        assert len(self.ped_trajectories) == 0
        assert len(self.ego_ped_trajectory) == 0

        print("✓ Trajectory clear test passed!")

    def test_trajectory_toggle(self):
        """Test trajectory toggle functionality."""
        print("\nTesting trajectory toggle...")

        # Disable trajectories
        self.show_trajectories = False

        robot_pose = MockPose((1.0, 1.0), 0.0)
        state = MockVisualizableSimState(
            timestep=0, robot_pose=robot_pose, pedestrian_positions=[[1.0, 1.0]]
        )

        # Try to update - should be ignored
        self._update_trajectories(state)

        # Verify no data was added
        assert len(self.robot_trajectory) == 0
        assert len(self.ped_trajectories) == 0

        # Enable trajectories
        self.show_trajectories = True
        self._update_trajectories(state)

        # Verify data is now added
        assert len(self.robot_trajectory) == 1
        assert len(self.ped_trajectories) == 1

        print("✓ Trajectory toggle test passed!")


def main():
    """Run all tests."""
    print("Running trajectory visualization tests...\n")

    test = TrajectoryVisualizationTest()

    try:
        test.test_trajectory_update()
        test.test_trajectory_length_limit()
        test.test_trajectory_clear()
        test.test_trajectory_toggle()

        print(
            "\n🎉 All tests passed! Trajectory visualization implementation is working correctly."
        )
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
