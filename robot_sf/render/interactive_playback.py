"""
Interactive playback module for recorded simulation states.

This module extends the SimulationView to provide interactive navigation through
recorded simulation states, allowing users to step forward/backward through frames
and control playback speed. Includes trajectory visualization for entities.
"""

from collections import deque
from typing import Deque, Dict, List, Tuple

import pygame
from loguru import logger

from robot_sf.nav.map_config import MapDefinition
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import (
    TEXT_BACKGROUND,
    TEXT_COLOR,
    SimulationView,
    VisualizableSimState,
)

# Trajectory visualization colors
ROBOT_TRAJECTORY_COLOR = (0, 100, 255)  # Blue
PED_TRAJECTORY_COLOR = (255, 100, 100)  # Light red
EGO_PED_TRAJECTORY_COLOR = (200, 0, 200)  # Magenta

# UI frame rate target (Hz)
UI_FPS = 60


class InteractivePlayback(SimulationView):
    """
    Interactive playback viewer that allows navigating through recorded simulation states.

    This class extends SimulationView to add playback controls and trajectory visualization:
    - Step forward/backward through frames
    - Jump to specific frames
    - Play/pause functionality
    - Display current frame index
    - Visualize entity trajectories during playback

    Attributes:
        states (List[VisualizableSimState]): List of states to playback
        current_frame (int): Index of the current frame being displayed
        is_playing (bool): Whether playback is currently running
        playback_speed (float): Multiplier for playback speed
        show_trajectories (bool): Whether to display entity trajectories
        max_trajectory_length (int): Maximum number of points in trajectory trails
        robot_trajectory (Deque[Tuple[float, float]]): Robot position history
        ped_trajectories (Dict[int, Deque[Tuple[float, float]]]): Pedestrian position histories
        ego_ped_trajectory (Deque[Tuple[float, float]]): Ego pedestrian position history
    """

    def __init__(
        self,
        states: List[VisualizableSimState],
        map_def: MapDefinition,
        sleep_time: float = 0.1,  # Default to 0.1 explicitly
        caption: str = "RobotSF Interactive Playback",
    ):
        """
        Initialize the interactive playback viewer.

        Args:
            states: List of VisualizableSimState objects to playback
            map_def: Map definition for the simulation
            sleep_time: Time to sleep between frames (default: 0.1s)
            caption: Window caption
        """
        super().__init__(map_def=map_def, caption=caption)
        self.states = states
        self.current_frame = 0
        self.sleep_time = sleep_time  # Store this directly as an instance variable
        self.is_playing = False
        self.playback_speed = 1.0
        self.total_frames = len(states)

        # Time tracking for playback
        self.last_update_time = 0

        # Trajectory visualization attributes
        self.show_trajectories = False
        # Use a private backing field during init to avoid triggering updates prematurely
        self._max_trajectory_length = 100  # Default trail length
        self.robot_trajectory: Deque[Tuple[float, float]] = deque(
            maxlen=self._max_trajectory_length
        )
        self.ped_trajectories: Dict[int, Deque[Tuple[float, float]]] = {}
        self.ego_ped_trajectory: Deque[Tuple[float, float]] = deque(
            maxlen=self._max_trajectory_length
        )

        # Add playback controls to the help text
        self._extend_help_text()

    def _extend_help_text(self):
        """Extend the help text with playback controls."""
        self._playback_help_lines = [
            "--- Playback Controls ---",
            "Space: Play/pause",
            "Period (.): Next frame",
            "Comma (,): Previous frame",
            "n: First frame",
            "m: Last frame",
            "k: Speed up",
            "j: Slow down",
            "--- Trajectory Controls ---",
            "v: Toggle trajectories",
            "b: Increase trail length",
            "c: Decrease trail length",
            "x: Clear trajectories",
        ]

    def _add_help_text(self):
        """Override the help text method to include playback controls."""
        # Call the parent method and get the rect of the standard help block
        parent_rect = super()._add_help_text()

        # Add playback specific help text below the standard help
        text_lines = self._playback_help_lines

        # Calculate position for the playback controls (below standard help),
        # using the actual height of the parent's help rect
        y_offset = parent_rect.bottom + 10

        max_width = max(self.font.size(line)[0] for line in text_lines)
        text_height = len(text_lines) * self.font.get_linesize()
        text_surface = pygame.Surface((max_width + 10, text_height + 10), pygame.SRCALPHA)
        text_surface.fill(TEXT_BACKGROUND)

        for i, text in enumerate(text_lines):
            text_render = self.font.render(text, True, TEXT_COLOR)
            text_outline = self.font.render(text, True, (0, 0, 0))

            pos = (5, i * self.font.get_linesize() + 5)

            # Draw text outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                text_surface.blit(text_outline, (pos[0] + dx, pos[1] + dy))

            # Draw main text
            text_surface.blit(text_render, pos)

        self.screen.blit(text_surface, (self.width - max_width - 10, y_offset))

    def _handle_keydown(self, e):
        """Handle key presses for both simulation view and playback controls."""
        # Quick toggle for play/pause
        if e.key == pygame.K_SPACE:
            self.is_playing = not self.is_playing
            return

        # Playback navigation and speed controls
        if self._handle_playback_key(e):
            return

        # Trajectory controls
        if self._handle_trajectory_key(e):
            return

        # If not a playback control key, let parent handle it
        super()._handle_keydown(e)

    def _handle_playback_key(self, e) -> bool:
        """Handle playback navigation and speed keys; return True if handled."""
        # Next frame (using period '.' instead of right arrow)
        if e.key == pygame.K_PERIOD and not self.is_playing:
            old_frame = self.current_frame
            self.current_frame = min(self.current_frame + 1, len(self.states) - 1)
            if self.current_frame != old_frame:
                self.redraw_needed = True
            return True

        # Previous frame (using comma ',' instead of left arrow)
        if e.key == pygame.K_COMMA and not self.is_playing:
            old_frame = self.current_frame
            self.current_frame = max(self.current_frame - 1, 0)
            if self.current_frame != old_frame:
                # When going backwards, rebuild trajectories
                self._rebuild_trajectories_up_to_frame(self.current_frame)
                self.redraw_needed = True
            return True

        # First frame
        if e.key == pygame.K_n:
            self.current_frame = 0
            self._rebuild_trajectories_up_to_frame(self.current_frame)
            self.redraw_needed = True
            return True

        # Last frame
        if e.key == pygame.K_m:
            self.current_frame = len(self.states) - 1
            self._rebuild_trajectories_up_to_frame(self.current_frame)
            self.redraw_needed = True
            return True

        # Speed up
        if e.key == pygame.K_k:
            self.playback_speed = min(self.playback_speed * 1.5, 10.0)
            return True

        # Slow down
        if e.key == pygame.K_j:
            self.playback_speed = max(self.playback_speed / 1.5, 0.1)
            return True

        return False

    def _handle_trajectory_key(self, e) -> bool:
        """Handle trajectory-related keys; return True if handled."""
        if e.key == pygame.K_v:
            self.show_trajectories = not self.show_trajectories
            if self.show_trajectories:
                self._rebuild_trajectories_up_to_frame(self.current_frame)
            logger.info(f"Trajectory display: {'ON' if self.show_trajectories else 'OFF'}")
            return True

        if e.key == pygame.K_b:
            self.set_trail_length(min(self.max_trajectory_length + 20, 500))
            logger.info(f"Trail length increased to: {self.max_trajectory_length}")
            return True

        if e.key == pygame.K_c:
            self.set_trail_length(max(self.max_trajectory_length - 20, 10))
            logger.info(f"Trail length decreased to: {self.max_trajectory_length}")
            return True

        if e.key == pygame.K_x:
            self._clear_trajectories()
            logger.info("Trajectories cleared")
            return True

        return False

    def _update_trajectory_maxlen(self):
        """Update the maximum length of trajectory deques."""
        # Update robot trajectory
        new_robot_trajectory = deque(self.robot_trajectory, maxlen=self.max_trajectory_length)
        self.robot_trajectory = new_robot_trajectory

        # Update pedestrian trajectories
        for ped_id in self.ped_trajectories:
            new_ped_trajectory = deque(
                self.ped_trajectories[ped_id], maxlen=self.max_trajectory_length
            )
            self.ped_trajectories[ped_id] = new_ped_trajectory

        # Update ego pedestrian trajectory
        new_ego_trajectory = deque(self.ego_ped_trajectory, maxlen=self.max_trajectory_length)
        self.ego_ped_trajectory = new_ego_trajectory

    def set_trail_length(self, length: int) -> None:
        """Set the maximum trajectory trail length and reconfigure existing deques.

        This is the preferred public API; it clamps to [10, 500] and reapplies
        the current maxlen to robot/ped/ego trajectories.
        """
        self.max_trajectory_length = length  # delegate to property setter

    # Property to ensure direct assignment also updates deques
    @property
    def max_trajectory_length(self) -> int:
        return getattr(self, "_max_trajectory_length", 100)

    @max_trajectory_length.setter
    def max_trajectory_length(self, value: int) -> None:
        clamped = max(10, min(int(value), 500))
        old = getattr(self, "_max_trajectory_length", None)
        if old == clamped:
            self._max_trajectory_length = clamped
            return
        self._max_trajectory_length = clamped
        # Reconfigure existing deques if they are already created
        if hasattr(self, "robot_trajectory"):
            self._update_trajectory_maxlen()

    def _clear_trajectories(self):
        """Clear all trajectory histories."""
        self.robot_trajectory.clear()
        self.ped_trajectories.clear()
        self.ego_ped_trajectory.clear()

    def _update_trajectories(self, state: VisualizableSimState):
        """Update trajectory histories with current state."""
        if not self.show_trajectories:
            return

        # Update robot trajectory
        if hasattr(state, "robot_pose") and state.robot_pose:
            robot_pos = state.robot_pose[0]  # Get position from pose
            self.robot_trajectory.append((robot_pos[0], robot_pos[1]))

        # Update pedestrian trajectories
        if hasattr(state, "pedestrian_positions") and state.pedestrian_positions is not None:
            for ped_id, pos in enumerate(state.pedestrian_positions):
                if ped_id not in self.ped_trajectories:
                    self.ped_trajectories[ped_id] = deque(maxlen=self.max_trajectory_length)
                self.ped_trajectories[ped_id].append((pos[0], pos[1]))

        # Update ego pedestrian trajectory
        if hasattr(state, "ego_ped_pose") and state.ego_ped_pose:
            ego_pos = state.ego_ped_pose[0]  # Get position from pose
            self.ego_ped_trajectory.append((ego_pos[0], ego_pos[1]))

    def _draw_trajectory(
        self, trajectory: Deque[Tuple[float, float]], color: Tuple[int, int, int], width: int = 2
    ):
        """Draw a trajectory as connected lines."""
        if len(trajectory) < 2:
            return

        points = []
        for pos in trajectory:
            scaled_pos = self._scale_tuple(pos)
            points.append(scaled_pos)

        # Draw trajectory as connected lines
        if len(points) >= 2:
            pygame.draw.lines(self.screen, color, False, points, width)

    def _draw_all_trajectories(self):
        """Draw all entity trajectories if enabled."""
        if not self.show_trajectories:
            return

        # Draw robot trajectory
        if len(self.robot_trajectory) > 1:
            self._draw_trajectory(self.robot_trajectory, ROBOT_TRAJECTORY_COLOR, 3)

        # Draw pedestrian trajectories
        for trajectory in self.ped_trajectories.values():
            if len(trajectory) > 1:
                self._draw_trajectory(trajectory, PED_TRAJECTORY_COLOR, 2)

        # Draw ego pedestrian trajectory
        if len(self.ego_ped_trajectory) > 1:
            self._draw_trajectory(self.ego_ped_trajectory, EGO_PED_TRAJECTORY_COLOR, 3)

    def _rebuild_trajectories_up_to_frame(self, target_frame: int):
        """Rebuild trajectory histories up to the specified frame."""
        if not self.show_trajectories:
            return

        # Clear existing trajectories
        self._clear_trajectories()

        # Ensure current deques reflect the active max length before refilling
        self._update_trajectory_maxlen()

        # Rebuild trajectories from frame 0 to target_frame
        for frame_idx in range(min(target_frame + 1, len(self.states))):
            state = self.states[frame_idx]

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

    def render_current_frame(self):
        """Render the current frame at a smooth UI framerate."""
        if 0 <= self.current_frame < len(self.states):
            current_state = self.states[self.current_frame]

            # Update trajectories with current state
            self._update_trajectories(current_state)

            # Prepare base frame (map, entities, UI)
            self._prepare_frame(current_state)

            # Draw trajectories on top of everything else before finalizing
            self._draw_all_trajectories()

            # Finalize at UI FPS (avoid using sleep_time as FPS)
            self._finalize_frame(UI_FPS)
        else:
            logger.error(f"Invalid frame index: {self.current_frame}")

    def update(self):
        """Update the playback state and render the current frame."""
        current_time = pygame.time.get_ticks()

        # Handle automatic playback
        if self.is_playing:
            time_since_update = (
                current_time - self.last_update_time
            ) / 1000.0  # Convert to seconds
            frames_to_advance = (
                time_since_update * self.playback_speed * 10
            )  # Assume 10 fps base rate

            if frames_to_advance >= 1:
                # Advance frames
                self.current_frame = min(
                    int(self.current_frame + frames_to_advance), len(self.states) - 1
                )
                self.last_update_time = current_time

                # Loop back to beginning if at the end
                if self.current_frame >= len(self.states) - 1:
                    self.current_frame = 0
        else:
            self.last_update_time = current_time

        # Render the current frame - no sleep_time needed
        self.render_current_frame()

    def run(self):
        """Run the interactive playback loop."""
        self.last_update_time = pygame.time.get_ticks()

        while not self.is_exit_requested:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._handle_quit()
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_video_resize(event)
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event)

            if self.is_exit_requested:
                break

            self.update()  # No sleep_time parameter needed

    def _add_text(self, timestep: int, state: VisualizableSimState):
        """Override parent _add_text to include playback information"""
        # First call the parent method to add standard text
        super()._add_text(timestep, state)

        # Then add our playback-specific text
        # This ensures text is rendered in one place consistently
        status_lines = [
            f"Frame: {self.current_frame + 1}/{self.total_frames}",
            f"Playing: {'Yes' if self.is_playing else 'No'}",
            f"Speed: {self.playback_speed:.1f}x",
            f"Trajectories: {'ON' if self.show_trajectories else 'OFF'}",
            f"Trail Length: {self.max_trajectory_length}",
        ]

        max_width = max(self.font.size(line)[0] for line in status_lines)
        text_height = len(status_lines) * self.font.get_linesize()
        text_surface = pygame.Surface((max_width + 10, text_height + 10), pygame.SRCALPHA)
        text_surface.fill(TEXT_BACKGROUND)

        for i, text in enumerate(status_lines):
            text_render = self.font.render(text, True, TEXT_COLOR)
            text_outline = self.font.render(text, True, (0, 0, 0))

            pos = (5, i * self.font.get_linesize() + 5)

            # Draw text outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                text_surface.blit(text_outline, (pos[0] + dx, pos[1] + dy))

            # Draw main text
            text_surface.blit(text_render, pos)

        # Position at the bottom right of the screen
        pos_x = self.width - max_width - 10
        pos_y = self.height - text_height - 10
        self.screen.blit(text_surface, (pos_x, pos_y))


def load_and_play_interactively(filename: str):
    """
    Load recorded states from a file and play them back interactively.

    Args:
        filename: Path to the pickle file containing states and map_def
    """
    logger.info(f"Loading states from {filename}")
    states, map_def = load_states(filename)

    logger.info(f"Starting interactive playback with {len(states)} states")
    player = InteractivePlayback(states, map_def)
    player.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m robot_sf.render.interactive_playback <state_file>")
        sys.exit(1)

    state_file = sys.argv[1]
    load_and_play_interactively(state_file)

    # Uncomment for testing with a specific recording
    # load_and_play_interactively("robot_sf/data/recording_2021-09-01_15-23-51.pkl")
