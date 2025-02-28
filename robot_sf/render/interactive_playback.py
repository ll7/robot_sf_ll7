"""
Interactive playback module for recorded simulation states.

This module extends the SimulationView to provide interactive navigation through
recorded simulation states, allowing users to step forward/backward through frames
and control playback speed.
"""

from typing import List

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


class InteractivePlayback(SimulationView):
    """
    Interactive playback viewer that allows navigating through recorded simulation states.

    This class extends SimulationView to add playback controls:
    - Step forward/backward through frames
    - Jump to specific frames
    - Play/pause functionality
    - Display current frame index

    Attributes:
        states (List[VisualizableSimState]): List of states to playback
        current_frame (int): Index of the current frame being displayed
        is_playing (bool): Whether playback is currently running
        playback_speed (float): Multiplier for playback speed
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
        ]

    def _add_help_text(self):
        """Override the help text method to include playback controls."""
        # Call the parent method to get standard help text
        super()._add_help_text()

        # Add playback specific help text below the standard help
        text_lines = self._playback_help_lines

        # Calculate position for the playback controls (below standard help)
        standard_help_height = 10 * self.font.get_linesize()  # Assuming 10 lines in standard help
        y_offset = self._timestep_text_pos[1] + standard_help_height + 20

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
        # Check for playback control keys
        if e.key == pygame.K_SPACE:
            # Toggle play/pause
            self.is_playing = not self.is_playing
            return

        elif e.key == pygame.K_PERIOD and not self.is_playing:
            # Next frame (using period '.' instead of right arrow)
            self.current_frame = min(self.current_frame + 1, len(self.states) - 1)
            self.redraw_needed = True
            return

        elif e.key == pygame.K_COMMA and not self.is_playing:
            # Previous frame (using comma ',' instead of left arrow)
            self.current_frame = max(self.current_frame - 1, 0)
            self.redraw_needed = True
            return

        elif e.key == pygame.K_n:
            # First frame
            self.current_frame = 0
            self.redraw_needed = True
            return

        elif e.key == pygame.K_m:
            # Last frame
            self.current_frame = len(self.states) - 1
            self.redraw_needed = True
            return

        elif e.key == pygame.K_k:
            # Speed up
            self.playback_speed = min(self.playback_speed * 1.5, 10.0)
            return

        elif e.key == pygame.K_j:
            # Slow down
            self.playback_speed = max(self.playback_speed / 1.5, 0.1)
            return

        # If not a playback control key, let parent handle it
        super()._handle_keydown(e)

    def render_current_frame(self):
        """Render the current frame."""
        if 0 <= self.current_frame < len(self.states):
            # Use the instance sleep_time variable
            super().render(self.states[self.current_frame], self.sleep_time)
            pygame.display.update()
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

            # Limit the frame rate to avoid excessive CPU usage
            self.clock.tick(60)

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

    # Uncommenting this line for testing with a specific recording
    load_and_play_interactively("robot_sf/data/recording_2021-09-01_15-23-51.pkl")
