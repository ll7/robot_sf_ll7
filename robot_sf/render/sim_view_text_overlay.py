"""Text/info-panel collaborator extracted from ``SimulationView``.

This module is part of the god-class split of ``robot_sf/render/sim_view.py``
(see issues #4770 / #4989 / characterization baseline #4965). It owns the
text-overlay cluster: the surface-independent content builders
(``_build_text_lines``, ``_get_display_info_lines``, ``_get_robot_info_lines``,
``_get_pedestrian_info_lines``, ``_has_pedestrian_data``) plus the rendering
methods (``_render_text_display``, ``_render_text_line``, ``_add_text``).

Behavior is preserved verbatim from the pre-split ``SimulationView``: a
``SimulationView`` holds a :class:`SimViewTextOverlay` instance and delegates
the text-overlay methods to it. The collaborator reads shared render state
(clock, font, screen, scaling, offset, recording flag, display mode) through a
back-reference to its host view, so dynamic state stays authoritative on the
host exactly as before the split. Public method signatures are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from robot_sf.common.geometry import euclid_dist

# Text-overlay colors must match the pre-split constants in ``sim_view`` so the
# rendered output is pixel-identical. They are duplicated here (rather than
# imported) deliberately: the collaborator must not import the heavyweight
# ``sim_view`` module (pygame init side effects, moviepy, map config) at import
# time, which would create a circular import and re-trigger pygame startup.
TEXT_COLOR = (255, 255, 255)  # White text
TEXT_BACKGROUND = (0, 0, 0, 180)  # Semi-transparent black background
TEXT_OUTLINE_COLOR = (0, 0, 0)  # Black outline

if TYPE_CHECKING:
    from robot_sf.render.sim_state import VisualizableSimState
    from robot_sf.render.sim_view import SimulationView


class SimViewTextOverlay:
    """Text/info-panel collaborator for :class:`~robot_sf.render.sim_view.SimulationView`.

    Owns the diagnostic text-overlay content builders and rendering. A single
    instance is held by its host ``SimulationView`` (created in
    ``__post_init__``) and reached through delegating methods on the host, so
    existing call sites (including the ``InteractivePlayback`` subclass override
    of ``_add_text`` and the #4965 characterization golden lines) keep working
    unchanged.

    Shared, mutable render state (``screen``, ``font``, ``clock``, ``scaling``,
    ``offset``, ``record_video``, ``display_robot_info``, ``current_target_fps``)
    is accessed through the ``host`` back-reference rather than copied, so the
    host remains the single source of truth exactly as before the split.
    """

    def __init__(self, host: SimulationView) -> None:
        """Bind the overlay to its host view.

        The host back-reference is how the collaborator reaches shared, mutable
        render state (screen, font, clock, scaling, offset, display mode).
        """
        self._host = host

    def _add_text(self, timestep: int, state: VisualizableSimState) -> None:
        """Render diagnostic overlay text for the current frame.

        Args:
            timestep: Simulation step index shown in the overlay.
            state: Current visualizable state used to build display lines.
        """
        info_lines = self._get_display_info_lines(state)
        text_lines = self._build_text_lines(timestep, state, info_lines)
        self._render_text_display(text_lines)

    def _get_display_info_lines(self, state: VisualizableSimState) -> list[str]:
        """Get lines for robot/pedestrian info display based on display mode.

        Returns:
            list[str]: Info lines for robot (mode 1), pedestrian (mode 2), or empty (mode 0).
        """
        if self._host.display_robot_info == 1:
            return self._get_robot_info_lines(state)
        elif self._host.display_robot_info == 2:
            return self._get_pedestrian_info_lines(state)
        return []

    def _get_robot_info_lines(self, state: VisualizableSimState) -> list[str]:
        """Get robot information lines for display.

        Returns:
            list[str]: Robot pose, action, and goal info lines, or empty list if unavailable.
        """
        if hasattr(state, "robot_action") and state.robot_action:
            return [
                f"RobotPose: {state.robot_pose}",
                f"RobotAction: {state.robot_action.action if state.robot_action else None}",
                f"RobotGoal: {state.robot_action.goal if state.robot_action else None}",
            ]
        return []

    def _get_pedestrian_info_lines(self, state: VisualizableSimState) -> list[str]:
        """Get pedestrian information lines for display.

        Returns:
            list[str]: Ego pedestrian pose, action, goal, and distance info, or empty list if unavailable.
        """
        if self._has_pedestrian_data(state):
            assert state.ego_ped_pose is not None, "ego_ped_pose must be set"
            distance_to_robot = euclid_dist(state.ego_ped_pose[0], state.robot_pose[0])
            assert state.ego_ped_action is not None, "ego_ped_action must be set"
            return [
                f"PedestrianPose: {state.ego_ped_pose}",
                f"PedestrianAction: {state.ego_ped_action.action}",
                f"PedestrianGoal: {state.ego_ped_action.goal}",
                f"DistanceRobot: {distance_to_robot:.2f}",
            ]
        else:
            self._host.display_robot_info = 0
            return []

    def _has_pedestrian_data(self, state: VisualizableSimState) -> bool:
        """Check if the state has complete pedestrian data.

        Returns:
            bool: True if state has valid ego_ped_pose and ego_ped_action.
        """
        return bool(
            hasattr(state, "ego_ped_pose")
            and state.ego_ped_pose
            and hasattr(state, "ego_ped_action")
            and state.ego_ped_action
        )

    def _build_text_lines(
        self, timestep: int, state: VisualizableSimState, info_lines: list[str]
    ) -> list[str]:
        """Build the complete list of text lines for display.

        Returns:
            list[str]: Combined list of timestep, scaling, speedup, and optional robot/ped info lines.
        """
        # Calculate speedup factor safely
        actual_fps = self._host.clock.get_fps()
        time_per_step = getattr(state, "time_per_step_in_secs", 0.1)
        speedup = actual_fps * time_per_step

        text_lines = [
            f"step: {timestep}",
            f"scaling: {self._host.scaling}",
        ]

        # Add FPS and speedup information if not recording
        if not self._host.record_video:
            text_lines += [
                f"target fps: {actual_fps:.1f}/{getattr(self._host, 'current_target_fps', 60):.1f}",
                f"speedup: {speedup:.1f}x",
            ]

        text_lines += [
            f"x-offset: {self._host.offset[0] / self._host.scaling:.2f}",
            f"y-offset: {self._host.offset[1] / self._host.scaling:.2f}",
        ]

        text_lines += info_lines
        text_lines += ["(Press h for help)"]
        return text_lines

    def _render_text_display(self, text_lines: list[str]) -> None:
        """Render the text display on screen."""
        # Create a surface for the text background
        max_width = max(self._host.font.size(line)[0] for line in text_lines)
        text_height = len(text_lines) * self._host.font.get_linesize()
        text_surface = pygame.Surface((max_width + 10, text_height + 10), pygame.SRCALPHA)
        text_surface.fill(TEXT_BACKGROUND)

        for i, text in enumerate(text_lines):
            self._render_text_line(text_surface, text, i)

        self._host.screen.blit(text_surface, self._host._timestep_text_pos)

    def _render_text_line(self, surface, text: str, line_index: int) -> None:
        """Render a single text line with outline effect."""
        text_render = self._host.font.render(text, True, TEXT_COLOR)
        text_outline = self._host.font.render(text, True, TEXT_OUTLINE_COLOR)

        pos = (5, line_index * self._host.font.get_linesize() + 5)

        # Draw text outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            surface.blit(text_outline, (pos[0] + dx, pos[1] + dy))

        # Draw main text
        surface.blit(text_render, pos)
