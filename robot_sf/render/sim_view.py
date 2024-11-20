from robot_sf.sensor.range_sensor import euclid_dist
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.map_config import Obstacle
from robot_sf.ped_ego.unicycle_drive import UnicycleAction
from robot_sf.robot.bicycle_drive import BicycleAction
from robot_sf.robot.differential_drive import DifferentialDriveAction
from loguru import logger
import numpy as np
import pygame
from time import sleep
from math import sin, cos
from typing import Tuple, Union, List
from dataclasses import dataclass, field
from threading import Thread
from signal import signal, SIGINT

import os
from datetime import datetime

# Make moviepy optional
try:
    from moviepy.editor import ImageSequenceClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning(
        "MoviePy is not available. Video recording is disabled. Have you installed ffmpeg?")

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


Vec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]
PedPose = Tuple[Vec2D, float]
RobotAction = Tuple[float, float]
RgbColor = Tuple[int, int, int]


BACKGROUND_COLOR = (255, 255, 255)
BACKGROUND_COLOR_TRANSP = (255, 255, 255, 128)
OBSTACLE_COLOR = (20, 30, 20, 128)
PED_SPAWN_COLOR = (255, 204, 203)
PED_GOAL_COLOR = (144, 238, 144)
PED_COLOR = (255, 50, 50)
EGO_PED_COLOR = (108, 70, 117)
EGO_PED_ACTION_COLOR = (108, 70, 117)
PED_ROUTE_COLOR = (0, 0, 255)
ROBOT_ROUTE_COLOR = (30, 30, 255)
ROBOT_COLOR = (0, 0, 200)
COLLISION_COLOR = (200, 0, 0)
ROBOT_ACTION_COLOR = (65, 105, 225)
PED_ACTION_COLOR = (255, 50, 50)
ROBOT_GOAL_COLOR = (0, 204, 102)
ROBOT_LIDAR_COLOR = (238, 160, 238, 128)
TEXT_COLOR = (255, 255, 255)  # White text
TEXT_BACKGROUND = (0, 0, 0, 180)  # Semi-transparent black background
TEXT_OUTLINE_COLOR = (0, 0, 0)  # Black outline


@dataclass
class VisualizableAction:
    pose: RobotPose
    action: Union[DifferentialDriveAction, BicycleAction, UnicycleAction]
    goal: Vec2D


@dataclass
class VisualizableSimState:
    """Representing a collection of properties to display
    the simulator's state at a discrete timestep."""

    timestep: int
    robot_action: Union[VisualizableAction, None]
    robot_pose: RobotPose
    pedestrian_positions: np.ndarray
    ray_vecs: np.ndarray
    ped_actions: np.ndarray
    ego_ped_pose: PedPose = None
    ego_ped_ray_vecs: np.ndarray = None
    ego_ped_action: Union[VisualizableAction, None] = None
    # obstacles: List[Obstacle]


@dataclass
class SimulationView:
    width: float = 1200
    height: float = 800
    scaling: float = 15
    robot_radius: float = 1.0
    ego_ped_radius: float = 0.4
    ped_radius: float = 0.4
    goal_radius: float = 1.0
    map_def: MapDefinition = field(default_factory=MapDefinition)
    obstacles: List[Obstacle] = field(default_factory=list)
    caption: str = "RobotSF Simulation"
    focus_on_robot: bool = False
    focus_on_ego_ped: bool = False
    record_video: bool = False
    video_path: str = None
    frames: List[np.ndarray] = field(default_factory=list)

    # Add UI state fields
    screen: pygame.surface.Surface = field(init=False)
    font: pygame.font.Font = field(init=False)
    size_changed: bool = field(init=False, default=False)
    redraw_needed: bool = field(init=False, default=False)
    is_exit_requested: bool = field(init=False, default=False)
    is_abortion_requested: bool = field(init=False, default=False)
    offset: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    display_robot_info: int = field(default=0)  # Add this line
    display_help: bool = field(default=False)  # Also add this for help text

    def __post_init__(self):
        """Initialize PyGame components."""
        logger.info("Initializing the simulation view.")
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (int(self.width), int(self.height)), pygame.RESIZABLE
        )
        pygame.display.set_caption(self.caption)
        self.font = pygame.font.Font(None, 36)

    def show(self):
        """Start the simulation view."""
        pass  # Remove threading since we'll handle events in render()

    def render(self, state: VisualizableSimState):
        """Render one frame and handle events."""
        # Handle events on main thread
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._handle_quit()
            elif event.type == pygame.VIDEORESIZE:
                self._handle_video_resize(event)
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

        if self.is_exit_requested:
            pygame.quit()
            if self.is_abortion_requested:
                exit()
            return

        sleep(0.01)  # limit UI update rate to 100 fps

        # Adjust the view based on the focus
        self._move_camera(state)

        self.screen.fill(BACKGROUND_COLOR)

        # static objects
        if self.map_def.obstacles:
            self._draw_obstacles()

        self._draw_grid()

        # dynamic objects
        if hasattr(state, 'ray_vecs'):
            self._augment_lidar(state.ray_vecs)
        if hasattr(state, 'ped_actions'):
            self._augment_ped_actions(state.ped_actions)
        if hasattr(state, 'robot_action') and state.robot_action:
            self._augment_action(state.robot_action, ROBOT_ACTION_COLOR)
            if hasattr(state.robot_action, 'goal'):
                self._augment_goal_position(state.robot_action.goal)
        if hasattr(state, 'pedestrian_positions'):
            self._draw_pedestrians(state.pedestrian_positions)
        if hasattr(state, 'robot_pose'):
            self._draw_robot(state.robot_pose)
        if hasattr(state, 'ego_ped_pose') and state.ego_ped_pose:
            if hasattr(state, 'ego_ped_ray_vecs'):
                self._augment_lidar(state.ego_ped_ray_vecs)
            if hasattr(state, 'ego_ped_action') and state.ego_ped_action:
                self._augment_action(state.ego_ped_action, EGO_PED_ACTION_COLOR)
            self._draw_ego_ped(state.ego_ped_pose)

        # information
        self._add_text(state.timestep, state)
        if self.display_help:
            self._add_help_text()

        if self.record_video:
            # Capture frame
            frame_data = pygame.surfarray.array3d(self.screen)
            frame_data = frame_data.swapaxes(0, 1)
            self.frames.append(frame_data)
        else:
            # Normal display update
            pygame.display.update()

    @property
    def _timestep_text_pos(self) -> Vec2D:
        return (16, 16)

    def _scale_tuple(self, tup: Tuple[float, float]) -> Tuple[float, float]:
        """scales a tuple of floats by the scaling factor and adds the offset."""
        x = tup[0] * self.scaling + self.offset[0]
        y = tup[1] * self.scaling + self.offset[1]
        return (x, y)

    def exit(self):
        self.is_exit_requested = True

    def _handle_quit(self, e=None):
        """Handle the quit event of the pygame window."""
        self.is_exit_requested = True
        self.is_abortion_requested = True
        if self.record_video and self.frames:
            if MOVIEPY_AVAILABLE:
                clip = ImageSequenceClip(self.frames, fps=100)
                clip.write_videofile(self.video_path)
                self.frames = []

    def _handle_video_resize(self, e):
        """Handle the resize event of the pygame window."""
        self.size_changed = True
        self.width, self.height = e.w, e.h

    def _handle_keydown(self, e):
        """Handle key presses for the simulation view."""
        new_offset = 100
        new_scaling = 1
        if pygame.key.get_mods() & pygame.KMOD_CTRL:
            new_offset = 250
            new_scaling = 10

        if pygame.key.get_mods() & pygame.KMOD_ALT:
            new_offset = 10

        key_action_map = {
            # scale the view
            pygame.K_PLUS: lambda: setattr(self, "scaling", self.scaling + new_scaling),
            pygame.K_MINUS: lambda: setattr(
                self, "scaling", max(self.scaling - new_scaling, 1)
            ),
            # move the view
            pygame.K_LEFT: lambda: self.offset.__setitem__(
                0, self.offset[0] + new_offset
            ),
            pygame.K_RIGHT: lambda: self.offset.__setitem__(
                0, self.offset[0] - new_offset
            ),
            pygame.K_UP: lambda: self.offset.__setitem__(
                1, self.offset[1] + new_offset
            ),
            pygame.K_DOWN: lambda: self.offset.__setitem__(
                1, self.offset[1] - new_offset
            ),
            # reset the view
            pygame.K_r: lambda: self.offset.__setitem__(slice(None), (0, 0)),
            # focus on the robot or ped
            pygame.K_f: lambda: setattr(
                self, "focus_on_robot", not self.focus_on_robot
            ),
            pygame.K_p: lambda: setattr(
                self, "focus_on_ego_ped", not self.focus_on_ego_ped
            ),
            # display help
            pygame.K_h: lambda: setattr(self, "display_help", not self.display_help),
            # display robotinfo
            pygame.K_q: lambda: setattr(
                self, "display_robot_info", (self.display_robot_info + 1) % 3
            ),
        }

        if e.key in key_action_map:
            key_action_map[e.key]()
            if e.key in (pygame.K_PLUS, pygame.K_MINUS):
                # for the scaling, we need to redraw the obstacles
                # this is not necessary for a fixed offset
                self.redraw_needed = True

    def _process_event_queue(self):
        """Process the event queue of the pygame window."""
        event_handler_map = {
            pygame.QUIT: self._handle_quit,
            pygame.VIDEORESIZE: self._handle_video_resize,
            pygame.KEYDOWN: self._handle_keydown,
        }
        while not self.is_exit_requested:
            for e in pygame.event.get():
                handler = event_handler_map.get(e.type)
                if handler:
                    handler(e)
            sleep(0.01)  # Consider removing or replacing with a frame rate clock

    def clear(self):
        """
        Clears the screen and updates the display.

        This method fills the screen with the background color,
        adds text at position 0, and updates the display.
        """
        self.screen.fill(BACKGROUND_COLOR)
        pygame.display.update()

    def _resize_window(self):
        logger.debug("Resizing the window.")
        old_surface = self.screen
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        self.screen.blit(old_surface, (0, 0))

    def _move_camera(self, state: VisualizableSimState):
        """Moves the camera based on the focused object."""
        if self.focus_on_robot:
            r_x, r_y = state.robot_pose[0]
            self.offset[0] = int(r_x * self.scaling - self.width / 2) * -1
            self.offset[1] = int(r_y * self.scaling - self.height / 2) * -1
        if self.focus_on_ego_ped and state.ego_ped_pose:
            r_x, r_y = state.ego_ped_pose[0]
            self.offset[0] = int(r_x * self.scaling - self.width / 2) * -1
            self.offset[1] = int(r_y * self.scaling - self.height / 2) * -1

    def _draw_robot(self, pose: RobotPose):
        # TODO: display robot with an image instead of a circle
        pygame.draw.circle(
            self.screen,
            ROBOT_COLOR,
            self._scale_tuple(pose[0]),
            self.robot_radius * self.scaling,
        )

    def _draw_ego_ped(self, pose: PedPose):
        # TODO: display robot with an image instead of a circle
        pygame.draw.circle(
            self.screen,
            EGO_PED_COLOR,
            self._scale_tuple(pose[0]),
            self.ego_ped_radius * self.scaling,
        )

    def _draw_pedestrians(self, ped_pos: np.ndarray):
        # TODO: display pedestrians with an image instead of a circle
        for ped_x, ped_y in ped_pos:
            pygame.draw.circle(
                self.screen,
                PED_COLOR,
                self._scale_tuple((ped_x, ped_y)),
                self.ped_radius * self.scaling,
            )

    def _draw_obstacles(self):
        # Iterate over each obstacle in the list of obstacles
        for obstacle in self.map_def.obstacles:
            # Scale and offset the vertices of the obstacle
            scaled_vertices = [
                (self._scale_tuple((x, y))) for x, y in obstacle.vertices_np
            ]
            # Draw the obstacle as a polygon on the screen
            pygame.draw.polygon(self.screen, OBSTACLE_COLOR, scaled_vertices)

    def _draw_spawn_zones(self):
        # Iterate over each spawn_zone in the list of spawn_zones
        for spawn_zone in self.map_def.ped_spawn_zones:
            # Scale and offset the vertices of the zones
            vertices_np = np.array(spawn_zone)
            scaled_vertices = [(self._scale_tuple((x, y))) for x, y in vertices_np]
            # Draw the spawn zone as a polygon on the screen
            pygame.draw.polygon(self.screen, PED_SPAWN_COLOR, scaled_vertices)

    def _draw_goal_zones(self):
        # Iterate over each goal_zone in the list of goal_zones
        for goal_zone in self.map_def.ped_goal_zones:
            # Scale and offset the vertices of the goal zones
            vertices_np = np.array(goal_zone)
            scaled_vertices = [(self._scale_tuple((x, y))) for x, y in vertices_np]
            # Draw the goal_zone as a polygon on the screen
            pygame.draw.polygon(self.screen, PED_GOAL_COLOR, scaled_vertices)

    def _augment_goal_position(self, robot_goal: Vec2D):
        # TODO: display pedestrians with an image instead of a circle
        pygame.draw.circle(
            self.screen,
            ROBOT_GOAL_COLOR,
            self._scale_tuple(robot_goal),
            self.goal_radius * self.scaling,
        )

    def _augment_lidar(self, ray_vecs: np.ndarray):
        for p1, p2 in ray_vecs:
            pygame.draw.line(
                self.screen,
                ROBOT_LIDAR_COLOR,
                self._scale_tuple(p1),
                self._scale_tuple(p2),
            )

    def _augment_action(self, action: VisualizableAction, color):
        r_x, r_y = action.pose[0]
        # scale vector length to be always visible
        vec_length = action.action[0] * self.scaling
        vec_orient = action.pose[1]

        def from_polar(length: float, orient: float) -> Vec2D:
            return cos(orient) * length, sin(orient) * length

        def add_vec(v_1: Vec2D, v_2: Vec2D) -> Vec2D:
            return v_1[0] + v_2[0], v_1[1] + v_2[1]

        vec_x, vec_y = add_vec((r_x, r_y), from_polar(vec_length, vec_orient))
        pygame.draw.line(
            self.screen,
            color,
            self._scale_tuple((r_x, r_y)),
            self._scale_tuple((vec_x, vec_y)),
            width=3,
        )

    def _augment_ped_actions(self, ped_actions: np.ndarray):
        """Draw the actions of the pedestrians as lines."""
        for p1, p2 in ped_actions:
            pygame.draw.line(
                self.screen,
                PED_ACTION_COLOR,
                self._scale_tuple(p1),
                self._scale_tuple(p2),
                width=3,
            )

    def _draw_pedestrian_routes(self):
        """
        draw the map_def.routes on the screen
        """
        for route in self.map_def.ped_routes:
            pygame.draw.lines(
                self.screen,
                PED_ROUTE_COLOR,
                False,
                [self._scale_tuple((x, y)) for x, y in route.waypoints],
                width=1,
            )

    def _draw_robot_routes(self):
        """
        draw the map_def.routes on the screen
        """
        for route in self.map_def.robot_routes:
            pygame.draw.lines(
                self.screen,
                ROBOT_ROUTE_COLOR,
                False,
                [self._scale_tuple((x, y)) for x, y in route.waypoints],
                width=1,
            )

    def _draw_coordinates(self, x, y):
        """
        Draws the coordinates (x, y) on the screen.
        """
        text = self.font.render(f"({x}, {y})", False, TEXT_COLOR)
        self.screen.blit(text, (x, y))

    def _add_text(self, timestep: int, state: VisualizableSimState):
        lines = []
        if self.display_robot_info == 1 and state.robot_action:
            lines += [
                f"RobotPose: {state.robot_pose}",
                f"RobotAction: {state.robot_action.action if state.robot_action else None}",
                f"RobotGoal: {state.robot_action.goal if state.robot_action else None}",
            ]
        elif self.display_robot_info == 2:
            if state.ego_ped_pose and state.ego_ped_action:
                distance_to_robot = euclid_dist(
                    state.ego_ped_pose[0], state.robot_pose[0]
                )
                lines += [
                    f"PedestrianPose: {state.ego_ped_pose}",
                    f"PedestrianAction: {state.ego_ped_action.action}",
                    f"PedestrianGoal: {state.ego_ped_action.goal}",
                    f"DistanceRobot: {distance_to_robot:.2f}",
                ]
            else:
                self.display_robot_info = 0
        text_lines = [
            f"step: {timestep}",
            f"scaling: {self.scaling}",
            f"x-offset: {self.offset[0]/self.scaling:.2f}",
            f"y-offset: {self.offset[1]/self.scaling:.2f}",
        ]
        text_lines += lines
        text_lines += [
            "(Press h for help)",
        ]

        # Create a surface for the text background
        max_width = max(self.font.size(line)[0] for line in text_lines)
        text_height = len(text_lines) * self.font.get_linesize()
        text_surface = pygame.Surface(
            (max_width + 10, text_height + 10), pygame.SRCALPHA
        )
        text_surface.fill(TEXT_BACKGROUND)

        for i, text in enumerate(text_lines):
            text_render = self.font.render(text, True, TEXT_COLOR)
            text_outline = self.font.render(text, True, TEXT_OUTLINE_COLOR)

            pos = (5, i * self.font.get_linesize() + 5)

            # Draw text outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                text_surface.blit(text_outline, (pos[0] + dx, pos[1] + dy))

            # Draw main text
            text_surface.blit(text_render, pos)

        self.screen.blit(text_surface, self._timestep_text_pos)

    def _add_help_text(self):
        text_lines = [
            "Move camera: arrow keys",
            "Move fast: CTRL + arrow keys",
            "Move slow: ALT + arrow keys",
            "Reset view: r",
            "Focus robot: f",
            "Focus ego_ped: p",
            "Scale up: +",
            "Scale down: -",
            "Display robot info: q",
            "Help: h",
        ]

        # Determine max width of the text
        text_surface = self.font.render(text_lines[1], False, TEXT_COLOR)

        max_width = max(self.font.size(line)[0] for line in text_lines)
        text_height = len(text_lines) * self.font.get_linesize()
        text_surface = pygame.Surface(
            (max_width + 10, text_height + 10), pygame.SRCALPHA
        )
        text_surface.fill(TEXT_BACKGROUND)

        for i, text in enumerate(text_lines):
            text_render = self.font.render(text, True, TEXT_COLOR)
            text_outline = self.font.render(text, True, TEXT_OUTLINE_COLOR)

            pos = (5, i * self.font.get_linesize() + 5)

            # Draw text outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                text_surface.blit(text_outline, (pos[0] + dx, pos[1] + dy))

            # Draw main text
            text_surface.blit(text_render, pos)

        self.screen.blit(
            text_surface, (self.width - max_width - 10, self._timestep_text_pos[1])
        )

    def _draw_grid(
        self, grid_increment: int = 50, grid_color: RgbColor = (200, 200, 200)
    ):
        """
        Draw a grid on the screen.
        :param grid_increment: The increment of the grid in pixels.
        :param grid_color: The color of the grid lines.
        """
        scaled_grid_size = grid_increment * self.scaling
        font = pygame.font.Font(None, 24)
        # draw the vertical lines
        start_x = ((-self.offset[0]) // scaled_grid_size) * scaled_grid_size
        for x in range(start_x, self.width - self.offset[0], scaled_grid_size):
            pygame.draw.line(
                self.screen,
                grid_color,
                (x + self.offset[0], 0),
                (x + self.offset[0], self.height),
            )
            label = font.render(str(int(x / self.scaling)), 1, grid_color)
            self.screen.blit(label, (x + self.offset[0], 0))

        # draw the horizontal lines
        start_y = ((-self.offset[1]) // scaled_grid_size) * scaled_grid_size
        for y in range(start_y, self.height - self.offset[1], scaled_grid_size):
            pygame.draw.line(
                self.screen,
                grid_color,
                (0, y + self.offset[1]),
                (self.width, y + self.offset[1]),
            )
            label = font.render(str(int(y / self.scaling)), 1, grid_color)
            self.screen.blit(label, (0, y + self.offset[1]))
