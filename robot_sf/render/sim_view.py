from time import sleep
from math import sin, cos
from typing import Tuple, Union, List
from dataclasses import dataclass, field
from threading import Thread
from signal import signal, SIGINT

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import numpy as np

from robot_sf.robot.differential_drive import DifferentialDriveAction
from robot_sf.robot.bicycle_drive import BicycleAction
from robot_sf.nav.map_config import Obstacle
from robot_sf.nav.map_config import MapDefinition

Vec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]
RobotAction = Tuple[float, float]
RgbColor = Tuple[int, int, int]


BACKGROUND_COLOR = (255, 255, 255)
BACKGROUND_COLOR_TRANSP = (255, 255, 255, 128)
OBSTACLE_COLOR = (20, 30, 20, 128)
PED_COLOR = (255, 50, 50)
ROBOT_COLOR = (0, 0, 200)
COLLISION_COLOR = (200, 0, 0)
ROBOT_ACTION_COLOR = (65, 105, 225)
PED_ACTION_COLOR = (255, 50, 50)
ROBOT_GOAL_COLOR = (0, 204, 102)
ROBOT_LIDAR_COLOR = (238, 160, 238, 128)
TEXT_COLOR = (0, 0, 0)


@dataclass
class VisualizableAction:
    robot_pose: RobotPose
    robot_action: Union[DifferentialDriveAction, BicycleAction]
    robot_goal: Vec2D


@dataclass
class VisualizableSimState:
    """Representing a collection of properties to display
    the simulator's state at a discrete timestep."""
    timestep: int
    action: Union[VisualizableAction, None]
    robot_pose: RobotPose
    pedestrian_positions: np.ndarray
    ray_vecs: np.ndarray
    ped_actions: np.ndarray
    # obstacles: List[Obstacle]


@dataclass
class SimulationView:
    width: float=1200
    height: float=800
    scaling: float=15
    robot_radius: float=1.0
    ped_radius: float=0.4
    goal_radius: float=1.0
    map_def: MapDefinition = field(default_factory=MapDefinition)
    obstacles: List[Obstacle] = field(default_factory=list)
    size_changed: bool = field(init=False, default=False)
    is_exit_requested: bool = field(init=False, default=False)
    is_abortion_requested: bool = field(init=False, default=False)
    screen: pygame.surface.Surface = field(init=False)
    font: pygame.font.Font = field(init=False)
    redraw_needed: bool = field(init=False, default=False)
    offset: np.array = field(init=False, default=np.array([0, 0]))
    """The offset is already uses `scaling` as a factor."""

    @property
    def _timestep_text_pos(self) -> Vec2D:
        return (16, 16)

    def __post_init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption('RobotSF Simulation')
        self.font = pygame.font.SysFont('Consolas', 14)
        self.surface_obstacles = self.preprocess_obstacles()
        self.clear()

    def _scale_tuple(self, tup: Tuple[float, float]) -> Tuple[float, float]:
        """scales a tuple of floats by the scaling factor and adds the offset."""
        x = tup[0] * self.scaling + self.offset[0]
        y = tup[1] * self.scaling + self.offset[1]
        return (x, y)

    def preprocess_obstacles(self) -> pygame.Surface:
        # Scale the vertices of the obstacles
        obst_vertices = [o.vertices_np * self.scaling for o in self.map_def.obstacles]

        # Initialize the minimum and maximum x and y coordinates
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

        # Find the minimum and maximum x and y coordinates among all the obstacles
        for vertices in obst_vertices:
            min_x = min(np.min(vertices[:, 0]), min_x)
            max_x = max(np.max(vertices[:, 0]), max_x)
            min_y = min(np.min(vertices[:, 1]), min_y)
            max_y = max(np.max(vertices[:, 1]), max_y)

        # Calculate the width and height of the surface needed to draw the obstacles
        width, height = max_x - min_x, max_y - min_y

        # Create a new surface with the calculated width and height
        surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Fill the surface with a transparent background color
        surface.fill(BACKGROUND_COLOR_TRANSP)

        # Draw each obstacle on the surface
        for vertices in obst_vertices:
            # Shift the vertices so that the minimum x and y coordinates are 0
            shifted_vertices = vertices - [min_x, min_y]
            # Draw the obstacle as a polygon with the shifted vertices
            pygame.draw.polygon(surface, OBSTACLE_COLOR, [(x, y) for x, y in shifted_vertices])

        # Return the surface with the drawn obstacles
        return surface

    def show(self):
        """
        Starts a separate thread to process the event queue and handles SIGINT signal.

        This method starts a new thread to process the event queue in the background.
        It also sets up a signal handler for SIGINT to handle the interruption of the program.
        """
        self.ui_events_thread = Thread(target=self._process_event_queue)
        self.ui_events_thread.start()

        def handle_sigint(signum, frame):
            self.is_exit_requested = True
            self.is_abortion_requested = True

        signal(SIGINT, handle_sigint)

    def exit(self):
        self.is_exit_requested = True
        self.ui_events_thread.join()

    def _handle_quit(self, e=None):
        """Handle the quit event of the pygame window."""
        self.is_exit_requested = True
        self.is_abortion_requested = True

    def _handle_video_resize(self, e):
        """Handle the resize event of the pygame window."""
        self.size_changed = True
        self.width, self.height = e.w, e.h

    def _handle_keydown(self, e):
        """Handle key presses for the simulation view."""
        key_action_map = {
            # scale the view
            pygame.K_PLUS: lambda: setattr(self, 'scaling', self.scaling + 1),
            pygame.K_MINUS: lambda: setattr(self, 'scaling', max(self.scaling - 1, 1)),
            # move the view
            pygame.K_LEFT: lambda: self.offset.__setitem__(0, self.offset[0] - 10),
            pygame.K_RIGHT: lambda: self.offset.__setitem__(0, self.offset[0] + 10),
            pygame.K_UP: lambda: self.offset.__setitem__(1, self.offset[1] - 10),
            pygame.K_DOWN: lambda: self.offset.__setitem__(1, self.offset[1] + 10),
            # reset the view
            pygame.K_r: lambda: self.offset.__setitem__(slice(None), (0, 0)),
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
        self._augment_timestep(0)
        pygame.display.update()

    def render(self, state: VisualizableSimState):
        sleep(0.01) # limit UI update rate to 100 fps

        # info: event handling needs to be processed
        #       in the main thread to access UI resources
        if self.is_exit_requested:
            pygame.quit()
            self.ui_events_thread.join()
            if self.is_abortion_requested:
                exit()
        if self.size_changed:
            self._resize_window()
            self.size_changed = False
        if self.redraw_needed:
            self.surface_obstacles = self.preprocess_obstacles()
            self.redraw_needed = False

        state = self._scale_pedestrian_state(state)
        # state, offset = self._zoom_camera(state)
        self.screen.fill(BACKGROUND_COLOR)

        # static objects
        self._draw_obstacles()
        self._draw_grid()

        # dynamic objects
        if self.map_def.ped_routes:
            self._draw_pedestrian_routes()
        self._augment_lidar(state.ray_vecs)
        self._augment_ped_actions(state.ped_actions)
        if state.action:
            self._augment_robot_action(state.action)
            self._augment_goal_position(state.action.robot_goal)
        self._draw_pedestrians(state.pedestrian_positions)
        self._draw_robot(state.robot_pose)
        self._augment_timestep(state.timestep)
        self._add_text(state.timestep, state)

        # update the display
        pygame.display.update()

    def _resize_window(self):
        old_surface = self.screen
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE)
        self.screen.blit(old_surface, (0, 0))

    def _scale_pedestrian_state(self, state: VisualizableSimState) \
            -> Tuple[VisualizableSimState, Tuple[float, float]]:
        state.pedestrian_positions *= self.scaling
        state.ped_actions *= self.scaling
        return state

    def _zoom_camera(self, state: VisualizableSimState) \
            -> Tuple[VisualizableSimState, Tuple[float, float]]:
        # TODO: remove this unused method
        r_x, r_y = state.robot_pose[0]
        x_offset = r_x * self.scaling - self.width / 2
        y_offset = r_y * self.scaling - self.height / 2
        state.pedestrian_positions *= self.scaling
        state.pedestrian_positions -= [x_offset, y_offset]
        state.ped_actions *= self.scaling
        state.ped_actions -= [x_offset, y_offset]
        state.ray_vecs *= self.scaling
        state.ray_vecs -= [x_offset, y_offset]
        state.robot_pose = ((
            state.robot_pose[0][0] * self.scaling - x_offset,
            state.robot_pose[0][1] * self.scaling - y_offset),
            state.robot_pose[1])
        if state.action:
            state.action.robot_pose = ((
                state.action.robot_pose[0][0] * self.scaling - x_offset,
                state.action.robot_pose[0][1] * self.scaling - y_offset),
                state.action.robot_pose[1])
            state.action.robot_goal = (
                state.action.robot_goal[0] * self.scaling - x_offset,
                state.action.robot_goal[1] * self.scaling - y_offset)
        return state, (x_offset, y_offset)

    def _draw_robot(self, pose: RobotPose):
        # TODO: display robot with an image instead of a circle
        pygame.draw.circle(
            self.screen,
            ROBOT_COLOR,
            self._scale_tuple(pose[0]),
            self.robot_radius * self.scaling)

    def _draw_pedestrians(self, ped_pos: np.ndarray):
        # TODO: display pedestrians with an image instead of a circle
        for ped_x, ped_y in ped_pos:
            pygame.draw.circle(
                self.screen,
                PED_COLOR,
                (ped_x+self.offset[0], ped_y+self.offset[1]),
                self.ped_radius * self.scaling
                )

    def _draw_obstacles(self):
        # Iterate over each obstacle in the list of obstacles
        for obstacle in self.map_def.obstacles:
            # Scale and offset the vertices of the obstacle
            scaled_vertices = [(
                self._scale_tuple((x, y))
                ) for x, y in obstacle.vertices_np]
            # Draw the obstacle as a polygon on the screen
            pygame.draw.polygon(self.screen, OBSTACLE_COLOR, scaled_vertices)

    def _augment_goal_position(self, robot_goal: Vec2D):
        # TODO: display pedestrians with an image instead of a circle
        pygame.draw.circle(self.screen, ROBOT_GOAL_COLOR, robot_goal, self.goal_radius * self.scaling)

    def _augment_lidar(self, ray_vecs: np.ndarray):
        for p1, p2 in ray_vecs:
            pygame.draw.line(self.screen, ROBOT_LIDAR_COLOR, p1, p2)

    def _augment_robot_action(self, action: VisualizableAction):
        r_x, r_y = action.robot_pose[0]
        # scale vector length to be always visible
        vec_length = action.robot_action[0] * self.scaling * 3
        vec_orient =  action.robot_pose[1]

        def from_polar(length: float, orient: float) -> Vec2D:
            return cos(orient) * length, sin(orient) * length

        def add_vec(v_1: Vec2D, v_2: Vec2D) -> Vec2D:
            return v_1[0] + v_2[0], v_1[1] + v_2[1]

        vec_x, vec_y = add_vec((r_x, r_y), from_polar(vec_length, vec_orient))
        pygame.draw.line(
            self.screen,
            ROBOT_ACTION_COLOR,
            self._scale_tuple((r_x, r_y)),
            self._scale_tuple((vec_x, vec_y)),
            width=3
            )

    def _augment_ped_actions(self, ped_actions: np.ndarray):
        """Draw the actions of the pedestrians as lines."""
        for p1, p2 in ped_actions:
            pygame.draw.line(
                self.screen,
                PED_ACTION_COLOR,
                p1+self.offset,
                p2+self.offset,
                width=3
                )
            
    def _draw_pedestrian_routes(self):
        """
        draw the map_def.routes on the screen
        """
        for route in self.map_def.ped_routes:
            pygame.draw.lines(
                self.screen,
                (0, 0, 255),
                False,
                [(x*self.scaling + self.offset[0], y*self.scaling + self.offset[1])
                for x, y in route.waypoints]
                )

    def _augment_timestep(self, timestep: int):
        # TODO: show map name as well
        text = f'step: {timestep}'
        text_surface = self.font.render(text, False, TEXT_COLOR)
        self.screen.blit(text_surface, self.timestep_text_pos)

    def _add_text(self, timestep: int, state: VisualizableSimState):
        text_lines = [
            f'step: {timestep}',
            f'scaling: {self.scaling}',
            f'x-offset: {self.offset[0]/self.scaling:.2f}',
            f'y-offset: {self.offset[1]/self.scaling:.2f}',
            f'RobotPose: {state.robot_pose}',
            f'RobotAction: {state.action.robot_action}'
        ]
        for i, text in enumerate(text_lines):
            text_surface = self.font.render(text, False, TEXT_COLOR)
            pos = self.timestep_text_pos[0], \
                self.timestep_text_pos[1] + i * self.font.get_linesize()
            self.screen.blit(text_surface, pos)

    def _draw_grid(
            self,
            grid_increment: int=50,
            grid_color: RgbColor=(200, 200, 200)
            ):
        """
        Draw a grid on the screen.
        :param grid_increment: The increment of the grid in pixels.
        :param grid_color: The color of the grid lines.
        """
        scaled_grid_size = grid_increment*self.scaling
        font = pygame.font.Font(None, 24)
        # draw the vertical lines
        start_x = ((-self.offset[0]) // scaled_grid_size) * scaled_grid_size
        for x in range(start_x, self.width-self.offset[0], scaled_grid_size):
            pygame.draw.line(
                self.screen,
                grid_color,
                (x + self.offset[0], 0),
                (x + self.offset[0], self.height)
                )
            label = font.render(str(int(x/self.scaling)), 1, grid_color)
            self.screen.blit(label, (x + self.offset[0], 0))

        # draw the horizontal lines
        start_y = ((-self.offset[1]) // scaled_grid_size) * scaled_grid_size
        for y in range(start_y, self.height-self.offset[1], scaled_grid_size):
            pygame.draw.line(
                self.screen,
                grid_color,
                (0, y + self.offset[1]),
                (self.width, y + self.offset[1])
                )
            label = font.render(str(int(y/self.scaling)), 1, grid_color)
            self.screen.blit(label, (0, y + self.offset[1]))
