from time import sleep
from math import sin, cos
from typing import Tuple, Union
from dataclasses import dataclass
from threading import Thread
from signal import signal, SIGINT

import pygame
import numpy as np


Vec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]
RobotAction = Tuple[float, float]
RgbColor = Tuple[int, int, int]


BACKGROUND_COLOR = (255, 255, 255)
OBSTACLE_COLOR = (20, 30, 20)
PED_COLOR = (255, 50, 50)
ROBOT_COLOR = (0, 0, 200)
COLLISION_COLOR = (200, 0, 0)
ROBOT_ACTION_COLOR = (65, 105, 225)
ROBOT_GOAL_COLOR = (0, 204, 102)
TEXT_COLOR = (0, 0, 0)


@dataclass
class VisualizableAction:
    robot_pose: RobotPose
    robot_action: RobotAction
    robot_goal: Vec2D


@dataclass
class VisualizableSimState:
    """Representing a collection of properties to display
    the simulator's state at a discrete timestep."""
    timestep: int
    action: Union[VisualizableAction, None]
    robot_pose: RobotPose
    pedestrian_positions: np.ndarray
    obstacles: np.ndarray


class SimulationView:
    """Representing a UI window for visualizing the simulation's state."""

    def __init__(
            self, width: float=1200, height: float=800, scaling: float=15,
            robot_radius: float=1.0, ped_radius: float=0.4, goal_radius: float=1.0):
        self.width = width
        self.height = height
        self.scaling = scaling
        self.robot_radius = robot_radius
        self.ped_radius = ped_radius
        self.goal_radius = goal_radius
        self.size_changed = False
        self.is_exit_requested = False
        self.is_abortion_requested = False

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE)
        self.font = pygame.font.SysFont('Consolas', 14)
        self.timestep_text_pos = (self.width - 100, 10)
        self.clear()

    def show(self):
        self.ui_events_thread = Thread(target=self._process_event_queue)
        self.ui_events_thread.start()

        def handle_sigint(signum, frame):
            self.is_exit_requested = True
            self.is_abortion_requested = True

        signal(SIGINT, handle_sigint)

    def exit(self):
        self.is_exit_requested = True
        self.ui_events_thread.join()

    def _process_event_queue(self):
        while not self.is_exit_requested:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.is_exit_requested = True
                    self.is_abortion_requested = True
                elif e.type == pygame.VIDEORESIZE:
                    self.size_changed = True
                    self.width, self.height = e.w, e.h
            sleep(0.01)

    def clear(self):
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

        state = self._zoom_camera(state)
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_robot(state.robot_pose)
        self._draw_pedestrians(state.pedestrian_positions)
        self._draw_obstacles(state.obstacles)
        if state.action:
            self._augment_action_vector(state.action)
            self._augment_goal_position(state.action.robot_goal)
        self._augment_timestep(state.timestep)
        pygame.display.update()

    def _resize_window(self):
        old_surface = self.screen
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE)
        self.screen.blit(old_surface, (0, 0))
        self.size_changed = False

    def _zoom_camera(self, state: VisualizableSimState) -> VisualizableSimState:
        r_x, r_y = state.robot_pose[0]
        x_offset = r_x * self.scaling - self.width / 2
        y_offset = r_y * self.scaling - self.height / 2
        state.pedestrian_positions *= self.scaling
        state.pedestrian_positions -= [x_offset, y_offset]
        state.obstacles *= self.scaling
        state.obstacles -= [x_offset, y_offset, x_offset, y_offset]
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
        return state

    def _draw_robot(self, pose: RobotPose):
        # TODO: display robot with an image instead of a circle
        pygame.draw.circle(self.screen, ROBOT_COLOR, pose[0], self.robot_radius * self.scaling)

    def _draw_pedestrians(self, ped_pos: np.ndarray):
        # TODO: display pedestrians with an image instead of a circle
        for ped_x, ped_y in ped_pos:
            pygame.draw.circle(self.screen, PED_COLOR, (ped_x, ped_y), self.ped_radius * self.scaling)

    def _draw_obstacles(self, obstacles: np.ndarray):
        for s_x, s_y, e_x, e_y in obstacles:
            pygame.draw.line(self.screen, OBSTACLE_COLOR, (s_x, s_y), (e_x, e_y))

    def _augment_goal_position(self, robot_goal: Vec2D):
        # TODO: display pedestrians with an image instead of a circle
        pygame.draw.circle(self.screen, ROBOT_GOAL_COLOR, robot_goal, self.goal_radius * self.scaling)

    def _augment_action_vector(self, action: VisualizableAction):
        r_x, r_y = action.robot_pose[0]
        vec_length, vec_orient = action.robot_action[0] * self.scaling * 3, action.robot_pose[1]

        def from_polar(length: float, orient: float) -> Vec2D:
            return cos(orient) * length, sin(orient) * length

        def add_vec(v_1: Vec2D, v_2: Vec2D) -> Vec2D:
            return v_1[0] + v_2[0], v_1[1] + v_2[1]

        vec_x, vec_y = add_vec((r_x, r_y), from_polar(vec_length, vec_orient))
        pygame.draw.line(self.screen, ROBOT_ACTION_COLOR, (r_x, r_y), (vec_x, vec_y))

    def _augment_timestep(self, timestep: int):
        # TODO: show map name as well
        text = f'step: {timestep}'
        text_surface = self.font.render(text, False, TEXT_COLOR)
        self.screen.blit(text_surface, self.timestep_text_pos)
