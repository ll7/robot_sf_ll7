from time import sleep
from typing import Callable, Tuple, Union
from dataclasses import dataclass, field

import pygame
import numpy as np

from robot_sf.vector import PolarVec2D, RobotPose


RgbColor = Tuple[int, int, int]
RobotAction = PolarVec2D
WorldPosition = Tuple[float, float]
GridPosition = Tuple[int, int]


# TODO: pick reasonable colors
BACKGROUND_COLOR = (255, 255, 255)
OBSTACLE_COLOR = (20, 30, 20)
PEDESTRIAN_COLOR = (0, 50, 100)
ROBOT_COLOR = (0, 0, 200)
COLLISION_COLOR = (200, 0, 0)
ROBOT_ACTION_COLOR = (0, 100, 0)
ROBOT_GOAL_COLOR = (0, 204, 102)
TEXT_COLOR = (0, 0, 0)


@dataclass
class VisualizableAction:
    robot_pose: RobotPose
    robot_action: RobotAction
    robot_goal: WorldPosition
    world_to_grid: Callable[[float, float], GridPosition]
    robot_goal_grid: GridPosition = field(init=False)
    start: GridPosition = field(init=False)
    end: GridPosition = field(init=False)

    def __post_init__(self):
        self.robot_goal_grid = self.world_to_grid(self.robot_goal[0], self.robot_goal[1])
        x_start, y_start = self.robot_pose.pos.as_list
        x_start, y_start = self.world_to_grid(x_start, y_start)
        x_diff, y_diff = self.robot_action.vector.as_list
        x_diff, y_diff = self.world_to_grid(x_diff, y_diff)
        x_end, y_end = x_start + x_diff, y_start + y_diff
        self.start, self.end = (x_start, y_start), (x_end, y_end)


@dataclass
class VisualizableSimState:
    """Representing a collection of properties to display
    the simulator's state at a discrete timestep."""
    timestep: int
    action: Union[VisualizableAction, None]
    robot_occupancy: np.ndarray
    pedestrians_occupancy: np.ndarray
    obstacles_occupancy: np.ndarray
    collisions_occupancy: np.ndarray = field(init=False)

    def __post_init__(self):
        coll_with_obstacle = np.bitwise_and(self.robot_occupancy, self.pedestrians_occupancy)
        coll_with_pedestrian = np.bitwise_and(self.robot_occupancy, self.obstacles_occupancy)
        self.collisions_occupancy = np.bitwise_or(coll_with_obstacle, coll_with_pedestrian)


class SimulationView:
    """Representing a UI window for visualizing the simulation's state."""

    def __init__(self, grid_width: int=600, grid_height: int=800, pixels_per_cell: int=1):
        self.width = grid_width * pixels_per_cell
        self.height = grid_height * pixels_per_cell
        self.pixels_per_cell = pixels_per_cell

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.font = pygame.font.SysFont('Consolas', 14)
        self.timestep_text_pos = (self.width - 100, 10)
        self.clear()

    def show_as_daemon(self, on_term: Callable[[], None]):
        exit_requested = lambda: any(
            e.type == pygame.QUIT for e in pygame.event.get())
        while not exit_requested():
            sleep(0.01)
        pygame.quit()
        on_term()

    def clear(self):
        self.screen.fill(BACKGROUND_COLOR)
        self._augment_timestep(0)
        pygame.display.update()

    def render(self, state: VisualizableSimState):
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_occupancy(state.robot_occupancy, ROBOT_COLOR)
        self._draw_occupancy(state.obstacles_occupancy, OBSTACLE_COLOR)
        self._draw_occupancy(state.pedestrians_occupancy, PEDESTRIAN_COLOR)
        self._draw_occupancy(state.collisions_occupancy, COLLISION_COLOR)
        if state.action:
            self._augment_action_vector(state.action)
            self._augment_goal_position(state.action.robot_goal_grid)
        self._augment_timestep(state.timestep)
        pygame.display.update()

    def _draw_occupancy(self, occupancy: np.ndarray, color: RgbColor):
        pos_x, pos_y = np.where(occupancy)
        for grid_x, grid_y in zip(pos_x, pos_y):
            pos_x, pos_y = grid_x * self.pixels_per_cell, grid_y * self.pixels_per_cell
            rect = pygame.Rect(pos_x, pos_y, self.pixels_per_cell, self.pixels_per_cell)
            pygame.draw.rect(self.screen, color, rect)

    def _augment_goal_position(self, robot_goal: GridPosition):
        pygame.draw.circle(self.screen, ROBOT_GOAL_COLOR, robot_goal, 5)

    def _augment_action_vector(self, action: VisualizableAction):
        start = (action.start[0] * self.pixels_per_cell, action.start[1] * self.pixels_per_cell)
        end = (action.end[0] * self.pixels_per_cell, action.end[1] * self.pixels_per_cell)
        pygame.draw.line(self.screen, ROBOT_ACTION_COLOR, start, end)

    def _augment_timestep(self, timestep: int):
        text = f'step: {timestep}'
        text_surface = self.font.render(text, False, TEXT_COLOR)
        self.screen.blit(text_surface, self.timestep_text_pos)
