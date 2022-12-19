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
PEDESTRIAN_COLOR = (255, 50, 50)
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
    # world_to_grid: Callable[[float, float], GridPosition]
    # start: GridPosition = field(init=False)
    # end: GridPosition = field(init=False)

    # def __post_init__(self):
    #     x_start, y_start = self.robot_pose.pos.as_list
    #     x_start, y_start = self.world_to_grid(x_start, y_start)
    #     x_diff, y_diff = self.robot_action.vector.as_list
    #     x_diff, y_diff = self.world_to_grid(x_diff, y_diff)
    #     x_end, y_end = x_start + x_diff, y_start + y_diff
    #     self.start, self.end = (x_start, y_start), (x_end, y_end)


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

    def __init__(self, box_width: float=10, box_height: float=10, scaling: float=20):
        self.width = box_width * scaling
        self.height = box_height * scaling
        self.scaling = scaling

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE)
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
        state = self._norm_state(state)
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_robot(state.robot_pose)
        self._draw_pedestrians(state.pedestrian_positions)
        self._draw_obstacles(state.obstacles)
        if state.action:
            # self._augment_action_vector(state.action)
            self._augment_goal_position(state.action.robot_goal)
        self._augment_timestep(state.timestep)
        pygame.display.update()

    def _norm_state(self, state: VisualizableSimState) -> VisualizableSimState:
        state.pedestrian_positions *= self.scaling
        state.pedestrian_positions[:, 0] += self.width / 2
        state.pedestrian_positions[:, 1] += self.height / 2
        state.obstacles *= self.scaling
        state.obstacles[:, 0] += self.width / 2
        state.obstacles[:, 1] += self.height / 2
        state.obstacles[:, 2] += self.width / 2
        state.obstacles[:, 3] += self.height / 2
        state.robot_pose.pos = (
            state.robot_pose.pos[0] * self.scaling + self.width / 2,
            state.robot_pose.pos[1] * self.scaling + self.height / 2)
        if state.action:
            state.action.robot_goal = (
                state.action.robot_goal[0] * self.scaling + self.width / 2,
                state.action.robot_goal[1] * self.scaling + self.height / 2)
        return state

    def _draw_robot(self, pose: RobotPose):
        pygame.draw.circle(self.screen, ROBOT_COLOR, pose.coords, 0.5 * self.scaling)

    def _draw_pedestrians(self, ped_pos: np.ndarray):
        for ped_x, ped_y in ped_pos:
            pygame.draw.circle(self.screen, PEDESTRIAN_COLOR, (ped_x, ped_y), 0.4 * self.scaling)

    def _draw_obstacles(self, obstacles: np.ndarray):
        for s_x, s_y, e_x, e_y in obstacles:
            pygame.draw.line(self.screen, OBSTACLE_COLOR, (s_x, s_y), (e_x, e_y))

    def _augment_goal_position(self, robot_goal: WorldPosition):
        pygame.draw.circle(self.screen, ROBOT_GOAL_COLOR, robot_goal, self.scaling)

    # def _augment_action_vector(self, action: VisualizableAction):
    #     start = (action.start[0] * self.scaling, action.start[1] * self.scaling)
    #     end = (action.end[0] * self.scaling, action.end[1] * self.scaling)
    #     pygame.draw.line(self.screen, ROBOT_ACTION_COLOR, start, end)

    def _augment_timestep(self, timestep: int):
        text = f'step: {timestep}'
        text_surface = self.font.render(text, False, TEXT_COLOR)
        self.screen.blit(text_surface, self.timestep_text_pos)
