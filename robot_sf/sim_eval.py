from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

from matplotlib import pyplot as plt
from robot_sf.robot_env import EnvState


class EpisodeOutcome(IntEnum):
    GOAL_REACHED = 0
    COLLISION_WITH_PEDESTRIAN = 1
    COLLISION_WITH_OBSTACLE = 2
    TIMEOUT = 3


@dataclass
class EpisodeSummary:
    id: int
    timesteps: int
    rewards: float
    outcome: EpisodeOutcome


@dataclass
class EpisodeLoggingCallback:
    episode: int = 0
    timesteps: int = 0
    total_rewards: float = 0
    results: List[EpisodeSummary] = field(default_factory=list)

    def __call__(self, reward: float, done: bool, meta: Dict):
        self.total_rewards += reward
        self.timesteps += 1

        if done:
            def get_outcome(state: EnvState) -> EpisodeOutcome:
                if state.is_pedestrian_collision:
                    return EpisodeOutcome.COLLISION_WITH_PEDESTRIAN
                elif state.is_obstacle_collision:
                    return EpisodeOutcome.COLLISION_WITH_OBSTACLE
                elif state.is_robot_at_goal:
                    return EpisodeOutcome.GOAL_REACHED
                elif state.is_timesteps_exceeded:
                    return EpisodeOutcome.TIMEOUT
                else:
                    raise ValueError((
                        'Invalid episode outcome! If episode is done, it must be ',
                        'within one of the previous 4 cases!'))

            outcome = get_outcome(meta['meta'])
            result = EpisodeSummary(meta['step'], self.total_rewards, self.timesteps, outcome)
            self.results.append(result)


@dataclass
class EvaluationPlotting:
    results: List[EpisodeSummary]

    def plot_episode_completion(self):
        pass

    def plot_episode_mean_steps(self):
        pass

    def plot_episode_mean_rewards(self):
        pass
