from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Callable, Tuple

import numpy as np
from matplotlib import pyplot as plt


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
    result_consumer: Callable[[EpisodeSummary], None]
    episode: int = field(init=False, default=0)
    timesteps: int = field(init=False, default=0)
    total_rewards: float = field(init=False, default=0)

    def __call__(self, reward: float, done: bool, meta: Dict):
        self.total_rewards += reward
        self.timesteps += 1

        if done:
            def get_outcome(state: dict) -> EpisodeOutcome:
                if state["is_pedestrian_collision"]:
                    return EpisodeOutcome.COLLISION_WITH_PEDESTRIAN
                elif state["is_obstacle_collision"]:
                    return EpisodeOutcome.COLLISION_WITH_OBSTACLE
                elif state["is_robot_at_goal"]:
                    return EpisodeOutcome.GOAL_REACHED
                elif state["is_timesteps_exceeded"]:
                    return EpisodeOutcome.TIMEOUT
                else:
                    raise ValueError((
                        'Invalid episode outcome! If episode is done, it must be ',
                        'within one of the previous 4 cases!'))

            outcome = get_outcome(meta)
            result = EpisodeSummary(meta['episode'], self.timesteps, self.total_rewards, outcome)
            self.result_consumer(result)


@dataclass
class ResultBuffer:
    buffered_result_consumer: Callable[[List[EpisodeSummary]], None]
    buffer_size: int = 200
    results_buffer: List[EpisodeSummary] = field(init=False, default_factory=list)

    def add_result(self, result: EpisodeSummary):
        self.results_buffer.append(result)

        if len(self.results_buffer) >= self.buffer_size:
            results = self.results_buffer[:self.buffer_size]
            self.results_buffer.clear()
            self.buffered_result_consumer(results)


@dataclass
class EpisodeCompletionMetric:
    num_completed: int = field(init=False, default=0)
    num_timeout: int = field(init=False, default=0)
    num_ped_coll: int = field(init=False, default=0)
    num_obst_coll: int = field(init=False, default=0)

    def __call__(self, result: EpisodeSummary):
        if result.outcome == EpisodeOutcome.GOAL_REACHED:
            self.num_completed += 1
        elif result.outcome == EpisodeOutcome.TIMEOUT:
            self.num_timeout += 1
        elif result.outcome == EpisodeOutcome.COLLISION_WITH_PEDESTRIAN:
            self.num_ped_coll += 1
        elif result.outcome == EpisodeOutcome.COLLISION_WITH_OBSTACLE:
            self.num_obst_coll += 1

    def reset_states(self):
        self.num_completed = 0
        self.num_timeout = 0
        self.num_ped_coll = 0
        self.num_obst_coll = 0


CompletionDistribution = Tuple[float, float, float, float]

@dataclass
class EpisodeCompletionHistogram:
    buffer_size: int
    comptation_rates_over_time: List[CompletionDistribution] = field(default_factory=list)

    def add_data_point(self, rates: CompletionDistribution):
        self.comptation_rates_over_time.append(rates)

    def plot(self, out_file: str):
        labels = ['Goal Reached', 'Timeout', 'Pedestrian Collision', 'Obstacle Collision']
        histogram = np.array(self.comptation_rates_over_time)
        timeline = np.arange(histogram.shape[0]) * self.buffer_size
        fig, ax = plt.subplots()
        ax.stackplot(timeline, histogram, labels=labels, alpha=0.8)
        ax.set_title('Completion rates development during training')
        ax.set_xlabel('Step')
        ax.set_ylabel('Completion rates')
        plt.savefig(out_file)


def plot_episode_completion(results: List[EpisodeSummary], out_file: str):
    num_completed = sum([1 for r in results if r.outcome == EpisodeOutcome.GOAL_REACHED])
    num_timeout = sum([1 for r in results if r.outcome == EpisodeOutcome.TIMEOUT])
    num_ped_coll = sum([1 for r in results if r.outcome == EpisodeOutcome.COLLISION_WITH_PEDESTRIAN])
    num_obst_coll = sum([1 for r in results if r.outcome == EpisodeOutcome.COLLISION_WITH_OBSTACLE])
    sizes = np.array([num_completed, num_timeout, num_ped_coll, num_obst_coll]) / len(results)
    labels = ['Reach Goal', 'Timeout', 'Ped Coll.', 'Obstacle Coll.']
    fig1, ax1 = plt.subplots()
    plt.bar(labels, sizes)
    plt.savefig(out_file)
