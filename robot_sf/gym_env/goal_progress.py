"""Episode-local goal progress history for training reward shaping."""

from collections import deque


class GoalProgressWindow:
    """Measure net goal progress over ten simulation steps (1 s at default dt).

    Hold the initial goal distance through warmup; a new instance starts each episode.
    """

    def __init__(self, initial_distance: float) -> None:
        """Prime the window with the episode's initial distance to its final goal."""
        self._distances: deque[float] = deque([initial_distance], maxlen=11)

    def update(self, distance: float) -> dict[str, float | bool]:
        """Return net progress in metres and whether the ten-step window is full."""
        self._distances.append(distance)
        return {
            "goal_progress_window": self._distances[0] - distance,
            "goal_progress_window_ready": len(self._distances) == 11,
        }
