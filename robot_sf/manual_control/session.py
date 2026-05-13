"""Manual-control session state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ManualSessionState(StrEnum):
    """High-level state for a manual-control attempt."""

    IDLE = "idle"
    COUNTDOWN = "countdown"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class AttemptKey:
    """Stable identifier for one scenario/seed attempt target."""

    scenario_id: str
    seed: int


@dataclass
class AttemptProgress:
    """Progress metadata for the active manual-control attempt."""

    key: AttemptKey
    retry_count: int = 0
    beat_baseline: bool = False
    success: bool = False
    failure_reason: str | None = None


@dataclass
class ManualSessionController:
    """Pure controller for manual-control attempt lifecycle state."""

    countdown_steps: int = 3
    stop_between_episodes: bool = True
    speed_multiplier: float = 1.0
    state: ManualSessionState = ManualSessionState.IDLE
    countdown_remaining: int = 0
    active_attempt: AttemptProgress | None = None
    completed: dict[AttemptKey, AttemptProgress] = field(default_factory=dict)
    unresolved: dict[AttemptKey, AttemptProgress] = field(default_factory=dict)

    def start_attempt(self, scenario_id: str, seed: int) -> AttemptProgress:
        """Start a new attempt and enter countdown before simulation stepping.

        Returns
        -------
        AttemptProgress
            New active attempt progress record.
        """
        attempt = AttemptProgress(key=AttemptKey(scenario_id=scenario_id, seed=seed))
        self.active_attempt = attempt
        self.countdown_remaining = max(0, int(self.countdown_steps))
        self.state = (
            ManualSessionState.COUNTDOWN
            if self.countdown_remaining > 0
            else ManualSessionState.RUNNING
        )
        return attempt

    def advance_countdown(self) -> ManualSessionState:
        """Advance one countdown tick and switch to running when it reaches zero.

        Returns
        -------
        ManualSessionState
            Current state after the countdown tick.
        """
        if self.state != ManualSessionState.COUNTDOWN:
            return self.state
        self.countdown_remaining = max(0, self.countdown_remaining - 1)
        if self.countdown_remaining == 0:
            self.state = ManualSessionState.RUNNING
        return self.state

    def pause(self) -> None:
        """Pause a running attempt."""
        if self.state == ManualSessionState.RUNNING:
            self.state = ManualSessionState.PAUSED

    def resume(self) -> None:
        """Resume a paused attempt."""
        if self.state == ManualSessionState.PAUSED:
            self.state = ManualSessionState.RUNNING

    def toggle_pause(self) -> ManualSessionState:
        """Toggle between running and paused states.

        Returns
        -------
        ManualSessionState
            Current state after toggling.
        """
        if self.state == ManualSessionState.RUNNING:
            self.pause()
        elif self.state == ManualSessionState.PAUSED:
            self.resume()
        return self.state

    def set_speed_multiplier(self, value: float) -> None:
        """Set the simulation speed multiplier used by the interactive runner."""
        if value <= 0:
            raise ValueError("speed multiplier must be positive")
        self.speed_multiplier = float(value)

    def mark_terminal(
        self,
        *,
        success: bool,
        beat_baseline: bool,
        failure_reason: str | None = None,
    ) -> AttemptProgress:
        """Mark the active attempt terminal and update progress indexes.

        Returns
        -------
        AttemptProgress
            Completed active attempt progress.
        """
        if self.active_attempt is None:
            raise RuntimeError("cannot mark terminal without an active attempt")
        self.active_attempt.success = success
        self.active_attempt.beat_baseline = beat_baseline
        self.active_attempt.failure_reason = failure_reason
        self.completed[self.active_attempt.key] = self.active_attempt
        if beat_baseline:
            self.unresolved.pop(self.active_attempt.key, None)
        else:
            self.unresolved[self.active_attempt.key] = self.active_attempt
        self.state = ManualSessionState.TERMINAL
        return self.active_attempt

    def retry_active(self) -> AttemptProgress:
        """Restart the same scenario/seed with incremented retry count.

        Returns
        -------
        AttemptProgress
            New retry attempt progress.
        """
        if self.active_attempt is None:
            raise RuntimeError("cannot retry without an active attempt")
        retry = AttemptProgress(
            key=self.active_attempt.key,
            retry_count=self.active_attempt.retry_count + 1,
        )
        self.active_attempt = retry
        self.countdown_remaining = max(0, int(self.countdown_steps))
        self.state = (
            ManualSessionState.COUNTDOWN
            if self.countdown_remaining > 0
            else ManualSessionState.RUNNING
        )
        return retry

    @property
    def should_step(self) -> bool:
        """Return whether the runner should advance the environment now.

        Returns
        -------
        bool
            True when the session is in the running state.
        """
        return self.state == ManualSessionState.RUNNING
