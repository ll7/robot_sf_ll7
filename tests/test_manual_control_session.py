"""Tests for manual-control session state management."""

import math

import pytest

from robot_sf.manual_control.session import ManualSessionController, ManualSessionState


def test_session_starts_with_countdown_then_runs():
    """Starting an attempt should gate stepping behind the countdown."""
    controller = ManualSessionController(countdown_steps=2)

    attempt = controller.start_attempt("scenario-a", 7)

    assert attempt.retry_count == 0
    assert controller.state == ManualSessionState.COUNTDOWN
    assert controller.should_step is False
    assert controller.advance_countdown() == ManualSessionState.COUNTDOWN
    assert controller.advance_countdown() == ManualSessionState.RUNNING
    assert controller.should_step is True


def test_session_pause_resume_toggles_stepping():
    """Pause and resume should only affect running attempts."""
    controller = ManualSessionController(countdown_steps=0)
    controller.start_attempt("scenario-a", 7)

    assert controller.toggle_pause() == ManualSessionState.PAUSED
    assert controller.should_step is False
    assert controller.toggle_pause() == ManualSessionState.RUNNING
    assert controller.should_step is True


def test_terminal_attempt_tracks_unresolved_when_baseline_not_beaten():
    """Terminal attempts should record unresolved scenario/seeds."""
    controller = ManualSessionController(countdown_steps=0)
    attempt = controller.start_attempt("scenario-a", 7)

    result = controller.mark_terminal(
        success=False,
        beat_baseline=False,
        failure_reason="collision",
    )

    assert result is attempt
    assert controller.state == ManualSessionState.TERMINAL
    assert controller.completed[attempt.key].failure_reason == "collision"
    assert controller.unresolved[attempt.key].beat_baseline is False


def test_terminal_attempt_clears_unresolved_when_baseline_is_beaten():
    """A later successful retry should clear the unresolved case."""
    controller = ManualSessionController(countdown_steps=0)
    attempt = controller.start_attempt("scenario-a", 7)
    controller.mark_terminal(success=False, beat_baseline=False, failure_reason="timeout")
    retry = controller.retry_active()

    controller.mark_terminal(success=True, beat_baseline=True)

    assert retry.retry_count == 1
    assert attempt.key not in controller.unresolved
    assert controller.completed[attempt.key].beat_baseline is True


def test_speed_multiplier_must_be_positive():
    """Speed multipliers should fail closed for invalid values."""
    controller = ManualSessionController()

    with pytest.raises(ValueError, match="positive"):
        controller.set_speed_multiplier(0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_speed_multiplier_must_be_finite(value: float) -> None:
    """Speed multipliers should reject non-finite values."""
    controller = ManualSessionController()

    with pytest.raises(ValueError, match="positive"):
        controller.set_speed_multiplier(value)
