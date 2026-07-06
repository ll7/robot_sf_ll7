"""Tests for the stateful deadlock-recovery stage of the issue #3501 safety wrapper.

These exercise the fourth predeclared safety stage: it must break the *frozen robot* failure mode
by permitting a bounded in-place rotation, while never adding forward speed (so the hard stop/yield
veto is preserved and recovery cannot inject forward motion into a hazard).
"""

from __future__ import annotations

import pytest

from robot_sf.robot.safety_wrapper import (
    DEADLOCK_RECOVERY_SCHEMA,
    DeadlockRecoveryConfig,
    DeadlockRecoveryMonitor,
    SafetyContext,
)

# A context where the robot is boxed in by a nearby pedestrian (hazard-blocked).
BLOCKED = SafetyContext(min_pedestrian_distance_m=0.8, min_clearance_m=0.2, min_ttc_s=0.5)
# A context with clear surroundings (e.g. goal reached): a stop here is not a deadlock.
CLEAR = SafetyContext(min_pedestrian_distance_m=25.0, min_clearance_m=24.0, min_ttc_s=None)


def _run_frozen(monitor: DeadlockRecoveryMonitor, n: int, context: SafetyContext = BLOCKED):
    """Feed ``n`` frozen (zero forward speed) steps and return the per-step records."""
    return [monitor.step(0.0, 0.0, context) for _ in range(n)]


def test_disabled_monitor_is_passthrough():
    """A disabled monitor never recovers and passes the command through unchanged."""
    monitor = DeadlockRecoveryMonitor()  # disabled by default
    record = monitor.step(0.0, 0.0, BLOCKED)
    assert record["enabled"] is False
    assert record["recovery_active"] is False
    assert record["final_linear_velocity"] == 0.0
    assert record["final_angular_velocity"] == 0.0
    assert record["schema_version"] == DEADLOCK_RECOVERY_SCHEMA


def test_no_recovery_before_patience():
    """Recovery must not engage while the frozen run is below ``patience_steps``."""
    config = DeadlockRecoveryConfig(enabled=True, patience_steps=5, recovery_steps=3)
    monitor = DeadlockRecoveryMonitor(config)
    records = _run_frozen(monitor, 4)
    assert all(not r["recovery_active"] for r in records)
    assert [r["frozen_run"] for r in records] == [1, 2, 3, 4]
    assert all(not r["deadlock_detected"] for r in records)


def test_recovery_engages_at_patience_and_rotates_in_place():
    """At ``patience_steps`` a deadlock is detected and an in-place rotation is applied."""
    config = DeadlockRecoveryConfig(
        enabled=True,
        patience_steps=3,
        recovery_steps=2,
        recovery_angular_velocity_rad_s=0.7,
        recovery_turn_sign=1,
    )
    monitor = DeadlockRecoveryMonitor(config)
    records = _run_frozen(monitor, 5)
    # Steps 1-2: below patience, no recovery. Step 3 hits patience -> recovery engages.
    assert not records[0]["recovery_active"]
    assert not records[1]["recovery_active"]
    assert records[2]["deadlock_detected"] is True
    assert records[2]["recovery_active"] is True
    # Recovery overrides angular velocity but preserves forward speed (0.0 here).
    assert records[2]["final_angular_velocity"] == pytest.approx(0.7)
    assert records[2]["final_linear_velocity"] == 0.0


def test_recovery_cycles_maneuver_then_re_arms_patience():
    """A completed maneuver re-arms patience, so a still-stuck robot pauses then retries."""
    config = DeadlockRecoveryConfig(enabled=True, patience_steps=2, recovery_steps=2)
    monitor = DeadlockRecoveryMonitor(config)
    records = _run_frozen(monitor, 6)
    active = [r["recovery_active"] for r in records]
    # Patience reached at index 1 -> a 2-step maneuver (indices 1,2). The completed maneuver
    # re-arms the patience window, so the still-frozen robot pauses one step (index 3) to let the
    # planner react, then a fresh maneuver runs (indices 4,5).
    assert active == [False, True, True, False, True, True]


def test_negative_turn_sign_rotates_the_other_way():
    """``recovery_turn_sign=-1`` rotates in the opposite direction during recovery."""
    config = DeadlockRecoveryConfig(
        enabled=True, patience_steps=1, recovery_steps=1, recovery_turn_sign=-1
    )
    monitor = DeadlockRecoveryMonitor(config)
    record = monitor.step(0.0, 0.0, BLOCKED)
    assert record["final_angular_velocity"] == pytest.approx(-0.5)


def test_movement_resets_frozen_run_and_ends_recovery():
    """Resumed forward motion clears the frozen run and cancels any active maneuver."""
    config = DeadlockRecoveryConfig(enabled=True, patience_steps=2, recovery_steps=5)
    monitor = DeadlockRecoveryMonitor(config)
    _run_frozen(monitor, 3)  # engage recovery
    moving = monitor.step(0.9, 0.1, BLOCKED)  # robot moves again
    assert moving["frozen_run"] == 0
    assert moving["recovery_active"] is False
    assert moving["recovery_steps_remaining"] == 0
    assert moving["final_linear_velocity"] == pytest.approx(0.9)
    assert moving["final_angular_velocity"] == pytest.approx(0.1)


def test_clear_surroundings_is_not_a_deadlock():
    """A frozen robot with clear surroundings (goal reached) is never treated as deadlocked."""
    config = DeadlockRecoveryConfig(enabled=True, patience_steps=2, recovery_steps=3)
    monitor = DeadlockRecoveryMonitor(config)
    records = _run_frozen(monitor, 10, context=CLEAR)
    # Frozen but not hazard-blocked (goal reached): never counts as a deadlock.
    assert all(r["frozen_run"] == 0 for r in records)
    assert all(not r["recovery_active"] for r in records)
    assert all(not r["deadlock_detected"] for r in records)


def test_recovery_never_adds_forward_speed_during_hard_stop():
    """Recovery only rotates; it never re-introduces forward speed under a persistent hard stop."""
    # Simulate a persistent hard stop: the wrapper zeroes forward speed each step. Recovery must
    # only add rotation and must never re-introduce forward motion into the hazard.
    config = DeadlockRecoveryConfig(enabled=True, patience_steps=2, recovery_steps=4)
    monitor = DeadlockRecoveryMonitor(config)
    records = _run_frozen(monitor, 6)
    for record in records:
        assert record["final_linear_velocity"] == 0.0


def test_obstacle_only_freeze_counts_as_hazard_blocked():
    """A tight predicted clearance marks a freeze as hazard-blocked without a nearby pedestrian."""
    context = SafetyContext(min_pedestrian_distance_m=50.0, min_clearance_m=0.1, min_ttc_s=None)
    config = DeadlockRecoveryConfig(enabled=True, patience_steps=1, recovery_steps=1)
    monitor = DeadlockRecoveryMonitor(config)
    record = monitor.step(0.0, 0.0, context)
    assert record["deadlock_detected"] is True
    assert record["recovery_active"] is True


def test_reset_clears_counters():
    """``reset`` clears the frozen-run and recovery counters for a fresh episode."""
    config = DeadlockRecoveryConfig(enabled=True, patience_steps=2, recovery_steps=3)
    monitor = DeadlockRecoveryMonitor(config)
    _run_frozen(monitor, 3)
    monitor.reset()
    first = monitor.step(0.0, 0.0, BLOCKED)
    assert first["frozen_run"] == 1
    assert first["recovery_active"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"patience_steps": 0},
        {"recovery_steps": 0},
        {"recovery_turn_sign": 0},
        {"recovery_angular_velocity_rad_s": 0.0},
        {"frozen_speed_eps_m_s": -0.1},
        {"hazard_proximity_m": -1.0},
        {"hazard_clearance_m": -0.5},
    ],
)
def test_config_rejects_invalid_thresholds(kwargs):
    """Out-of-range recovery thresholds are rejected at construction time."""
    with pytest.raises(ValueError):
        DeadlockRecoveryConfig(enabled=True, **kwargs)
