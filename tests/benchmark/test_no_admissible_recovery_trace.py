"""Fast recovery telemetry contracts: no planner/environment stepping."""

import pytest


@pytest.mark.parametrize("kind", ["brake", "least_bad_clearance", "progress_escape"])
def test_no_admissible_recovery_kind_reaches_guard_and_planner_trace(kind: str) -> None:
    """Native recovery telemetry must survive the real stats and trace projection."""
    from types import SimpleNamespace

    from robot_sf.benchmark.map_runner.map_runner import _attach_guard_decision_stats
    from robot_sf.benchmark.map_runner.map_runner_episode import _step_planner_decision_dwa_keys

    policy = SimpleNamespace(_planner_stats=lambda: {})
    recovery = {
        "no_admissible_command": True,
        "no_admissible_command_count": 3,
        "recovery_command_count": 0,
        "recovery_kind": kind,
    }
    adapter = SimpleNamespace(diagnostics=lambda: dict(recovery))
    _attach_guard_decision_stats(
        policy,
        {"shield_stats": {"last_decision": {"decision_label": "fallback_best_effort"}}},
        adapter,
    )
    trace = {}
    _step_planner_decision_dwa_keys(trace, policy._planner_stats()["last_decision"])
    assert trace == recovery
