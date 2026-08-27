"""Tests for bounded delegated-worker inactivity classification."""

from __future__ import annotations

from scripts.dev.autopilot_state_snapshot import classify_worker_inactivity


def _observation(
    elapsed_seconds: float,
    *,
    activity: str = "waiting",
    changed_paths: tuple[str, ...] = (),
    present_paths: tuple[str, ...] = (),
    last_input: str = "implement the assigned change",
    state: str = "running",
    productive: bool = False,
) -> dict[str, object]:
    """Build a synthetic wait/status, Git, and artifact snapshot."""
    return {
        "agent_id": "worker-7959",
        "elapsed_seconds": elapsed_seconds,
        "wait_status": {
            "state": state,
            "activity": activity,
            "last_input": last_input,
            "output_quiet": True,
            "productive": productive,
        },
        "scoped_git": {"scope_ok": True, "changed_paths": changed_paths},
        "required_artifacts": {"present_paths": present_paths},
    }


def test_two_consecutive_no_progress_observations_report_stalled_worker() -> None:
    """A running worker needs two quiet observations before interrupt is recommended."""
    report = classify_worker_inactivity(
        [_observation(0), _observation(120)],
        inactivity_after_seconds=60,
    )

    assert report["schema"] == "worker_inactivity.v1"
    assert report["classification"] == "stalled"
    assert report["recommended_action"] == "interrupt"
    assert report["agent_id"] == "worker-7959"
    assert report["elapsed_interval_seconds"] == 120
    assert report["consecutive_no_progress_observations"] == 2
    assert report["last_input"] == "implement the assigned change"
    assert report["observed_signals"]["progress"] is False
    assert report["observed_signals"]["scoped_git"]["changed_paths"] == []
    assert report["observed_signals"]["required_artifacts"]["present_paths"] == []


def test_productive_long_running_test_resets_no_progress_streak() -> None:
    """Quiet output does not stall a status-marked productive test or edit."""
    report = classify_worker_inactivity(
        [_observation(0), _observation(120), _observation(240, activity="test_running")],
        inactivity_after_seconds=60,
    )

    assert report["classification"] == "productive"
    assert report["recommended_action"] == "continue"
    assert report["consecutive_no_progress_observations"] == 0
    assert report["observed_signals"]["wait_status"]["quiet_output"] is True
    assert report["observed_signals"]["progress_reasons"] == ["wait_status"]


def test_scoped_git_or_required_artifact_progress_prevents_stall() -> None:
    """Scoped edits and required-artifact creation are productive signals."""
    report = classify_worker_inactivity(
        [
            _observation(0),
            _observation(60, changed_paths=("scripts/dev/example.py",)),
            _observation(120, present_paths=("result.json",)),
        ],
        inactivity_after_seconds=30,
    )

    assert report["classification"] == "productive"
    assert report["recommended_action"] == "continue"
    assert report["consecutive_no_progress_observations"] == 0
    assert report["observation_history"][1]["progress_reasons"] == ["scoped_git"]
    assert report["observation_history"][2]["progress_reasons"] == ["required_artifacts"]


def test_recovery_action_closes_or_falls_back_for_non_running_states() -> None:
    """Deterministic recovery actions distinguish closeable and unsafe states."""
    close_report = classify_worker_inactivity(
        [_observation(0, state="interrupted"), _observation(60, state="interrupted")],
        inactivity_after_seconds=30,
    )
    fallback_report = classify_worker_inactivity(
        [_observation(0, state="unknown"), _observation(60, state="unknown")],
        inactivity_after_seconds=30,
    )

    assert close_report["recommended_action"] == "close"
    assert fallback_report["recommended_action"] == "parent_fallback"


def test_malformed_or_unbounded_input_fails_closed_without_stalled_claim() -> None:
    """Malformed snapshots cannot be mistaken for inactivity evidence."""
    report = classify_worker_inactivity(
        [_observation(0), {"agent_id": "worker-7959", "elapsed_seconds": -1}],
        inactivity_after_seconds=30,
    )

    assert report["classification"] == "insufficient_evidence"
    assert report["recommended_action"] == "parent_fallback"
    assert report["errors"] == [
        "observation 1: elapsed_seconds must be a finite non-negative number"
    ]
