"""Tests for bounded delegated-worker inactivity classification."""

from __future__ import annotations

import json

from scripts.dev import autopilot_state_snapshot as snapshot
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


def test_malformed_signal_shapes_fail_closed() -> None:
    """Malformed signal objects and path entries cannot become inactivity evidence."""
    malformed_wait = _observation(0)
    malformed_wait["wait_status"] = []
    missing_scope = _observation(0)
    missing_scope["scoped_git"] = {"changed_paths": []}
    missing_artifacts = _observation(0)
    missing_artifacts["required_artifacts"] = {}
    malformed_path = _observation(0)
    malformed_path["scoped_git"] = {"scope_ok": True, "changed_paths": [7]}

    for observation, expected_error in (
        (malformed_wait, "observation 0: wait_status must be an object"),
        (missing_scope, "observation 0: scoped_git.scope_ok must be boolean"),
        (
            missing_artifacts,
            "observation 0: required_artifacts.present_paths is required",
        ),
        (malformed_path, "observation 0: scoped_git.changed_paths[0] must be a string"),
    ):
        report = classify_worker_inactivity([observation])

        assert report["classification"] == "insufficient_evidence"
        assert report["recommended_action"] == "parent_fallback"
        assert report["errors"] == [expected_error]


def test_preexisting_git_and_artifact_paths_are_not_progress_without_a_delta() -> None:
    """Initial snapshots need an explicit progress signal or a later observed delta."""
    report = classify_worker_inactivity(
        [
            _observation(0, changed_paths=("preexisting.py",), present_paths=("result.json",)),
            _observation(60, changed_paths=("preexisting.py",), present_paths=("result.json",)),
        ],
        inactivity_after_seconds=30,
    )

    assert report["classification"] == "stalled"
    assert report["recommended_action"] == "interrupt"
    assert report["consecutive_no_progress_observations"] == 2


def test_signal_payload_limits_fail_closed() -> None:
    """Path and last-input payload limits keep diagnostic output bounded."""
    long_path = _observation(0, changed_paths=("x" * 513,))
    long_input = _observation(0, last_input="x" * 4097)

    path_report = classify_worker_inactivity([long_path])
    input_report = classify_worker_inactivity([long_input])

    assert path_report["classification"] == "insufficient_evidence"
    assert path_report["errors"] == [
        "observation 0: scoped_git.changed_paths[0] exceeds 512 characters"
    ]
    assert input_report["classification"] == "insufficient_evidence"
    assert input_report["errors"] == [
        "observation 0: wait_status.last_input exceeds 4096 characters"
    ]


def test_missing_worker_observation_file_fails_closed(tmp_path) -> None:
    """Unavailable observation files produce diagnostic fallback evidence."""
    report = snapshot.worker_inactivity_snapshot(tmp_path / "missing.json")

    assert report["status"] == "unavailable"
    assert report["classification"] == "insufficient_evidence"
    assert report["recommended_action"] == "parent_fallback"
    assert report["errors"]


def test_invalid_utf8_worker_observation_fails_closed(tmp_path) -> None:
    """Unreadable observation bytes produce bounded unavailable evidence."""
    path = tmp_path / "invalid-utf8.json"
    path.write_bytes(b"\xff")

    report = snapshot.worker_inactivity_snapshot(path)

    assert report["status"] == "unavailable"
    assert report["classification"] == "insufficient_evidence"
    assert report["recommended_action"] == "parent_fallback"
    assert "worker observation unavailable" in report["errors"][0]


def test_worker_observation_file_is_a_bounded_snapshot_entrypoint(tmp_path) -> None:
    """The canonical state snapshot can emit a stalled worker diagnostic from JSON input."""
    path = tmp_path / "worker-observations.json"
    path.write_text(
        json.dumps(
            {
                "observations": [_observation(0), _observation(360)],
            }
        ),
        encoding="utf-8",
    )

    report = snapshot.worker_inactivity_snapshot(path)

    assert report["status"] == "ok"
    assert report["classification"] == "stalled"
    assert report["recommended_action"] == "interrupt"
    assert report["route_evidence_only"] is True
    assert snapshot._build_parser().parse_args(
        ["--worker-observation", str(path)]
    ).worker_observation == [str(path)]


def test_build_snapshot_includes_worker_inactivity_report(monkeypatch, tmp_path) -> None:
    """Full autopilot snapshots carry the worker diagnostic without lifecycle side effects."""
    path = tmp_path / "worker-observations.json"
    path.write_text(
        json.dumps({"observations": [_observation(0), _observation(360)]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        snapshot,
        "git_snapshot",
        lambda **_kwargs: (
            {"branch": "test", "head_sha": "head", "origin_main_sha": "main"},
            [],
            [],
        ),
    )
    monkeypatch.setattr(snapshot, "claim_snapshot", lambda *_args, **_kwargs: ([], [], []))
    monkeypatch.setattr(
        snapshot, "issue_queue_snapshot", lambda *_args, **_kwargs: ([], [], [], [])
    )
    monkeypatch.setattr(snapshot, "pr_snapshot", lambda *_args, **_kwargs: ([], [], []))
    args = snapshot._build_parser().parse_args(["--worker-observation", str(path)])

    payload = snapshot.build_snapshot(args)

    assert payload["ok"] is True
    assert payload["worker_inactivity"][0]["classification"] == "stalled"
