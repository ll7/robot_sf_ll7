"""Tests for the exact-head stability snapshot mode (deterministic, no live GitHub)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.check_pr_ci_status import (
    STABILITY_SNAPSHOT_SCHEMA,
    StabilitySnapshotEvidence,
    _fetch_ci_status,
    _fetch_rate_limit_info,
    _fetch_stability_snapshot,
    _is_rate_limit_error_text,
    _parse_retry_after,
    _resolve_ci_state,
    _snapshot_resume_command,
    _validate_expected_main_sha,
    _validate_expected_metadata_digest,
    evaluate_stability_snapshot,
    main,
)
from scripts.dev.pr_metadata import metadata_digest

HEAD = "a" * 40
MAIN = "b" * 40
OLD_HEAD = "c" * 40
OLD_MAIN = "d" * 40
DIGEST = "e" * 64
OLD_DIGEST = "f" * 64
QUOTA_STDERR = "HTTP 403: API rate limit exceeded for user"


def test_snapshot_ci_read_is_single_attempt_without_rest_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot CI reads do not retry transient failures or expand into REST reads."""
    runner = MagicMock(
        return_value=MagicMock(
            returncode=1,
            stdout="",
            stderr="HTTP 503: service unavailable",
        )
    )
    rest_fallback = MagicMock(side_effect=AssertionError("snapshot must not use REST fallback"))
    monkeypatch.setattr("scripts.dev.check_pr_ci_status._gh", runner)
    monkeypatch.setattr("scripts.dev.check_pr_ci_status._fetch_ci_status_rest", rest_fallback)

    result = _fetch_ci_status("42", max_attempts=1, allow_rest_fallback=False)

    assert result["status"] == "error"
    assert result["error_kind"] == "graphql_transient_exhausted"
    assert runner.call_count == 1
    rest_fallback.assert_not_called()


def test_snapshot_requests_single_ci_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stability snapshot selects the no-retry, no-fallback CI path."""
    ci_fetch = MagicMock(return_value={"status": "error", "error": "HTTP 503: service unavailable"})
    monkeypatch.setattr("scripts.dev.check_pr_ci_status._fetch_ci_status", ci_fetch)

    result = _fetch_stability_snapshot(
        "42",
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
        expected_metadata_digest=DIGEST,
    )

    assert result["status"] == "error"
    ci_fetch.assert_called_once_with(
        "42",
        repo="ll7/robot_sf_ll7",
        max_attempts=1,
        allow_rest_fallback=False,
    )


def _evidence(**overrides: Any) -> StabilitySnapshotEvidence:
    """Build a stable-evidence fixture with per-test overrides."""
    base = {
        "observed_head_sha": HEAD,
        "observed_main_sha": MAIN,
        "base_sha": MAIN,
        "base_ref": "main",
        "observed_metadata_digest": DIGEST,
        "ci_overall": "success",
        "ci_pending_reason": "",
        "expected_head_sha": HEAD,
        "expected_main_sha": MAIN,
        "expected_metadata_digest": DIGEST,
    }
    base.update(overrides)
    return StabilitySnapshotEvidence(**base)


def _evaluate(
    evidence: StabilitySnapshotEvidence,
    **kwargs: Any,
) -> dict[str, Any]:
    return evaluate_stability_snapshot(evidence, pr="42", repo="ll7/robot_sf_ll7", **kwargs)


def test_snapshot_stable_when_everything_matches() -> None:
    """Matching head/main/digest and green CI produce a stable route-evidence snapshot."""
    result = _evaluate(_evidence())
    assert result["schema"] == STABILITY_SNAPSHOT_SCHEMA
    assert result["status"] == "stable"
    assert result["route_evidence_only"] is True
    assert result["invalidated"] is False
    assert result["invalidated_reasons"] == []
    assert result["head_sha_matches"] is True
    assert result["main_sha_matches"] is True
    assert result["metadata"]["digest_matches"] is True
    assert result["ci_state"] == "success"
    assert result["resume"] == {"command": None, "reason": "none", "min_delay_seconds": None}


def test_snapshot_head_movement_invalidates_with_observed_values() -> None:
    """Head movement invalidates the snapshot with observed values and a safe resume."""
    result = _evaluate(_evidence(observed_head_sha=OLD_HEAD))
    assert result["status"] == "changed"
    assert result["invalidated"] is True
    assert result["invalidated_reasons"] == ["head_sha_changed"]
    assert result["head_sha"] == OLD_HEAD
    assert result["head_sha_matches"] is False
    resume = result["resume"]
    assert resume["reason"] == "refresh_expecteds_and_rerun"
    assert "check_pr_ci_status.py 42 --stability-snapshot" in resume["command"]
    assert f"--expected-head-sha {OLD_HEAD}" in resume["command"]
    assert "&&" not in resume["command"]


def test_snapshot_main_movement_invalidates() -> None:
    """Current-main movement invalidates the snapshot without retrying or authorizing."""
    result = _evaluate(_evidence(observed_main_sha=OLD_MAIN))
    assert result["status"] == "changed"
    assert result["invalidated_reasons"] == ["main_sha_changed"]
    assert result["main_sha"] == OLD_MAIN
    assert result["main_sha_matches"] is False
    assert f"--expected-main-sha {OLD_MAIN}" in result["resume"]["command"]


def test_snapshot_metadata_drift_invalidates() -> None:
    """A drifted title/body digest invalidates the snapshot with a refresh resume."""
    result = _evaluate(_evidence(observed_metadata_digest=OLD_DIGEST))
    assert result["status"] == "changed"
    assert result["invalidated_reasons"] == ["metadata_digest_changed"]
    assert result["metadata"]["observed_digest"] == OLD_DIGEST
    assert result["metadata"]["digest_matches"] is False
    assert f"--expected-metadata-digest {OLD_DIGEST}" in result["resume"]["command"]


def test_snapshot_metadata_drift_resume_suggests_reconcile_with_desired_pair() -> None:
    """With a desired title/body pair, drift resumes through the reconcile helper."""
    result = _evaluate(
        _evidence(observed_metadata_digest=OLD_DIGEST),
        desired_metadata_digest=DIGEST,
        desired_hint=f"--title {HEAD!r} --body-file body.md",
    )
    assert result["status"] == "changed"
    assert result["metadata"]["desired_digest"] == DIGEST
    resume = result["resume"]
    assert resume["reason"] == "reconcile_metadata_then_rerun"
    assert "uv run python scripts/dev/gh_pr_body_rest.py 42 --reconcile" in resume["command"]
    assert "--body-file body.md" in resume["command"]
    assert "&&" in resume["command"]
    assert "check_pr_ci_status.py 42 --stability-snapshot" in resume["command"]


def test_snapshot_status_propagation_lag_is_distinct() -> None:
    """Completed-success workflow lag is distinct from pending work and failure."""
    lag = _evidence(ci_overall="pending", ci_pending_reason="status_propagation_lag")
    lag_result = _evaluate(lag)
    assert lag_result["status"] == "status_propagation_lag"
    assert lag_result["ci_state"] == "status_propagation_lag"
    assert lag_result["resume"]["reason"] == "bounded_ci_wait"
    assert "--poll-attempts 40" in lag_result["resume"]["command"]

    pending = _evaluate(_evidence(ci_overall="pending"))
    assert pending["status"] == "pending"
    assert pending["ci_state"] == "pending"

    failure = _evaluate(_evidence(ci_overall="failure"))
    assert failure["status"] == "failure"
    assert failure["ci_state"] == "failure"
    assert failure["resume"]["reason"] == "rerun_after_fix"


def test_snapshot_head_read_race_invalidates() -> None:
    """A head read that disagrees between CI and pull evidence invalidates the snapshot."""
    result = _evaluate(
        _evidence(expected_head_sha="", expected_main_sha="", expected_metadata_digest=""),
        head_read_race=True,
    )
    assert result["status"] == "changed"
    assert result["invalidated_reasons"] == ["head_sha_read_race"]


def test_snapshot_unknown_ci_state_fails_closed() -> None:
    """An unclassifiable CI rollup cannot become stable evidence."""
    result = _evaluate(_evidence(ci_overall="bogus"))
    assert result["status"] == "error"


def test_snapshot_without_expecteds_never_invalidates_metadata_only() -> None:
    """Missing expected guards leave match states None instead of claiming stability."""
    result = _evaluate(
        _evidence(expected_head_sha="", expected_main_sha="", expected_metadata_digest="")
    )
    assert result["status"] == "stable"
    assert result["head_sha_matches"] is None
    assert result["main_sha_matches"] is None
    assert result["metadata"]["digest_matches"] is None


def test_resolve_ci_state_distinguishes_lag_from_pending_and_failure() -> None:
    assert _resolve_ci_state("success", "") == "success"
    assert _resolve_ci_state("failure", "") == "failure"
    assert _resolve_ci_state("pending", "status_propagation_lag") == "status_propagation_lag"
    assert _resolve_ci_state("pending", "") == "pending"
    assert _resolve_ci_state("", "") == "unknown"


def test_resume_command_shapes() -> None:
    evidence = _evidence()
    stable = _snapshot_resume_command("stable", reasons=[], evidence=evidence, pr="42")
    assert stable == {"command": None, "reason": "none", "min_delay_seconds": None}

    pending = _snapshot_resume_command("pending", reasons=[], evidence=evidence, pr="42")
    assert pending["reason"] == "bounded_ci_wait"
    assert pending["command"].startswith("scripts/dev/check_pr_ci_status.py 42 --json")
    assert "--max-wall-seconds 1200" in pending["command"]

    quota = _snapshot_resume_command(
        "quota_blocked", reasons=[], evidence=evidence, pr="42", min_delay_seconds=60
    )
    assert quota["reason"] == "rest_rate_limit_reset"
    assert quota["min_delay_seconds"] == 60


def test_sha_and_digest_validators() -> None:
    assert _validate_expected_main_sha("") is None
    assert _validate_expected_main_sha(MAIN) is None
    assert "40-hex" in _validate_expected_main_sha(MAIN[:8])
    assert _validate_expected_metadata_digest("") is None
    assert _validate_expected_metadata_digest(DIGEST) is None
    assert "64-hex" in _validate_expected_metadata_digest(DIGEST[:8])


def test_is_rate_limit_error_text_and_retry_after_parsing() -> None:
    assert _is_rate_limit_error_text(QUOTA_STDERR)
    assert _is_rate_limit_error_text("too many requests")
    assert not _is_rate_limit_error_text("merge conflict")
    assert _parse_retry_after("Retry-After: 60") == 60
    assert _parse_retry_after("retry-after: 120") == 120
    assert _parse_retry_after("no header here") is None


def test_fetch_rate_limit_info_parses_core_and_graphql(monkeypatch: pytest.MonkeyPatch) -> None:
    """A healthy ``gh api rate_limit`` response normalizes through the shared parser."""
    payload = {
        "resources": {
            "core": {"limit": 5000, "remaining": 4000, "reset": 1712345678},
            "graphql": {"limit": 5000, "remaining": 2500, "reset": 1712345678},
        }
    }
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._gh",
        MagicMock(return_value=MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")),
    )
    info = _fetch_rate_limit_info()
    assert info["source"] == "gh_api_rate_limit"
    assert info["core_remaining"] == 4000
    assert info["core_reset_epoch_seconds"] == 1712345678
    assert info["graphql_remaining"] == 2500


def test_fetch_rate_limit_info_uses_retry_after_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked rate_limit read surfaces Retry-After as the bounded resume hint."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._gh",
        MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr="Retry-After: 60")),
    )
    info = _fetch_rate_limit_info()
    assert info == {"source": "retry_after", "retry_after_seconds": 60}


def test_fetch_rate_limit_info_unavailable_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecoverable rate_limit read never fabricates quota state."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._gh",
        MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr="connection refused")),
    )
    assert _fetch_rate_limit_info() == {"source": "unavailable"}


def test_fetch_snapshot_quota_blocked_resumes_without_retry_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quota-exhausted CI read yields quota_blocked with a bounded resume, once."""
    ci_fetch = MagicMock(
        return_value={
            "status": "error",
            "error_kind": "graphql_quota_exhausted",
            "error": "GraphQL quota exhausted and REST pull fallback failed",
        }
    )
    rate_fetch = MagicMock(
        return_value={
            "source": "gh_api_rate_limit",
            "core_remaining": 0,
            "core_reset_epoch_seconds": 1000,
        }
    )
    monkeypatch.setattr("scripts.dev.check_pr_ci_status._fetch_ci_status", ci_fetch)
    monkeypatch.setattr("scripts.dev.check_pr_ci_status._fetch_rate_limit_info", rate_fetch)
    monkeypatch.setattr("scripts.dev.check_pr_ci_status.time.time", lambda: 0)

    result = _fetch_stability_snapshot(
        "42",
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
        expected_metadata_digest=DIGEST,
    )

    assert result["status"] == "quota_blocked"
    assert result["route_evidence_only"] is True
    assert result["resume"]["reason"] == "rest_rate_limit_reset"
    assert result["resume"]["min_delay_seconds"] == 1000
    assert result["resume"]["resume_epoch_seconds"] == 1000
    assert "check_pr_ci_status.py 42 --stability-snapshot" in result["resume"]["command"]
    ci_fetch.assert_called_once()
    rate_fetch.assert_called_once()


def test_fetch_snapshot_quota_blocked_reuses_rate_limit_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REST-read rate-limit failure reuses the already-fetched quota state."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_ci_status",
        MagicMock(
            return_value={
                "status": "ok",
                "pr": 42,
                "head_sha": HEAD,
                "checks": {"overall": "success"},
                "reviews": {},
            }
        ),
    )
    rate_limit = {
        "source": "gh_api_rate_limit",
        "core_remaining": 0,
        "core_reset_epoch_seconds": 500,
    }
    rate_fetch = MagicMock(return_value=rate_limit)
    monkeypatch.setattr("scripts.dev.check_pr_ci_status._fetch_rate_limit_info", rate_fetch)
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._rest_api_get_detailed",
        MagicMock(return_value=(None, QUOTA_STDERR)),
    )
    monkeypatch.setattr("scripts.dev.check_pr_ci_status.time.time", lambda: 0)

    result = _fetch_stability_snapshot(
        "42",
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
        expected_metadata_digest=DIGEST,
    )

    assert result["status"] == "quota_blocked"
    assert result["resume"]["min_delay_seconds"] == 500
    rate_fetch.assert_called_once()


def test_fetch_snapshot_stable_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully matching live read produces a stable snapshot with checks and quota state."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_ci_status",
        MagicMock(
            return_value={
                "status": "ok",
                "pr": 42,
                "head_sha": HEAD,
                "checks": {
                    "overall": "success",
                    "total": 1,
                    "by_conclusion": {"success": 1},
                },
                "reviews": {"APPROVED": 1},
            }
        ),
    )
    pull = {
        "title": "fix: the thing",
        "body": "body",
        "head": {"sha": HEAD},
        "base": {"sha": MAIN, "ref": "main"},
    }
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._rest_api_get_detailed",
        MagicMock(
            side_effect=[
                (pull, ""),
                ({"commit": {"sha": MAIN}}, ""),
            ]
        ),
    )
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_rate_limit_info",
        MagicMock(
            return_value={
                "source": "gh_api_rate_limit",
                "core_remaining": 4000,
                "core_reset_epoch_seconds": 1712345678,
            }
        ),
    )

    result = _fetch_stability_snapshot(
        "42",
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
        expected_metadata_digest=metadata_digest("fix: the thing", "body"),
    )

    assert result["status"] == "stable"
    assert result["checks"]["overall"] == "success"
    assert result["reviews"] == {"APPROVED": 1}
    assert result["rate_limit"]["core_remaining"] == 4000
    assert result["metadata"]["observed_digest"] == metadata_digest("fix: the thing", "body")


def test_fetch_snapshot_head_read_race_through_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CI head that disagrees with the pull head invalidates the fetched snapshot."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_ci_status",
        MagicMock(return_value={"status": "ok", "head_sha": OLD_HEAD}),
    )
    pull = {"title": "t", "body": "b", "head": {"sha": HEAD}, "base": {"sha": MAIN, "ref": "main"}}
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._rest_api_get_detailed",
        MagicMock(side_effect=[(pull, ""), ({"commit": {"sha": MAIN}}, "")]),
    )
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_rate_limit_info",
        MagicMock(return_value={"source": "unavailable"}),
    )

    result = _fetch_stability_snapshot(
        "42",
        repo="ll7/robot_sf_ll7",
        expected_head_sha="",
        expected_main_sha="",
        expected_metadata_digest="",
    )

    assert result["status"] == "changed"
    assert "head_sha_read_race" in result["invalidated_reasons"]


def test_fetch_snapshot_metadata_drift_resume_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Desired title/body inputs produce a concrete reconcile resume command."""
    body_file = tmp_path / "body.md"
    body_file.write_text("new body", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_ci_status",
        MagicMock(return_value={"status": "ok", "head_sha": HEAD}),
    )
    pull = {
        "title": "fix: the thing",
        "body": "old body",
        "head": {"sha": HEAD},
        "base": {"sha": MAIN, "ref": "main"},
    }
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._rest_api_get_detailed",
        MagicMock(side_effect=[(pull, ""), ({"commit": {"sha": MAIN}}, "")]),
    )
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_rate_limit_info",
        MagicMock(return_value={"source": "unavailable"}),
    )

    result = _fetch_stability_snapshot(
        "42",
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
        expected_metadata_digest=DIGEST,
        metadata_title="fix: final title",
        metadata_body_file=body_file,
    )

    assert result["status"] == "changed"
    assert result["metadata"]["desired_digest"] == metadata_digest("fix: final title", "new body")
    resume = result["resume"]
    assert resume["reason"] == "reconcile_metadata_then_rerun"
    assert "uv run python scripts/dev/gh_pr_body_rest.py 42 --reconcile" in resume["command"]
    assert str(body_file) in resume["command"]


def test_fetch_snapshot_repo_derivation_failure_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snapshot without a resolvable repository fails closed before any read."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._git_remote_owner_name",
        MagicMock(return_value=("", "")),
    )
    result = _fetch_stability_snapshot(
        "42",
        repo="",
        expected_head_sha="",
        expected_main_sha="",
        expected_metadata_digest="",
    )
    assert result["status"] == "error"
    assert "--repo" in result["error"]


def test_fetch_snapshot_stable_with_exhausted_quota_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh evidence with zero REST quota stays stable but warns that resume fails."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_ci_status",
        MagicMock(
            return_value={"status": "ok", "head_sha": HEAD, "checks": {"overall": "success"}}
        ),
    )
    pull = {"title": "t", "body": "b", "head": {"sha": HEAD}, "base": {"sha": MAIN, "ref": "main"}}
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._rest_api_get_detailed",
        MagicMock(side_effect=[(pull, ""), ({"commit": {"sha": MAIN}}, "")]),
    )
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_rate_limit_info",
        MagicMock(return_value={"source": "gh_api_rate_limit", "core_remaining": 0}),
    )

    result = _fetch_stability_snapshot(
        "42",
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
        expected_metadata_digest="",
    )

    assert result["status"] == "stable"
    assert "quota" in result["warning"]


@pytest.mark.parametrize(
    ("snapshot_status", "expected_exit"),
    [
        ("stable", 0),
        ("changed", 1),
        ("failure", 1),
        ("error", 1),
        ("pending", 2),
        ("status_propagation_lag", 2),
        ("quota_blocked", 2),
    ],
)
def test_main_snapshot_exit_codes(
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_status: str,
    expected_exit: int,
) -> None:
    """Snapshot exit codes: 0 stable, 1 conclusive negative, 2 resume-later."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_stability_snapshot",
        MagicMock(
            return_value={
                "schema": STABILITY_SNAPSHOT_SCHEMA,
                "status": snapshot_status,
                "route_evidence_only": True,
                "pr": 42,
            }
        ),
    )
    rc = main(["42", "--stability-snapshot"])
    assert rc == expected_exit
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == snapshot_status


def test_main_snapshot_emits_json_without_json_flag(
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot always emits one JSON document even without --json."""
    monkeypatch.setattr(
        "scripts.dev.check_pr_ci_status._fetch_stability_snapshot",
        MagicMock(return_value={"schema": STABILITY_SNAPSHOT_SCHEMA, "status": "stable", "pr": 42}),
    )
    rc = main(["42", "--stability-snapshot"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "stable"


def test_main_snapshot_rejects_polling_flags(capsys: pytest.CaptureFixture) -> None:
    """The snapshot never polls; conflicting polling flags fail before any read."""
    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        with pytest.raises(SystemExit) as excinfo:
            main(["42", "--stability-snapshot", "--poll-attempts", "5"])
    assert excinfo.value.code == 2
    mock_run.assert_not_called()
    assert "never polls" in capsys.readouterr().err


def test_main_snapshot_rejects_partial_metadata_pair(capsys: pytest.CaptureFixture) -> None:
    """--metadata-title without --metadata-body-file is a preflight error."""
    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        with pytest.raises(SystemExit) as excinfo:
            main(["42", "--stability-snapshot", "--metadata-title", "fix: t"])
    assert excinfo.value.code == 2
    mock_run.assert_not_called()
    assert "provided together" in capsys.readouterr().err


def test_main_snapshot_rejects_short_main_sha_before_gh(
    capsys: pytest.CaptureFixture,
) -> None:
    """A short --expected-main-sha fails fast without invoking gh."""
    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        rc = main(["42", "--stability-snapshot", "--expected-main-sha", MAIN[:8]])
    assert rc == 1
    mock_run.assert_not_called()
    assert "40-hex" in capsys.readouterr().err


def test_main_rejects_snapshot_only_flags_in_plain_mode(
    capsys: pytest.CaptureFixture,
) -> None:
    """Snapshot-only flags outside snapshot mode are preflight errors."""
    with patch("scripts.dev.check_pr_ci_status.subprocess.run") as mock_run:
        with pytest.raises(SystemExit) as excinfo:
            main(["42", "--expected-main-sha", MAIN])
    assert excinfo.value.code == 2
    mock_run.assert_not_called()
    assert "require --stability-snapshot" in capsys.readouterr().err
