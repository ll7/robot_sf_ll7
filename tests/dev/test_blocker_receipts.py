"""Behavioral tests for the goal blocker receipt contract."""

from __future__ import annotations

import json

import pytest

from scripts.dev.autopilot_state_snapshot import blocker_receipt_snapshot
from scripts.dev.blocker_receipt import (
    blocker_fingerprint,
    build_receipt,
    compare_blocker_inputs,
    load_receipt,
    main,
    receipt_artifact_path,
    summarize_decisions,
    unavailable,
    validate_receipt,
    write_receipt,
)


def test_fingerprint_distinguishes_missing_empty_false_and_unavailable_values() -> None:
    """Blocker inputs retain the distinction between unavailable and real values."""
    empty = blocker_fingerprint({"required_input": ""})
    false = blocker_fingerprint({"required_input": False})
    unavailable_value = blocker_fingerprint({"required_input": unavailable("not observed")})
    missing_value = blocker_fingerprint({"required_input": None})
    missing = blocker_fingerprint({})

    assert len({empty, false, unavailable_value, missing_value, missing}) == 5


def test_build_receipt_produces_a_self_validating_exact_head_contract() -> None:
    """A blocked receipt records the owner, unblock condition, and verified digest."""
    receipt = build_receipt(
        repository="ll7/robot_sf_ll7",
        issue=7612,
        issue_revision="body-sha256:abc",
        origin_main_sha="a" * 40,
        pr_head_sha=unavailable("no PR exists yet"),
        blocker_class="dependency",
        required_transition="merge #7619",
        evidence=[{"source": "issue", "identity": "#7609"}],
        safe_work=["contract design"],
        next_owner="goal-autopilot",
        recommended_state="state:blocked",
        retryable=True,
        invalidating_fields=["issue.labels", "origin_main_sha"],
        fingerprint_inputs={"issue.labels": ["state:blocked"], "origin_main_sha": "a" * 40},
        observed_at="2026-08-20T00:00:00Z",
    )

    report = validate_receipt(receipt)

    assert report["valid"] is True
    assert receipt["fingerprint"] == blocker_fingerprint(receipt["fingerprint_inputs"])
    assert receipt["pr_head_sha"]["$state"] == "unavailable"


def test_receipt_storage_round_trips_atomically_to_external_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Validated receipts use a caller-selected private artifact, never the worktree."""
    receipt = build_receipt(
        repository="ll7/robot_sf_ll7",
        issue=7612,
        issue_revision="body-sha256:abc",
        origin_main_sha="a" * 40,
        blocker_class="dependency",
        required_transition="merge the prerequisite",
        evidence=[],
        safe_work=[],
        next_owner="goal-autopilot",
        recommended_state="state:blocked",
        retryable=True,
        invalidating_fields=["origin_main_sha"],
        fingerprint_inputs={"origin_main_sha": "a" * 40},
        observed_at="2026-08-20T00:00:00Z",
    )
    path = tmp_path / "active" / "issue-7612.json"

    assert write_receipt(receipt, path) == path
    assert load_receipt(path) == receipt
    assert list(path.parent.glob("*.tmp")) == []


def test_default_receipt_path_uses_common_active_artifact_owner(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The default path is resolved through the shared common-Git artifact helper."""
    monkeypatch.setattr(
        "scripts.dev.blocker_receipt.resolve_agent_artifact_dir",
        lambda subdir: tmp_path / subdir,
    )

    assert receipt_artifact_path(7612) == tmp_path / "goal-blocker-receipts" / "issue-7612.json"


def test_receipt_storage_rejects_invalid_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Invalid receipts cannot be persisted as suppression artifacts."""
    with pytest.raises(ValueError, match="invalid blocker receipt"):
        write_receipt({"schema": "goal_blocker_receipt.v1"}, tmp_path / "invalid.json")


def test_compare_blocker_inputs_suppresses_only_an_unchanged_blocker() -> None:
    """A changed invalidating input re-enters evaluation instead of being suppressed."""
    inputs = {"issue.labels": ["state:blocked"], "origin_main_sha": "a" * 40}
    receipt = build_receipt(
        repository="ll7/robot_sf_ll7",
        issue=7612,
        issue_revision="body-sha256:abc",
        origin_main_sha="a" * 40,
        blocker_class="dependency",
        required_transition="merge #7619",
        evidence=[],
        safe_work=[],
        next_owner="goal-autopilot",
        recommended_state="state:blocked",
        retryable=True,
        invalidating_fields=["issue.labels", "origin_main_sha"],
        fingerprint_inputs=inputs,
        observed_at="2026-08-20T00:00:00Z",
    )

    unchanged = compare_blocker_inputs(inputs, receipt)
    changed = compare_blocker_inputs(
        {"issue.labels": ["state:ready"], "origin_main_sha": "a" * 40}, receipt
    )

    assert unchanged["status"] == "blocked_unchanged"
    assert unchanged["next_owner"] == "goal-autopilot"
    assert changed["status"] == "blocker_changed"
    assert changed["invalidating_fields_changed"] == ["issue.labels"]


def test_compare_preserves_explicit_missing_value_stability() -> None:
    """Repeated explicit missing inputs do not create a false blocker change."""
    inputs = {"required_input": None}
    receipt = build_receipt(
        repository="ll7/robot_sf_ll7",
        issue=7612,
        issue_revision="body-sha256:abc",
        origin_main_sha="a" * 40,
        blocker_class="external_input",
        required_transition="stage the named input",
        evidence=[],
        safe_work=[],
        next_owner="data-owner",
        recommended_state="state:blocked",
        retryable=True,
        invalidating_fields=["required_input"],
        fingerprint_inputs=inputs,
        observed_at="2026-08-20T00:00:00Z",
    )

    decision = compare_blocker_inputs(inputs, receipt)

    assert decision["status"] == "blocked_unchanged"


def test_invalid_receipt_fails_open_to_re_evaluation() -> None:
    """Malformed or tampered receipts cannot silently suppress a dispatch."""
    decision = compare_blocker_inputs(
        {"issue.labels": ["state:blocked"]},
        {"schema": "goal_blocker_receipt.v1", "fingerprint": "0" * 64},
    )

    assert decision["status"] == "re_evaluate"
    assert decision["reason"] == "invalid_or_stale_receipt"


def test_receipt_rejects_noncanonical_sha_lengths() -> None:
    """Partial SHA values cannot masquerade as exact repository identities."""
    receipt = build_receipt(
        repository="ll7/robot_sf_ll7",
        issue=7612,
        issue_revision="body-sha256:abc",
        origin_main_sha="a" * 40,
        blocker_class="dependency",
        required_transition="merge prerequisite",
        evidence=[],
        safe_work=[],
        next_owner="goal-autopilot",
        recommended_state="state:blocked",
        retryable=True,
        invalidating_fields=["origin_main_sha"],
        fingerprint_inputs={"origin_main_sha": "a" * 40},
        observed_at="2026-08-20T00:00:00Z",
    )
    receipt["origin_main_sha"] = "a" * 41

    report = validate_receipt(receipt)

    assert report["valid"] is False
    assert any("origin_main_sha must be a full SHA" in error for error in report["errors"])


def test_summary_exposes_suppressed_redispatch_counts_and_reasons() -> None:
    """The loop-facing summary makes suppression and re-evaluation visible."""
    summary = summarize_decisions(
        [
            {"status": "blocked_unchanged", "reason": "blocker_fingerprint_unchanged"},
            {
                "status": "blocker_changed",
                "reason": "fingerprint_changed",
                "blocker_class": "compute",
            },
            {"status": "re_evaluate", "reason": "invalid_or_stale_receipt"},
        ]
    )

    assert summary["schema"] == "goal_blocker_summary.v1"
    assert summary["suppressed_redispatch_count"] == 1
    assert summary["re_evaluation_count"] == 2
    assert summary["by_reason"]["invalid_or_stale_receipt"] == 1


def test_autopilot_snapshot_reads_external_decision_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The loop snapshot exposes receipt decisions without mutating the artifact."""
    artifact = tmp_path / "blocker-decisions.json"
    artifact.write_text(
        json.dumps(
            [
                {"status": "blocked_unchanged", "reason": "blocker_fingerprint_unchanged"},
                {"status": "re_evaluate", "reason": "invalid_or_stale_receipt"},
            ]
        ),
        encoding="utf-8",
    )

    snapshot = blocker_receipt_snapshot([artifact])

    assert snapshot["status"] == "ok"
    assert snapshot["summary"]["suppressed_redispatch_count"] == 1
    assert snapshot["summary"]["re_evaluation_count"] == 1


def test_cli_fingerprint_is_side_effect_free(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The offline fingerprint command emits deterministic JSON without writing files."""
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"origin_main_sha": "a" * 40}), encoding="utf-8")

    assert main(["fingerprint", str(inputs)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["fingerprint"] == blocker_fingerprint({"origin_main_sha": "a" * 40})


def test_cli_compare_requires_re_evaluation_for_changed_blocker(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    """A changed blocker must return a nonzero CLI status for dispatch guards."""
    inputs = tmp_path / "inputs.json"
    receipt_path = tmp_path / "receipt.json"
    inputs.write_text(json.dumps({"dependency": "merged"}), encoding="utf-8")
    receipt_path.write_text(
        json.dumps(
            build_receipt(
                repository="ll7/robot_sf_ll7",
                issue=7612,
                issue_revision="body-sha256:abc",
                origin_main_sha="a" * 40,
                blocker_class="dependency",
                required_transition="merge the prerequisite",
                evidence=[],
                safe_work=[],
                next_owner="goal-autopilot",
                recommended_state="state:blocked",
                retryable=True,
                invalidating_fields=["dependency"],
                fingerprint_inputs={"dependency": "open"},
                observed_at="2026-08-20T00:00:00Z",
            )
        ),
        encoding="utf-8",
    )

    assert main(["compare", "--inputs", str(inputs), "--receipt", str(receipt_path)]) == 2
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "blocker_changed"
    assert output["re_evaluate"] is True
    assert output["issue"] == 7612
    assert output["repository"] == "ll7/robot_sf_ll7"
    assert output["receipt_digest"] == json.loads(receipt_path.read_text())["receipt_digest"]
