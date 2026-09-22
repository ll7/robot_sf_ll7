"""Offline regression tests for the report-only recovery-cycle driver."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.dev import autopilot_recovery_cycle as cycle

ORIGIN = "a" * 40
DIGESTS = {
    "issue_state_digest": "b" * 64,
    "claim_state_digest": "c" * 64,
    "pr_head_digest": "d" * 64,
    "preparation_audit_digest": "e" * 64,
    "discovery_relevant_paths_digest": "f" * 64,
}


def _queue(*, claimable=0, complete=True, authoritative=True, histogram=None):
    """Build a canned ready-candidate queue payload."""
    return {
        "queue_completeness": "complete" if complete else "incomplete",
        "zero_work_authoritative": authoritative,
        "claimable_count": claimable,
        "claimable_issues": [],
        "admission_reason_histogram": histogram or {},
        "issues": [],
    }


def _prep(*, promotable=0, formalizable=0):
    """Build a canned preparation-plan payload."""
    return {
        "schema": "open_issue_preparation_plan.v1",
        "summary": {"promotable_count": promotable, "formalizable_count": formalizable},
    }


def _prs():
    """Build an empty active-PR queue payload."""
    return {"prs": []}


def _lane_payload(text, *, queue, prep, audit, prs):
    """Serve the canned payload for one canonical-owner command."""
    if "snapshot_issue_batch" in text:
        return cycle.LaneResult(ok=True, payload=queue)
    if "prepare_open_issue_contracts" in text:
        return cycle.LaneResult(ok=True, payload=prep)
    if "snapshot_pr_queue" in text:
        return cycle.LaneResult(ok=True, payload=_prs() if prs is None else prs)
    raise AssertionError(f"unexpected lane command: {text}")


def _runner_for(queue, prep, *, audit_ok=True, hygiene_ok=True, prs=None, fail_lanes=()):
    """Return a fake canonical-owner runner serving canned lane payloads."""
    audit = {"complete": True} if audit_ok else None

    def run(command):
        text = " ".join(command)
        if "open_state_label_hygiene" in text:
            if not hygiene_ok:
                return cycle.LaneResult(ok=False, error="boom")
            return cycle.LaneResult(ok=True, payload={"ok": True, "issues": []})
        if "audit_open_issue_contracts" in text:
            if not audit_ok:
                return cycle.LaneResult(ok=False, error="boom")
            return cycle.LaneResult(ok=True, payload=audit)
        return _lane_payload(text, queue=queue, prep=prep, audit=audit, prs=prs)

    def guarded(command):
        for lane in fail_lanes:
            if lane in " ".join(command):
                return cycle.LaneResult(ok=False, error="injected lane failure")
        return run(command)

    return guarded


def test_false_empty_with_formalizable_routes_formalize() -> None:
    """0 claimable but formalizable work continues to formalization."""
    runner = _runner_for(_queue(), _prep(formalizable=1))
    receipt = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert receipt["schema"] == cycle.RECEIPT_SCHEMA
    assert receipt["next_action"] == "formalize_issue"
    assert receipt["counts"]["formalizable_count"] == 1
    assert receipt["terminal_refused"] is True


def test_false_empty_with_promotable_routes_gate_readiness() -> None:
    """0 claimable but promotable work continues to readiness gating."""
    runner = _runner_for(_queue(), _prep(promotable=2))
    receipt = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert receipt["next_action"] == "gate_readiness"
    assert receipt["counts"]["promotable_count"] == 2


def test_stale_queue_never_terminal() -> None:
    """Incomplete queue evidence refreshes instead of stopping."""
    runner = _runner_for(_queue(complete=False, authoritative=False), _prep())
    receipt = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert receipt["next_action"] == "refresh_issue_queue"
    assert receipt["terminal_refused"] is True


def test_stale_lifecycle_rows_route_lifecycle_reconciliation() -> None:
    """A running-without-claim row blocks exhaustion with a named lane."""
    queue = _queue()
    queue["issues"] = [
        {"number": 1, "labels": ["state:running"], "claim": {"claimed": False}, "linked_prs": []}
    ]
    runner = _runner_for(queue, _prep())
    receipt = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert receipt["next_action"] == "reconcile_lifecycle"
    assert receipt["counts"]["stale_state_count"] == 1
    assert receipt["stale_lifecycle_rows"] == [
        {"issue": 1, "reason": "stale_running_without_claim_or_pr"}
    ]


def test_lane_failure_refuses_terminal_receipt() -> None:
    """An unavailable recovery lane fails closed, never zero-work."""
    runner = _runner_for(_queue(), _prep(), fail_lanes=("audit_open_issue_contracts",))
    receipt = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert receipt["terminal_refused"] is True
    assert receipt["next_action"] in ("reconcile_blockers", "run_preparation")
    assert any(entry["lane"] == "contract_audit" for entry in receipt["skipped"])


def test_true_zero_work_requires_every_lane_fresh() -> None:
    """A fully fresh, empty evidence set continues to discovery, never terminal.

    The report-only cycle records the discovery decision instead of executing
    scouts, so even the clean case refuses terminal and routes onward.
    """
    runner = _runner_for(_queue(), _prep())
    receipt = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert receipt["terminal_refused"] is True
    assert receipt["next_action"] == "discover"
    assert receipt["discovery_decision"]["next_lane"] == cycle.DISCOVERY_LANES[0]
    assert receipt["lanes_evaluated"] == [
        "admission_rerun",
        "contract_audit",
        "issue_queue",
        "lifecycle",
        "pr_queue",
        "preparation",
    ]


def test_receipt_is_json_stable() -> None:
    """The receipt serializes deterministically for ledger reuse."""
    runner = _runner_for(_queue(), _prep())
    first = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)
    second = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_default_staging_uses_helper_owned_temporary_directory(tmp_path: Path, monkeypatch) -> None:
    """Default runs expose the audit only to preparation and never dirty the caller's CWD."""
    monkeypatch.chdir(tmp_path)
    base_runner = _runner_for(_queue(), _prep())
    staged_paths: list[Path] = []

    def runner(command):
        if any("prepare_open_issue_contracts" in part for part in command):
            audit_path = Path(command[command.index("--audit-json") + 1])
            assert json.loads(audit_path.read_text(encoding="utf-8")) == {"complete": True}
            staged_paths.append(audit_path)
        return base_runner(command)

    cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert not (tmp_path / "recovery_cycle_audit.json").exists()
    assert len(staged_paths) == 1
    assert not staged_paths[0].exists()


def test_explicit_work_dir_preserves_staged_audit(tmp_path: Path) -> None:
    """An explicit work directory retains the caller-owned audit artifact for inspection."""
    runner = _runner_for(_queue(), _prep())

    cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner, work_dir=tmp_path)

    audit_path = tmp_path / "recovery_cycle_audit.json"
    assert json.loads(audit_path.read_text(encoding="utf-8")) == {"complete": True}


def test_temporary_staging_failure_does_not_fall_back_to_cwd(tmp_path: Path, monkeypatch) -> None:
    """Unavailable helper-owned scratch fails closed instead of writing in the caller's CWD."""
    monkeypatch.chdir(tmp_path)
    runner = _runner_for(_queue(), _prep())

    def fail_temporary_directory(*args, **kwargs):
        raise OSError("scratch unavailable")

    monkeypatch.setattr(cycle.tempfile, "TemporaryDirectory", fail_temporary_directory)
    receipt = cycle.run_cycle(repo="o/r", origin_main_sha=ORIGIN, runner=runner)

    assert receipt["next_action"] == "run_preparation"
    assert receipt["terminal_refused"] is True
    assert receipt["lane_errors"] == ["preparation_audit_staging_failed: scratch unavailable"]
    assert not (tmp_path / "recovery_cycle_audit.json").exists()
