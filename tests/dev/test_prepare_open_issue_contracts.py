"""Offline tests for scripts/dev/prepare_open_issue_contracts.py.

Covers the issue #7929 contract: plan-mode zero writes, deterministic
fixtures, idempotence, at-most-one marker, byte preservation outside the
marker region, and fail-closed behavior on drift and malformed inputs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.dev import prepare_open_issue_contracts as prep

if TYPE_CHECKING:
    from pathlib import Path

# --- Fixtures ----------------------------------------------------------------


def _audit_fixture(*, mutation_authorized: bool = False) -> dict:
    """A complete report-only audit fixture with representative classifications."""
    items = []
    for number, classification in (
        (1001, "ready"),
        (1002, "needs_spec"),
        (1003, "parent"),
        (1004, "blocked"),
        (1005, "error"),
        (1006, "assigned"),
    ):
        items.append(
            {
                "number": number,
                "title": f"issue {number}",
                "url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
                "labels": [],
                "assignees": [],
                "claim": None,
                "observed_classification": classification,
                "classification": classification,
                "reasons": [],
                "body_sha256": f"deadbeef{number}",
                "contract_fields": {},
                "missing_fields": [],
                "dependency_gate": None,
                "listing_drift": [],
                "applicable": True,
                "dispatch_eligible": classification == "ready",
                "next_action": f"next-{classification}",
                "authority": f"authority-{classification}",
            }
        )
    return {
        "schema": "open_issue_contract_audit.v1",
        "repository": "ll7/robot_sf_ll7",
        "base_sha": "a" * 40,
        "complete": True,
        "mutation_authorized": mutation_authorized,
        "pagination": {
            "excluded_pull_requests": 0,
            "issue_rows": len(items),
            "pages_read": 1,
            "page_size": 100,
            "raw_rows": len(items),
        },
        "items": items,
    }


def _audit_path(tmp_path: Path) -> Path:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_audit_fixture()), encoding="utf-8")
    return path


# --- Plan mode ---------------------------------------------------------------


def test_plan_mode_is_report_only_and_complete(tmp_path: Path) -> None:
    """Plan mode covers every item and never enables mutation."""
    audit = _load_audit(tmp_path)
    plan = prep.build_plan(audit, batch_id="b1")
    assert plan["schema"] == prep.PLAN_SCHEMA
    assert plan["mutation_authorized"] is False
    assert plan["item_count"] == 6
    assert plan["summary"]["dispatch_eligible"] == 1
    assert plan["content_sha256"]


def test_plan_classification_roundtrip(tmp_path: Path) -> None:
    """Every classification routes to its execution mode and worker."""
    plan = prep.build_plan(_load_audit(tmp_path), batch_id="b1")
    by_issue = {e["issue"]: e for e in plan["entries"]}
    assert by_issue[1001]["execution_mode"] == "implementation"
    assert by_issue[1001]["worker_route"] == "LunaRunner"
    assert by_issue[1002]["execution_mode"] == "formalization"
    assert by_issue[1003]["execution_mode"] == "decomposition"
    assert by_issue[1003]["worker_route"] == "MaxRunner"
    assert by_issue[1004]["execution_mode"] == "blocker"
    assert by_issue[1005]["execution_mode"] == "error-repair"


def test_plan_skips_active_and_error_items(tmp_path: Path) -> None:
    """Active-owner and error rows must not be proposed for mutation."""
    plan = prep.build_plan(_load_audit(tmp_path), batch_id="b1")
    by_issue = {e["issue"]: e for e in plan["entries"]}
    assert by_issue[1006]["skip_reason"] == "active_owner"
    assert by_issue[1005]["skip_reason"] == "error_row"


def test_complete_unready_leaf_plans_canonical_readiness_gate() -> None:
    """An ordinary complete leaf gets a gate operation, never a generic add."""
    _audit, plan = _label_plan_fixture()
    entry = plan["entries"][0]

    assert entry["preparation_action"] == "gate_readiness"
    assert entry["state_ready_change_proposed"] is True
    assert entry["label_plan"] == [
        {
            "issue": "1001",
            "action": "gate_readiness",
            "expected_body_sha256": prep._sha256_text("body-1001"),
            "expected_labels": ["priority:1"],
        }
    ]
    assert not any(operation.get("action") == "add" for operation in entry["label_plan"])


def test_state_conflict_with_missing_fields_routes_to_formalization() -> None:
    """Canonical state precedence must not hide an under-specified contract."""
    audit = _audit_fixture()
    item = audit["items"][0]
    item["observed_classification"] = "state_conflict"
    item["classification"] = "state_conflict"
    item["admission_reason"] = "state_label_conflict"
    item["missing_fields"] = ["scope"]
    plan = prep.build_plan(audit, batch_id="b1")
    entry = plan["entries"][0]

    assert entry["preparation_action"] == "formalize_issue"
    assert entry["execution_mode"] == "formalization"
    assert entry["worker_route"] == "LunaRunner"
    assert entry["readiness_gate"] is None


def test_state_conflict_preserves_parent_authority() -> None:
    """A missing execution-state label must not turn a parent into a leaf."""
    audit = _audit_fixture()
    item = audit["items"][0]
    item["title"] = "[parent] bounded work"
    item["labels"] = ["parent"]
    item["observed_classification"] = "state_conflict"
    item["classification"] = "state_conflict"
    item["admission_reason"] = "state_label_conflict"
    plan = prep.build_plan(audit, batch_id="b1")
    entry = plan["entries"][0]

    assert entry["preparation_action"] == "decompose_issue"
    assert entry["execution_mode"] == "decomposition"
    assert entry["skip_reason"] == "authority_held"
    assert entry["readiness_gate"] is None


def test_plan_cli_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The plan CLI performs zero writes and emits a plan JSON on stdout."""
    audit_path = _audit_path(tmp_path)
    out_path = tmp_path / "plan.json"
    rc = prep.main(["--audit-json", str(audit_path), "--plan-json", str(out_path)])
    assert rc == 0
    plan = json.loads(out_path.read_text(encoding="utf-8"))
    assert plan["schema"] == prep.PLAN_SCHEMA
    assert plan["item_count"] == 6


def test_plan_rejects_mutation_authorized_audit(tmp_path: Path) -> None:
    """A non-report-only audit must fail closed."""
    path = tmp_path / "bad-audit.json"
    path.write_text(
        json.dumps({**_audit_fixture(mutation_authorized=True), "complete": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        prep._load_audit(str(path))


def test_plan_rejects_unknown_schema(tmp_path: Path) -> None:
    """An audit with the wrong schema must fail closed."""
    path = tmp_path / "wrong-schema.json"
    path.write_text(json.dumps({"schema": "other.v1", "items": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        prep._load_audit(str(path))


# --- Rendering ---------------------------------------------------------------


def test_render_single_marker_and_envelope() -> None:
    """Rendered packet has exactly one marker and a valid envelope."""
    item = {
        "number": 1001,
        "classification": "ready",
        "next_action": "claim",
        "authority": "goal_issue_admission",
        "dispatch_eligible": True,
        "labels": [],
        "body_sha256": "abc",
    }
    block = prep._render_marker_block(item, audit_digest="d1", batch_id="b1")
    assert block.count(prep.MARKER_START) == 1
    assert block.count(prep.MARKER_END) == 1
    assert "schema: goal_autopilot_preparation.v1" in block
    assert "execution_mode: implementation" in block
    assert "preferred_worker: LunaRunner" in block
    assert "implementation_admitted: True" in block


def test_render_max_runner_route() -> None:
    """Parent/decision classifications route to MaxRunner."""
    item = {
        "number": 1003,
        "classification": "parent",
        "next_action": "split",
        "authority": "parent_owner",
        "dispatch_eligible": False,
        "labels": [],
        "body_sha256": "abc",
    }
    block = prep._render_marker_block(item, audit_digest="d1", batch_id="b1")
    assert "execution_mode: decomposition" in block
    assert "preferred_worker: MaxRunner" in block
    assert "implementation_admitted: False" in block


def test_render_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Render mode prints exactly one packet for a selected issue."""
    audit_path = _audit_path(tmp_path)
    rc = prep.main(
        [
            "--audit-json",
            str(audit_path),
            "--mode",
            "render",
            "--issue",
            "1001",
            "--batch-id",
            "b1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count(prep.MARKER_START) == 1


# --- Verify ------------------------------------------------------------------


def _body_with_marker(body: str, number: int, *, separator: str = "\n\n") -> str:
    item = {
        "number": number,
        "classification": "ready",
        "next_action": "claim",
        "authority": "goal_issue_admission",
        "dispatch_eligible": True,
        "labels": [],
        "body_sha256": prep._sha256_text(body),
    }
    block = prep._render_marker_block(item, audit_digest="d1", batch_id="b1")
    return body + separator + block


def test_verify_ok_on_single_marker() -> None:
    """One marker with byte-preserved outside content verifies clean."""
    original = "# Title\n\nbody text\n"
    body = _body_with_marker(original, 1001)
    fixture = _audit_fixture()
    fixture["items"][0]["body_sha256"] = prep._sha256_text(original)
    plan = prep.build_plan(fixture, batch_id="b1")
    findings = prep._verify_batch(plan, {"1001": body})
    assert all(f["ok"] for f in findings)


def test_verify_fails_on_duplicate_marker() -> None:
    """Two markers in one body must fail closed."""
    original = "# Title\n\nbody text\n"
    body = _body_with_marker(original, 1001) + _body_with_marker(original, 1001)
    plan = prep.build_plan(_audit_fixture(), batch_id="b1")
    findings = prep._verify_batch(plan, {"1001": body})
    assert not findings[0]["ok"]
    assert findings[0]["reason"] == "duplicate marker"


def test_verify_fails_on_content_drift() -> None:
    """Changed content outside the marker must fail closed."""
    original = "# Title\n\nbody text\n"
    body = _body_with_marker(original, 1001).replace("body text", "changed text")
    fixture = _audit_fixture()
    fixture["items"][0]["body_sha256"] = prep._sha256_text(original)
    plan = prep.build_plan(fixture, batch_id="b1")
    findings = prep._verify_batch(plan, {"1001": body})
    assert not findings[0]["ok"]
    assert "content drift" in findings[0]["reason"]


def test_verify_accepts_body_without_trailing_newline() -> None:
    """Exact body preservation also works when the source has no final newline."""
    original = "# Title\n\nbody text"
    body = _body_with_marker(original, 1001, separator="")
    fixture = _audit_fixture()
    fixture["items"][0]["body_sha256"] = prep._sha256_text(original)
    plan = prep.build_plan(fixture, batch_id="b1")
    findings = prep._verify_batch(plan, {"1001": body})
    assert all(f["ok"] for f in findings)


# --- Apply -------------------------------------------------------------------


def _recording_writer(log: list[tuple[int, str]]):
    def writer(issue: int, block: str) -> None:
        log.append((issue, block))

    return writer


def _label_plan_fixture(*, numbers: tuple[int, ...] = (1001,)) -> tuple[dict, dict]:
    """Build selected complete unlabeled leaves for readiness-gate tests."""
    audit = _audit_fixture()
    for item in audit["items"]:
        if item["number"] not in numbers:
            continue
        item["observed_classification"] = "state_conflict"
        item["classification"] = "state_conflict"
        item["dispatch_eligible"] = False
        item["labels"] = ["priority:1"]
        item["body_sha256"] = prep._sha256_text(f"body-{item['number']}")
        item["claim"] = {
            "ok": True,
            "claimed": False,
            "claim_ref": None,
            "sha": None,
        }
        item["missing_fields"] = []
        item["admission_reason"] = "state_label_conflict"
        item["execution_contract"] = {
            "valid": True,
            "owning_repo": "ll7/robot_sf_ll7",
            "mutation_repos": ["ll7/robot_sf_ll7"],
            "route_required": "local",
            "external_inputs": [],
        }
    full_plan = prep.build_plan(audit, batch_id="b1")
    return audit, prep._restrict_plan(full_plan, numbers)


def test_apply_dry_run_writes_nothing() -> None:
    """Dry-run apply produces would-write entries and calls no writer."""
    plan = prep.build_plan(_audit_fixture(), batch_id="b1")
    log: list[tuple[int, str]] = []
    receipt = prep._apply_bodies(
        _audit_fixture(),
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=True,
        body_writer=_recording_writer(log),
    )
    assert receipt["dry_run"] is True
    assert receipt["would_write"] > 0
    assert log == []


def test_apply_respects_mutation_ceiling() -> None:
    """Apply stops at the mutation ceiling and never exceeds it."""
    plan = prep.build_plan(_audit_fixture(), batch_id="b1")
    receipt = prep._apply_bodies(
        _audit_fixture(),
        plan,
        mutation_ceiling=1,
        batch_id="b1",
        dry_run=True,
        body_writer=_recording_writer([]),
    )
    assert receipt["would_write"] == 1


def test_apply_skips_active_and_error_items() -> None:
    """Assigned and error items are never written."""
    plan = prep.build_plan(_audit_fixture(), batch_id="b1")
    log: list[tuple[int, str]] = []
    receipt = prep._apply_bodies(
        _audit_fixture(),
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=True,
        body_writer=_recording_writer(log),
    )
    written_issues = [
        op["issue"] for op in receipt["operations"] if op["operation"] == "would_write"
    ]
    assert 1006 not in written_issues
    assert 1005 not in written_issues


def test_apply_writer_receives_block() -> None:
    """The real writer path receives the rendered marker block."""
    plan = prep.build_plan(_audit_fixture(), batch_id="b1")
    log: list[tuple[int, str]] = []
    prep._apply_bodies(
        _audit_fixture(),
        plan,
        mutation_ceiling=2,
        batch_id="b1",
        dry_run=False,
        body_writer=_recording_writer(log),
    )
    assert len(log) == 2
    for issue, block in log:
        assert block.count(prep.MARKER_START) == 1
        assert prep.MARKER_END in block


@pytest.mark.parametrize("current_body", ["# Issue\n\nbody\n", "# Issue\n\nbody"])
def test_live_body_writer_uses_rest_path_and_verifies_readback(
    monkeypatch: pytest.MonkeyPatch,
    current_body: str,
) -> None:
    """The live writer uses the shared REST signature and checks the returned body."""
    from scripts.dev import _gh_rest

    calls: list[dict[str, object]] = []

    def fake_run_gh_api(
        path: str,
        payload: object | None = None,
        *,
        method: str | None = None,
        extra_args: list[str] | None = None,
        **_: object,
    ) -> object:
        nonlocal current_body
        calls.append({"path": path, "payload": payload, "method": method, "extra_args": extra_args})
        if method == "PATCH":
            assert isinstance(payload, dict)
            current_body = str(payload["body"])
            return type("Result", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()
        return type(
            "Result",
            (),
            {"returncode": 0, "stdout": json.dumps({"body": current_body}), "stderr": ""},
        )()

    monkeypatch.setattr(_gh_rest, "run_gh_api", fake_run_gh_api)
    block = f"{prep.MARKER_START}\npacket\n{prep.MARKER_END}"

    prep._live_body_writer(1001, block)

    assert [call["path"] for call in calls] == [
        "repos/ll7/robot_sf_ll7/issues/1001",
        "repos/ll7/robot_sf_ll7/issues/1001",
        "repos/ll7/robot_sf_ll7/issues/1001",
    ]
    assert calls[0]["extra_args"] is None
    assert calls[1]["method"] == "PATCH"
    assert calls[2]["extra_args"] is None
    assert current_body.endswith(block)


def test_apply_readiness_gate_uses_canonical_gate_and_verifies_body_labels() -> None:
    """Readiness uses the canonical gate and never calls the generic label writer."""
    audit, plan = _label_plan_fixture()
    issue = 1001
    snapshots = {
        issue: {
            "number": issue,
            "state": "open",
            "body": f"body-{issue}",
            "labels": ["priority:1"],
            "assignees": [],
        }
    }
    body_calls: list[int] = []
    label_calls: list[tuple[int, str, str]] = []
    gate_calls: list[int] = []

    def read_issue(number: int) -> dict:
        snapshot = snapshots[number]
        return {**snapshot, "labels": list(snapshot["labels"])}

    def write_body(number: int, block: str) -> None:
        body_calls.append(number)
        snapshots[number]["body"] = prep._compose_body(snapshots[number]["body"], block)

    def gate(number: int) -> dict:
        gate_calls.append(number)
        snapshots[number]["labels"].append("state:ready")
        return {"outcome": "ready", "ready_added": True, "verified": True}

    def write_label(number: int, action: str, label: str) -> dict:
        label_calls.append((number, action, label))
        return {"status": "ok"}

    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=write_body,
        issue_reader=read_issue,
        label_writer=write_label,
        readiness_gater=gate,
    )

    assert body_calls == [issue]
    assert gate_calls == [issue]
    assert label_calls == []
    assert receipt["written"] == 2
    assert receipt["drifted"] == 0
    assert snapshots[issue]["labels"] == ["priority:1", "state:ready"]
    assert receipt["safe_order"] == "readiness_gate_then_body_then_labels"
    assert (
        next(op for op in receipt["operations"] if op.get("kind") == "readiness_gate")["outcome"]
        == "ready"
    )


def test_apply_readiness_gate_aborts_on_gate_readback_drift() -> None:
    """A gate that cannot preserve the reviewed body/labels fails closed."""
    audit, plan = _label_plan_fixture()
    issue = 1001
    snapshots = {
        issue: {
            "number": issue,
            "state": "open",
            "body": f"body-{issue}",
            "labels": ["priority:1"],
            "assignees": [],
        }
    }
    body_calls: list[int] = []

    def read_issue(number: int) -> dict:
        snapshot = snapshots[number]
        return {**snapshot, "labels": list(snapshot["labels"])}

    def write_body(number: int, block: str) -> None:
        body_calls.append(number)

    def gate(number: int) -> dict:
        snapshots[number]["body"] = "changed-during-gate"
        snapshots[number]["labels"].append("state:ready")
        return {"outcome": "ready", "ready_added": True, "verified": True}

    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=write_body,
        issue_reader=read_issue,
        readiness_gater=gate,
    )

    assert receipt["aborted"] is True
    assert receipt["abort_reason"] == "readiness_gate_readback_drift"
    assert receipt["drifted"] == 1
    assert body_calls == []


def test_apply_aborts_before_writes_on_label_drift() -> None:
    """Any expected-label drift aborts the whole batch before mutation."""
    audit, plan = _label_plan_fixture()
    body_calls: list[int] = []
    label_calls: list[tuple[int, str, str]] = []

    def read_issue(number: int) -> dict:
        return {
            "number": number,
            "state": "open",
            "body": f"body-{number}",
            "labels": ["unexpected"],
            "assignees": [],
        }

    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=lambda issue, block: body_calls.append(issue),
        issue_reader=read_issue,
        label_catalog={"priority:1", "state:ready", "unexpected"},
        label_writer=lambda issue, action, label: label_calls.append((issue, action, label)),
    )

    assert receipt["aborted"] is True
    assert receipt["abort_reason"] == "preflight_drift_or_unavailable"
    assert receipt["drifted"] == 1
    assert body_calls == []
    assert label_calls == []


def test_apply_label_plan_supports_exact_remove() -> None:
    """Reviewed removal uses the same exact-set and readback protocol."""
    audit, plan = _label_plan_fixture()
    entry = plan["entries"][0]
    entry["labels"] = ["priority:1", "state:ready"]
    entry["label_plan"] = [{"issue": "1001", "action": "remove", "label": "state:ready"}]
    snapshots = {
        1001: {
            "number": 1001,
            "state": "open",
            "body": "body-1001",
            "labels": ["priority:1", "state:ready"],
            "assignees": [],
        }
    }

    def read_issue(number: int) -> dict:
        snapshot = snapshots[number]
        return {**snapshot, "labels": list(snapshot["labels"])}

    def write_label(number: int, action: str, label: str) -> dict:
        snapshots[number]["labels"].remove(label)
        return {"status": "ok"}

    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=lambda issue, block: None,
        issue_reader=read_issue,
        label_catalog={"priority:1", "state:ready"},
        label_writer=write_label,
    )

    assert receipt["failed"] == 0
    assert receipt["drifted"] == 0
    assert snapshots[1001]["labels"] == ["priority:1"]
    assert any(
        op.get("action") == "remove" and op.get("operation") == "written"
        for op in receipt["operations"]
    )


def test_apply_rejects_runner_label_before_any_body_write() -> None:
    """Issue preparation cannot mutate PR runner labels or bypass the catalog."""
    audit, plan = _label_plan_fixture()
    plan["entries"][0]["label_plan"] = [{"issue": "1001", "action": "add", "label": "runner:luna"}]
    body_calls: list[int] = []
    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=lambda issue, block: body_calls.append(issue),
        label_catalog={"runner:luna"},
        label_writer=lambda issue, action, label: {"status": "ok"},
    )

    assert receipt["abort_reason"] == "invalid_label_plan"
    assert receipt["failed"] >= 1
    assert any(op.get("reason") == "runner_label_forbidden" for op in receipt["operations"])
    assert body_calls == []


def test_apply_rejects_missing_label_before_any_body_write() -> None:
    """A label absent from the repository catalog cannot be created implicitly."""
    audit, plan = _label_plan_fixture()
    plan["entries"][0]["label_plan"] = [
        {"issue": "1001", "action": "add", "label": "not-in-catalog"}
    ]
    body_calls: list[int] = []
    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=lambda issue, block: body_calls.append(issue),
        label_catalog={"priority:1", "state:ready"},
        label_writer=lambda issue, action, label: {"status": "ok"},
    )

    assert receipt["abort_reason"] == "invalid_label_plan"
    assert any(
        op.get("reason") == "label_not_in_repository_catalog" for op in receipt["operations"]
    )
    assert body_calls == []


def test_apply_rejects_more_than_hard_label_ceiling() -> None:
    """The label budget is checked before the first body or label mutation."""
    audit, plan = _label_plan_fixture()
    entry = plan["entries"][0]
    entry["body_patch"]["proposed"] = False
    entry["label_plan"] = [
        {"issue": "1001", "action": "add", "label": f"custom:{index}"}
        for index in range(prep.HARD_LABEL_CEILING + 1)
    ]
    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=lambda issue, block: pytest.fail("body writer called"),
        label_catalog={f"custom:{index}" for index in range(prep.HARD_LABEL_CEILING + 1)},
        label_writer=lambda issue, action, label: pytest.fail("label writer called"),
    )

    assert receipt["abort_reason"] == "label_operation_ceiling_exceeded"
    assert receipt["label_operations_planned"] == prep.HARD_LABEL_CEILING + 1
    assert receipt["written"] == 0


def test_apply_records_partial_failure_after_body_write() -> None:
    """A generic cleanup failure remains explicit and resumable after gate work."""
    audit, plan = _label_plan_fixture(numbers=(1001, 1002))
    plan["entries"][1]["body_patch"]["proposed"] = False
    plan["entries"][1]["labels"] = ["priority:1", "stale"]
    plan["entries"][1]["label_plan"] = [{"issue": "1002", "action": "remove", "label": "stale"}]
    snapshots = {
        1001: {
            "number": 1001,
            "state": "open",
            "body": "body-1001",
            "labels": ["priority:1"],
            "assignees": [],
        },
        1002: {
            "number": 1002,
            "state": "open",
            "body": "body-1002",
            "labels": ["priority:1", "stale"],
            "assignees": [],
        },
    }
    body_calls: list[int] = []
    gate_calls: list[int] = []

    def read_issue(number: int) -> dict:
        snapshot = snapshots[number]
        return {**snapshot, "labels": list(snapshot["labels"])}

    def write_body(number: int, block: str) -> None:
        body_calls.append(number)

    def gate(number: int) -> dict:
        gate_calls.append(number)
        snapshots[number]["labels"].append("state:ready")
        return {"outcome": "ready", "ready_added": True, "verified": True}

    def write_label(number: int, action: str, label: str) -> dict:
        raise RuntimeError("simulated label service failure")

    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=write_body,
        issue_reader=read_issue,
        label_catalog={"priority:1", "state:ready", "stale"},
        label_writer=write_label,
        readiness_gater=gate,
    )

    assert body_calls == [1001]
    assert gate_calls == [1001]
    assert receipt["partial_failure"] is True
    assert receipt["written"] == 2
    assert receipt["failed"] == 1
    assert any(
        op.get("issue") == 1002 and op.get("operation") == "failed" for op in receipt["operations"]
    )


def test_apply_dry_run_includes_readiness_gate_without_writes() -> None:
    """Dry-run renders gate and body operations without requiring live issue reads."""
    audit, plan = _label_plan_fixture()
    body_calls: list[int] = []
    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=True,
        body_writer=lambda issue, block: body_calls.append(issue),
    )

    assert receipt["would_write"] == 2
    assert receipt["label_operations_attempted"] == 1
    assert body_calls == []
    assert any(
        op.get("kind") == "readiness_gate" and op.get("operation") == "would_write"
        for op in receipt["operations"]
    )


def test_readiness_gate_failure_writes_no_body_or_generic_label() -> None:
    """A failed canonical gate leaves the issue untouched by preparation."""
    audit, plan = _label_plan_fixture()
    body_calls: list[int] = []
    label_calls: list[tuple[int, str, str]] = []
    receipt = prep._apply_bodies(
        audit,
        plan,
        mutation_ceiling=10,
        batch_id="b1",
        dry_run=False,
        body_writer=lambda issue, block: body_calls.append(issue),
        readiness_gater=lambda issue: {"outcome": "needs_spec", "verified": False},
        label_writer=lambda issue, action, label: label_calls.append((issue, action, label)),
    )

    assert receipt["abort_reason"] == "readiness_gate_failed"
    assert receipt["failed"] == 1
    assert body_calls == []
    assert label_calls == []


def test_apply_ceiling_too_high_fails_closed(tmp_path: Path) -> None:
    """A ceiling above the hard max must fail closed."""
    audit_path = _audit_path(tmp_path)
    rc = prep.main(
        [
            "--audit-json",
            str(audit_path),
            "--mode",
            "apply",
            "--apply",
            "--mutation-ceiling",
            "99",
        ]
    )
    assert rc == 2


# --- CLI plumbing ------------------------------------------------------------


def test_main_requires_apply_for_apply_mode(tmp_path: Path) -> None:
    """Apply mode without --apply must fail closed."""
    audit_path = _audit_path(tmp_path)
    rc = prep.main(["--audit-json", str(audit_path), "--mode", "apply"])
    assert rc == 2


def test_main_real_apply_requires_reviewed_digest_and_issue_list(tmp_path: Path) -> None:
    """CLI mutation requires both exact membership and explicit review evidence."""
    audit_path = _audit_path(tmp_path)
    rc = prep.main(["--audit-json", str(audit_path), "--mode", "apply", "--apply"])
    assert rc == 2

    rc = prep.main(
        [
            "--audit-json",
            str(audit_path),
            "--mode",
            "apply",
            "--apply",
            "--issues",
            "1001",
        ]
    )
    assert rc == 2


def test_main_selected_plan_recomputes_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Explicit issue selection freezes a new exact plan digest."""
    audit_path = _audit_path(tmp_path)
    rc = prep.main(["--audit-json", str(audit_path), "--issues", "1001"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["item_count"] == 1
    assert plan["entries"][0]["issue"] == 1001
    assert plan["content_sha256"] == prep._content_digest(plan)


def test_main_plan_emits_stdout_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Plan mode prints the plan JSON on stdout."""
    audit_path = _audit_path(tmp_path)
    rc = prep.main(["--audit-json", str(audit_path)])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == prep.PLAN_SCHEMA


def _load_audit(tmp_path: Path) -> dict:
    return prep._load_audit(str(_audit_path(tmp_path)))
