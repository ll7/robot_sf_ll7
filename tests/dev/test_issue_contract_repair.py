"""Tests for the semantics-preserving contract repair lane (issue #9536).

Packet planning is fully offline. Live paths are exercised through injected
readers, writers, and admission results: no GitHub access, no label writes,
no claims.
"""

from __future__ import annotations

import json
from typing import Any

from scripts.dev import issue_contract_repair as repair

REPAIRABLE_BODY = """## Goal / Problem

Fix the stale running label.

## Scope

- In scope: the hint text.

## Inputs and files

- `scripts/dev/check_prepublication_state.py`.

## Acceptance

- A documented roundtrip exists in `--help`.

## Verification

- Focused helper tests pass.
"""


def test_plan_body_file_emits_packet(tmp_path: Any, capsys: Any) -> None:
    """Offline planning prints a versioned packet without any I/O beyond the file."""
    body_file = tmp_path / "body.md"
    body_file.write_text(REPAIRABLE_BODY, encoding="utf-8")

    assert repair.main(["plan", "--body-file", str(body_file)]) == 0

    packet = json.loads(capsys.readouterr().out)
    assert packet["repairable"] is True
    assert packet["missing_fields"] == ["inputs"]
    assert packet["renames"] == [
        {"field": "inputs", "old_heading": "inputs and files", "new_heading": "inputs"}
    ]


def test_plan_body_file_reports_refusal_for_complete_body(tmp_path: Any, capsys: Any) -> None:
    """Complete bodies report no repair instead of failing the plan."""
    body_file = tmp_path / "body.md"
    body_file.write_text(REPAIRABLE_BODY.replace("## Inputs and files", "## Inputs"))

    assert repair.main(["plan", "--body-file", str(body_file)]) == 0

    packet = json.loads(capsys.readouterr().out)
    assert packet["repairable"] is False


def test_plan_requires_a_source(capsys: Any) -> None:
    """Planning without a body source fails closed with usage guidance."""
    assert repair.main(["plan"]) == 2
    assert "needs --body-file or --issue" in capsys.readouterr().err


def test_apply_repairs_and_verifies_with_injected_owners(monkeypatch: Any, capsys: Any) -> None:
    """Apply writes once, verifies the readback, and reruns admission."""
    from scripts.dev import issue_implementability as implementability

    packet = implementability.build_repair_packet(REPAIRABLE_BODY)
    repaired = implementability.apply_repair_packet(REPAIRABLE_BODY, packet)
    bodies = iter([REPAIRABLE_BODY, repaired])
    reads = {"count": 0}

    def _read(number: int, *, repo: str) -> dict[str, Any]:
        reads["count"] += 1
        return {"body": next(bodies), "labels": []}

    writes: list[str] = []

    def _patch(number: int, *, repo: str, expected_sha256: str, body: str) -> None:
        writes.append(body)

    monkeypatch.setattr(repair, "_exact_read_body", _read)
    monkeypatch.setattr(repair, "_patch_body", _patch)
    monkeypatch.setattr(
        repair.goal_issue_admission,
        "admit_issue",
        lambda *_, **__: {"ok": False, "outcome": "not_admitted", "preflight": {}},
    )

    assert repair.main(["apply", "--issue", "7601"]) == 0

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["applied"] is True
    assert len(writes) == 1
    assert "## inputs\n" in writes[0].lower()
    assert reads["count"] == 2
    assert receipt["admission"]["outcome"] == "not_admitted"
    assert "labels" not in receipt or receipt.get("packet", {}).get("labels") == []
    assert set(receipt) >= {"schema", "issue", "repo", "applied", "packet", "admission"}


def test_apply_refuses_unrepairable_body(monkeypatch: Any, capsys: Any) -> None:
    """Invented content never reaches the write path."""
    monkeypatch.setattr(
        repair,
        "_exact_read_body",
        lambda number, *, repo: {"body": "## Goal\nOnly an objective.\n", "labels": []},
    )
    writes: list[str] = []
    monkeypatch.setattr(
        repair, "_patch_body", lambda number, *, repo, expected_sha256, body: writes.append(body)
    )

    assert repair.main(["apply", "--issue", "7601"]) == 1

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["applied"] is False
    assert writes == []


def test_apply_fails_closed_on_readback_mismatch(monkeypatch: Any) -> None:
    """A readback that differs from the repaired body aborts the receipt."""
    bodies = iter([REPAIRABLE_BODY, REPAIRABLE_BODY + "\nConcurrent edit.\n"])

    def _shifting_read(number: int, *, repo: str) -> dict[str, Any]:
        return {"body": next(bodies), "labels": []}

    monkeypatch.setattr(repair, "_exact_read_body", _shifting_read)
    monkeypatch.setattr(repair, "_patch_body", lambda number, *, repo, expected_sha256, body: None)

    assert repair.main(["apply", "--issue", "7601"]) == 2
