"""Fast-lane coverage for the merge-receipt blocked-verification remedy (#9218).

The remedy is diagnostic only: it names the canonical recovery for a base-drift block
without changing any verification semantics, so these tests stay small and
deterministic.
"""

from __future__ import annotations

from scripts.dev.single_account_merge_receipt import verify_receipt
from tests.dev.test_single_account_merge_receipt import _live_evidence, _receipt


def _base_drifted_evidence(receipt: dict) -> dict:
    """Return live evidence that differs only in the base-bound fields."""
    evidence = _live_evidence(receipt)
    evidence["current_base_sha"] = "c" * 40
    gate_audit = dict(evidence.get("gate_audit") or {})
    gate_audit["passed"] = False
    evidence["gate_audit"] = gate_audit
    return evidence


def test_base_drift_block_names_regenerate_and_reapply() -> None:
    receipt = _receipt()
    result = verify_receipt(receipt, live_evidence=_base_drifted_evidence(receipt))
    assert result["passed"] is False
    remedy = result["remedy"]
    assert remedy["kind"] == "regenerate_and_reapply"
    assert remedy["reason_class"] == "concurrent_base_advance"
    assert remedy["head_sha"] == receipt["head_sha"]
    assert all(receipt["head_sha"] in command for command in remedy["commands"])
    assert any("--mode report-only" in command for command in remedy["commands"])
    assert any("--mode apply" in command for command in remedy["commands"])
    assert all(str(receipt["pr_number"]) in command for command in remedy["commands"])


def test_non_drift_block_stays_an_inspection() -> None:
    receipt = _receipt()
    evidence = _live_evidence(receipt)
    evidence["holds"] = {
        **{key: dict(value) for key, value in (evidence.get("holds") or {}).items()},
        "merge": {"status": "held", "reason_codes": ["manual_pause"], "source": "labels"},
    }
    result = verify_receipt(receipt, live_evidence=evidence)
    assert result["passed"] is False
    remedy = result["remedy"]
    assert remedy["kind"] == "inspect_reasons"
    assert remedy["reason_class"] == "other"
    assert remedy["reasons"] == result["reasons"]
    assert "regenerate_and_reapply" != remedy["kind"]


def test_mixed_drift_and_other_reasons_stay_fail_closed() -> None:
    receipt = _receipt()
    evidence = _base_drifted_evidence(receipt)
    evidence["holds"] = {
        **{key: dict(value) for key, value in (evidence.get("holds") or {}).items()},
        "legal_release": {"status": "held", "reason_codes": ["counsel"], "source": "labels"},
    }
    result = verify_receipt(receipt, live_evidence=evidence)
    assert result["passed"] is False
    assert result["remedy"]["kind"] == "inspect_reasons"


def test_passing_verification_has_no_remedy() -> None:
    receipt = _receipt()
    result = verify_receipt(receipt, live_evidence=_live_evidence(receipt))
    assert result["passed"] is True
    assert "remedy" not in result


def test_structural_block_has_no_blanket_remedy() -> None:
    broken = dict(_receipt())
    broken["head_sha"] = "not-a-sha"
    result = verify_receipt(broken)
    assert result["passed"] is False
    assert result["remedy"]["kind"] == "inspect_reasons"
