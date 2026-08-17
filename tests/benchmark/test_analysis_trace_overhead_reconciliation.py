"""Tests for fail-closed analysis-trace overhead receipt reconciliation."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.analysis_trace_overhead_reconciliation import (
    THREAD_SETTING_KEYS,
    reconcile_receipts,
)


def _receipt(*, target_decision: str = "met", platform: str = "Linux-test") -> dict[str, Any]:
    """Build a compact receipt with complete reconciliation context."""

    return {
        "schema_version": "analysis_trace_overhead_measurement_receipt.v2",
        "issue": 6987,
        "source_issue": 6972,
        "status": "diagnostic_only",
        "repository_commit": "a" * 40,
        "environment": {"platform": platform, "python": "3.13.13", "machine": "x86_64"},
        "fixture": {"scenario": {"id": "issue-6972-overhead"}, "seed": 123},
        "method": {
            "warmups_per_arm_per_batch": 1,
            "samples_per_arm_per_batch": 6,
            "batch_count": 2,
            "arm_order": "alternating off/on, then on/off by batch",
            "stability_tolerance_fraction": 0.25,
            "timer": "time.perf_counter",
            "serialization": "analysis_trace.canonical_json",
            "compression": "gzip.compress(mtime=0)",
        },
        "execution_context": {
            "execution_order": "alternating off/on, then on/off by batch",
            "warmup_state": {"warmups_per_arm_per_batch": 1, "excluded_from_timing": True},
            "cache_state": {
                "process_warmup": "per_arm_per_batch",
                "external_cache": "uncontrolled",
            },
            "numerical_thread_settings": dict.fromkeys(THREAD_SETTING_KEYS),
        },
        "batches": [{"derived": {"overhead_fraction": 0.05}}],
        "arms": {},
        "checks": {
            "paired_outcomes_and_metrics_equal": True,
            "control_sequence_digest_stable": True,
            "trace_git_hash_matches_commit": True,
            "trace_artifact_matches_provenance": True,
        },
        "derived": {
            "batch_overhead_fractions": [0.05, 0.06],
            "median_overhead_fraction": 0.055,
            "target_decision": target_decision,
            "target_met": target_decision == "met",
            "stability_status": "stable",
            "stability_passed": True,
        },
    }


def _write_receipt(tmp_path: Path, name: str, receipt: dict[str, Any]) -> Path:
    """Write a deterministic JSON receipt for a reconciliation test."""

    path = tmp_path / name
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_reconciliation_keeps_compatible_receipts_separate(tmp_path: Path) -> None:
    """Compatible receipts can be classified without a cross-receipt average."""

    first = _write_receipt(tmp_path, "first.json", _receipt())
    second_receipt = _receipt()
    second_receipt["derived"]["batch_overhead_fractions"] = [0.07, 0.08]
    second = _write_receipt(tmp_path, "second.json", second_receipt)

    packet = reconcile_receipts([first, second])

    assert packet["compatibility"] == {
        "compatible": True,
        "cross_receipt_aggregation": "forbidden",
        "reasons": [],
    }
    assert packet["reconciliation"]["classification"] == "measurement_stable"
    assert packet["reconciliation"]["target_decisions"] == ["met", "met"]
    assert "aggregate" not in json.dumps(packet)


def test_reconciliation_rejects_incompatible_environment(tmp_path: Path) -> None:
    """A host mismatch is unavailable rather than an averaged result."""

    first = _write_receipt(tmp_path, "first.json", _receipt())
    second = _write_receipt(tmp_path, "second.json", _receipt(platform="macOS-test"))

    packet = reconcile_receipts([first, second])

    assert packet["reconciliation"]["classification"] == "unavailable"
    assert packet["reconciliation"]["target_decision"] == "unavailable"
    assert "receipt_1_environment_mismatch" in packet["compatibility"]["reasons"]


def test_reconciliation_marks_disagreeing_targets_measurement_unstable(tmp_path: Path) -> None:
    """Compatible stable receipts with different target decisions stay inconclusive."""

    first = _write_receipt(tmp_path, "first.json", _receipt(target_decision="met"))
    second = _write_receipt(
        tmp_path,
        "second.json",
        _receipt(target_decision="not_met"),
    )

    packet = reconcile_receipts([first, second])

    assert packet["compatibility"]["compatible"] is True
    assert packet["reconciliation"]["classification"] == "measurement_unstable"
    assert packet["reconciliation"]["target_decision"] == "inconclusive"


def test_legacy_receipts_without_context_are_unavailable(tmp_path: Path) -> None:
    """Legacy receipts cannot be promoted into a host/order comparison."""

    first_receipt = _receipt()
    second_receipt = copy.deepcopy(first_receipt)
    first_receipt.pop("execution_context")
    second_receipt.pop("execution_context")
    first = _write_receipt(tmp_path, "first.json", first_receipt)
    second = _write_receipt(tmp_path, "second.json", second_receipt)

    packet = reconcile_receipts([first, second])

    assert packet["reconciliation"]["classification"] == "unavailable"
    assert "missing_execution_context" in packet["compatibility"]["reasons"]
    assert packet["reconciliation"]["context_complete"] is False
