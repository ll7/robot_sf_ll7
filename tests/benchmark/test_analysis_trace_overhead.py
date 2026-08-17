"""Tests for the fail-closed analysis-trace overhead measurement harness."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmark import measure_analysis_trace_overhead_issue_6972 as measurement

GIT_HASH = "a" * 40
ARTIFACT_SHA256 = "b" * 64


def _record(*, trace_enabled: bool) -> dict[str, Any]:
    """Return a compact deterministic fake episode record."""

    record: dict[str, Any] = {
        "outcome": {"timeout_event": True},
        "metrics": {"progress": 0.5},
        "algorithm_metadata": {},
        "provenance": {"artifact_sha256": ARTIFACT_SHA256},
    }
    if trace_enabled:
        record["algorithm_metadata"] = {
            "analysis_trace": {
                "git_hash": GIT_HASH,
                "artifact_sha256": ARTIFACT_SHA256,
                "steps": [
                    {"step": 0, "controls": {"requested": None, "applied": None}},
                    {
                        "step": 1,
                        "controls": {"requested": [0.1, 0.0], "applied": [0.1, 0.0]},
                    },
                ],
            }
        }
    return record


def _patch_deterministic_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    elapsed_seconds: list[float],
) -> list[bool]:
    """Patch the runner and clock while preserving arm-order observability."""

    arm_calls: list[bool] = []
    monkeypatch.setattr(measurement, "_git_hash", lambda: GIT_HASH)

    def fake_run_episode(*, trace_enabled: bool) -> dict[str, Any]:
        arm_calls.append(trace_enabled)
        return _record(trace_enabled=trace_enabled)

    monkeypatch.setattr(measurement, "_run_episode", fake_run_episode)
    ticks = iter(tick for elapsed in elapsed_seconds for tick in (0.0, elapsed))

    # The caller supplies non-negative durations; resetting each pair to zero keeps
    # the fake clock simple while preserving every elapsed interval exactly.
    monkeypatch.setattr(measurement, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    return arm_calls


def test_measurement_alternates_batches_and_allows_stable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable repeated paired batches can produce a target decision."""

    arm_calls = _patch_deterministic_fixture(monkeypatch, elapsed_seconds=[1.0] * 8)

    receipt = measurement.measure(samples=2, warmups=1, batches=2)

    assert arm_calls == [
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [batch["arm_order"] for batch in receipt["batches"]] == [
        ["off", "on"],
        ["on", "off"],
    ]
    assert receipt["checks"]["same_commit_repeated_batch_stable"] is True
    assert receipt["derived"]["stability_status"] == "stable"
    assert receipt["derived"]["target_met"] is True
    assert receipt["derived"]["median_runtime_delta_ms"] == pytest.approx(0.0)


def test_measurement_with_unstable_batches_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spread across repeated batches cannot support a target claim."""

    _patch_deterministic_fixture(monkeypatch, elapsed_seconds=[1.0, 2.0, 1.0, 1.0])

    receipt = measurement.measure(samples=1, warmups=0, batches=2)

    assert receipt["derived"]["batch_overhead_fractions"] == pytest.approx([1.0, 0.0])
    assert receipt["derived"]["stability_status"] == "unstable"
    assert receipt["checks"]["same_commit_repeated_batch_stable"] is False
    assert receipt["derived"]["target_met"] is None
    assert receipt["derived"]["target_decision"] == "inconclusive"


def test_measurement_does_not_hide_a_failed_batch_behind_the_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stable aggregate cannot pass when one repeated batch exceeds 10 percent."""

    _patch_deterministic_fixture(monkeypatch, elapsed_seconds=[1.0, 1.15, 1.05, 1.0])

    receipt = measurement.measure(samples=1, warmups=0, batches=2)

    assert receipt["derived"]["batch_overhead_fractions"] == pytest.approx([0.15, 0.05])
    assert receipt["derived"]["stability_status"] == "stable"
    assert receipt["derived"]["batch_target_met"] is False
    assert receipt["derived"]["target_met"] is False
    assert receipt["derived"]["target_decision"] == "not_met"
    assert receipt["issue"] == 6987
    assert receipt["source_issue"] == 6972


def test_measurement_records_reconciliation_execution_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future receipts record order, warm-up/cache, and thread context explicitly."""

    _patch_deterministic_fixture(monkeypatch, elapsed_seconds=[1.0] * 8)

    receipt = measurement.measure(samples=1, warmups=1, batches=2)

    assert receipt["execution_context"] == {
        "execution_order": "alternating off/on, then on/off by batch",
        "warmup_state": {"warmups_per_arm_per_batch": 1, "excluded_from_timing": True},
        "cache_state": {
            "process_warmup": "per_arm_per_batch",
            "external_cache": "uncontrolled",
        },
        "numerical_thread_settings": {
            key: measurement.os.environ.get(key) for key in measurement.THREAD_SETTING_KEYS
        },
    }


def test_measurement_blocks_target_decision_on_integrity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Divergent paired outcomes remain inconclusive despite stable timing."""

    monkeypatch.setattr(measurement, "_git_hash", lambda: GIT_HASH)

    def fake_run_episode(*, trace_enabled: bool) -> dict[str, Any]:
        record = _record(trace_enabled=trace_enabled)
        if trace_enabled:
            record["outcome"] = {"timeout_event": False}
        return record

    monkeypatch.setattr(measurement, "_run_episode", fake_run_episode)
    ticks = iter(tick for _ in range(4) for tick in (0.0, 1.0))
    monkeypatch.setattr(measurement, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))

    receipt = measurement.measure(samples=1, warmups=0, batches=2)

    assert receipt["checks"]["paired_outcomes_and_metrics_equal"] is False
    assert receipt["derived"]["batch_target_met"] is True
    assert receipt["derived"]["target_met"] is None
    assert receipt["derived"]["target_decision"] == "inconclusive"
