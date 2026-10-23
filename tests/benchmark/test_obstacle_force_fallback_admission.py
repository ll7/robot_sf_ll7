"""Fail-closed admission through the scanner and real JSONL metadata producer."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.gym_env.robot_env import _jsonl_runtime_metadata


def _metadata() -> dict[str, object]:
    """Return minimal law metadata without relying on the receipt builder."""
    return {
        "site": "fast_pysf",
        "law_version": "legacy_shifted_gradient_v1",
        "resolution_mode": "defaulted_missing",
        "enabled": True,
        "applied": False,
    }


def _receipt() -> dict[str, object]:
    """Construct an independently digest-bound neutral persisted receipt."""
    identity = {
        "config_hash": "regression-config",
        "source_commit": "regression-source",
        "site": "fast_pysf",
        "law_version": "legacy_shifted_gradient_v1",
        "selection_source": "defaulted_missing",
        "parameters_sha256": None,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "obstacle_force_law_diagnostic_receipt.v1",
        "claim_boundary": "diagnostic_only",
        **{
            key: identity[key]
            for key in ("site", "law_version", "selection_source", "parameters_sha256")
        },
        "enabled": True,
        "applied": False,
        "fallback": {"used": False, "count": 0, "first_reason": None, "reasons": {}},
        "input_identity": {
            "config_hash": identity["config_hash"],
            "source_commit": identity["source_commit"],
            "input_sha256": digest,
        },
    }


@pytest.mark.parametrize("malformed", [None, [], "malformed", True, {}, {"used": False}])
def test_scanner_rejects_malformed_persisted_fallback(malformed: object) -> None:
    """Malformed or incomplete persisted fallback blocks cannot become neutral."""
    forged = _receipt()
    forged["fallback"] = malformed
    assert runtime_fallback_or_degraded_marker({"diagnostic_receipt": forged}) is not None


def test_jsonl_refuses_to_erase_contradictory_positive_fallback() -> None:
    """The production sidecar path must reject or expose conflicting telemetry."""
    metadata = {**_metadata(), "fallback": False, "fallback_triggered": True, "fallback_count": 0}
    try:
        emitted = _jsonl_runtime_metadata({"obstacle_force_law": metadata}, "regression-config")
    except ValueError as exc:
        assert "fallback" in str(exc) and "disagree" in str(exc)
    else:
        assert runtime_fallback_or_degraded_marker(emitted) is not None


@pytest.mark.parametrize(
    "fallback",
    [
        {"used": False, "fallback_triggered": True},
        {
            "fallback": {"used": False, "count": 0, "first_reason": None, "reasons": {}},
            "fallback_count": 1,
            "fallback_reason": "dropped",
        },
        {"fallback": {"used": False}, "fallback_triggered": True},
        {"used": True, "count": 0, "fallback_count": 1},
        {"used": True, "first_reason": None, "fallback_reason": "dropped"},
        {"used": True, "reasons": {}, "fallback_reasons": {"dropped": 1}},
        {"used": False, "fallback_triggered": 0},
        {"used": True, "count": 1, "fallback_count": True},
        {"fallback": None},
        {"fallback": []},
        [],
        ["malformed"],
        "malformed",
        False,
    ],
)
def test_builder_rejects_conflicting_aliases_and_malformed_containers(fallback: Any) -> None:
    """Every supplied alias and container is checked before destructive projection."""
    from robot_sf.benchmark.obstacle_force_diagnostic_receipt import (
        ObstacleForceDiagnosticReceiptError,
        attach_obstacle_force_diagnostic_receipt,
    )

    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="fallback"):
        attach_obstacle_force_diagnostic_receipt(
            _metadata(), config_hash="c", source_commit="s", fallback=fallback
        )


@pytest.mark.parametrize(
    "telemetry",
    [
        {"fallback": False, "fallback_triggered": True},
        {"fallback": True, "fallback_count": 1},
    ],
)
def test_explicit_neutral_diagnostics_cannot_hide_positive_law_metadata(telemetry: dict) -> None:
    """Explicit defaults cannot hide positive or contradictory law telemetry."""
    from robot_sf.benchmark.obstacle_force_diagnostic_receipt import (
        ObstacleForceDiagnosticReceiptError,
        attach_obstacle_force_diagnostic_receipt,
    )

    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="fallback.*disagree"):
        attach_obstacle_force_diagnostic_receipt(
            {**_metadata(), **telemetry}, config_hash="c", source_commit="s", fallback={}
        )
