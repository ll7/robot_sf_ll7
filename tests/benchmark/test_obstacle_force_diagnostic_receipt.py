"""Regression tests for the diagnostic-only obstacle-force receipt contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from robot_sf.benchmark.obstacle_force_diagnostic_receipt import (
    ObstacleForceDiagnosticReceiptError,
    build_obstacle_force_diagnostic_receipt,
    validate_obstacle_force_diagnostic_receipt,
)
from robot_sf.gym_env.robot_env import _jsonl_runtime_metadata

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf/benchmark/schemas/obstacle_force_law_diagnostic_receipt.v1.json"
)


def _metadata() -> dict[str, object]:
    """Return a small resolved-law payload with a self-consistent parameter hash."""
    parameters = {"distance_floor": 1e-5, "factor": 1.0}
    parameter_bytes = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": "obstacle_force_law_metadata.v2",
        "law_version": "legacy_shifted_gradient_v1",
        "site": "fast_pysf",
        "geometry_convention": "map_line_endpoints_orthogonal_vector",
        "radius_convention": "threshold_plus_agent_radius_sigma",
        "compatibility_mode": "legacy_compatible",
        "enabled": True,
        "applied": False,
        "resolution_mode": "defaulted_missing",
        "parameters": parameters,
        "parameters_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
    }


def test_receipt_is_schema_valid_json_roundtrippable_and_preserves_fallback() -> None:
    """The receipt carries selection, fallback, and explicit input identity."""
    receipt = build_obstacle_force_diagnostic_receipt(
        _metadata(),
        config_hash="config-8277",
        source_commit="commit-8277",
        fallback={
            "fallback": True,
            "fallback_count": 2,
            "fallback_reason": "obstacle_force_dropped",
            "fallback_reasons": {"obstacle_force_dropped": 2},
        },
    )

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(receipt)) == []
    assert validate_obstacle_force_diagnostic_receipt(receipt) == receipt
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt
    assert receipt["claim_boundary"] == "diagnostic_only"
    assert receipt["fallback"] == {
        "used": True,
        "count": 2,
        "first_reason": "obstacle_force_dropped",
        "reasons": {"obstacle_force_dropped": 2},
    }
    assert receipt["input_identity"]["config_hash"] == "config-8277"
    assert len(receipt["input_identity"]["input_sha256"]) == 64


def test_receipt_rejects_nonfinite_parameters() -> None:
    """A diagnostic receipt must never hide non-finite numerical inputs."""
    metadata = _metadata()
    metadata["parameters"] = {"distance_floor": float("nan")}
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="JSON-safe"):
        build_obstacle_force_diagnostic_receipt(
            metadata,
            config_hash="config-8277",
            source_commit="commit-8277",
        )


def test_receipt_preserves_socnav_fallback_triggered_alias() -> None:
    """SocNav's existing fallback_triggered diagnostic remains observable."""
    receipt = build_obstacle_force_diagnostic_receipt(
        _metadata(),
        config_hash="config-8277",
        source_commit="commit-8277",
        fallback={
            "fallback_triggered": True,
            "fallback_count": 1,
            "fallback_reason": "socnav_fallback",
            "fallback_reasons": {"socnav_fallback": 1},
        },
    )

    assert receipt["fallback"] == {
        "used": True,
        "count": 1,
        "first_reason": "socnav_fallback",
        "reasons": {"socnav_fallback": 1},
    }


def test_jsonl_runtime_metadata_attaches_receipt_and_fallback(monkeypatch) -> None:
    """The JSONL sidecar path emits the same validated receipt shape."""
    from robot_sf.gym_env import robot_env

    monkeypatch.setattr(robot_env, "_git_hash_fallback", lambda: "jsonl-commit")
    payload = _jsonl_runtime_metadata(
        {
            "obstacle_force_law": _metadata(),
            "obstacle_force_law_diagnostics": {
                "fallback": True,
                "fallback_count": 1,
                "fallback_reason": "synthetic",
                "fallback_reasons": {"synthetic": 1},
            },
        },
        "jsonl-config",
    )

    assert payload is not None
    metadata = payload["obstacle_force_law"]
    receipt = metadata["diagnostic_receipt"]
    assert validate_obstacle_force_diagnostic_receipt(receipt)["fallback"]["used"] is True
    assert receipt["input_identity"]["config_hash"] == "jsonl-config"
    assert receipt["input_identity"]["source_commit"] == "jsonl-commit"
    assert metadata["source_commit"] == "jsonl-commit"
