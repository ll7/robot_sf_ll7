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


def _valid_receipt() -> dict[str, object]:
    """Build one valid receipt for negative validation cases."""
    return build_obstacle_force_diagnostic_receipt(
        _metadata(),
        config_hash="config-8277",
        source_commit="commit-8277",
    )


def test_validate_rejects_non_mapping_receipt() -> None:
    """Only mappings can be diagnostic receipts."""
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a mapping"):
        validate_obstacle_force_diagnostic_receipt(["not", "a", "mapping"])


def test_validate_rejects_shape_violations() -> None:
    """Missing/extra keys, schema version, and claim boundary fail closed."""
    receipt = dict(_valid_receipt())
    del receipt["site"]
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="missing="):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    receipt["unexpected"] = 1
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="extra="):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    receipt["schema_version"] = "other.v9"
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="unsupported diagnostic"):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    receipt["claim_boundary"] = "benchmark"
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="diagnostic_only"):
        validate_obstacle_force_diagnostic_receipt(receipt)


def test_validate_rejects_selection_violations() -> None:
    """Unknown laws, sources, digests, and flags fail closed."""
    receipt = dict(_valid_receipt())
    receipt["law_version"] = "unknown_law"
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="unsupported law_version"):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    receipt["selection_source"] = "unknown_source"
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="unsupported selection_source"):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    receipt["parameters_sha256"] = "not-a-digest"
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="SHA-256 digest"):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    receipt["enabled"] = "yes"
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a boolean"):
        validate_obstacle_force_diagnostic_receipt(receipt)


def test_validate_rejects_input_identity_violations() -> None:
    """The input-identity block must be complete, textual, and digest-bound."""
    receipt = dict(_valid_receipt())
    receipt["input_identity"] = []
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a mapping"):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    identity = dict(receipt["input_identity"])
    del identity["source_commit"]
    receipt["input_identity"] = identity
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="incomplete or unknown"):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    identity = dict(receipt["input_identity"])
    identity["config_hash"] = "  "
    receipt["input_identity"] = identity
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="non-empty string"):
        validate_obstacle_force_diagnostic_receipt(receipt)

    receipt = dict(_valid_receipt())
    identity = dict(receipt["input_identity"])
    identity["input_sha256"] = "0" * 64
    receipt["input_identity"] = identity
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="does not match receipt"):
        validate_obstacle_force_diagnostic_receipt(receipt)


def test_build_rejects_metadata_violations() -> None:
    """Malformed law metadata fails closed before any receipt exists."""
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a mapping"):
        build_obstacle_force_diagnostic_receipt(
            ["not", "a", "mapping"], config_hash="c", source_commit="s"
        )

    metadata = _metadata()
    metadata["site"] = ""
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="non-empty string"):
        build_obstacle_force_diagnostic_receipt(metadata, config_hash="c", source_commit="s")

    metadata = _metadata()
    metadata["enabled"] = 1
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be booleans"):
        build_obstacle_force_diagnostic_receipt(metadata, config_hash="c", source_commit="s")

    metadata = _metadata()
    metadata["parameters_sha256"] = "0" * 64
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="does not match"):
        build_obstacle_force_diagnostic_receipt(metadata, config_hash="c", source_commit="s")

    metadata = _metadata()
    metadata["parameters"] = ["not", "a", "mapping"]
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a mapping"):
        build_obstacle_force_diagnostic_receipt(metadata, config_hash="c", source_commit="s")


def test_build_rejects_fallback_violations() -> None:
    """Malformed fallback payloads fail closed instead of recording guesses."""
    base = {"config_hash": "c", "source_commit": "s"}
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a boolean"):
        build_obstacle_force_diagnostic_receipt(_metadata(), fallback={"used": "yes"}, **base)
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="non-negative integer"):
        build_obstacle_force_diagnostic_receipt(
            _metadata(), fallback={"used": True, "count": -1}, **base
        )
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="non-negative integer"):
        build_obstacle_force_diagnostic_receipt(
            _metadata(), fallback={"used": True, "count": True}, **base
        )
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="requires fallback.used=true"):
        build_obstacle_force_diagnostic_receipt(
            _metadata(), fallback={"used": False, "count": 2}, **base
        )
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="cannot be present"):
        build_obstacle_force_diagnostic_receipt(
            _metadata(),
            fallback={"used": False, "first_reason": "stale"},
            **base,
        )
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a mapping"):
        build_obstacle_force_diagnostic_receipt(_metadata(), fallback={"reasons": ["x"]}, **base)
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="non-negative integers"):
        build_obstacle_force_diagnostic_receipt(
            _metadata(), fallback={"reasons": {"x": -1}}, **base
        )


def test_build_defaults_absent_fallback_to_unused() -> None:
    """A receipt without fallback signals records an explicit unused default."""
    receipt = build_obstacle_force_diagnostic_receipt(
        _metadata(), config_hash="config-8277", source_commit="commit-8277"
    )

    assert receipt["fallback"] == {
        "used": False,
        "count": 0,
        "first_reason": None,
        "reasons": {},
    }


def test_validate_rejects_malformed_input_sha256() -> None:
    """A non-digest input SHA fails closed before digest comparison."""
    receipt = dict(_valid_receipt())
    identity = dict(receipt["input_identity"])
    identity["input_sha256"] = "not-a-digest"
    receipt["input_identity"] = identity
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a SHA-256 digest"):
        validate_obstacle_force_diagnostic_receipt(receipt)


def test_build_rejects_malformed_standalone_parameters_sha256() -> None:
    """A provided digest without parameters must still be a real digest."""
    metadata = _metadata()
    del metadata["parameters"]
    metadata["parameters_sha256"] = "not-a-digest"
    with pytest.raises(ObstacleForceDiagnosticReceiptError, match="must be a SHA-256 digest"):
        build_obstacle_force_diagnostic_receipt(metadata, config_hash="c", source_commit="s")


def test_fallback_from_mapping_prefers_nested_fallback_block() -> None:
    """A nested fallback block passes through verbatim for normalization."""
    from robot_sf.benchmark.obstacle_force_diagnostic_receipt import (
        obstacle_force_fallback_from_mapping,
    )

    nested = {"used": True, "count": 3}
    assert obstacle_force_fallback_from_mapping({"fallback": nested}) == nested
    assert obstacle_force_fallback_from_mapping(None) is None
