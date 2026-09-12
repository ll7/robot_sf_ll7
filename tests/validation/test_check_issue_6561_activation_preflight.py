"""Adversarial tests for the issue #6561 activation-preflight classifier.

Fixtures use only the reserved disjoint seed block and the checker-owned
integrity envelope. They never execute registered seeds or assert native
benchmark evidence; the complete synthetic path must remain blocked until a
canonical native diagnostics owner exists.
"""

from __future__ import annotations

import copy
import json
import sys
from typing import Any

import pytest

from scripts.validation import check_issue_6561_activation_preflight as preflight

PREFLIGHT_SEEDS = (311, 312, 313, 314)
TREATED_REGIMES = {"slow_distributed": 0.65, "typical_distributed": 1.3}


def _diagnostics(  # noqa: PLR0913 - fixture exposes each diagnostic override
    payload: dict[str, Any],
    protocol: dict[str, Any],
    *,
    fraction: float = 0.9,
    time_to_target: float = 0.8,
    transient_steps: int = 5,
    spawn_mean: float = 0.5,
    spawn_peak: float = 0.52,
    mean_offset: float = 0.01,
    row_status: str = "native",
) -> dict[str, Any]:
    manifest = preflight.compile_preflight_manifest(payload, protocol)
    rows: list[dict[str, Any]] = []
    for identity in manifest["identities"]:
        configured_mean = identity["runtime_controls"]["desired_speed_mean"]
        row = dict(identity)
        row.update(
            {
                "row_status": row_status,
                "diagnostics": {
                    "configured_desired_speed_mean_m_s": configured_mean,
                    "configured_desired_speed_std_m_s": identity["runtime_controls"][
                        "desired_speed_std"
                    ],
                    "realized_desired_speed_mean_m_s": configured_mean + mean_offset,
                    "realized_desired_speed_std_m_s": 0.21,
                    "initial_spawn_speed_mean_m_s": spawn_mean,
                    "initial_spawn_speed_peak_m_s": spawn_peak,
                    "time_to_desired_speed_target_seconds": time_to_target,
                    "acceleration_transient_steps": transient_steps,
                    "desired_speed_activation_fraction": fraction,
                },
            }
        )
        rows.append(row)
    return {
        "schema_version": preflight.DIAGNOSTICS_SCHEMA_VERSION,
        "status": "complete",
        "seeds": list(PREFLIGHT_SEEDS),
        "provenance": {
            "protocol_semantic_hash": payload["protocol_semantic_hash"],
            "preflight_config_semantic_hash": payload["preflight_config_semantic_hash"],
            "preflight_manifest_hash": manifest["preflight_manifest_hash"],
        },
        "rows": rows,
    }


def _terminal_diagnostics(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    """Build an explicit terminal outcome with frozen checker provenance."""
    manifest = preflight.compile_preflight_manifest(payload, protocol)
    return {
        "schema_version": preflight.DIAGNOSTICS_SCHEMA_VERSION,
        "status": status,
        "seeds": list(PREFLIGHT_SEEDS),
        "provenance": {
            "protocol_semantic_hash": payload["protocol_semantic_hash"],
            "preflight_config_semantic_hash": payload["preflight_config_semantic_hash"],
            "preflight_manifest_hash": manifest["preflight_manifest_hash"],
        },
        "reason": f"synthetic {status} fixture",
    }


def test_check_only_compiles_disjoint_192_cells() -> None:
    """The check-only manifest must be deterministic, unique, and disjoint."""
    payload, protocol = preflight.load_preflight()
    first = preflight.compile_preflight_manifest(payload, protocol)
    second = preflight.compile_preflight_manifest(payload, protocol)

    assert first["identity_count"] == 192
    assert first["unique_identity_count"] == 192
    assert first["preflight_manifest_hash"] == second["preflight_manifest_hash"]
    assert first["banner"] == preflight.NOT_EVIDENCE_BANNER
    assert {row["seed"] for row in first["identities"]} == set(PREFLIGHT_SEEDS)
    assert all(
        row["registered"] is False and row["preflight"] is True for row in first["identities"]
    )
    assert all(
        {row["regime_id"] for row in first["identities"] if row["seed"] == seed}
        == set(TREATED_REGIMES)
        for seed in PREFLIGHT_SEEDS
    )


def test_validate_rejects_registered_seed_overlap() -> None:
    """A preflight seed inside 111-140 must fail closed."""
    payload, protocol = preflight.load_preflight()
    mutated = copy.deepcopy(payload)
    mutated["preflight_seed_block"]["seeds"] = [140, 311, 312, 313]

    with pytest.raises(preflight.PreflightError, match="overlap"):
        preflight.validate_preflight(mutated, protocol)


def test_validate_rejects_threshold_drift() -> None:
    """Activation thresholds must match the frozen protocol."""
    payload, protocol = preflight.load_preflight()
    mutated = copy.deepcopy(payload)
    mutated["activation_rule"]["target_tolerance_m_s"] = 0.5

    with pytest.raises(preflight.PreflightError, match="tolerance"):
        preflight.validate_preflight(mutated, protocol)


def test_validate_rejects_production_enablement() -> None:
    """The preflight packet must remain check-only."""
    payload, protocol = preflight.load_preflight()
    mutated = copy.deepcopy(payload)
    mutated["validation_contract"]["registered_rows_allowed"] = True

    with pytest.raises(preflight.PreflightError, match="registered rows"):
        preflight.validate_preflight(mutated, protocol)


def test_classify_activation_passes_synthetic_native_rows() -> None:
    """Synthetic native diagnostics within every threshold classify as activated."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(payload, protocol, _diagnostics(payload, protocol))

    assert result["ok"] is False
    assert result["activation_ok"] is True
    assert result["activation_verdict"] == "activation_pass"
    assert result["status"] == "blocked"
    assert result["admission_status"] == "not_admitted"
    assert result["availability_status"] == "not_available"
    assert result["registered_seed_overlap"] is False
    assert set(result["per_regime"]) == set(TREATED_REGIMES)
    assert all(
        entry["classification"] == "intervention_activated"
        for entry in result["per_regime"].values()
    )


def test_classify_activation_flags_inactive_intervention() -> None:
    """An activation fraction below the frozen minimum must be inactive."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(
        payload,
        protocol,
        _diagnostics(payload, protocol, fraction=0.5),
    )

    assert result["ok"] is False
    assert result["activation_ok"] is False
    assert all(
        entry["classification"] == "intervention_inactive"
        for entry in result["per_regime"].values()
    )


def test_classify_activation_flags_invalid_transient() -> None:
    """A spawn transient beyond the frozen window must be invalid."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(
        payload,
        protocol,
        _diagnostics(payload, protocol, transient_steps=25),
    )

    assert result["ok"] is False
    assert all(
        entry["classification"] == "invalid_transient" for entry in result["per_regime"].values()
    )


def test_classify_activation_flags_invalid_initial_speed() -> None:
    """A perturbed initial spawn speed must be invalid, not inactive."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(
        payload,
        protocol,
        _diagnostics(payload, protocol, spawn_mean=0.7, spawn_peak=0.72),
    )

    assert result["ok"] is False
    assert all(
        entry["classification"] == "invalid_initial_speed"
        for entry in result["per_regime"].values()
    )


@pytest.mark.parametrize("row_status", preflight.EXPECTED_FORBIDDEN_ROW_STATUSES)
def test_classify_fails_closed_on_forbidden_row_status(row_status: str) -> None:
    """Forbidden row statuses cannot become activation evidence."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(
        payload,
        protocol,
        _diagnostics(payload, protocol, row_status=row_status),
    )

    assert result["ok"] is False
    assert all(
        entry["classification"] == "not_evaluable" for entry in result["per_regime"].values()
    )


def test_classify_rejects_non_preflight_seed() -> None:
    """Diagnostics from registered seeds must be rejected outright."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["rows"][0]["seed"] = 120

    with pytest.raises(preflight.PreflightError, match="seed drifted"):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_classify_rejects_partial_identity_set() -> None:
    """A complete status cannot hide a missing manifest identity."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["rows"].pop()

    with pytest.raises(preflight.PreflightError, match="missing_count=1"):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_classify_rejects_duplicate_identity() -> None:
    """Duplicate rows are rejected before any per-regime aggregation."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["rows"].append(copy.deepcopy(diagnostics["rows"][0]))

    with pytest.raises(preflight.PreflightError, match="duplicate diagnostics identity"):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_classify_rejects_unknown_identity() -> None:
    """An arbitrary row identifier cannot be admitted into the frozen grid."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["rows"][0]["identity_key"] = "unknown__identity__311"

    with pytest.raises(preflight.PreflightError, match="unknown diagnostics identity"):
        preflight.classify_activation(payload, protocol, diagnostics)


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("scenario_source_sha256", "scenario source hash drifted"),
        ("planner_config_sha256", "planner config hash drifted"),
        ("runtime_controls", "runtime controls drifted"),
        ("dt_seconds", "dt drifted"),
        ("robot_speed_cap_m_s", "robot speed cap drifted"),
    ],
)
def test_classify_rejects_mutated_identity_inputs(field: str, expected_message: str) -> None:
    """Scenario, planner, and runtime identity inputs stay bound to the manifest."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    row = diagnostics["rows"][0]
    if field == "planner_config_sha256":
        row = next(
            candidate
            for candidate in diagnostics["rows"]
            if candidate["planner_config_sha256"] is not None
        )
    if field == "runtime_controls":
        row[field]["desired_speed_mean"] = 0.66
    elif field == "dt_seconds":
        row[field] = 0.2
    elif field == "robot_speed_cap_m_s":
        row[field] = 1.5
    else:
        row[field] = "0" * 64

    with pytest.raises(preflight.PreflightError, match=expected_message):
        preflight.classify_activation(payload, protocol, diagnostics)


@pytest.mark.parametrize("field", sorted(preflight.EXPECTED_PROVENANCE_FIELDS))
def test_classify_rejects_mutated_provenance(field: str) -> None:
    """Every checker-owned hash must point at the frozen protocol and manifest."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["provenance"][field] = "0" * 64

    with pytest.raises(preflight.PreflightError, match="provenance"):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_classify_rejects_mutated_diagnostics_seed_block() -> None:
    """A payload seed declaration cannot silently select a different block."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["seeds"][-1] = 315

    with pytest.raises(preflight.PreflightError, match="diagnostics seeds"):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_validate_rejects_mutated_preflight_seed_block() -> None:
    """The versioned config pins the reserved disjoint seed block exactly."""
    payload, protocol = preflight.load_preflight()
    mutated = copy.deepcopy(payload)
    mutated["preflight_seed_block"]["seeds"][-1] = 315

    with pytest.raises(preflight.PreflightError, match="preflight seeds must remain exactly"):
        preflight.validate_preflight(mutated, protocol)


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("preflight_config_semantic_hash", "preflight config semantic hash drifted"),
        ("preflight_manifest_hash", "preflight manifest hash declaration drifted"),
    ],
)
def test_validate_rejects_mutated_preflight_identity_hash(
    field: str,
    expected_message: str,
) -> None:
    """The versioned config cannot be detached from its frozen identity hashes."""
    payload, protocol = preflight.load_preflight()
    mutated = copy.deepcopy(payload)
    mutated[field] = "0" * 64

    with pytest.raises(preflight.PreflightError, match=expected_message):
        preflight.validate_preflight(mutated, protocol)


@pytest.mark.parametrize("status", ["adapter", "unknown"])
def test_classify_rejects_unknown_or_adapter_row_status(status: str) -> None:
    """Only native or explicitly forbidden statuses belong to the row contract."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol, row_status=status)

    with pytest.raises(preflight.PreflightError, match="unsupported row_status"):
        preflight.classify_activation(payload, protocol, diagnostics)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("desired_speed_activation_fraction", float("nan")),
        ("time_to_desired_speed_target_seconds", float("inf")),
        ("realized_desired_speed_std_m_s", float("-inf")),
    ],
)
def test_classify_rejects_non_finite_diagnostics(field: str, value: float) -> None:
    """NaN and infinities cannot pass through the activation classifier."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["rows"][0]["diagnostics"][field] = value

    with pytest.raises(preflight.PreflightError, match="must be finite"):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_classify_rejects_non_finite_forbidden_row_diagnostics() -> None:
    """Forbidden status must not become a loophole for malformed diagnostic values."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol, row_status="degraded")
    diagnostics["rows"][0]["diagnostics"]["desired_speed_activation_fraction"] = float("nan")

    with pytest.raises(preflight.PreflightError, match="must be finite"):
        preflight.classify_activation(payload, protocol, diagnostics)


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("desired_speed_activation_fraction", 1.1, "activation fraction is out of range"),
        (
            "time_to_desired_speed_target_seconds",
            -0.1,
            "time to desired-speed target is out of range",
        ),
        ("initial_spawn_speed_peak_m_s", 0.4, "initial spawn speed peak is out of range"),
    ],
)
def test_classify_rejects_out_of_range_diagnostics(
    field: str,
    value: float,
    expected_message: str,
) -> None:
    """Finite values still need physically meaningful contract bounds."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["rows"][0]["diagnostics"][field] = value

    with pytest.raises(preflight.PreflightError, match=expected_message):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_classify_rejects_mutated_configured_speed() -> None:
    """Reported configured speed must equal the frozen manifest runtime input."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["rows"][0]["diagnostics"]["configured_desired_speed_mean_m_s"] = 0.66

    with pytest.raises(preflight.PreflightError, match="configured desired-speed mean drifted"):
        preflight.classify_activation(payload, protocol, diagnostics)


@pytest.mark.parametrize("status", ["not_available", "failed"])
def test_classify_preserves_explicit_terminal_fail_closed_status(status: str) -> None:
    """Explicit unavailable/failed inputs remain non-admitted outcomes."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(
        payload,
        protocol,
        _terminal_diagnostics(payload, protocol, status=status),
    )

    assert result["ok"] is False
    assert result["activation_ok"] is False
    assert result["status"] == status
    assert result["input_status"] == status
    assert result["availability_status"] == status
    assert result["admission_status"] == "not_admitted"
    assert result["canonical_native_diagnostics_status"] == "blocked"


def test_classify_rejects_diagnostics_schema_drift() -> None:
    """The checker-owned envelope version is explicit and fail-closed."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["schema_version"] = "robot_sf.issue_6561_pedestrian_speed_activation_diagnostics.v1"

    with pytest.raises(preflight.PreflightError, match="schema_version is unsupported"):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_classify_rejects_missing_diagnostics_envelope_field() -> None:
    """A complete envelope must carry all checker-owned provenance fields."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics.pop("provenance")

    with pytest.raises(preflight.PreflightError, match="fields do not match"):
        preflight.classify_activation(payload, protocol, diagnostics)


def test_error_result_is_structured_failed_and_not_admitted() -> None:
    """Malformed CLI input must render a failed, non-admitted result."""
    result = preflight._error_result(preflight.PreflightError("malformed fixture"))

    assert result["status"] == "failed"
    assert result["input_status"] == "failed"
    assert result["ok"] is False
    assert result["admission_status"] == "not_admitted"
    assert result["reason_code"] == "invalid_diagnostics_contract"
    assert result["registered_seed_overlap"] is None


def test_cli_rejects_oversized_transient_steps_as_structured_failure(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """An oversized transient-step integer must not escape as a traceback."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(payload, protocol)
    diagnostics["rows"][0]["diagnostics"]["acceleration_transient_steps"] = 10**1000
    diagnostics_path = tmp_path / "oversized.json"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_issue_6561_activation_preflight.py",
            "--diagnostics",
            str(diagnostics_path),
        ],
    )

    assert preflight.main() == 2

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["reason_code"] == "invalid_diagnostics_contract"
    assert result["admission_status"] == "not_admitted"
    assert result["benchmark_success"] is False
    assert result["canonical_native_diagnostics_owner"] == "unavailable"
    assert result["canonical_native_diagnostics_status"] == "blocked"
    assert result["registered_seed_overlap"] is None
    assert "int too large to convert to float" in result["reason"]
