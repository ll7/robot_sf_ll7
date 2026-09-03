"""Tests for the pre-submit evidence-contract preflight (scripts/validation).

These pin the cheap CPU gate that blocks a SLURM submission when the public
commit would emit an incomplete evidence contract (the issue #1475 / job 12913
fail-closed waste). They prove:

* a representative on-main row PASSES the ``orca_residual_smoke`` contract;
* a row missing a signal FAILS CLOSED and names the missing field;
* an unknown contract id is a clear error;
* the registry's required-field list comes from the **canonical owner**
  (``orca_residual_lineage_packet``), so there is no second source of truth.
"""

from __future__ import annotations

import json
from typing import Any

from robot_sf.training.orca_residual_lineage_packet import (
    REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS,
    REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS,
)
from scripts.validation.preflight_evidence_contract import (
    _CONTRACT_REGISTRY,
    _evaluate,
    derive_claim_identity,
    main,
)


def _representative_row() -> dict[str, Any]:
    """Return the built-in synthetic on-main row for the smoke contract."""
    return _CONTRACT_REGISTRY["orca_residual_smoke"].representative_row()


def test_representative_row_passes_contract() -> None:
    """The post-#1475 representative row conforms with no missing fields."""
    spec = _CONTRACT_REGISTRY["orca_residual_smoke"]
    report = _evaluate(spec, _representative_row())

    assert report["conforms"] is True
    assert report["missing_required_fields"] == []
    for field in spec.required_fields:
        assert report["evidence"].get(field) is not None


def test_unbound_claim_identity_cannot_be_added_by_caller() -> None:
    """Caller text cannot turn an unbound representative row into claim evidence."""
    identity = {
        "campaign_id": "campaign_fixture",
        "question": "Which contract is checked?",
        "estimand": "Required evidence fields",
    }

    report = _evaluate(
        _CONTRACT_REGISTRY["orca_residual_smoke"],
        _representative_row(),
        claim_identity=identity,
    )

    assert report["conforms"] is False
    assert "not bound to canonical evidence" in report["claim_identity_error"]
    assert "claim_identity" not in report


def test_claim_identity_is_derived_from_canonical_row() -> None:
    """A matching identity is accepted only when the evidence row carries it."""
    identity = {
        "campaign_id": "campaign_fixture",
        "question": "Which contract is checked?",
        "estimand": "Required evidence fields",
    }
    row = _representative_row()
    row["claim_identity"] = identity

    assert derive_claim_identity(row) == identity
    report = _evaluate(
        _CONTRACT_REGISTRY["orca_residual_smoke"],
        row,
        claim_identity=identity,
    )

    assert report["conforms"] is True
    assert report["claim_identity"] == identity


def test_mismatched_claim_identity_is_rejected() -> None:
    """A canonical row cannot be attributed to a different caller identity."""
    row = _representative_row()
    row["claim_identity"] = {
        "campaign_id": "canonical_campaign",
        "question": "Canonical question",
        "estimand": "Canonical estimand",
    }

    report = _evaluate(
        _CONTRACT_REGISTRY["orca_residual_smoke"],
        row,
        claim_identity={
            "campaign_id": "forged_campaign",
            "question": "Unrelated question",
            "estimand": "Unrelated estimand",
        },
    )

    assert report["conforms"] is False
    assert "does not match the canonical evidence row" in report["claim_identity_error"]


def test_cli_exit_zero_on_pass(capsys) -> None:
    """The CLI exits 0 (safe to submit) on the default representative row."""
    exit_code = main(["orca_residual_smoke"])
    assert exit_code == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_rejects_forged_claim_identity(capsys) -> None:
    """The public checker fails closed when CLI identity has no canonical row binding."""
    exit_code = main(
        [
            "orca_residual_smoke",
            "--json",
            "--campaign-id",
            "forged_campaign",
            "--question",
            "Unrelated question",
            "--estimand",
            "Unrelated estimand",
        ]
    )

    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["conforms"] is False
    assert "claim_identity" not in report
    assert "not bound to canonical evidence" in report["claim_identity_error"]


def test_row_missing_signal_fails_closed(tmp_path) -> None:
    """A row whose residual signal is absent blocks and names the missing field."""
    # Strip the shield_stats payload the residual extractor reads, mirroring the
    # historical job-12913 null path (no residual_clipping evidence anywhere).
    row = _representative_row()
    row["algorithm_metadata"] = {}
    row["metrics"] = {}

    row_path = tmp_path / "row.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")

    spec = _CONTRACT_REGISTRY["orca_residual_smoke"]
    report = _evaluate(spec, row)
    assert report["conforms"] is False
    assert "residual_clipping_rate" in report["missing_required_fields"]

    # And the CLI blocks with a non-zero exit naming the field.
    exit_code = main(["orca_residual_smoke", "--row", str(row_path), "--json"])
    assert exit_code == 1


def test_cli_names_missing_field_in_json(tmp_path, capsys) -> None:
    """The blocking JSON report enumerates exactly which fields would be missing."""
    row = _representative_row()
    row["algorithm_metadata"] = {}
    row["metrics"] = {}
    row_path = tmp_path / "row.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")

    exit_code = main(["orca_residual_smoke", "--row", str(row_path), "--json"])
    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["conforms"] is False
    assert "residual_clipping_rate" in report["missing_required_fields"]
    assert "orca_residual_lineage_packet" in report["owner"]


def test_unknown_contract_is_clear_error(capsys) -> None:
    """An unknown contract id exits non-zero with an explanatory message."""
    exit_code = main(["does_not_exist"])
    assert exit_code == 2
    assert "unknown contract" in capsys.readouterr().err.lower()


def test_malformed_row_is_structured_cli_error(tmp_path, capsys) -> None:
    """Malformed row input exits with a structured command error."""
    row_path = tmp_path / "row.json"
    row_path.write_text("not json", encoding="utf-8")

    exit_code = main(["orca_residual_smoke", "--row", str(row_path), "--json"])

    assert exit_code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["contract_id"] == "orca_residual_smoke"
    assert "preflight evaluation failed" in report["error"]


def test_required_fields_come_from_canonical_owner() -> None:
    """The registry's required fields must be the owner's exported tuple, not a copy.

    This proves there is a single source of truth: the smoke contract's required
    fields are the owner's ``REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS``, and the three
    diagnostic members are a subset of the owner's diagnostics contract.
    """
    spec = _CONTRACT_REGISTRY["orca_residual_smoke"]
    # Identity, not just equality: same object imported from the owner module.
    assert spec.required_fields is REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS

    diagnostic_members = set(spec.required_fields) - {"artifact_pointer_status"}
    assert diagnostic_members.issubset(REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS)
