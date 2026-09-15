"""Focused tests for the diagnostic-only issue #9308 intervention contract."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.intervention_spec import (
    CLAIM_BOUNDARY,
    EVIDENCE_TIER,
    INTERVENTION_SPEC_ISSUE,
    INTERVENTION_SPEC_SCHEMA_VERSION,
    STOP_CONDITIONS,
    InterventionSpecValidationError,
    compute_intervention_spec_digest,
    load_intervention_spec,
    load_intervention_spec_schema,
    validate_intervention_spec,
)

if TYPE_CHECKING:
    from pathlib import Path

_BASE_COMMIT = "33823ae01264181acd0bdbff4e4d86c00244a9be"
_DIGEST = "a" * 64


def _payload() -> dict[str, object]:
    """Return a valid matched-start stage-1 payload."""

    return {
        "schema_version": INTERVENTION_SPEC_SCHEMA_VERSION,
        "spec_id": "issue-9308-visibility-001",
        "issue": INTERVENTION_SPEC_ISSUE,
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": CLAIM_BOUNDARY,
        "status": "specification_only",
        "hypothesis": {
            "mechanism": "occlusion_exposure",
            "statement": "Reduced visibility delays the planner response at the selected near miss.",
        },
        "factor": {
            "name": "visibility",
            "path": "observation.visibility",
            "unit": "category",
            "baseline": "occluded",
            "intervention": "visible",
        },
        "held_fixed": ["initial_state", "planner_id", "scenario_id", "seed"],
        "known_unfixable": ["pedestrian_response"],
        "comparison": {
            "classification": "matched_start_replay",
            "match_basis": ["initial_state", "scenario_id", "seed"],
            "required_shared_prefix_steps": 0,
            "verification_status": "not_verified",
        },
        "negative_control": {
            "id": "visibility-no-op",
            "factor_path": "observation.visibility",
            "value": "occluded",
            "expected": "no_factor_activation",
            "rationale": "A no-op arm checks that the harness does not activate the selected factor.",
        },
        "stop_rule": {
            "action": "stop",
            "outcome": "not_available",
            "conditions": list(STOP_CONDITIONS),
            "no_substitution": True,
        },
        "provenance": {
            "claim_boundary": CLAIM_BOUNDARY,
            "execution_status": "not_executed",
            "source_identity": {
                "scenario_id": "classic_doorway_medium",
                "planner_id": "ppo",
                "episode_id": "classic_doorway_medium--113--fixture",
                "seed": 113,
                "source_kind": "existing_diagnostic_trace_or_dossier",
                "source_refs": [
                    {
                        "path": "docs/case_workbench.md",
                        "sha256": _DIGEST,
                        "role": "case_dossier",
                    }
                ],
            },
            "config_identity": {
                "config_id": "issue-9308-local-stage-1",
                "path": "configs/analysis/case_workbench.v1.yaml",
                "sha256": _DIGEST,
            },
            "contract_identity": {
                "owner": "robot_sf.benchmark.intervention_spec",
                "schema_version": INTERVENTION_SPEC_SCHEMA_VERSION,
                "base_commit": _BASE_COMMIT,
            },
        },
    }


def test_schema_loads_and_valid_payload_is_normalized() -> None:
    """The schema is valid and declared sets have deterministic ordering."""

    schema = load_intervention_spec_schema()
    assert schema["properties"]["schema_version"]["const"] == INTERVENTION_SPEC_SCHEMA_VERSION

    payload = _payload()
    payload["held_fixed"] = ["seed", "initial_state", "scenario_id", "planner_id"]
    normalized = validate_intervention_spec(payload)

    assert normalized["held_fixed"] == ["initial_state", "planner_id", "scenario_id", "seed"]
    assert normalized["stop_rule"]["conditions"] == sorted(STOP_CONDITIONS)
    assert payload["held_fixed"] == ["seed", "initial_state", "scenario_id", "planner_id"]


def test_digest_is_stable_for_mapping_order() -> None:
    """Canonical digesting must not depend on YAML/dict insertion order."""

    first = _payload()
    second = json.loads(json.dumps(first, sort_keys=True))
    assert compute_intervention_spec_digest(first) == compute_intervention_spec_digest(second)


def test_yaml_loader_is_validation_only(tmp_path: Path) -> None:
    """Loading a spec parses and validates it without invoking any execution path."""

    path = tmp_path / "intervention.yaml"
    path.write_text(yaml.safe_dump(_payload(), sort_keys=False), encoding="utf-8")

    loaded = load_intervention_spec(path)
    assert loaded["status"] == "specification_only"
    assert loaded["provenance"]["execution_status"] == "not_executed"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_loader_rejects_duplicate_mapping_keys(tmp_path: Path, suffix: str) -> None:
    """Duplicate input keys cannot silently replace a contract field last-wins."""

    if suffix == ".json":
        text = json.dumps(_payload())
        text = text.replace(
            '"status": "specification_only"',
            '"status": "specification_only", "status": "specification_only"',
            1,
        )
    else:
        text = yaml.safe_dump(_payload(), sort_keys=False)
        text = text.replace(
            "status: specification_only\n",
            "status: specification_only\nstatus: specification_only\n",
            1,
        )

    path = tmp_path / f"duplicate{suffix}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InterventionSpecValidationError, match="duplicate"):
        load_intervention_spec(path)


def test_recursive_yaml_alias_fails_with_validation_error(tmp_path: Path) -> None:
    """Recursive aliases must be rejected as validation errors, not leak RecursionError."""

    text = yaml.safe_dump(_payload(), sort_keys=False)
    assert "baseline: occluded\n" in text
    path = tmp_path / "recursive.yaml"
    path.write_text(
        text.replace("baseline: occluded\n", "baseline: &cycle {self: *cycle}\n", 1),
        encoding="utf-8",
    )

    with pytest.raises(InterventionSpecValidationError, match="recursive"):
        load_intervention_spec(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("factor", {"name": "visibility", "path": "observation.visibility"}),
        ("provenance", {"claim_boundary": CLAIM_BOUNDARY}),
    ],
)
def test_missing_required_sections_fail_closed(field: str, replacement: object) -> None:
    """Partial sections cannot be accepted as an implicit default."""

    payload = _payload()
    payload[field] = replacement
    with pytest.raises(InterventionSpecValidationError):
        validate_intervention_spec(payload)


def test_factor_must_change_and_must_not_be_declared_fixed() -> None:
    """The one factor cannot be a no-op or appear in a fixed/unfixable set."""

    unchanged = _payload()
    unchanged["factor"]["intervention"] = "occluded"
    with pytest.raises(InterventionSpecValidationError, match="changed value"):
        validate_intervention_spec(unchanged)

    overlap = _payload()
    overlap["held_fixed"].append("observation.visibility")
    with pytest.raises(InterventionSpecValidationError, match="factor.path"):
        validate_intervention_spec(overlap)

    missing_identity = _payload()
    missing_identity["held_fixed"] = ["scenario_id", "seed", "initial_state"]
    with pytest.raises(InterventionSpecValidationError, match="planner_id"):
        validate_intervention_spec(missing_identity)


def test_comparison_classification_carries_no_unverified_shared_prefix() -> None:
    """Shared-prefix wording is a declared design, never an observed result."""

    genuine = _payload()
    genuine["comparison"].update(
        {
            "classification": "genuine_shared_prefix",
            "required_shared_prefix_steps": 3,
        }
    )
    normalized = validate_intervention_spec(genuine)
    assert normalized["comparison"]["verification_status"] == "not_verified"

    invalid = _payload()
    invalid["comparison"]["classification"] = "genuine_shared_prefix"
    with pytest.raises(InterventionSpecValidationError, match="at least one"):
        validate_intervention_spec(invalid)


def test_negative_control_is_explicit_no_op_for_the_selected_factor() -> None:
    """A control that changes the factor or expected behavior is rejected."""

    wrong_value = _payload()
    wrong_value["negative_control"]["value"] = "visible"
    with pytest.raises(InterventionSpecValidationError, match="factor.baseline"):
        validate_intervention_spec(wrong_value)

    wrong_expectation = _payload()
    wrong_expectation["negative_control"]["expected"] = "no_outcome_change"
    with pytest.raises(InterventionSpecValidationError):
        validate_intervention_spec(wrong_expectation)


def test_stop_rule_is_fixed_and_complete() -> None:
    """The contract has no adaptive substitute for a failed guard."""

    payload = _payload()
    payload["stop_rule"]["conditions"] = list(STOP_CONDITIONS[:-1])
    with pytest.raises(InterventionSpecValidationError, match="exactly"):
        validate_intervention_spec(payload)


def test_bound_files_require_matching_hashes(tmp_path: Path) -> None:
    """Optional local binding verifies the declared config and source bytes."""

    source = tmp_path / "source.json"
    config = tmp_path / "config.yaml"
    source.write_text("source\n", encoding="utf-8")
    config.write_text("config: true\n", encoding="utf-8")

    payload = _payload()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    payload["provenance"]["source_identity"]["source_refs"] = [
        {"path": "source.json", "sha256": source_hash, "role": "mechanism_trace"}
    ]
    payload["provenance"]["config_identity"].update({"path": "config.yaml", "sha256": config_hash})

    assert validate_intervention_spec(payload, repo_root=tmp_path)["spec_id"] == payload["spec_id"]
    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(InterventionSpecValidationError, match="does not match source bytes"):
        validate_intervention_spec(payload, repo_root=tmp_path)
