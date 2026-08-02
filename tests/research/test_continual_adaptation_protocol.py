"""Tests for the bounded continual-adaptation protocol manifest validator (issue #6582).

The validator is a metadata-only contract. It must never launch training, alter a
checkpoint, mutate the safety wrapper, or promote a policy. These tests cover one
valid experimental manifest, the promotion-gated promote path, deterministic
identifier derivation, and every fail-closed case named in the issue.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
    CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
    PROTOCOL_STATUS_INVALID,
    PROTOCOL_STATUS_VALID,
    ContinualAdaptationProtocolError,
    check_continual_adaptation_run,
    derive_adapted_policy_identifier,
    load_continual_adaptation_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST_PATH = (
    REPO_ROOT / "configs" / "training" / "continual_adaptation_run_issue_6582.yaml"
)

_BASELINE_DIGEST = "a" * 64
_WRAPPER_DIGEST = "b" * 64
_RESULT_DIGEST = "c" * 64


def _checksum(digest: str = _BASELINE_DIGEST) -> dict:
    return {"algorithm": "sha256", "digest": digest}


def _threshold(bound: float, direction: str = "at_most") -> dict:
    return {"metric": "success_rate_delta", "bound": bound, "direction": direction}


def _result_ref(uri: str = "runs/nominal.json") -> dict:
    return {"uri": uri, "checksum": _checksum(_RESULT_DIGEST)}


def _evidence_ref() -> dict:
    return {
        "identifier": "evidence_adapted_v1",
        "uri": "evidence/adapted_v1.json",
        "checksum": _checksum(_RESULT_DIGEST),
    }


def _results() -> dict:
    return {
        "nominal_result": _result_ref("runs/nominal.json"),
        "shift_result": _result_ref("runs/shift.json"),
        "forgetting_result": _result_ref("runs/forgetting.json"),
        "evidence_bundle": _evidence_ref(),
    }


def _manifest(**overrides: object) -> dict:
    """Return a minimal valid manifest with promotion_decision 'experimental'."""
    manifest = {
        "schema_version": CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
        "run_id": "test_run",
        "issue": 6582,
        "claim_boundary": "metadata-only protocol contract; no execution or promotion",
        "baseline_policy": {
            "identifier": "ppo_baseline_v1",
            "checksum": _checksum(),
        },
        "safety_wrapper": {
            "identifier": "robot_sf.gym_env.safety_wrapper",
            "checksum": _checksum(_WRAPPER_DIGEST),
            "mutation_permitted": False,
        },
        "adaptation": {
            "allowed_parameters": ["policy_net.head."],
            "experience_budget": {
                "bounded": True,
                "steps": 100000,
                "units": "gradient_steps",
            },
        },
        "scenarios": {
            "adaptation": ["train_a", "train_b"],
            "evaluation": ["eval_a", "eval_b"],
        },
        "shifts": [
            {
                "id": "friction_low",
                "kind": "friction",
                "description": "lowered floor friction",
                "parameters": {"friction_coefficient": 0.4},
            }
        ],
        "thresholds": {
            "nominal": _threshold(-0.02, "at_most"),
            "shift": _threshold(0.05, "at_least"),
            "forgetting": _threshold(-0.02, "at_most"),
        },
        "promotion_decision": {
            "decision": "experimental",
            "rationale": "contract example only",
        },
    }
    manifest.update(overrides)
    return manifest


def test_valid_experimental_manifest_is_valid_and_not_promotable() -> None:
    """A well-formed experimental manifest is valid but not promotion-ready."""
    report = check_continual_adaptation_run(_manifest())
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.blockers == []
    assert report.promotion_decision == "experimental"
    assert report.promotion_ready is False
    assert report.evidence_boundary == CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier
    assert report.safety_wrapper_mutation_permitted is False
    assert report.experience_budget_bounded is True
    assert report.adaptation_evaluation_disjoint is True


def test_promote_with_complete_results_is_promotion_ready() -> None:
    """A 'promote' decision with all result/evidence references is promotion-ready."""
    manifest = _manifest()
    manifest["promotion_decision"] = {"decision": "promote", "rationale": "all gates passed"}
    manifest["results"] = _results()
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.promotion_decision == "promote"
    assert report.promotion_ready is True
    assert report.blockers == []


def test_promote_without_results_block_fails_closed() -> None:
    """promotion_decision='promote' with no results block fails closed."""
    manifest = _manifest()
    manifest["promotion_decision"] = {"decision": "promote", "rationale": "want to ship"}
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert report.promotion_ready is False
    assert any("no results block" in blocker for blocker in report.blockers)


def test_promote_with_incomplete_results_fails_closed() -> None:
    """promotion_decision='promote' missing one required reference fails closed."""
    manifest = _manifest()
    manifest["promotion_decision"] = {"decision": "promote", "rationale": "want to ship"}
    incomplete = _results()
    del incomplete["forgetting_result"]
    manifest["results"] = incomplete
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert report.promotion_ready is False
    assert any("forgetting_result" in blocker for blocker in report.blockers)


def test_promote_with_empty_reference_fails_closed() -> None:
    """A result reference missing its checksum fails closed even under 'promote'."""
    manifest = _manifest()
    manifest["promotion_decision"] = {"decision": "promote", "rationale": "want to ship"}
    incomplete = _results()
    incomplete["shift_result"] = {"uri": "runs/shift.json"}  # no checksum
    manifest["results"] = incomplete
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert any("shift_result" in blocker for blocker in report.blockers)


def test_promote_with_unsupported_result_checksum_fails_closed() -> None:
    """A promotion result must use a supported checksum with a hexadecimal digest."""
    manifest = _manifest()
    manifest["promotion_decision"] = {"decision": "promote", "rationale": "want to ship"}
    incomplete = _results()
    incomplete["nominal_result"]["checksum"] = {"algorithm": "md5", "digest": "not-a-hash"}
    manifest["results"] = incomplete
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert report.promotion_ready is False
    assert any("nominal_result" in blocker for blocker in report.blockers)


def test_promote_with_baseline_named_evidence_bundle_fails_closed() -> None:
    """A new evidence bundle must not reuse the immutable baseline identifier."""
    manifest = _manifest()
    manifest["promotion_decision"] = {"decision": "promote", "rationale": "want to ship"}
    results = _results()
    results["evidence_bundle"]["identifier"] = manifest["baseline_policy"]["identifier"]
    manifest["results"] = results
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert report.promotion_ready is False
    assert any("evidence_bundle" in blocker for blocker in report.blockers)


def test_safety_wrapper_mutation_permitted_fails_closed() -> None:
    """A manifest granting safety-wrapper mutation permission fails closed."""
    manifest = _manifest()
    manifest["safety_wrapper"]["mutation_permitted"] = True
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert report.safety_wrapper_mutation_permitted is True
    assert any("safety wrapper" in blocker for blocker in report.blockers)


def test_missing_required_baseline_hash_raises() -> None:
    """Omitting the required baseline checksum fails closed at schema validation."""
    manifest = _manifest()
    del manifest["baseline_policy"]["checksum"]
    with pytest.raises(ContinualAdaptationProtocolError):
        check_continual_adaptation_run(manifest)


def test_missing_required_safety_wrapper_hash_raises() -> None:
    """Omitting the required safety-wrapper checksum fails closed at schema validation."""
    manifest = _manifest()
    del manifest["safety_wrapper"]["checksum"]
    with pytest.raises(ContinualAdaptationProtocolError):
        check_continual_adaptation_run(manifest)


def test_empty_checksum_digest_raises() -> None:
    """A trivially empty digest fails closed at schema validation."""
    manifest = _manifest()
    manifest["baseline_policy"]["checksum"]["digest"] = "not-hex"
    with pytest.raises(ContinualAdaptationProtocolError):
        check_continual_adaptation_run(manifest)


def test_overlapping_scenario_ids_fail_closed() -> None:
    """Overlapping adaptation and evaluation scenario IDs fail closed."""
    manifest = _manifest()
    manifest["scenarios"]["evaluation"] = ["train_a", "eval_b"]
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert report.adaptation_evaluation_disjoint is False
    assert any("disjoint" in blocker and "train_a" in blocker for blocker in report.blockers)


def test_unbounded_budget_fails_closed() -> None:
    """An unbounded experience budget fails closed."""
    manifest = _manifest()
    manifest["adaptation"]["experience_budget"] = {
        "bounded": False,
        "steps": None,
        "units": "gradient_steps",
    }
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert report.experience_budget_bounded is False
    assert any("bounded" in blocker for blocker in report.blockers)


def test_bounded_budget_with_null_steps_fails_closed() -> None:
    """A budget claiming 'bounded' with null steps still fails closed."""
    manifest = _manifest()
    manifest["adaptation"]["experience_budget"]["steps"] = None
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert report.experience_budget_bounded is False
    assert any("steps" in blocker for blocker in report.blockers)


def test_bounded_budget_with_nonpositive_steps_fails_closed() -> None:
    """A bounded budget with a non-positive step count fails closed."""
    manifest = _manifest()
    manifest["adaptation"]["experience_budget"]["steps"] = 0
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_INVALID
    assert any("positive integer" in blocker for blocker in report.blockers)


def test_derived_identifier_is_deterministic() -> None:
    """The same manifest always derives the same adapted-policy identifier."""
    manifest = _manifest()
    first = derive_adapted_policy_identifier(manifest)
    second = derive_adapted_policy_identifier(copy.deepcopy(manifest))
    assert first == second
    assert first != manifest["baseline_policy"]["identifier"]


def test_derived_identifier_reflects_adaptation_manifest() -> None:
    """Changing the adaptation recipe changes the derived identifier."""
    manifest = _manifest()
    base_derived = derive_adapted_policy_identifier(manifest)
    changed = copy.deepcopy(manifest)
    changed["adaptation"]["experience_budget"]["steps"] = 999999
    assert derive_adapted_policy_identifier(changed) != base_derived


def test_derived_identifier_never_equals_baseline() -> None:
    """The derived identifier is guaranteed to differ from the baseline identifier."""
    for baseline in [
        "ppo_baseline_v1",
        "a",
        "ppo_baseline_v1#continual-adaptation@sha256:deadbeef",
    ]:
        manifest = _manifest()
        manifest["baseline_policy"]["identifier"] = baseline
        report = check_continual_adaptation_run(manifest)
        assert report.derived_adapted_policy_identifier != baseline
        assert report.derived_adapted_policy_identifier.startswith(
            f"{baseline}#continual-adaptation@"
        )


def test_schema_violation_raises() -> None:
    """An invalid payload (bad enum) raises a schema error."""
    bad = _manifest()
    bad["promotion_decision"]["decision"] = "ship-it"
    with pytest.raises(ContinualAdaptationProtocolError):
        check_continual_adaptation_run(bad)


def test_missing_top_level_required_field_raises() -> None:
    """A manifest missing a required top-level block raises a schema error."""
    bad = _manifest()
    del bad["thresholds"]
    with pytest.raises(ContinualAdaptationProtocolError):
        check_continual_adaptation_run(bad)


def test_report_to_dict_is_json_serializable() -> None:
    """The report serializes to JSON (CLI contract)."""
    report = check_continual_adaptation_run(_manifest())
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["schema_version"] == CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION
    assert payload["protocol_status"] == PROTOCOL_STATUS_VALID
    assert payload["evidence_boundary"] == CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent manifest path raises an actionable error."""
    with pytest.raises(ContinualAdaptationProtocolError):
        load_continual_adaptation_run(tmp_path / "nope.yaml")


def test_load_roundtrip(tmp_path: Path) -> None:
    """A manifest written to disk loads and validates."""
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_manifest(), sort_keys=False), encoding="utf-8")
    loaded = load_continual_adaptation_run(path)
    assert loaded["run_id"] == "test_run"


def test_repo_example_manifest_is_valid_experimental_and_not_promotable() -> None:
    """The shipped #6582 example manifest validates as 'experimental', not 'promote'."""
    manifest = load_continual_adaptation_run(EXAMPLE_MANIFEST_PATH)
    report = check_continual_adaptation_run(manifest, source=EXAMPLE_MANIFEST_PATH)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.blockers == []
    assert report.promotion_decision == "experimental"
    assert report.promotion_ready is False
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier
    # The shipped example must never start in a promotable state.
    assert manifest["promotion_decision"]["decision"] in {"reject", "experimental"}


def test_repo_example_manifest_path_matches() -> None:
    """The example manifest ships at the documented path with the right schema."""
    assert EXAMPLE_MANIFEST_PATH.is_file()
    payload = yaml.safe_load(EXAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION


def test_copy_independence() -> None:
    """The checker does not mutate the input manifest mapping."""
    manifest = _manifest()
    snapshot = copy.deepcopy(manifest)
    check_continual_adaptation_run(manifest)
    assert manifest == snapshot


def test_invalid_promotion_decision_in_results_only_path() -> None:
    """A 'reject' decision with results present is valid but not promotion-ready."""
    manifest = _manifest()
    manifest["promotion_decision"] = {"decision": "reject", "rationale": "no improvement"}
    manifest["results"] = _results()
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.promotion_decision == "reject"
    assert report.promotion_ready is False
