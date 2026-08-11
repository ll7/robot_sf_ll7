"""Integration tests: campaign evidence satisfies the protocol promotion gate (issue #6657).

Verifies that the benchmark campaign integration produces evidence bundles
that the merged validator accepts for promotion, and that fallback/degraded
execution fails closed at both the campaign and protocol layers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.continual_adaptation_campaign import (
    ContinualAdaptationCampaignError,
    ContinualAdaptationEvidenceBundle,
    prepare_promotion_manifest,
    validate_promotion_readiness,
)
from robot_sf.benchmark.continual_adaptation_campaign import (
    build_continual_adaptation_evidence as _build_continual_adaptation_evidence,
)
from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
    PROTOCOL_STATUS_VALID,
    check_continual_adaptation_run,
    derive_adapted_policy_identifier,
    load_continual_adaptation_run,
)
from robot_sf.research.ppo_continual_adaptation_manifest import (
    PPOContinualAdaptationSpec,
    build_ppo_continual_adaptation_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTION_FIXTURE_PATH = (
    REPO_ROOT / "configs" / "benchmark" / "continual_adaptation_promotion_fixture.yaml"
)

_BASELINE_DIGEST = "a" * 64
_WRAPPER_DIGEST = "b" * 64


def _checksum(digest: str = _BASELINE_DIGEST) -> dict:
    return {"algorithm": "sha256", "digest": digest}


def _build_evidence(
    manifest: dict,
    **kwargs: object,
) -> ContinualAdaptationEvidenceBundle:
    """Build evidence from exact deterministic integration-fixture bytes."""
    options = dict(kwargs)
    options.setdefault("nominal_content", b'{"result_type":"nominal"}\n')
    options.setdefault("shift_content", b'{"result_type":"shift"}\n')
    options.setdefault("forgetting_content", b'{"result_type":"forgetting"}\n')
    return _build_continual_adaptation_evidence(manifest, **options)


def _manifest() -> dict:
    return {
        "schema_version": CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
        "run_id": "integration_test_run",
        "issue": 6657,
        "claim_boundary": "integration test; metadata-only",
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
            "adaptation": ["train_a"],
            "evaluation": ["eval_a"],
        },
        "shifts": [
            {
                "id": "friction_low",
                "kind": "friction",
                "description": "lowered friction",
                "parameters": {"friction_coefficient": 0.4},
            }
        ],
        "thresholds": {
            "nominal": {"metric": "success_rate_delta", "bound": -0.02, "direction": "at_most"},
            "shift": {"metric": "success_rate_delta", "bound": 0.05, "direction": "at_least"},
            "forgetting": {"metric": "success_rate_delta", "bound": -0.02, "direction": "at_most"},
        },
        "promotion_decision": {"decision": "experimental", "rationale": "test"},
    }


def test_campaign_evidence_satisfies_protocol_promotion_gate() -> None:
    """End-to-end: campaign evidence wired into a manifest passes the protocol gate."""
    manifest = _manifest()
    evidence = _build_evidence(
        manifest,
        nominal_uri="runs/nominal.json",
        shift_uri="runs/shift.json",
        forgetting_uri="runs/forgetting.json",
        evidence_bundle_uri="evidence/bundle.yaml",
        evidence_bundle_identifier="evidence_integration_v1",
    )
    promoted = prepare_promotion_manifest(manifest, evidence)
    report = check_continual_adaptation_run(promoted)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.promotion_ready is True
    assert report.promotion_decision == "promote"
    assert report.blockers == []
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier


def test_evidence_bundle_names_derived_identifier_accepted_by_validator() -> None:
    """The bundle policy_identifier matches the validator-derived identifier."""
    manifest = _manifest()
    derived = derive_adapted_policy_identifier(manifest)
    evidence = _build_evidence(
        manifest,
        nominal_uri="runs/nominal.json",
        shift_uri="runs/shift.json",
        forgetting_uri="runs/forgetting.json",
        evidence_bundle_uri="evidence/bundle.yaml",
        evidence_bundle_identifier="evidence_v1",
    )
    assert evidence.evidence_bundle_ref["policy_identifier"] == derived
    promoted = prepare_promotion_manifest(manifest, evidence)
    report = check_continual_adaptation_run(promoted)
    assert report.protocol_status == PROTOCOL_STATUS_VALID


def test_fallback_execution_fails_closed_at_campaign_layer() -> None:
    """Fallback execution mode is rejected before reaching the protocol layer."""
    manifest = _manifest()
    with pytest.raises(ContinualAdaptationCampaignError, match="allowed native record"):
        _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
            execution_mode="fallback",
        )


def test_degraded_execution_fails_closed_at_campaign_layer() -> None:
    """Degraded execution mode is rejected before reaching the protocol layer."""
    manifest = _manifest()
    with pytest.raises(ContinualAdaptationCampaignError, match="allowed native record"):
        _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
            execution_mode="degraded",
        )


def test_merged_ppo_manifest_generator_connects_to_metadata_bundle() -> None:
    """The merged PPO manifest builder is a supported upstream source."""
    manifest = build_ppo_continual_adaptation_manifest(
        PPOContinualAdaptationSpec(
            run_id="continual_adaptation_campaign_issue_6657",
            issue=6657,
        )
    )
    evidence = _build_evidence(
        manifest,
        nominal_uri="runs/nominal.json",
        shift_uri="runs/shift.json",
        forgetting_uri="runs/forgetting.json",
        evidence_bundle_uri="evidence/bundle.yaml",
        evidence_bundle_identifier="evidence_v1",
    )
    promoted = prepare_promotion_manifest(manifest, evidence)
    report = check_continual_adaptation_run(promoted)

    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.promotion_ready is True
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier


def test_validate_promotion_readiness_on_valid_manifest() -> None:
    """validate_promotion_readiness confirms a correctly wired manifest."""
    manifest = _manifest()
    evidence = _build_evidence(
        manifest,
        nominal_uri="runs/nominal.json",
        shift_uri="runs/shift.json",
        forgetting_uri="runs/forgetting.json",
        evidence_bundle_uri="evidence/bundle.yaml",
        evidence_bundle_identifier="evidence_v1",
    )
    promoted = prepare_promotion_manifest(manifest, evidence)
    validation = validate_promotion_readiness(promoted)
    assert validation.is_promotion_ready
    assert validation.blockers == []
    assert validation.derived_adapted_policy_identifier != "ppo_baseline_v1"


def test_validate_promotion_readiness_on_incomplete_manifest() -> None:
    """validate_promotion_readiness reports blockers for an incomplete manifest."""
    manifest = _manifest()
    manifest["promotion_decision"] = {"decision": "promote", "rationale": "want to ship"}
    validation = validate_promotion_readiness(manifest)
    assert not validation.is_promotion_ready
    assert len(validation.blockers) > 0


def test_experimental_manifest_is_valid_but_not_promotion_ready() -> None:
    """A blocker-free experimental manifest must not be mislabeled ready."""
    validation = validate_promotion_readiness(_manifest())

    assert not validation.is_promotion_ready
    assert validation.blockers == []


def test_committed_fixture_passes_promotion_gate() -> None:
    """The committed fixture YAML passes the protocol promotion gate."""
    manifest = load_continual_adaptation_run(PROMOTION_FIXTURE_PATH)
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.promotion_decision == "promote"
    assert report.promotion_ready is True
    assert report.blockers == []


def test_committed_fixture_evidence_names_derived_identifier() -> None:
    """The committed fixture evidence_bundle names the validator-derived identifier."""
    manifest = load_continual_adaptation_run(PROMOTION_FIXTURE_PATH)
    derived = derive_adapted_policy_identifier(manifest)
    evidence_ref = manifest["results"]["evidence_bundle"]
    assert evidence_ref["policy_identifier"] == derived
    assert evidence_ref["identifier"] != manifest["baseline_policy"]["identifier"]
    assert evidence_ref["identifier"].strip()
