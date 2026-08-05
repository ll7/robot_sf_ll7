"""Tests for the reviewed PPO backend continual-adaptation manifest builder (issue #6658).

The builder is metadata-only: it wires ``scripts/training/train_ppo.py --config``
behind the merged continual-adaptation protocol contract and validates fail-closed,
so it never emits a manifest that fails ``check_continual_adaptation_run``. It must
never launch training, write a checkpoint, mutate the safety wrapper, run an
evaluation, or promote a policy. These tests cover the valid default manifest, the
distinct derived identifier, the shipped example manifest, and every fail-closed case
reachable through the spec.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
    CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
    PROTOCOL_STATUS_VALID,
    ContinualAdaptationProtocolError,
    check_continual_adaptation_run,
    load_continual_adaptation_run,
)
from robot_sf.research.ppo_continual_adaptation_manifest import (
    DEFAULT_PPO_BASELINE_IDENTIFIER,
    DEFAULT_SAFETY_WRAPPER_IDENTIFIER,
    PPO_TRAINING_ENTRY_POINT,
    PPOContinualAdaptationSpec,
    ShiftSpec,
    ThresholdSpec,
    build_ppo_continual_adaptation_manifest,
    derive_ppo_adapted_policy_identifier,
    sha256_checksum,
    write_ppo_continual_adaptation_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST_PATH = (
    REPO_ROOT / "configs" / "training" / "continual_adaptation_run_ppo_issue_6658.yaml"
)


def test_default_manifest_is_valid_and_not_promotable() -> None:
    """The reviewed PPO backend default manifest validates but is not promotion-ready."""
    manifest = build_ppo_continual_adaptation_manifest()
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.blockers == []
    assert report.promotion_decision == "experimental"
    assert report.promotion_ready is False
    assert report.evidence_boundary == CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY
    assert report.baseline_policy_identifier == DEFAULT_PPO_BASELINE_IDENTIFIER
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier
    assert report.safety_wrapper_mutation_permitted is False
    assert report.experience_budget_bounded is True
    assert report.adaptation_evaluation_disjoint is True


def test_default_manifest_is_metadata_only_and_wired_to_ppo_backend() -> None:
    """The built manifest declares the schema, the PPO entry point, and immutability."""
    manifest = build_ppo_continual_adaptation_manifest()
    assert manifest["schema_version"] == CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION
    assert manifest["issue"] == 6658
    assert PPO_TRAINING_ENTRY_POINT in manifest["claim_boundary"]
    assert "Metadata-only" in manifest["claim_boundary"]
    assert manifest["safety_wrapper"]["identifier"] == DEFAULT_SAFETY_WRAPPER_IDENTIFIER
    assert manifest["safety_wrapper"]["mutation_permitted"] is False
    assert manifest["adaptation"]["experience_budget"]["bounded"] is True


def test_derived_identifier_differs_from_baseline_and_is_deterministic() -> None:
    """The derived adapted identifier differs from the baseline and is reproducible."""
    derived = derive_ppo_adapted_policy_identifier()
    assert derived != DEFAULT_PPO_BASELINE_IDENTIFIER
    assert derived.startswith(f"{DEFAULT_PPO_BASELINE_IDENTIFIER}#continual-adaptation@")
    assert derived == derive_ppo_adapted_policy_identifier(PPOContinualAdaptationSpec())


def test_derived_identifier_reflects_adaptation_recipe() -> None:
    """Changing the adaptation budget changes the derived adapted-policy identifier."""
    base = derive_ppo_adapted_policy_identifier()
    changed = derive_ppo_adapted_policy_identifier(
        replace(PPOContinualAdaptationSpec(), budget_steps=999999)
    )
    assert changed != base


def test_builder_fails_closed_on_parameter_prefix_overlapping_safety_wrapper() -> None:
    """A mutable prefix overlapping the immutable safety wrapper fails closed."""
    spec = replace(
        PPOContinualAdaptationSpec(),
        allowed_parameters=("robot_sf.robot.safety_wrapper.",),
    )
    with pytest.raises(ContinualAdaptationProtocolError, match="safety wrapper"):
        build_ppo_continual_adaptation_manifest(spec)


def test_builder_fails_closed_on_overlapping_scenarios() -> None:
    """Overlapping adaptation and evaluation scenario IDs fail closed."""
    spec = replace(
        PPOContinualAdaptationSpec(),
        evaluation_scenarios=("ppo_friction_low_train_a", "ppo_friction_low_eval_holdout_b"),
    )
    with pytest.raises(ContinualAdaptationProtocolError, match="disjoint"):
        build_ppo_continual_adaptation_manifest(spec)


def test_builder_fails_closed_on_nonpositive_budget() -> None:
    """A bounded budget with a non-positive step count fails closed."""
    spec = replace(PPOContinualAdaptationSpec(), budget_steps=0)
    with pytest.raises(ContinualAdaptationProtocolError, match="positive integer"):
        build_ppo_continual_adaptation_manifest(spec)


def test_builder_fails_closed_on_non_finite_threshold_bound() -> None:
    """A non-finite acceptance threshold bound fails closed."""
    spec = replace(
        PPOContinualAdaptationSpec(),
        nominal_threshold=ThresholdSpec("success_rate_delta", math.nan, "at_most"),
    )
    with pytest.raises(ContinualAdaptationProtocolError, match="finite"):
        build_ppo_continual_adaptation_manifest(spec)


def test_builder_fails_closed_on_truncated_baseline_checksum() -> None:
    """A baseline checksum without the full canonical digest length fails closed."""
    spec = replace(
        PPOContinualAdaptationSpec(),
        baseline_checksum={"algorithm": "sha256", "digest": "deadbeef"},
    )
    with pytest.raises(ContinualAdaptationProtocolError):
        build_ppo_continual_adaptation_manifest(spec)


def test_builder_fails_closed_on_unsupported_shift_kind() -> None:
    """A shift kind outside the schema enum fails closed at schema validation."""
    spec = replace(
        PPOContinualAdaptationSpec(),
        shifts=(ShiftSpec(id="x", kind="teleport", description="not a real shift"),),
    )
    with pytest.raises(ContinualAdaptationProtocolError):
        build_ppo_continual_adaptation_manifest(spec)


def test_write_roundtrip_loads_and_validates(tmp_path: Path) -> None:
    """A written manifest loads from disk and validates as a valid experimental run."""
    out = write_ppo_continual_adaptation_manifest(tmp_path / "manifest.yaml")
    loaded = load_continual_adaptation_run(out)
    report = check_continual_adaptation_run(loaded, source=out)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.promotion_decision == "experimental"
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier


def test_shipped_example_manifest_is_valid_experimental_and_not_promotable() -> None:
    """The shipped PPO backend example manifest validates as 'experimental'."""
    manifest = load_continual_adaptation_run(EXAMPLE_MANIFEST_PATH)
    report = check_continual_adaptation_run(manifest, source=EXAMPLE_MANIFEST_PATH)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.blockers == []
    assert report.promotion_decision == "experimental"
    assert report.promotion_ready is False
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier
    # The shipped example must never start in a promotable state.
    assert manifest["promotion_decision"]["decision"] in {"reject", "experimental"}


def test_shipped_example_manifest_is_coherent_with_default_spec() -> None:
    """The shipped example and the builder default describe the same PPO backend."""
    example = yaml.safe_load(EXAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    built = build_ppo_continual_adaptation_manifest()
    assert example["schema_version"] == CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION
    assert example["baseline_policy"]["identifier"] == built["baseline_policy"]["identifier"]
    assert example["safety_wrapper"]["identifier"] == built["safety_wrapper"]["identifier"]
    assert example["adaptation"]["allowed_parameters"] == built["adaptation"]["allowed_parameters"]


def test_sha256_checksum_matches_hashlib_and_validates(tmp_path: Path) -> None:
    """sha256_checksum computes the real digest and yields a valid baseline checksum."""
    artifact = tmp_path / "baseline.pt"
    artifact.write_bytes(b"reviewed-ppo-baseline-bytes")
    checksum = sha256_checksum(artifact)
    assert checksum["algorithm"] == "sha256"
    assert checksum["digest"] == hashlib.sha256(b"reviewed-ppo-baseline-bytes").hexdigest()

    spec = replace(PPOContinualAdaptationSpec(), baseline_checksum=checksum)
    report = check_continual_adaptation_run(build_ppo_continual_adaptation_manifest(spec))
    assert report.protocol_status == PROTOCOL_STATUS_VALID


def test_sha256_checksum_missing_file_raises(tmp_path: Path) -> None:
    """Checksumming a missing artifact fails closed with FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        sha256_checksum(tmp_path / "absent.pt")


def test_builder_does_not_mutate_spec() -> None:
    """Building a manifest leaves the frozen spec unchanged."""
    spec = PPOContinualAdaptationSpec()
    snapshot_allowed = tuple(spec.allowed_parameters)
    snapshot_scenarios = tuple(spec.adaptation_scenarios)
    build_ppo_continual_adaptation_manifest(spec)
    assert spec.allowed_parameters == snapshot_allowed
    assert spec.adaptation_scenarios == snapshot_scenarios
