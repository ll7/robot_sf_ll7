"""Metadata-only continual-adaptation manifest builder for the reviewed PPO backend.

Child 1 of #6648 (parent #6582; contract landed via merged PR #6627). The bounded
continual-adaptation protocol contract
(:mod:`robot_sf.research.continual_adaptation_protocol`) and its JSON schema define
what a trusted adaptation run must declare before it may preserve safety evidence.
This module wires the **reviewed PPO backend** (``scripts/training/train_ppo.py
--config``) behind that contract: it builds a ``continual_adaptation_run.v1``
manifest describing a bounded adaptation of the PPO expert policy and validates it
fail-closed against the merged validator, so the builder never emits a manifest
that fails ``check_continual_adaptation_run``.

It is metadata-only. It deliberately does **not** launch training, write a
checkpoint, mutate the safety wrapper, run an evaluation, or promote a policy, and
it makes no benchmark or paper claim. The adapted-policy identifier is **derived**
by the validator from the baseline identity plus the normalized adaptation manifest;
this module never pre-declares an adapted identifier that could overwrite the
baseline.

The immutable safety wrapper named here is the planner-agnostic safety wrapper
(:mod:`robot_sf.robot.safety_wrapper`, issue #3501) that post-processes the PPO
policy's commanded action. The declared mutable parameter prefixes are the policy
and value heads only and never overlap that wrapper namespace. This contract mirrors
the :mod:`robot_sf.research.scenario_prior_staging_contract` pattern.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
    PROTOCOL_STATUS_VALID,
    ContinualAdaptationProtocolError,
    check_continual_adaptation_run,
    derive_adapted_policy_identifier,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Reviewed PPO backend training entry point this manifest is wired behind.
PPO_TRAINING_ENTRY_POINT = "scripts/training/train_ppo.py"

#: Default immutable baseline checkpoint identifier for the reviewed PPO backend.
DEFAULT_PPO_BASELINE_IDENTIFIER = "ppo_ammv_baseline_v3"

#: Immutable safety wrapper the PPO backend's commanded action is post-processed by;
#: the declared mutable parameter prefixes must never overlap this namespace.
DEFAULT_SAFETY_WRAPPER_IDENTIFIER = "robot_sf.robot.safety_wrapper"

# Placeholder digests for the metadata-only contract example only; a real run pins
# the actual baseline-checkpoint and safety-wrapper content hashes. They match the
# parent #6582 example so the two contract examples stay visually consistent.
_PLACEHOLDER_BASELINE_DIGEST = "0123456789abcdef" * 4
_PLACEHOLDER_WRAPPER_DIGEST = "fedcba9876543210" * 4


@dataclass(frozen=True, slots=True)
class ShiftSpec:
    """One declared synthetic shift tested under the revalidation protocol."""

    id: str
    kind: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    """One pre-declared nominal/shift/forgetting acceptance threshold."""

    metric: str
    bound: float
    direction: str


@dataclass(frozen=True, slots=True)
class PPOContinualAdaptationSpec:
    """Metadata-only inputs for a bounded PPO continual-adaptation manifest.

    Defaults reflect the reviewed PPO backend contract example. A real run pins the
    baseline-checkpoint and safety-wrapper content checksums; the defaults carry
    documented placeholder digests so the example stays metadata-only.
    """

    run_id: str = "continual_adaptation_run_ppo_issue_6658"
    issue: int = 6658
    baseline_identifier: str = DEFAULT_PPO_BASELINE_IDENTIFIER
    baseline_checksum: Mapping[str, str] = field(
        default_factory=lambda: {
            "algorithm": "sha256",
            "digest": _PLACEHOLDER_BASELINE_DIGEST,
        }
    )
    safety_wrapper_identifier: str = DEFAULT_SAFETY_WRAPPER_IDENTIFIER
    safety_wrapper_checksum: Mapping[str, str] = field(
        default_factory=lambda: {
            "algorithm": "sha256",
            "digest": _PLACEHOLDER_WRAPPER_DIGEST,
        }
    )
    allowed_parameters: tuple[str, ...] = ("policy_net.head.", "value_net.head.")
    budget_steps: int = 200000
    budget_units: str = "gradient_steps"
    adaptation_scenarios: tuple[str, ...] = (
        "ppo_friction_low_train_a",
        "ppo_friction_low_train_b",
    )
    evaluation_scenarios: tuple[str, ...] = (
        "ppo_friction_low_eval_holdout_a",
        "ppo_friction_low_eval_holdout_b",
    )
    shifts: tuple[ShiftSpec, ...] = (
        ShiftSpec(
            id="friction_low",
            kind="friction",
            description=(
                "Lowered floor friction coefficient relative to the baseline calibration surface."
            ),
            parameters={"friction_coefficient": 0.4},
        ),
    )
    nominal_threshold: ThresholdSpec = ThresholdSpec("success_rate_delta", -0.02, "at_most")
    shift_threshold: ThresholdSpec = ThresholdSpec("success_rate_delta", 0.05, "at_least")
    forgetting_threshold: ThresholdSpec = ThresholdSpec("success_rate_delta", -0.02, "at_most")
    promotion_decision: str = "experimental"
    promotion_rationale: str = (
        "Metadata-only PPO backend contract example. Revalidation results and a new evidence "
        "bundle are not declared, so promotion is not requested; this manifest is not benchmark "
        "or paper evidence."
    )


def build_ppo_continual_adaptation_manifest(
    spec: PPOContinualAdaptationSpec | None = None,
) -> dict[str, Any]:
    """Build and fail-closed validate a PPO continual-adaptation manifest.

    Args:
        spec: Metadata-only adaptation inputs. Defaults to the reviewed PPO backend
            contract example.

    Returns:
        A ``continual_adaptation_run.v1`` manifest mapping that passes
        ``check_continual_adaptation_run`` with ``protocol_status='valid'``.

    Raises:
        ContinualAdaptationProtocolError: when the assembled manifest violates the
            schema or any fail-closed protocol invariant.
    """
    manifest = _assemble_manifest(spec or PPOContinualAdaptationSpec())
    report = check_continual_adaptation_run(manifest)
    if report.protocol_status != PROTOCOL_STATUS_VALID:
        raise ContinualAdaptationProtocolError(report.blockers)
    return manifest


def derive_ppo_adapted_policy_identifier(
    spec: PPOContinualAdaptationSpec | None = None,
) -> str:
    """Derive the adapted-policy identifier for a PPO adaptation spec.

    Pure derivation: never writes a checkpoint or promotes a policy. The result is
    guaranteed to differ from the baseline identifier.

    Args:
        spec: Metadata-only adaptation inputs. Defaults to the reviewed PPO backend
            contract example.

    Returns:
        The validator-derived adapted-policy identifier.

    Raises:
        ContinualAdaptationProtocolError: when the assembled manifest is invalid.
    """
    return derive_adapted_policy_identifier(build_ppo_continual_adaptation_manifest(spec))


def write_ppo_continual_adaptation_manifest(
    path: str | Path,
    spec: PPOContinualAdaptationSpec | None = None,
) -> Path:
    """Build, validate, and write a PPO continual-adaptation manifest as YAML.

    Writing the metadata-only manifest file is not a checkpoint write and does not
    launch training.

    Args:
        path: Destination YAML path.
        spec: Metadata-only adaptation inputs. Defaults to the reviewed PPO backend
            contract example.

    Returns:
        The written manifest path.

    Raises:
        ContinualAdaptationProtocolError: when the assembled manifest is invalid.
    """
    manifest = build_ppo_continual_adaptation_manifest(spec)
    out_path = Path(path)
    out_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return out_path


def sha256_checksum(path: str | Path) -> dict[str, str]:
    """Compute a ``continual_adaptation_run.v1`` sha256 checksum for a file.

    Convenience for real runs that pin the actual baseline-checkpoint or
    safety-wrapper content hash. Reading an existing file is metadata-only: it never
    launches training or writes a checkpoint.

    Args:
        path: Path to the checksummed artifact.

    Returns:
        A ``{"algorithm": "sha256", "digest": ...}`` mapping.

    Raises:
        FileNotFoundError: when the artifact does not exist.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"algorithm": "sha256", "digest": digest}


def _assemble_manifest(spec: PPOContinualAdaptationSpec) -> dict[str, Any]:
    """Return the assembled manifest mapping from a spec (no validation side effects)."""
    return {
        "schema_version": CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
        "run_id": spec.run_id,
        "issue": spec.issue,
        "claim_boundary": _claim_boundary(),
        "baseline_policy": {
            "identifier": spec.baseline_identifier,
            "checksum": dict(spec.baseline_checksum),
        },
        "safety_wrapper": {
            "identifier": spec.safety_wrapper_identifier,
            "checksum": dict(spec.safety_wrapper_checksum),
            "mutation_permitted": False,
        },
        "adaptation": {
            "allowed_parameters": list(spec.allowed_parameters),
            "experience_budget": {
                "bounded": True,
                "steps": spec.budget_steps,
                "units": spec.budget_units,
            },
        },
        "scenarios": {
            "adaptation": list(spec.adaptation_scenarios),
            "evaluation": list(spec.evaluation_scenarios),
        },
        "shifts": [
            {
                "id": shift.id,
                "kind": shift.kind,
                "description": shift.description,
                "parameters": dict(shift.parameters),
            }
            for shift in spec.shifts
        ],
        "thresholds": {
            "nominal": _threshold_dict(spec.nominal_threshold),
            "shift": _threshold_dict(spec.shift_threshold),
            "forgetting": _threshold_dict(spec.forgetting_threshold),
        },
        "promotion_decision": {
            "decision": spec.promotion_decision,
            "rationale": spec.promotion_rationale,
        },
    }


def _threshold_dict(threshold: ThresholdSpec) -> dict[str, Any]:
    """Return the schema-shaped view of one acceptance threshold."""
    return {
        "metric": threshold.metric,
        "bound": threshold.bound,
        "direction": threshold.direction,
    }


def _claim_boundary() -> str:
    """Return the metadata-only claim boundary stamped on the built manifest."""
    return (
        "Metadata-only continual-adaptation protocol contract for the reviewed PPO backend "
        f"({PPO_TRAINING_ENTRY_POINT} --config). Declares the immutable baseline identity, the "
        "immutable safety-wrapper checksum, the bounded adaptation experience budget, the "
        "declared mutable parameter prefixes, the disjoint adaptation/evaluation scenario IDs, "
        "the synthetic shift(s), and the nominal/shift/forgetting thresholds. It does not launch "
        "training, alter a checkpoint, mutate the safety wrapper, run an evaluation, or promote "
        "a policy, and it is not benchmark or paper evidence."
    )
