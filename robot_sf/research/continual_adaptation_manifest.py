"""Metadata-only continual-adaptation manifest generator for the PPO backend (issue #6656).

This module is the manifest-generation half that wires *behind* the merged
continual-adaptation protocol validator
(:mod:`robot_sf.research.continual_adaptation_protocol`) and its
``continual_adaptation_run.v1`` schema. It assembles, for the reviewed
Stable-Baselines3 PPO backend (``scripts/training/train_ppo.py``; example
baseline ``ppo_ammv_baseline_v3``; safety wrapper
``robot_sf.gym_env.safety_vel_controller``), a manifest that declares:

* **immutable baseline identity** -- the baseline checkpoint ``identifier`` plus a
  content ``checksum``;
* **immutable safety wrapper** -- the wrapper ``identifier`` plus ``checksum`` with
  ``mutation_permitted=false``;
* **literal mutable parameter prefixes** -- no wildcards or pattern syntax;
* **bounded finite experience budget** -- a positive integer step count;
* **disjoint adaptation/evaluation scenario IDs**;
* **at least one synthetic shift**;
* **finite nominal/shift/forgetting thresholds**.

The generated manifest validates with ``protocol_status='valid'`` and an initial
``promotion_decision='experimental'``. The builder always emits an
``experimental`` decision: a metadata-only helper can never legitimately declare
``promote``, because promotion requires executed nominal/shift/forgetting results
and a fresh evidence bundle that this helper does not and cannot produce.

This module is strictly metadata-only. It does **not** launch training, write a
checkpoint, mutate the safety wrapper, run an evaluation, promote a policy, or
make any benchmark or paper claim. Baseline and safety-wrapper digests are
clearly-labeled placeholders (mirroring
``configs/training/continual_adaptation_run_issue_6582.yaml``): the PPO backend
cannot supply real checksums without executing training or reading a checkpoint
that does not exist yet. A real adaptation run must replace both placeholders
with the checksums of the frozen artifacts.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
    PROMOTION_EXPERIMENTAL,
    PROTOCOL_STATUS_VALID,
    ContinualAdaptationProtocolError,
    check_continual_adaptation_run,
)

# Reviewed SB3 PPO backend identity (scripts/training/train_ppo.py).
PPO_CONTINUAL_ADAPTATION_RUN_ID = "continual_adaptation_run_issue_6655"
PPO_CONTINUAL_ADAPTATION_ISSUE = 6655
PPO_BASELINE_IDENTIFIER = "ppo_ammv_baseline_v3"
PPO_SAFETY_WRAPPER_IDENTIFIER = "robot_sf.gym_env.safety_vel_controller"

# Clearly-labeled placeholder digests (NOT real checksums). The PPO backend cannot
# supply real baseline/safety-wrapper checksums without executing training or
# reading a checkpoint that does not exist yet; issue #6656 explicitly permits
# placeholders mirroring configs/training/continual_adaptation_run_issue_6582.yaml.
# A real adaptation run MUST replace both with the sha256 of the frozen artifacts.
PPO_BASELINE_CHECKSUM_PLACEHOLDER = (
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)
PPO_SAFETY_WRAPPER_CHECKSUM_PLACEHOLDER = (
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
)

# Literal dotted mutable parameter prefixes the adaptation job may update. No
# wildcards; everything outside this list is frozen. These are the reviewed SB3
# PPO output-head namespaces (see ``robot_sf.training.ppo_diagnostics``) and are
# disjoint from the immutable safety-wrapper namespace
# (robot_sf.gym_env.safety_vel_controller).
PPO_ALLOWED_PARAMETERS = ("action_net.", "value_net.")

# Declared bounded finite experience budget. This is a pre-declared bound on a
# future adaptation job, not an executed step count.
PPO_EXPERIENCE_BUDGET_STEPS = 200000
PPO_EXPERIENCE_BUDGET_UNITS = "env_steps"

# Disjoint calibration/adaptation and held-out evaluation scenario IDs. These are
# declared placeholders for the protocol contract; a real run pins the concrete
# scenario split. Adaptation IDs must never overlap evaluation IDs.
PPO_ADAPTATION_SCENARIOS = ("ammv_friction_low_adapt_a", "ammv_friction_low_adapt_b")
PPO_EVALUATION_SCENARIOS = (
    "ammv_friction_low_eval_holdout_a",
    "ammv_friction_low_eval_holdout_b",
)

# At least one declared synthetic shift is required for revalidation.
PPO_SHIFTS: tuple[dict[str, Any], ...] = (
    {
        "id": "friction_low",
        "kind": "friction",
        "description": (
            "Lowered floor friction coefficient relative to the baseline calibration surface."
        ),
        "parameters": {"friction_coefficient": 0.4},
    },
)

# Pre-declared finite nominal/shift/forgetting acceptance thresholds.
PPO_THRESHOLDS: dict[str, dict[str, Any]] = {
    "nominal": {"metric": "success_rate_delta", "bound": -0.02, "direction": "at_most"},
    "shift": {"metric": "success_rate_delta", "bound": 0.05, "direction": "at_least"},
    "forgetting": {"metric": "success_rate_delta", "bound": -0.02, "direction": "at_most"},
}

_CLAIM_BOUNDARY = (
    "Metadata-only continual-adaptation protocol contract for the reviewed SB3 PPO backend "
    "(scripts/training/train_ppo.py; baseline ppo_ammv_baseline_v3; safety wrapper "
    "robot_sf.gym_env.safety_vel_controller). Declares the immutable baseline identity and "
    "checksum, the immutable safety-wrapper checksum with mutation_permitted=false, the literal "
    "mutable parameter prefixes, the bounded finite experience budget, the disjoint "
    "adaptation/evaluation scenario IDs, the synthetic shift(s), and the finite "
    "nominal/shift/forgetting thresholds. It does not launch training, alter a checkpoint, mutate "
    "the safety wrapper, run an evaluation, or promote a policy; the initial promotion_decision "
    "is 'experimental'. This manifest is not benchmark or paper evidence."
)

_PROMOTION_RATIONALE = (
    "Metadata-only contract for the reviewed PPO backend (issue #6656 wiring; residual child of "
    "#6655; root #6582). Baseline and safety-wrapper digests are clearly-labeled placeholders; "
    "revalidation results and a new evidence bundle are not declared, so promotion is not "
    "requested. This manifest is not benchmark or paper evidence."
)

_YAML_HEADER = """\
# Continual-adaptation and revalidation protocol manifest -- reviewed SB3 PPO backend.
#
# Generated by robot_sf.research.continual_adaptation_manifest (issue #6656 wiring; residual child
# of #6655; root #6582). It wires BEHIND the merged validator
# robot_sf.research.continual_adaptation_protocol and the continual_adaptation_run.v1 schema; it
# does not modify either. Regenerate deterministically with:
#   uv run python -c "from robot_sf.research.continual_adaptation_manifest import write_ppo_continual_adaptation_manifest; write_ppo_continual_adaptation_manifest('configs/training/continual_adaptation_run_issue_6655.yaml')"
#
# It is checked by:
#   uv run python -c "from robot_sf.research.continual_adaptation_protocol import load_continual_adaptation_run, check_continual_adaptation_run; print(check_continual_adaptation_run(load_continual_adaptation_run('configs/training/continual_adaptation_run_issue_6655.yaml')).to_dict())"
#
# This manifest neither launches training, alters a checkpoint, mutates the safety wrapper, runs an
# evaluation, nor promotes a policy, and it is NOT benchmark or paper evidence. Baseline and
# safety-wrapper digests are clearly-labeled placeholders; a real adaptation run replaces both with
# the checksums of the frozen artifacts. The adapted-policy identifier is DERIVED by the validator
# from the baseline identity plus a normalized adaptation manifest; it is never pre-declared here.
"""


def build_ppo_continual_adaptation_manifest(
    *,
    run_id: str = PPO_CONTINUAL_ADAPTATION_RUN_ID,
    issue: int = PPO_CONTINUAL_ADAPTATION_ISSUE,
    baseline_checksum_digest: str = PPO_BASELINE_CHECKSUM_PLACEHOLDER,
    safety_wrapper_checksum_digest: str = PPO_SAFETY_WRAPPER_CHECKSUM_PLACEHOLDER,
) -> dict[str, Any]:
    """Assemble the metadata-only continual-adaptation manifest for the PPO backend.

    The returned manifest validates with ``protocol_status='valid'`` and an initial
    ``promotion_decision='experimental'``. The adaptation recipe (mutable parameter
    prefixes, bounded budget, scenario split, synthetic shift, thresholds) is fixed
    to the reviewed backend's declared contract; only the run identity and the
    baseline/safety-wrapper digests are parameterized so a future real run can pin
    the true checksums without changing the recipe.

    This is a pure assembly: it does not launch training, write a checkpoint, mutate
    the safety wrapper, run an evaluation, or promote a policy.

    Args:
        run_id: Immutable run identifier stamped on the manifest.
        issue: Issue number stamped on the manifest.
        baseline_checksum_digest: sha256 hex digest of the frozen baseline
            checkpoint. Defaults to a clearly-labeled placeholder.
        safety_wrapper_checksum_digest: sha256 hex digest of the frozen safety
            wrapper. Defaults to a clearly-labeled placeholder.

    Returns:
        A ``continual_adaptation_run.v1`` manifest mapping.
    """
    return {
        "schema_version": CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "issue": issue,
        "claim_boundary": _CLAIM_BOUNDARY,
        "baseline_policy": {
            "identifier": PPO_BASELINE_IDENTIFIER,
            "checksum": {"algorithm": "sha256", "digest": baseline_checksum_digest},
        },
        "safety_wrapper": {
            "identifier": PPO_SAFETY_WRAPPER_IDENTIFIER,
            "checksum": {"algorithm": "sha256", "digest": safety_wrapper_checksum_digest},
            "mutation_permitted": False,
        },
        "adaptation": {
            "allowed_parameters": list(PPO_ALLOWED_PARAMETERS),
            "experience_budget": {
                "bounded": True,
                "steps": PPO_EXPERIENCE_BUDGET_STEPS,
                "units": PPO_EXPERIENCE_BUDGET_UNITS,
            },
        },
        "scenarios": {
            "adaptation": list(PPO_ADAPTATION_SCENARIOS),
            "evaluation": list(PPO_EVALUATION_SCENARIOS),
        },
        "shifts": copy.deepcopy(list(PPO_SHIFTS)),
        "thresholds": copy.deepcopy(PPO_THRESHOLDS),
        "promotion_decision": {
            "decision": PROMOTION_EXPERIMENTAL,
            "rationale": _PROMOTION_RATIONALE,
        },
    }


def write_ppo_continual_adaptation_manifest(
    path: str | Path,
    *,
    run_id: str = PPO_CONTINUAL_ADAPTATION_RUN_ID,
    issue: int = PPO_CONTINUAL_ADAPTATION_ISSUE,
    baseline_checksum_digest: str = PPO_BASELINE_CHECKSUM_PLACEHOLDER,
    safety_wrapper_checksum_digest: str = PPO_SAFETY_WRAPPER_CHECKSUM_PLACEHOLDER,
) -> Path:
    """Write the PPO continual-adaptation manifest to ``path`` as documented YAML.

    The assembled manifest is validated against the merged protocol before writing;
    an invalid manifest fails closed and nothing is written. Writing is metadata
    only: it does not launch training, write a checkpoint, mutate the safety
    wrapper, run an evaluation, or promote a policy.

    Args:
        path: Destination YAML manifest path.
        run_id: Immutable run identifier stamped on the manifest.
        issue: Issue number stamped on the manifest.
        baseline_checksum_digest: sha256 hex digest of the frozen baseline
            checkpoint. Defaults to a clearly-labeled placeholder.
        safety_wrapper_checksum_digest: sha256 hex digest of the frozen safety
            wrapper. Defaults to a clearly-labeled placeholder.

    Returns:
        The written manifest path.

    Raises:
        ContinualAdaptationProtocolError: when the assembled manifest fails schema
            or semantic protocol validation (nothing is written).
    """
    manifest = build_ppo_continual_adaptation_manifest(
        run_id=run_id,
        issue=issue,
        baseline_checksum_digest=baseline_checksum_digest,
        safety_wrapper_checksum_digest=safety_wrapper_checksum_digest,
    )
    report = check_continual_adaptation_run(manifest, source=path)
    if report.protocol_status != PROTOCOL_STATUS_VALID:
        raise ContinualAdaptationProtocolError(report.blockers, source=path)
    out_path = Path(path)
    out_path.write_text(_YAML_HEADER + yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return out_path
