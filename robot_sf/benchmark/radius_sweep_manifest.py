"""Pre-submission radius-sweep manifest builder for issue #6642 (Gate 2 of #6600).

This module enumerates the planned collision-envelope radius sensitivity sweep
(0.5 / 0.8 / 1.0 m) over the complete 14-planner release roster and the 48-cell
``classic_interactions_francis2023`` matrix. It is strictly a dry-run
pre-submission manifest: it does NOT submit, run, or authorize any production
SLURM compute and does NOT promote benchmark evidence. It may carry an
independently verified Gate 1 runtime-binding receipt, but production compute
stays blocked until the remaining Gate 2 campaign gates pass.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RADIUS_SWEEP_MANIFEST_SCHEMA = "issue-6642-radius-sweep-manifest.v1"
RADIUS_SWEEP_MANIFEST_CHECK_SCHEMA = "issue-6642-radius-sweep-manifest-check.v1"

# Treatment constants: the three production radii, with 1.0 m the release baseline.
PRODUCTION_RADII: tuple[float, ...] = (0.5, 0.8, 1.0)
PRODUCTION_RADIUS_KEYS: tuple[str, ...] = ("r0p5", "r0p8", "r1p0")
BASELINE_RADIUS: float = 1.0
ISSUE_6642 = 6642
PARENT_ISSUE_6600 = 6600
GATE1_CANARY_ISSUE = 6641

# The release roster is frozen to the 0.0.3.post1 baseline; the checker rejects a
# sweep whose arm config does not reproduce this exact 14-key roster in order.
RELEASE_PLANNER_KEYS: tuple[str, ...] = (
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
)

# The scenario matrix is a fixed release surface, not an arbitrary 48-entry
# placeholder.  Keep the canonical expanded roster here so a serialized
# manifest cannot replace a cell with a different scenario while preserving
# only the count.
EXPECTED_SCENARIO_MATRIX = "configs/scenarios/classic_interactions_francis2023.yaml"
EXPECTED_RELEASE_BASELINE_CONFIG = (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended_post1.yaml"
)
EXPECTED_ARM_CAMPAIGN_CONFIG = "configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml"
EXPECTED_ARM_CAMPAIGN_CONFIG_0P5M = "configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml"
EXPECTED_ARM_CAMPAIGN_CONFIG_0P8M = "configs/benchmarks/issue_6642_radius_sweep_arm_0p8m.yaml"
EXPECTED_MANIFEST_CONFIG = "configs/benchmarks/issue_6642_radius_sweep_manifest_v1.yaml"
EXPECTED_ARM_RELEASE_TAG = "issue-6642-radius-sweep-1p0m"
EXPECTED_ARM_RELEASE_TAG_0P5M = "issue-6642-radius-sweep-0p5m"
EXPECTED_ARM_RELEASE_TAG_0P8M = "issue-6642-radius-sweep-0p8m"

# Manifest-config key and frozen campaign identity per radius arm key. Every arm
# has its own tracked campaign config and issue-scoped release tag; the builder
# resolves all three and the checker rejects any drift on either surface.
ARM_CONFIG_KEYS: dict[str, str] = {
    "r0p5": "arm_campaign_config_0p5m",
    "r0p8": "arm_campaign_config_0p8m",
    "r1p0": "arm_campaign_config_1p0m",
}
EXPECTED_ARM_CAMPAIGN_CONFIGS: dict[str, str] = {
    "r0p5": EXPECTED_ARM_CAMPAIGN_CONFIG_0P5M,
    "r0p8": EXPECTED_ARM_CAMPAIGN_CONFIG_0P8M,
    "r1p0": EXPECTED_ARM_CAMPAIGN_CONFIG,
}
EXPECTED_ARM_RELEASE_TAGS: dict[str, str] = {
    "r0p5": EXPECTED_ARM_RELEASE_TAG_0P5M,
    "r0p8": EXPECTED_ARM_RELEASE_TAG_0P8M,
    "r1p0": EXPECTED_ARM_RELEASE_TAG,
}
EXPECTED_SCENARIO_NAMES: tuple[str, ...] = tuple(
    sorted(
        (
            "classic_bottleneck_low",
            "classic_bottleneck_medium",
            "classic_bottleneck_high",
            "classic_realworld_double_bottleneck_high",
            "classic_station_platform_medium",
            "classic_cross_trap_low",
            "classic_cross_trap_medium",
            "classic_cross_trap_high",
            "classic_doorway_low",
            "classic_doorway_medium",
            "classic_doorway_high",
            "classic_group_crossing_low",
            "classic_group_crossing_medium",
            "classic_group_crossing_high",
            "classic_head_on_corridor_low",
            "classic_head_on_corridor_medium",
            "classic_merging_low",
            "classic_merging_medium",
            "classic_overtaking_low",
            "classic_overtaking_medium",
            "classic_t_intersection_low",
            "classic_t_intersection_medium",
            "classic_urban_crossing_medium",
            "francis2023_frontal_approach",
            "francis2023_pedestrian_obstruction",
            "francis2023_pedestrian_overtaking",
            "francis2023_robot_overtaking",
            "francis2023_down_path",
            "francis2023_intersection_no_gesture",
            "francis2023_blind_corner",
            "francis2023_narrow_hallway",
            "francis2023_narrow_doorway",
            "francis2023_entering_room",
            "francis2023_exiting_room",
            "francis2023_entering_elevator",
            "francis2023_exiting_elevator",
            "francis2023_intersection_wait",
            "francis2023_intersection_proceed",
            "francis2023_following_human",
            "francis2023_leading_human",
            "francis2023_accompanying_peer",
            "francis2023_join_group",
            "francis2023_leave_group",
            "francis2023_crowd_navigation",
            "francis2023_parallel_traffic",
            "francis2023_perpendicular_traffic",
            "francis2023_circular_crossing",
            "francis2023_robot_crowding",
        )
    )
)
EXPECTED_SCENARIO_COUNT = len(EXPECTED_SCENARIO_NAMES)
EXPECTED_SEED_SET = "paper_eval_s30"
EXPECTED_SEED_RANGE: tuple[int, int] = (111, 140)
EXPECTED_SEEDS: tuple[int, ...] = tuple(range(EXPECTED_SEED_RANGE[0], EXPECTED_SEED_RANGE[1] + 1))
EXPECTED_HORIZON = 600
EXPECTED_DT = 0.1
EXPECTED_KINEMATICS = "differential_drive"
EXPECTED_ROWS_PER_ARM = len(RELEASE_PLANNER_KEYS) * EXPECTED_SCENARIO_COUNT * len(EXPECTED_SEEDS)
EXPECTED_TOTAL_ROWS = len(PRODUCTION_RADII) * EXPECTED_ROWS_PER_ARM

# Runtime-binding and gate statuses. A pending arm is metadata only. A bound arm
# must carry the exact Gate 1 receipt and source commit that admitted the runtime
# binding. Neither status authorizes production compute.
RUNTIME_BINDING_PENDING_GATE1 = "pending_gate1_canary"
RUNTIME_BINDING_BOUND_RUNTIME = "bound_runtime"
RUNTIME_BINDING_CONTRACT_VERSION = "radius_binding_canary.v1"
RUNTIME_BINDING_STATUSES: tuple[str, ...] = (
    RUNTIME_BINDING_PENDING_GATE1,
    RUNTIME_BINDING_BOUND_RUNTIME,
)
GATE1_STATUS_NOT_YET_PASSED = "not_yet_passed"
GATE1_STATUS_PASSED = "passed"
GATE1_STATUSES: tuple[str, ...] = (GATE1_STATUS_NOT_YET_PASSED, GATE1_STATUS_PASSED)
IMMUTABLE_COMMIT_POLICY = "pinned_at_launch_across_all_arms"
ROW_IDENTITY_CONTRACT = "complete_row_identities_or_explicit_fail_closed_missingness_ledger"
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# Evidence-exclusion row classes (never counted as evidence per #6600 stop rules).
EVIDENCE_EXCLUSIONS: tuple[str, ...] = (
    "unavailable",
    "degraded",
    "fallback",
    "failed",
    "missing",
    "duplicate",
    "provenance_invalid",
)

MANIFEST_STATUS = "preparation_manifest_only"
MANIFEST_CHECK_STATUS = "manifest_check_only"
EVIDENCE_STATUS = "not_benchmark_evidence"

CLAIM_BOUNDARY = (
    "Pre-submission admission manifest only: declares the radius-sensitivity sweep "
    "treatment and fixed factors for issue #6642 (Gate 2 of #6600). Not benchmark "
    "evidence, not a realism result, not a sim-to-real result, and not a safety "
    "guarantee. Gate 1 runtime binding is admitted, but production compute remains "
    "blocked by the remaining Gate 2 campaign gates. Degraded, fallback, failed, "
    "missing, duplicate, and provenance-invalid rows are never counted as evidence."
)


class RadiusSweepManifestError(ValueError):
    """Raised when a radius-sweep manifest fails closed on a contract violation."""


@dataclass(frozen=True)
class ManifestOptions:
    """Stable metadata stamped onto a preparation manifest build."""

    config_path: str
    git_head: str = "pending_launch"
    dry_run: bool = True


@dataclass(frozen=True)
class FixedFactors:
    """Fixed factors resolved from the arm campaign config at build time."""

    scenario_matrix: str
    scenario_count: int
    scenario_names: tuple[str, ...]
    planner_keys: tuple[str, ...]
    seed_set: str
    seeds: tuple[int, ...]
    horizon: int
    dt: float
    kinematics: str
    release_tag: str


@dataclass(frozen=True)
class ArmCampaignIdentity:
    """Resolved campaign identity for one radius arm of the sweep.

    Every radius arm runs from its own tracked campaign config with its own
    issue-scoped release tag. The builder resolves all three arms and fails
    closed unless each identity matches the frozen per-arm constants and the
    manifest config's declared arm campaign config keys.
    """

    arm_key: str
    campaign_config: str
    release_tag: str
    runtime_binding_status: str = RUNTIME_BINDING_PENDING_GATE1
    binding_contract_version: str | None = None
    gate1_canary_issue: int | None = None
    gate1_receipt_sha256: str | None = None
    gate1_source_commit: str | None = None
    runtime_binding_note: str | None = None


def build_radius_sweep_manifest(
    manifest_config: Mapping[str, Any],
    *,
    fixed_factors: FixedFactors,
    arm_identities: Sequence[ArmCampaignIdentity],
    options: ManifestOptions,
) -> dict[str, Any]:
    """Build a deterministic pre-submission radius-sweep manifest.

    Args:
        manifest_config: Parsed ``issue-6642-radius-sweep-manifest.v1`` YAML mapping.
        fixed_factors: Fixed factors resolved from the baseline (1.0 m) arm config.
        arm_identities: Resolved campaign identities for all three radius arms, in
            radius order (0.5/0.8/1.0 m). Each identity must match the frozen
            per-arm campaign config and release tag constants.
        options: Stable manifest-build metadata (config path, git head).

    Returns:
        JSON-serializable pre-submission manifest payload.
    """
    if not options.dry_run:
        raise RadiusSweepManifestError(
            "radius sweep manifest builder only supports dry-run preparation manifests"
        )
    _validate_manifest_config_shape(manifest_config)
    if not isinstance(fixed_factors, FixedFactors):
        raise RadiusSweepManifestError("fixed_factors must be a FixedFactors instance")
    _validate_fixed_factors(fixed_factors)
    _validate_declared_fixed_factors(manifest_config, fixed_factors)
    radii = _normalize_radii(manifest_config.get("radii"))
    identities = _validate_arm_identities(manifest_config, arm_identities)
    arms = [
        _arm_manifest(radius_entry, fixed_factors=fixed_factors, identity=identity)
        for radius_entry, identity in zip(radii, identities, strict=True)
    ]
    expected_episode_count_per_arm = (
        len(fixed_factors.planner_keys) * fixed_factors.scenario_count * len(fixed_factors.seeds)
    )
    return {
        "schema_version": RADIUS_SWEEP_MANIFEST_SCHEMA,
        "issue": ISSUE_6642,
        "parent_issue": PARENT_ISSUE_6600,
        "status": MANIFEST_STATUS,
        "dry_run": True,
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "config_path": options.config_path,
        "git_head": options.git_head,
        "release_baseline_config": str(manifest_config.get("release_baseline_config")),
        "arm_campaign_config_1p0m": str(manifest_config.get("arm_campaign_config_1p0m")),
        "fixed_factors": _fixed_factors_manifest(fixed_factors),
        "radii": [_radius_summary(radius_entry) for radius_entry in radii],
        "arm_count": len(arms),
        "arms": arms,
        "immutable_campaign_commit": _immutable_commit_manifest(manifest_config, options),
        "gate_preconditions": _gate_preconditions_manifest(manifest_config),
        "missingness_policy": _missingness_policy_manifest(manifest_config),
        "row_identity_ledger_template": {
            "dimensions": ("radius_arm", "planner_key", "scenario_name", "seed"),
            "expected_rows_per_arm": expected_episode_count_per_arm,
            "expected_total_rows": expected_episode_count_per_arm * len(arms),
            "completeness": "template_only_no_episodes_run",
            "fail_closed_contract": ROW_IDENTITY_CONTRACT,
        },
    }


def check_radius_sweep_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a preparation manifest against the issue #6642 contract.

    Args:
        manifest: Manifest payload produced by :func:`build_radius_sweep_manifest`.

    Returns:
        JSON-serializable checker payload with ``violations`` and ``passes``.
    """
    if not isinstance(manifest, Mapping):
        raise RadiusSweepManifestError("radius sweep manifest must be a mapping")
    violations: list[str] = []

    _check_boundary_fields(manifest, violations)
    radii = manifest.get("radii")
    _check_radii(radii, violations)
    _check_arms(manifest.get("arms"), violations)
    _check_fixed_factors(manifest.get("fixed_factors"), violations)
    _check_immutable_commit(manifest.get("immutable_campaign_commit"), violations)
    _check_provenance_identity(manifest, violations)
    _check_gate_preconditions(manifest.get("gate_preconditions"), manifest.get("arms"), violations)
    _check_missingness_policy(manifest.get("missingness_policy"), violations)
    _check_row_identity_ledger(manifest.get("row_identity_ledger_template"), violations)

    return {
        "schema_version": RADIUS_SWEEP_MANIFEST_CHECK_SCHEMA,
        "status": MANIFEST_CHECK_STATUS,
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": (
            "checker summary only: reviews the radius-sweep preparation manifest "
            "against the issue #6642 boundary and no-evidence contract; does not run "
            "the sweep or establish radius-sensitivity conclusions."
        ),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_status": manifest.get("status"),
        "arm_count": len(manifest.get("arms", [])) if isinstance(manifest.get("arms"), list) else 0,
        "expected_total_rows": (
            manifest.get("row_identity_ledger_template", {}).get("expected_total_rows")
            if isinstance(manifest.get("row_identity_ledger_template"), Mapping)
            else None
        ),
        "violations": violations,
        "passes": not violations,
    }


def write_radius_sweep_manifest(manifest: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a deterministic JSON preparation manifest.

    Returns:
        Path to the written JSON manifest.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "radius_sweep_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_radius_sweep_manifest_check(
    check_summary: Mapping[str, Any], output_dir: str | Path
) -> Path:
    """Write a deterministic JSON radius-sweep manifest checker summary.

    Returns:
        Path to the written JSON checker summary.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    check_path = out / "radius_sweep_manifest_check.json"
    check_path.write_text(
        json.dumps(check_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return check_path


def _validate_manifest_config_shape(manifest_config: Mapping[str, Any]) -> None:  # noqa: C901
    """Fail closed on manifest-config shape before enumerating arms."""
    if not isinstance(manifest_config, Mapping):
        raise RadiusSweepManifestError("manifest config must be a mapping")
    if manifest_config.get("schema_version") != RADIUS_SWEEP_MANIFEST_SCHEMA:
        raise RadiusSweepManifestError(
            f"manifest config schema_version must be {RADIUS_SWEEP_MANIFEST_SCHEMA!r}"
        )
    if manifest_config.get("issue") != ISSUE_6642:
        raise RadiusSweepManifestError(f"manifest config issue must be {ISSUE_6642}")
    if manifest_config.get("parent_issue") != PARENT_ISSUE_6600:
        raise RadiusSweepManifestError(f"manifest config parent_issue must be {PARENT_ISSUE_6600}")
    if not isinstance(manifest_config.get("radii"), list) or not manifest_config["radii"]:
        raise RadiusSweepManifestError("manifest config requires a non-empty radii list")
    expected_paths = {
        "release_baseline_config": EXPECTED_RELEASE_BASELINE_CONFIG,
        "arm_campaign_config_0p5m": EXPECTED_ARM_CAMPAIGN_CONFIG_0P5M,
        "arm_campaign_config_0p8m": EXPECTED_ARM_CAMPAIGN_CONFIG_0P8M,
        "arm_campaign_config_1p0m": EXPECTED_ARM_CAMPAIGN_CONFIG,
    }
    for key, expected in expected_paths.items():
        value = manifest_config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RadiusSweepManifestError(f"manifest config requires a non-empty {key!r}")
        if value != expected:
            raise RadiusSweepManifestError(
                f"manifest config {key} must be {expected!r}, got {value!r}"
            )
    fixed_factors = manifest_config.get("fixed_factors")
    if not isinstance(fixed_factors, Mapping):
        raise RadiusSweepManifestError("manifest config requires a fixed_factors mapping")
    required_fixed_factor_keys = (
        "scenario_matrix",
        "expected_scenario_count",
        "seed_set",
        "expected_seed_range",
        "horizon",
        "dt",
        "kinematics",
        "expected_planner_count",
    )
    missing = [key for key in required_fixed_factor_keys if key not in fixed_factors]
    if missing:
        raise RadiusSweepManifestError(
            f"manifest config fixed_factors is missing required keys: {missing!r}"
        )


def validate_arm_fixed_factors(factors: FixedFactors, *, arm_key: str) -> None:  # noqa: C901
    """Fail closed when one arm's resolved factors drift from the frozen contract.

    Every non-radius factor must equal the frozen release surface; only the
    issue-scoped release tag is arm-specific.

    Args:
        factors: Fixed factors resolved from one arm campaign config.
        arm_key: Radius arm key (``r0p5``, ``r0p8``, or ``r1p0``).
    """
    expected_release_tag = EXPECTED_ARM_RELEASE_TAGS.get(arm_key)
    if expected_release_tag is None:
        raise RadiusSweepManifestError(f"unknown radius arm key {arm_key!r}")
    violations: list[str] = []
    if factors.scenario_matrix != EXPECTED_SCENARIO_MATRIX:
        violations.append(f"scenario_matrix must be {EXPECTED_SCENARIO_MATRIX!r}")
    if factors.scenario_count != EXPECTED_SCENARIO_COUNT:
        violations.append(f"scenario_count must be {EXPECTED_SCENARIO_COUNT}")
    if not _sequence_matches(factors.scenario_names, EXPECTED_SCENARIO_NAMES):
        violations.append("scenario_names must equal the frozen 48-cell release roster")
    if not _sequence_matches(factors.planner_keys, RELEASE_PLANNER_KEYS):
        violations.append("planner_keys must equal the frozen 14-key release roster")
    if factors.seed_set != EXPECTED_SEED_SET:
        violations.append(f"seed_set must be {EXPECTED_SEED_SET!r}")
    if not _seed_sequence_matches(factors.seeds, EXPECTED_SEEDS):
        violations.append("seeds must equal the frozen paper_eval_s30 seeds 111-140")
    if factors.horizon != EXPECTED_HORIZON:
        violations.append(f"horizon must be {EXPECTED_HORIZON}")
    if factors.dt != EXPECTED_DT:
        violations.append(f"dt must be {EXPECTED_DT}")
    if factors.kinematics != EXPECTED_KINEMATICS:
        violations.append(f"kinematics must be {EXPECTED_KINEMATICS!r}")
    if factors.release_tag != expected_release_tag:
        violations.append(f"release_tag must be {expected_release_tag!r}")
    if violations:
        raise RadiusSweepManifestError(
            f"resolved fixed factors for arm {arm_key!r} violate the radius-sweep contract: "
            + "; ".join(violations)
        )


def _validate_fixed_factors(fixed_factors: FixedFactors) -> None:
    """Fail closed when the baseline (1.0 m) arm factors miss the frozen contract."""
    validate_arm_fixed_factors(fixed_factors, arm_key=PRODUCTION_RADIUS_KEYS[-1])


def _validate_declared_fixed_factors(
    manifest_config: Mapping[str, Any], fixed_factors: FixedFactors
) -> None:
    """Require manifest-declared expectations to match the resolved arm config."""
    declared = manifest_config["fixed_factors"]
    expected_values = {
        "scenario_matrix": fixed_factors.scenario_matrix,
        "expected_scenario_count": fixed_factors.scenario_count,
        "seed_set": fixed_factors.seed_set,
        "expected_seed_range": list(EXPECTED_SEED_RANGE),
        "horizon": fixed_factors.horizon,
        "dt": fixed_factors.dt,
        "kinematics": fixed_factors.kinematics,
        "expected_planner_count": len(fixed_factors.planner_keys),
    }
    mismatches = [
        f"{key}={declared.get(key)!r} (resolved {expected!r})"
        for key, expected in expected_values.items()
        if declared.get(key) != expected
    ]
    if mismatches:
        raise RadiusSweepManifestError(
            "manifest fixed_factors do not match the resolved arm config: " + "; ".join(mismatches)
        )


def _runtime_binding_violations(metadata: Mapping[str, Any], *, label: str) -> list[str]:
    """Return fail-closed violations for one runtime-binding metadata block."""
    violations: list[str] = []
    status = metadata.get("runtime_binding_status")
    if status not in RUNTIME_BINDING_STATUSES:
        violations.append(
            f"{label}.runtime_binding_status must be one of {list(RUNTIME_BINDING_STATUSES)!r}, "
            f"got {status!r}"
        )
        return violations

    provenance_fields = (
        "binding_contract_version",
        "gate1_canary_issue",
        "gate1_receipt_sha256",
        "gate1_source_commit",
    )
    if status == RUNTIME_BINDING_PENDING_GATE1:
        supplied = [field for field in provenance_fields if metadata.get(field) is not None]
        if supplied:
            violations.append(
                f"{label} pending status cannot carry admitted Gate 1 provenance: {supplied!r}"
            )
        return violations

    if metadata.get("binding_contract_version") != RUNTIME_BINDING_CONTRACT_VERSION:
        violations.append(
            f"{label}.binding_contract_version must be "
            f"{RUNTIME_BINDING_CONTRACT_VERSION!r} for bound_runtime arms"
        )
    gate1_issue = metadata.get("gate1_canary_issue")
    if isinstance(gate1_issue, bool) or gate1_issue != GATE1_CANARY_ISSUE:
        violations.append(
            f"{label}.gate1_canary_issue must be {GATE1_CANARY_ISSUE} for bound_runtime arms"
        )
    receipt = metadata.get("gate1_receipt_sha256")
    if not isinstance(receipt, str) or _SHA256_PATTERN.fullmatch(receipt) is None:
        violations.append(
            f"{label}.gate1_receipt_sha256 must be a lowercase 64-character SHA-256 digest"
        )
    source_commit = metadata.get("gate1_source_commit")
    if not isinstance(source_commit, str) or _GIT_SHA_PATTERN.fullmatch(source_commit) is None:
        violations.append(
            f"{label}.gate1_source_commit must be a lowercase 40-character commit hash"
        )
    return violations


def validate_arm_campaign_payload(
    payload: Mapping[str, Any], *, arm_key: str, radius_m: float, baseline: bool
) -> None:
    """Fail closed when an arm campaign config's radius_sweep metadata drifts.

    The ``radius_sweep`` block carries the declared treatment and, after Gate 1
    admission, its receipt-backed runtime-binding provenance. The sweep builder
    must validate both states instead of trusting a silently divergent declaration.

    Args:
        payload: Parsed arm campaign config mapping.
        arm_key: Expected radius arm key (``r0p5``, ``r0p8``, or ``r1p0``).
        radius_m: Expected declared radius in metres.
        baseline: Expected baseline-arm flag.
    """
    if not isinstance(payload, Mapping):
        raise RadiusSweepManifestError("arm campaign config payload must be a mapping")
    raw = payload.get("radius_sweep")
    if not isinstance(raw, Mapping):
        raise RadiusSweepManifestError(
            f"arm campaign config for {arm_key!r} requires a radius_sweep metadata mapping"
        )
    violations: list[str] = []
    if raw.get("issue") != ISSUE_6642:
        violations.append(f"radius_sweep.issue must be {ISSUE_6642}")
    if raw.get("parent_issue") != PARENT_ISSUE_6600:
        violations.append(f"radius_sweep.parent_issue must be {PARENT_ISSUE_6600}")
    if raw.get("arm_key") != arm_key:
        violations.append(f"radius_sweep.arm_key must be {arm_key!r}, got {raw.get('arm_key')!r}")
    declared_radius = raw.get("radius_m")
    if not _radius_matches(declared_radius, radius_m):
        violations.append(f"radius_sweep.radius_m must be {radius_m!r}, got {declared_radius!r}")
    if raw.get("baseline_arm") is not baseline:
        violations.append(f"radius_sweep.baseline_arm must be {baseline}")
    violations.extend(_runtime_binding_violations(raw, label="radius_sweep"))
    if violations:
        raise RadiusSweepManifestError(
            f"arm campaign config radius_sweep metadata for {arm_key!r} violates the "
            "radius-sweep contract: " + "; ".join(violations)
        )


def _validate_arm_identities(
    manifest_config: Mapping[str, Any], arm_identities: Sequence[ArmCampaignIdentity]
) -> list[ArmCampaignIdentity]:
    """Fail closed unless all three arm identities match the frozen campaign surface.

    Returns:
        The validated arm identities in radius order.
    """
    if not isinstance(arm_identities, (list, tuple)) or len(arm_identities) != len(
        PRODUCTION_RADIUS_KEYS
    ):
        raise RadiusSweepManifestError(
            f"arm_identities must provide exactly {len(PRODUCTION_RADIUS_KEYS)} "
            "ArmCampaignIdentity entries in radius order"
        )
    validated: list[ArmCampaignIdentity] = []
    for index, (identity, arm_key) in enumerate(
        zip(arm_identities, PRODUCTION_RADIUS_KEYS, strict=True)
    ):
        if not isinstance(identity, ArmCampaignIdentity):
            raise RadiusSweepManifestError(
                f"arm_identities[{index}] must be an ArmCampaignIdentity"
            )
        if identity.arm_key != arm_key:
            raise RadiusSweepManifestError(
                f"arm_identities[{index}].arm_key must be {arm_key!r}, got {identity.arm_key!r}"
            )
        expected_config = EXPECTED_ARM_CAMPAIGN_CONFIGS[arm_key]
        if identity.campaign_config != expected_config:
            raise RadiusSweepManifestError(
                f"arm {arm_key!r} campaign_config must be {expected_config!r}, "
                f"got {identity.campaign_config!r}"
            )
        declared = manifest_config.get(ARM_CONFIG_KEYS[arm_key])
        if declared != identity.campaign_config:
            raise RadiusSweepManifestError(
                f"manifest config {ARM_CONFIG_KEYS[arm_key]} must match the resolved "
                f"arm {arm_key!r} campaign_config {expected_config!r}, got {declared!r}"
            )
        expected_tag = EXPECTED_ARM_RELEASE_TAGS[arm_key]
        if identity.release_tag != expected_tag:
            raise RadiusSweepManifestError(
                f"arm {arm_key!r} release_tag must be {expected_tag!r}, "
                f"got {identity.release_tag!r}"
            )
        binding_violations = _runtime_binding_violations(
            {
                "runtime_binding_status": identity.runtime_binding_status,
                "binding_contract_version": identity.binding_contract_version,
                "gate1_canary_issue": identity.gate1_canary_issue,
                "gate1_receipt_sha256": identity.gate1_receipt_sha256,
                "gate1_source_commit": identity.gate1_source_commit,
            },
            label=f"arm {arm_key!r}",
        )
        if binding_violations:
            raise RadiusSweepManifestError(
                f"arm {arm_key!r} runtime-binding metadata violates the contract: "
                + "; ".join(binding_violations)
            )
        validated.append(identity)
    return validated


def _sequence_matches(value: Any, expected: tuple[Any, ...]) -> bool:
    """Return whether a list/tuple has exactly the expected ordered values."""
    return isinstance(value, (list, tuple)) and tuple(value) == expected


def _seed_sequence_matches(value: Any, expected: tuple[int, ...]) -> bool:
    """Return whether a sequence contains exactly the expected integer seeds."""
    return (
        isinstance(value, (list, tuple))
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in value)
        and tuple(value) == expected
    )


def _normalize_radii(raw_radii: Any) -> list[dict[str, Any]]:  # noqa: C901
    """Normalize and validate the radius-treatment list.

    Returns:
        List of normalized radius-entry mappings in declared order.
    """
    if not isinstance(raw_radii, list) or len(raw_radii) != len(PRODUCTION_RADII):
        raise RadiusSweepManifestError(
            f"radii must declare exactly {len(PRODUCTION_RADII)} arms "
            f"({list(PRODUCTION_RADII)!r}), got {len(raw_radii) if isinstance(raw_radii, list) else 'non-list'}"
        )
    normalized: list[dict[str, Any]] = []
    declared_values: list[float] = []
    for index, entry in enumerate(raw_radii):
        if not isinstance(entry, Mapping):
            raise RadiusSweepManifestError(f"radius entry {index} must be a mapping")
        radius_m = entry.get("radius_m")
        if not isinstance(radius_m, int | float) or isinstance(radius_m, bool):
            raise RadiusSweepManifestError(
                f"radius entry {index} radius_m must be a number, got {radius_m!r}"
            )
        radius_value = float(radius_m)
        if radius_value != PRODUCTION_RADII[index]:
            raise RadiusSweepManifestError(
                f"radius entry {index} must be {PRODUCTION_RADII[index]!r} m, got {radius_value!r}"
            )
        expected_key = PRODUCTION_RADIUS_KEYS[index]
        if entry.get("key") != expected_key:
            raise RadiusSweepManifestError(
                f"radius entry {index} key must be {expected_key!r}, got {entry.get('key')!r}"
            )
        baseline = entry.get("baseline", False)
        if not isinstance(baseline, bool):
            raise RadiusSweepManifestError(
                f"radius entry {index} baseline must be a boolean, got {baseline!r}"
            )
        declared_values.append(radius_value)
        normalized.append(
            {
                "key": expected_key,
                "radius_m": radius_value,
                "baseline": baseline,
            }
        )
    if tuple(declared_values) != PRODUCTION_RADII:
        raise RadiusSweepManifestError(
            f"radii must be exactly {list(PRODUCTION_RADII)!r} in order, got {declared_values!r}"
        )
    baseline_entries = [entry for entry in normalized if entry["baseline"]]
    if len(baseline_entries) != 1:
        raise RadiusSweepManifestError(
            "exactly one radius entry must carry baseline=true (the 1.0 m release comparator)"
        )
    if not baseline_entries[0]["radius_m"] == BASELINE_RADIUS:
        raise RadiusSweepManifestError(
            f"baseline radius must be {BASELINE_RADIUS!r} m, got {baseline_entries[0]['radius_m']!r}"
        )
    return normalized


def _radius_summary(radius_entry: Mapping[str, Any]) -> dict[str, Any]:
    """Project a radius entry into a compact manifest summary.

    Returns:
        JSON-serializable radius summary mapping.
    """
    return {
        "key": str(radius_entry["key"]),
        "radius_m": float(radius_entry["radius_m"]),
        "baseline": bool(radius_entry["baseline"]),
    }


def _radius_matches(value: Any, expected: float) -> bool:
    """Return whether a serialized radius is numeric and equals the expected value."""
    return (
        isinstance(value, int | float) and not isinstance(value, bool) and float(value) == expected
    )


def _arm_manifest(
    radius_entry: Mapping[str, Any],
    *,
    fixed_factors: FixedFactors,
    identity: ArmCampaignIdentity,
) -> dict[str, Any]:
    """Enumerate one radius arm with its planner roster and expected row count.

    Returns:
        JSON-serializable radius-arm manifest mapping.
    """
    radius_m = float(radius_entry["radius_m"])
    manifest_arm = {
        "key": str(radius_entry["key"]),
        "radius_m": radius_m,
        "baseline": bool(radius_entry["baseline"]),
        "arm_campaign_config": identity.campaign_config,
        "release_tag": identity.release_tag,
        "runtime_binding_status": identity.runtime_binding_status,
        "runtime_binding_note": identity.runtime_binding_note
        or (
            "Runtime binding admitted after the Gate 1 receipt. Production compute "
            "remains blocked until the separate Gate 2 campaign gates pass."
            if identity.runtime_binding_status == RUNTIME_BINDING_BOUND_RUNTIME
            else "Declared treatment metadata only. No runtime radius binding is admitted."
        ),
        "planner_keys": list(fixed_factors.planner_keys),
        "planner_count": len(fixed_factors.planner_keys),
        "scenario_count": fixed_factors.scenario_count,
        "seed_count": len(fixed_factors.seeds),
        "expected_episode_count": (
            len(fixed_factors.planner_keys)
            * fixed_factors.scenario_count
            * len(fixed_factors.seeds)
        ),
    }
    if identity.runtime_binding_status == RUNTIME_BINDING_BOUND_RUNTIME:
        manifest_arm.update(
            {
                "binding_contract_version": identity.binding_contract_version,
                "gate1_canary_issue": identity.gate1_canary_issue,
                "gate1_receipt_sha256": identity.gate1_receipt_sha256,
                "gate1_source_commit": identity.gate1_source_commit,
            }
        )
    return manifest_arm


def _fixed_factors_manifest(fixed_factors: FixedFactors) -> dict[str, Any]:
    """Project resolved fixed factors into a manifest summary.

    Returns:
        JSON-serializable fixed-factors summary mapping.
    """
    return {
        "scenario_matrix": fixed_factors.scenario_matrix,
        "scenario_count": fixed_factors.scenario_count,
        "scenario_names": list(fixed_factors.scenario_names),
        "planner_keys": list(fixed_factors.planner_keys),
        "planner_count": len(fixed_factors.planner_keys),
        "seed_set": fixed_factors.seed_set,
        "seeds": list(fixed_factors.seeds),
        "seed_range": [min(fixed_factors.seeds), max(fixed_factors.seeds)],
        "horizon": fixed_factors.horizon,
        "dt": fixed_factors.dt,
        "kinematics": fixed_factors.kinematics,
        "release_tag": fixed_factors.release_tag,
    }


def _immutable_commit_manifest(
    manifest_config: Mapping[str, Any], options: ManifestOptions
) -> dict[str, Any]:
    """Record the one-commit-across-arms policy and the launch git head.

    Returns:
        JSON-serializable immutable-commit manifest mapping.
    """
    raw = manifest_config.get("immutable_campaign_commit")
    policy = (
        str(raw.get("policy", IMMUTABLE_COMMIT_POLICY))
        if isinstance(raw, Mapping)
        else IMMUTABLE_COMMIT_POLICY
    )
    return {
        "policy": policy,
        "git_head": options.git_head,
        "one_commit_across_all_arms": True,
        "note": (
            "All radius arms (0.5/0.8/1.0 m) must run at one immutable campaign "
            "commit. Production compute remains blocked until the remaining Gate 2 "
            "campaign gates pass, so no commit is frozen as production evidence here."
        ),
    }


def _gate_preconditions_manifest(manifest_config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the gate precedence and production-submission block.

    Returns:
        JSON-serializable gate-preconditions manifest mapping.
    """
    raw = manifest_config.get("gate_preconditions")
    raw = raw if isinstance(raw, Mapping) else {}
    result = {
        "gate1_canary_issue": int(raw.get("gate1_canary_issue", GATE1_CANARY_ISSUE)),
        "gate1_canary_status": str(raw.get("gate1_canary_status", GATE1_STATUS_NOT_YET_PASSED)),
        "production_submission_authorized": bool(
            raw.get("production_submission_authorized", False)
        ),
        "unblock_condition": str(
            raw.get(
                "unblock_condition",
                "Gate 1 binding-canary child #6641 reports a passing verdict AND a "
                "runtime radius-binding surface exists.",
            )
        ),
    }
    for key in (
        "gate1_receipt_sha256",
        "gate1_source_commit",
        "runtime_binding_contract_version",
    ):
        if raw.get(key) is not None:
            result[key] = raw[key]
    return result


def _missingness_policy_manifest(manifest_config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the evidence-exclusion and row-identity contract.

    Returns:
        JSON-serializable missingness-policy manifest mapping.
    """
    raw = manifest_config.get("missingness_policy")
    raw = raw if isinstance(raw, Mapping) else {}
    exclusions = raw.get("evidence_exclusions")
    exclusions = (
        [str(value) for value in exclusions]
        if isinstance(exclusions, list)
        else list(EVIDENCE_EXCLUSIONS)
    )
    return {
        "evidence_exclusions": exclusions,
        "row_identity_contract": str(
            raw.get(
                "row_identity_contract",
                ROW_IDENTITY_CONTRACT,
            )
        ),
    }


def _check_boundary_fields(manifest: Mapping[str, Any], violations: list[str]) -> None:  # noqa: C901
    """Assert the manifest keeps its no-evidence / dry-run boundary intact."""
    if manifest.get("schema_version") != RADIUS_SWEEP_MANIFEST_SCHEMA:
        violations.append(f"manifest schema_version must remain {RADIUS_SWEEP_MANIFEST_SCHEMA!r}")
    if manifest.get("dry_run") is not True:
        violations.append("manifest must remain dry_run=true")
    if manifest.get("status") != MANIFEST_STATUS:
        violations.append(f"manifest status must remain {MANIFEST_STATUS!r}")
    if manifest.get("evidence_status") != EVIDENCE_STATUS:
        violations.append(f"manifest evidence_status must remain {EVIDENCE_STATUS!r}")
    if manifest.get("issue") != ISSUE_6642:
        violations.append(f"manifest issue must remain {ISSUE_6642}")
    if manifest.get("parent_issue") != PARENT_ISSUE_6600:
        violations.append(f"manifest parent_issue must remain {PARENT_ISSUE_6600}")
    if manifest.get("release_baseline_config") != EXPECTED_RELEASE_BASELINE_CONFIG:
        violations.append(
            f"manifest release_baseline_config must remain {EXPECTED_RELEASE_BASELINE_CONFIG!r}"
        )
    if manifest.get("arm_campaign_config_1p0m") != EXPECTED_ARM_CAMPAIGN_CONFIG:
        violations.append(
            f"manifest arm_campaign_config_1p0m must remain {EXPECTED_ARM_CAMPAIGN_CONFIG!r}"
        )
    if manifest.get("arm_count") != len(PRODUCTION_RADII):
        violations.append(f"manifest arm_count must be {len(PRODUCTION_RADII)}")
    if manifest.get("claim_boundary") != CLAIM_BOUNDARY:
        violations.append("manifest claim_boundary must equal the canonical preparation boundary")
    if manifest.get("config_path") != EXPECTED_MANIFEST_CONFIG:
        violations.append(f"manifest config_path must remain {EXPECTED_MANIFEST_CONFIG!r}")


def _check_radii(radii: Any, violations: list[str]) -> None:
    """Assert the radius treatment is exactly 0.5/0.8/1.0 m with one baseline."""
    if not isinstance(radii, list) or len(radii) != len(PRODUCTION_RADII):
        violations.append(
            f"radii must declare exactly {len(PRODUCTION_RADII)} arms ({list(PRODUCTION_RADII)!r})"
        )
        return
    declared_values: list[Any] = []
    for index, entry in enumerate(radii):
        if not isinstance(entry, Mapping):
            violations.append(f"radius entry {index} must be a mapping")
            continue
        radius_m = entry.get("radius_m")
        if not isinstance(radius_m, int | float) or isinstance(radius_m, bool):
            violations.append(f"radius entry {index} radius_m must be a number")
            declared_values.append(None)
        else:
            declared_values.append(radius_m)
        expected_key = PRODUCTION_RADIUS_KEYS[index]
        if entry.get("key") != expected_key:
            violations.append(f"radius entry {index} key must be {expected_key!r}")
        if not isinstance(entry.get("baseline", False), bool):
            violations.append(f"radius entry {index} baseline must be a boolean")
    if declared_values != list(PRODUCTION_RADII):
        violations.append(
            f"radii must be exactly {list(PRODUCTION_RADII)!r} in order, got {declared_values!r}"
        )
    baseline_entries = [
        entry for entry in radii if isinstance(entry, Mapping) and entry.get("baseline") is True
    ]
    if len(baseline_entries) != 1:
        violations.append("exactly one radius entry must carry baseline=true")
    elif baseline_entries[0].get("radius_m") != BASELINE_RADIUS:
        violations.append("baseline radius entry must be the 1.0 m arm")


def _check_arms(arms: Any, violations: list[str]) -> None:
    """Assert each arm carries a valid binding state and the full planner roster."""
    if not isinstance(arms, list) or len(arms) != len(PRODUCTION_RADII):
        violations.append(f"arms must enumerate exactly {len(PRODUCTION_RADII)} radius arms")
        return
    for index, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            violations.append("each arm must be a mapping")
            continue
        _check_arm_identity(arm, index, violations)
        _check_arm_roster_and_counts(arm, violations)


def _check_arm_identity(arm: Mapping[str, Any], index: int, violations: list[str]) -> None:
    """Assert one arm's radius identity and receipt-backed binding metadata."""
    expected_key = PRODUCTION_RADIUS_KEYS[index]
    arm_key = arm.get("key")
    if arm_key != expected_key:
        violations.append(f"arm {index} key must be {expected_key!r}")
    if not _radius_matches(arm.get("radius_m"), PRODUCTION_RADII[index]):
        violations.append(f"arm {arm_key!r} radius_m must be {PRODUCTION_RADII[index]!r}")
    expected_baseline = index == len(PRODUCTION_RADII) - 1
    if arm.get("baseline") is not expected_baseline:
        violations.append(f"arm {arm_key!r} baseline must be {expected_baseline}")
    expected_config = EXPECTED_ARM_CAMPAIGN_CONFIGS.get(expected_key)
    if arm.get("arm_campaign_config") != expected_config:
        violations.append(f"arm {arm_key!r} arm_campaign_config must be {expected_config!r}")
    expected_tag = EXPECTED_ARM_RELEASE_TAGS.get(expected_key)
    if arm.get("release_tag") != expected_tag:
        violations.append(f"arm {arm_key!r} release_tag must be {expected_tag!r}")
    violations.extend(
        _runtime_binding_violations(
            arm,
            label=f"arm {arm_key!r}",
        )
    )


def _check_arm_roster_and_counts(arm: Mapping[str, Any], violations: list[str]) -> None:
    """Assert one arm carries the complete planner, scenario, seed, and row grid."""
    arm_key = arm.get("key")
    planner_keys = arm.get("planner_keys")
    if not isinstance(planner_keys, list) or tuple(planner_keys) != RELEASE_PLANNER_KEYS:
        violations.append(
            f"arm {arm_key!r} planner_keys must equal the frozen "
            f"{len(RELEASE_PLANNER_KEYS)}-key release roster in order"
        )
    if arm.get("planner_count") != len(RELEASE_PLANNER_KEYS):
        violations.append(f"arm {arm_key!r} planner_count must be {len(RELEASE_PLANNER_KEYS)}")
    if arm.get("scenario_count") != EXPECTED_SCENARIO_COUNT:
        violations.append(f"arm {arm_key!r} scenario_count must be {EXPECTED_SCENARIO_COUNT}")
    if arm.get("seed_count") != len(EXPECTED_SEEDS):
        violations.append(f"arm {arm_key!r} seed_count must be {len(EXPECTED_SEEDS)}")
    if arm.get("expected_episode_count") != EXPECTED_ROWS_PER_ARM:
        violations.append(f"arm {arm_key!r} expected_episode_count must be {EXPECTED_ROWS_PER_ARM}")


def _check_fixed_factors(fixed_factors: Any, violations: list[str]) -> None:
    """Assert fixed factors match the release baseline except the radius."""
    if not isinstance(fixed_factors, Mapping):
        violations.append("manifest must carry a fixed_factors mapping")
        return
    _check_fixed_factor_scenario(fixed_factors, violations)
    _check_fixed_factor_planners(fixed_factors, violations)
    _check_fixed_factor_seeds(fixed_factors, violations)
    _check_fixed_factor_dynamics(fixed_factors, violations)


def _check_fixed_factor_scenario(fixed_factors: Mapping[str, Any], violations: list[str]) -> None:
    """Assert the scenario matrix and its complete 48-cell roster stay fixed."""
    if fixed_factors.get("scenario_matrix") != EXPECTED_SCENARIO_MATRIX:
        violations.append(f"fixed_factors.scenario_matrix must be {EXPECTED_SCENARIO_MATRIX!r}")
    if fixed_factors.get("scenario_count") != EXPECTED_SCENARIO_COUNT:
        violations.append(f"fixed_factors.scenario_count must be {EXPECTED_SCENARIO_COUNT}")
    names = fixed_factors.get("scenario_names")
    if not isinstance(names, list) or tuple(names) != EXPECTED_SCENARIO_NAMES:
        violations.append(
            "fixed_factors.scenario_names must equal the frozen 48-cell release roster in order"
        )


def _check_fixed_factor_planners(fixed_factors: Mapping[str, Any], violations: list[str]) -> None:
    """Assert the 14-key release roster is preserved in order."""
    if fixed_factors.get("planner_count") != len(RELEASE_PLANNER_KEYS):
        violations.append(f"fixed_factors.planner_count must be {len(RELEASE_PLANNER_KEYS)}")
    if not _sequence_matches(fixed_factors.get("planner_keys"), RELEASE_PLANNER_KEYS):
        violations.append(
            "fixed_factors.planner_keys must equal the frozen release roster in order"
        )


def _check_fixed_factor_seeds(fixed_factors: Mapping[str, Any], violations: list[str]) -> None:
    """Assert the complete paper_eval_s30 seed set (seeds 111-140) stays fixed."""
    if fixed_factors.get("seed_set") != EXPECTED_SEED_SET:
        violations.append(f"fixed_factors.seed_set must be {EXPECTED_SEED_SET!r}")
    seeds = fixed_factors.get("seeds")
    if not _seed_sequence_matches(seeds, EXPECTED_SEEDS):
        violations.append("fixed_factors.seeds must equal the frozen paper_eval_s30 seeds 111-140")
    seed_range = fixed_factors.get("seed_range")
    if not _sequence_matches(seed_range, EXPECTED_SEED_RANGE):
        violations.append(f"fixed_factors.seed_range must be {list(EXPECTED_SEED_RANGE)!r}")


def _check_fixed_factor_dynamics(fixed_factors: Mapping[str, Any], violations: list[str]) -> None:
    """Assert horizon, timestep, and kinematics stay fixed across arms."""
    if fixed_factors.get("horizon") != EXPECTED_HORIZON:
        violations.append(f"fixed_factors.horizon must be {EXPECTED_HORIZON}")
    if fixed_factors.get("dt") != EXPECTED_DT:
        violations.append(f"fixed_factors.dt must be {EXPECTED_DT}")
    if fixed_factors.get("kinematics") != EXPECTED_KINEMATICS:
        violations.append(f"fixed_factors.kinematics must be {EXPECTED_KINEMATICS!r}")
    if fixed_factors.get("release_tag") != EXPECTED_ARM_RELEASE_TAG:
        violations.append(f"fixed_factors.release_tag must be {EXPECTED_ARM_RELEASE_TAG!r}")


def _check_immutable_commit(immutable_commit: Any, violations: list[str]) -> None:
    """Assert all arms are pinned to one immutable campaign commit."""
    if not isinstance(immutable_commit, Mapping):
        violations.append("manifest must carry an immutable_campaign_commit mapping")
        return
    if immutable_commit.get("policy") != IMMUTABLE_COMMIT_POLICY:
        violations.append(f"immutable_campaign_commit.policy must be {IMMUTABLE_COMMIT_POLICY!r}")
    if immutable_commit.get("one_commit_across_all_arms") is not True:
        violations.append("immutable_campaign_commit.one_commit_across_all_arms must be true")
    git_head = immutable_commit.get("git_head")
    if not isinstance(git_head, str) or _GIT_SHA_PATTERN.fullmatch(git_head) is None:
        violations.append(
            "immutable_campaign_commit.git_head must be a 40-character lowercase git SHA"
        )


def _check_provenance_identity(manifest: Mapping[str, Any], violations: list[str]) -> None:
    """Assert duplicated manifest and immutable-commit identities remain consistent."""
    git_head = manifest.get("git_head")
    if not isinstance(git_head, str) or _GIT_SHA_PATTERN.fullmatch(git_head) is None:
        violations.append("manifest git_head must be a 40-character lowercase git SHA")

    immutable_commit = manifest.get("immutable_campaign_commit")
    if isinstance(immutable_commit, Mapping) and git_head != immutable_commit.get("git_head"):
        violations.append("manifest git_head must match immutable_campaign_commit.git_head")


def _gate_binding_metadata(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Project gate-precondition fields into the arm-binding contract shape.

    Returns:
        Mapping with the field names expected by the runtime-binding validator.
    """
    return {
        "runtime_binding_status": RUNTIME_BINDING_BOUND_RUNTIME,
        "binding_contract_version": gate.get("runtime_binding_contract_version"),
        "gate1_canary_issue": gate.get("gate1_canary_issue"),
        "gate1_receipt_sha256": gate.get("gate1_receipt_sha256"),
        "gate1_source_commit": gate.get("gate1_source_commit"),
    }


def _check_passed_gate_arms(gate: Mapping[str, Any], arms: Any, violations: list[str]) -> None:
    """Require every arm to carry provenance matching a passed Gate 1 state."""
    if not isinstance(arms, list):
        return
    gate_fields = {
        "binding_contract_version": "runtime_binding_contract_version",
        "gate1_canary_issue": "gate1_canary_issue",
        "gate1_receipt_sha256": "gate1_receipt_sha256",
        "gate1_source_commit": "gate1_source_commit",
    }
    for index, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            continue
        if arm.get("runtime_binding_status") != RUNTIME_BINDING_BOUND_RUNTIME:
            violations.append(
                f"arm {index} runtime_binding_status must be "
                f"{RUNTIME_BINDING_BOUND_RUNTIME!r} when Gate 1 is passed"
            )
            continue
        for arm_field, gate_field in gate_fields.items():
            if arm.get(arm_field) != gate.get(gate_field):
                violations.append(
                    f"arm {index} {arm_field} must match gate_preconditions.{gate_field}"
                )


def _check_pending_gate_arms(arms: Any, violations: list[str]) -> None:
    """Require every arm to remain pending when Gate 1 has not passed."""
    if not isinstance(arms, list):
        return
    for index, arm in enumerate(arms):
        if (
            isinstance(arm, Mapping)
            and arm.get("runtime_binding_status") != RUNTIME_BINDING_PENDING_GATE1
        ):
            violations.append(
                f"arm {index} runtime_binding_status must be "
                f"{RUNTIME_BINDING_PENDING_GATE1!r} when Gate 1 is pending"
            )


def _check_gate_preconditions(gate: Any, arms: Any, violations: list[str]) -> None:
    """Assert receipt-backed Gate 1 state and the continuing production block."""
    if not isinstance(gate, Mapping):
        violations.append("manifest must carry a gate_preconditions mapping")
        return
    if gate.get("gate1_canary_issue") != GATE1_CANARY_ISSUE:
        violations.append(f"gate_preconditions.gate1_canary_issue must be {GATE1_CANARY_ISSUE}")
    gate_status = gate.get("gate1_canary_status")
    if gate_status not in GATE1_STATUSES:
        violations.append(
            "gate_preconditions.gate1_canary_status must be one of "
            f"{list(GATE1_STATUSES)!r}, got {gate_status!r}"
        )
    if gate.get("production_submission_authorized") is not False:
        violations.append("production_submission_authorized must remain false for this manifest")
    if gate_status == GATE1_STATUS_PASSED:
        violations.extend(
            _runtime_binding_violations(
                _gate_binding_metadata(gate),
                label="gate_preconditions",
            )
        )
        _check_passed_gate_arms(gate, arms, violations)
    elif gate_status == GATE1_STATUS_NOT_YET_PASSED:
        supplied = [
            field
            for field in (
                "gate1_receipt_sha256",
                "gate1_source_commit",
                "runtime_binding_contract_version",
            )
            if gate.get(field) is not None
        ]
        if supplied:
            violations.append(
                "gate_preconditions pending status cannot carry admitted Gate 1 provenance: "
                f"{supplied!r}"
            )
        _check_pending_gate_arms(arms, violations)


def _check_missingness_policy(policy: Any, violations: list[str]) -> None:
    """Assert the evidence-exclusion and row-identity contract stay intact."""
    if not isinstance(policy, Mapping):
        violations.append("manifest must carry a missingness_policy mapping")
        return
    exclusions = policy.get("evidence_exclusions")
    if not isinstance(exclusions, list):
        violations.append("missingness_policy.evidence_exclusions must be a list")
    else:
        missing = [value for value in EVIDENCE_EXCLUSIONS if value not in exclusions]
        if missing:
            violations.append(f"missingness_policy.evidence_exclusions must include {missing!r}")
    if policy.get("row_identity_contract") != ROW_IDENTITY_CONTRACT:
        violations.append(
            f"missingness_policy.row_identity_contract must equal {ROW_IDENTITY_CONTRACT!r}"
        )


def _check_row_identity_ledger(ledger: Any, violations: list[str]) -> None:
    """Assert the row-identity ledger template carries the complete expected grid."""
    if not isinstance(ledger, Mapping):
        violations.append("manifest must carry a row_identity_ledger_template mapping")
        return
    dimensions = ledger.get("dimensions")
    expected = ("radius_arm", "planner_key", "scenario_name", "seed")
    if not _sequence_matches(dimensions, expected):
        violations.append(f"row_identity_ledger_template.dimensions must be {list(expected)!r}")
    if ledger.get("completeness") != "template_only_no_episodes_run":
        violations.append(
            "row_identity_ledger_template.completeness must remain "
            "'template_only_no_episodes_run' for a preparation manifest"
        )
    expected_total = ledger.get("expected_total_rows")
    if expected_total != EXPECTED_TOTAL_ROWS:
        violations.append(
            f"row_identity_ledger_template.expected_total_rows must be {EXPECTED_TOTAL_ROWS}"
        )
    if ledger.get("expected_rows_per_arm") != EXPECTED_ROWS_PER_ARM:
        violations.append(
            f"row_identity_ledger_template.expected_rows_per_arm must be {EXPECTED_ROWS_PER_ARM}"
        )
    if ledger.get("fail_closed_contract") != ROW_IDENTITY_CONTRACT:
        violations.append(
            "row_identity_ledger_template.fail_closed_contract must require complete "
            "row identities or an explicit fail-closed missingness ledger"
        )


__all__ = [
    "ARM_CONFIG_KEYS",
    "BASELINE_RADIUS",
    "CLAIM_BOUNDARY",
    "EVIDENCE_EXCLUSIONS",
    "EVIDENCE_STATUS",
    "EXPECTED_ARM_CAMPAIGN_CONFIG",
    "EXPECTED_ARM_CAMPAIGN_CONFIGS",
    "EXPECTED_ARM_CAMPAIGN_CONFIG_0P5M",
    "EXPECTED_ARM_CAMPAIGN_CONFIG_0P8M",
    "EXPECTED_ARM_RELEASE_TAG",
    "EXPECTED_ARM_RELEASE_TAGS",
    "EXPECTED_ARM_RELEASE_TAG_0P5M",
    "EXPECTED_ARM_RELEASE_TAG_0P8M",
    "EXPECTED_DT",
    "EXPECTED_HORIZON",
    "EXPECTED_KINEMATICS",
    "EXPECTED_MANIFEST_CONFIG",
    "EXPECTED_RELEASE_BASELINE_CONFIG",
    "EXPECTED_ROWS_PER_ARM",
    "EXPECTED_SCENARIO_COUNT",
    "EXPECTED_SCENARIO_MATRIX",
    "EXPECTED_SCENARIO_NAMES",
    "EXPECTED_SEEDS",
    "EXPECTED_SEED_RANGE",
    "EXPECTED_SEED_SET",
    "EXPECTED_TOTAL_ROWS",
    "GATE1_CANARY_ISSUE",
    "GATE1_STATUSES",
    "GATE1_STATUS_NOT_YET_PASSED",
    "GATE1_STATUS_PASSED",
    "IMMUTABLE_COMMIT_POLICY",
    "ISSUE_6642",
    "MANIFEST_CHECK_STATUS",
    "MANIFEST_STATUS",
    "PARENT_ISSUE_6600",
    "PRODUCTION_RADII",
    "PRODUCTION_RADIUS_KEYS",
    "RADIUS_SWEEP_MANIFEST_CHECK_SCHEMA",
    "RADIUS_SWEEP_MANIFEST_SCHEMA",
    "RELEASE_PLANNER_KEYS",
    "ROW_IDENTITY_CONTRACT",
    "RUNTIME_BINDING_BOUND_RUNTIME",
    "RUNTIME_BINDING_CONTRACT_VERSION",
    "RUNTIME_BINDING_PENDING_GATE1",
    "RUNTIME_BINDING_STATUSES",
    "ArmCampaignIdentity",
    "FixedFactors",
    "ManifestOptions",
    "RadiusSweepManifestError",
    "build_radius_sweep_manifest",
    "check_radius_sweep_manifest",
    "validate_arm_campaign_payload",
    "validate_arm_fixed_factors",
    "write_radius_sweep_manifest",
    "write_radius_sweep_manifest_check",
]
