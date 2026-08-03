"""Gate 0 post-hoc feasibility audit for the collision-envelope radius campaign (issue #6640).

This module records the *post-hoc-versus-replay boundary* for the radius-sensitivity campaign
defined in parent issue #6600. Gate 0 inspects the tracked ``0.0.3.post1`` release evidence,
canonical frozen row schema, and metric contract, then emits a machine-readable decision that classifies each
radius-sensitivity outcome as either:

- ``re-derivable`` -- the outcome at a new radius is an exact deterministic function of fields
  retained in the frozen release rows plus the radius metadata, under the frozen metric
  semantics, *and* it is not a trajectory-dependent planner/obstacle-contact/feasibility/collision
  outcome; or
- ``replay-required`` -- the outcome depends on the radius-arm trajectory (which differs across
  arms because the collision-envelope radius changes planner behaviour and simulator collision
  geometry), on per-timestep geometry that the aggregate frozen rows do not retain, or on
  effective radius/map provenance that the frozen release does not pin.

The module is diagnostic and read-only with respect to benchmark execution. It does **not** run
benchmark episodes, change any frozen ``0.0.3.post1`` metric semantics, release config, or
manifest, run production compute, or establish a planner ranking or a radius-sensitivity result.
It inspects the tracked release evidence and canonical row schema, records that the external
bundle is unavailable locally, and emits no re-derivable outcome when the retained provenance does
not establish the effective radius or pin the map asset bytes.

Stop conditions enforced programmatically by :func:`validate_gate0_decision`:

1. Trajectory-dependent planner behaviour, obstacle contact, feasibility, and collision outcomes
   are always ``replay-required``.
2. A re-derivable outcome requires its source geometry and semantics to be retained exactly.
3. The decision lists every outcome with exactly one classification.
4. Threshold reclassification of static geometry does not, by itself, infer a full radius sweep.
5. An unretained effective radius or unpinned map asset is not enough evidence for a re-derivable
   outcome.

See parent issue #6600 (Gate 0 spec), validity study #3207 (clearance-semantics foundation in
:mod:`robot_sf.benchmark.clearance_semantics`), the frozen release pointer under
``docs/context/evidence/issue_4364_release_0_0_3_post1/``, and the canonical row schema at
``robot_sf/benchmark/schemas/episode.schema.v1.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.constants import COLLISION_DIST, NEAR_MISS_DIST
from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest
from robot_sf.evidence.writers import write_json

GATE0_DECISION_SCHEMA = "radius_sensitivity_gate0_decision.v1"
GATE0_REVIEW_MARKER = "AI-GENERATED NEEDS-REVIEW"

# Campaign axis is the robot collision-envelope (planning proxy) radius in metres. The frozen
# ``0.0.3.post1`` baseline arm is 1.0 m; the sensitivity arms are 0.5 m and 0.8 m. See parent
# issue #6600 and the dissertation ruling referenced there.
CAMPAIGN_AXIS = "robot_collision_envelope_radius_m"
CAMPAIGN_BASELINE_ARM_M = 1.0
CAMPAIGN_ARMS_M: tuple[float, ...] = (0.5, 0.8, 1.0)

# Frozen ``0.0.3.post1`` release provenance. Sourced from the tracked artifact pointer and the
# diagnostic-only collision-consumer reconciliation; the episode rows themselves live in the
# attached release bundle, not in git.
FROZEN_RELEASE_TAG = "0.0.3.post1"
FROZEN_RELEASE_EPISODE_ROWS = 20160
FROZEN_RELEASE_ARMS = 14
FROZEN_RELEASE_ROWS_PER_ARM = 1440
FROZEN_RELEASE_EXECUTION_COMMIT = "a307ef276d701f8d14dead1aa0513f44ee97c0b0"
FROZEN_RELEASE_BUNDLE_SHA256 = "9bf6ea35a17ce812f0a9c841c3681bc072dcf7ba8c121cbcf05113b8514f4de1"
FROZEN_RELEASE_POINTER = (
    "docs/context/evidence/issue_4364_release_0_0_3_post1/artifact_pointer.json"
)
FROZEN_RELEASE_COLLISION_RECONCILIATION = (
    "docs/context/evidence/issue_4364_release_0_0_3_post1/collision_reconciliation.json"
)
FROZEN_RELEASE_CHECKSUMS = "docs/context/evidence/issue_4364_release_0_0_3_post1/SHA256SUMS"
FROZEN_RELEASE_PUBLICATION_PREFLIGHT = (
    "docs/context/evidence/issue_4364_release_0_0_3_post1/publication_preflight.json"
)
FROZEN_RELEASE_MANIFEST = (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml"
)
FROZEN_RELEASE_ROW_SCHEMA = "robot_sf/benchmark/schemas/episode.schema.v1.json"

REQUIRED_FROZEN_ROW_SCHEMA_FIELDS = (
    "version",
    "episode_id",
    "scenario_id",
    "seed",
    "metrics",
    "termination_reason",
    "outcome",
    "integrity",
)
EFFECTIVE_RADIUS_FIELDS = frozenset(
    {
        "robot_radius",
        "ped_radius",
        "pedestrian_radius",
        "effective_robot_radius_m",
        "effective_pedestrian_radius_m",
        "collision_envelope_radius_m",
    }
)
MAP_PROVENANCE_FIELDS = frozenset(
    {
        "map_asset_sha256",
        "map_asset_bytes_sha256",
        "map_asset_hashes",
        "map_bytes_sha256",
    }
)

# Metric contract grounding (see robot_sf/benchmark/metrics.py and constants.py).
#
# The collision-envelope radius binds the robot-pedestrian *clearance* family only:
#   clearance[t, k] = center_distance[t, k] - (robot_radius_m + ped_radius_m)
# Negative clearance is contact; ``human_collisions`` counts timesteps with negative minimum
# clearance, and ``near_misses`` counts timesteps in ``[0, NEAR_MISS_DIST)``.
RADIUS_AWARE_CLEARANCE_METRICS: tuple[str, ...] = (
    "human_collisions",
    "near_misses",
    "min_clearance",
    "mean_clearance",
)
CLEARANCE_FORMULA = "clearance[t,k] = center_distance[t,k] - (robot_radius_m + ped_radius_m)"

# Wall and agent collision predicates use the *fixed* ``COLLISION_DIST`` centre-distance
# threshold against point obstacles/agents; they do not subtract the collision-envelope radius
# in the metric formula. The radius still binds the *simulator* collision geometry during
# trajectory generation, so these counts remain trajectory-dependent across radius arms.
FIXED_THRESHOLD_COLLISION_METRICS: tuple[str, ...] = ("wall_collisions", "agent_collisions")

# Geometry metrics whose formula never references a radius. They are radius-independent for a
# fixed trajectory but still differ across radius arms because the trajectory itself changes.
RADIUS_INDEPENDENT_GEOMETRY_METRICS: tuple[str, ...] = (
    "clearing_distance_min",
    "clearing_distance_avg",
    "min_distance",
    "mean_distance",
    "robot_ped_within_5m_frac",
)

# The collision-envelope radius default is not uniform across the metric contract. This is a
# Gate 0 finding: post-hoc reclassification must confirm the per-row effective radius (from
# retained metadata / frozen config) before use; it must never assume a single default. Gate 1
# (binding canary) resolves the effective runtime binding.
METRICS_EPISODE_DATA_DEFAULT_ROBOT_RADIUS_M = 1.0
METRICS_EPISODE_DATA_DEFAULT_PED_RADIUS_M = 0.4
RUNNER_DEFAULT_ROBOT_RADIUS_M = 0.3
RUNNER_DEFAULT_PED_RADIUS_M = 0.35

CLAIM_BOUNDARY = (
    "radius_sensitivity_gate0_decision_diagnostic_only: a machine-readable decision recording the "
    "post-hoc-versus-replay boundary for the collision-envelope radius campaign. It does not run "
    "benchmark episodes, change frozen 0.0.3.post1 metric semantics, configs, or manifests, run "
    "production compute, or establish a radius-sensitivity result, planner ranking, feasibility "
    "verdict, or paper-facing benchmark evidence. No current outcome qualifies as re-derivable "
    "because the effective radius and map asset provenance are not retained/pinned; this decision "
    "must not be read as a radius sweep."
)

# Outcome categories that are unconditionally replay-required by stop condition #3.
TRAJECTORY_DEPENDENT_CATEGORIES = frozenset(
    {
        "scalar_metric_clearance_family",
        "scalar_metric_fixed_threshold",
        "scalar_metric_radius_independent_geometry",
        "binary_success",
        "aggregate_collision_count",
        "simulator_contact_outcome",
        "trajectory_feasibility_outcome",
        "planner_behaviour_outcome",
        "planner_ranking_outcome",
        "scenario_family_outcome",
        "scalar_metric_trajectory_dependent",
    }
)

RE_DERIVABLE = "re-derivable"
REPLAY_REQUIRED = "replay-required"
VALID_CLASSIFICATIONS = frozenset({RE_DERIVABLE, REPLAY_REQUIRED})
REQUIRED_OUTCOME_FIELDS = frozenset(
    {
        "outcome_id",
        "outcome",
        "category",
        "radius_binding",
        "source_geometry_retained_in_frozen_rows",
        "is_collision_contact_feasibility_or_planner_outcome",
        "classification",
        "rationale",
        "caveats",
    }
)


@dataclass(frozen=True)
class RadiusSensitivityOutcome:
    """One radius-sensitivity outcome and its Gate 0 classification."""

    outcome_id: str
    outcome: str
    category: str
    radius_binding: str
    source_geometry_retained_in_frozen_rows: bool
    is_collision_contact_feasibility_or_planner_outcome: bool
    classification: str
    rationale: str
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the outcome."""
        payload = asdict(self)
        payload["caveats"] = list(self.caveats)
        return payload


class FrozenReleaseEvidenceError(ValueError):
    """Raised when the tracked frozen-release evidence cannot be inspected safely."""


def _repository_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the repository root used for tracked evidence inspection.

    Returns:
        Repository root path.
    """
    return (
        Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    )


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON/YAML mapping and fail closed on malformed evidence.

    Returns:
        Parsed mapping payload.
    """
    try:
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.suffix.lower() == ".json"
            else yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise FrozenReleaseEvidenceError(f"could not load {label} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrozenReleaseEvidenceError(f"{label} must be a mapping: {path}")
    return payload


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Return a nested mapping or raise a provenance error."""
    if not isinstance(value, dict):
        raise FrozenReleaseEvidenceError(f"{label} must be a mapping")
    return value


def _require_int(value: Any, label: str) -> int:
    """Return an integer evidence field, rejecting booleans and missing values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenReleaseEvidenceError(f"{label} must be an integer")
    return value


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a tracked evidence file."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FrozenReleaseEvidenceError(f"could not hash tracked evidence {path}: {exc}") from exc
    return digest.hexdigest()


def _load_checksums(path: Path) -> dict[str, str]:
    """Load the compact checksum ledger shipped with the tracked release packet.

    Returns:
        Mapping from tracked filename to expected SHA-256 digest.
    """
    checksums: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FrozenReleaseEvidenceError(f"could not load checksum ledger {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise FrozenReleaseEvidenceError(f"malformed checksum entry at {path}:{line_number}")
        checksums[Path(parts[1]).name] = parts[0]
    return checksums


def _declared_schema_fields(schema: dict[str, Any]) -> set[str]:
    """Collect fields explicitly declared by nested JSON-schema ``properties`` blocks.

    Returns:
        Set of explicitly declared schema field names.
    """
    fields: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                fields.update(str(key) for key in properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return fields


def _mapping_keys(payload: Any) -> set[str]:
    """Collect keys from nested ordinary mappings such as YAML campaign metadata.

    Returns:
        Set of nested mapping keys.
    """
    keys: set[str] = set()
    if isinstance(payload, dict):
        keys.update(str(key) for key in payload)
        for value in payload.values():
            keys.update(_mapping_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            keys.update(_mapping_keys(value))
    return keys


def _assert_equal(label: str, *values: Any) -> Any:
    """Require all provenance copies of a value to agree and return the value.

    Returns:
        The shared evidence value.
    """
    if not values or any(value != values[0] for value in values[1:]):
        raise FrozenReleaseEvidenceError(f"tracked release evidence disagrees for {label}")
    return values[0]


def inspect_frozen_release_evidence(  # noqa: C901, PLR0915
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect and cross-check the tracked evidence backing the Gate 0 decision.

    The release episode bundle is an external asset and is intentionally not downloaded by this
    diagnostic builder. The tracked pointer, reconciliation summaries, release manifest, checksum
    ledger, and canonical row schema are still inspected and cross-checked. If any of those
    sources disagree, the builder fails closed instead of emitting a decision from constants.

    Returns:
        JSON-serialisable evidence and inspection blocks used by the decision builder.
    """
    root = _repository_root(repo_root)
    pointer_path = root / FROZEN_RELEASE_POINTER
    reconciliation_path = root / FROZEN_RELEASE_COLLISION_RECONCILIATION
    checksums_path = root / FROZEN_RELEASE_CHECKSUMS
    preflight_path = root / FROZEN_RELEASE_PUBLICATION_PREFLIGHT
    row_schema_path = root / FROZEN_RELEASE_ROW_SCHEMA

    pointer = _load_mapping(pointer_path, "artifact pointer")
    reconciliation = _load_mapping(reconciliation_path, "collision reconciliation")
    preflight = _load_mapping(preflight_path, "publication preflight")
    row_schema = _load_mapping(row_schema_path, "frozen episode row schema")

    checksums = _load_checksums(checksums_path)
    for path in (pointer_path, reconciliation_path, preflight_path):
        expected_digest = checksums.get(path.name)
        if expected_digest is None or expected_digest != _sha256(path):
            raise FrozenReleaseEvidenceError(
                f"checksum ledger does not verify tracked evidence file {path}"
            )

    pointer_artifact = _require_mapping(pointer.get("artifact"), "artifact_pointer.artifact")
    pointer_contract = _require_mapping(
        pointer.get("row_contract"), "artifact_pointer.row_contract"
    )
    pointer_provenance = _require_mapping(pointer.get("provenance"), "artifact_pointer.provenance")
    reconciliation_counts = _require_mapping(
        reconciliation.get("counts"), "collision_reconciliation.counts"
    )
    reconciliation_source = _require_mapping(
        reconciliation.get("source"), "collision_reconciliation.source"
    )
    reconciliation_provenance = _require_mapping(
        reconciliation.get("provenance"), "collision_reconciliation.provenance"
    )
    episode_commits = _require_mapping(
        reconciliation_provenance.get("episode_software_commits"),
        "collision_reconciliation.provenance.episode_software_commits",
    )

    release_tag = _assert_equal(
        "release_tag",
        pointer.get("release_tag"),
        reconciliation_source.get("release_tag"),
        FROZEN_RELEASE_TAG,
    )
    bundle_sha256 = _assert_equal(
        "bundle_sha256",
        pointer_artifact.get("sha256"),
        reconciliation_source.get("bundle_sha256"),
        FROZEN_RELEASE_BUNDLE_SHA256,
    )
    episode_rows = _assert_equal(
        "episode_rows",
        _require_int(
            pointer_contract.get("episode_rows"), "artifact_pointer.row_contract.episode_rows"
        ),
        _require_int(reconciliation_counts.get("rows"), "collision_reconciliation.counts.rows"),
        FROZEN_RELEASE_EPISODE_ROWS,
    )
    arms = _assert_equal(
        "arms",
        _require_int(pointer_contract.get("arms"), "artifact_pointer.row_contract.arms"),
        _require_int(reconciliation_counts.get("arms"), "collision_reconciliation.counts.arms"),
        FROZEN_RELEASE_ARMS,
    )
    rows_per_arm = _assert_equal(
        "rows_per_arm",
        _require_int(
            pointer_contract.get("rows_per_arm"), "artifact_pointer.row_contract.rows_per_arm"
        ),
        FROZEN_RELEASE_ROWS_PER_ARM,
    )
    release_manifest_path = str(pointer_provenance.get("release_manifest", "")).strip()
    if release_manifest_path != FROZEN_RELEASE_MANIFEST:
        raise FrozenReleaseEvidenceError(
            "artifact pointer release_manifest does not match the tracked 0.0.3.post1 manifest"
        )
    manifest_path = root / release_manifest_path
    try:
        manifest = load_release_manifest(manifest_path)
        manifest_validation = validate_release_manifest(manifest)
    except (OSError, ValueError) as exc:
        raise FrozenReleaseEvidenceError(
            f"could not validate frozen release manifest {manifest_path}: {exc}"
        ) from exc
    if manifest_validation.get("status") != "valid":
        raise FrozenReleaseEvidenceError(
            f"frozen release manifest validation failed: {manifest_validation.get('problems', [])}"
        )

    execution_commit = _assert_equal(
        "execution_commit",
        pointer_provenance.get("execution_commit"),
        next(iter(episode_commits)) if len(episode_commits) == 1 else None,
        FROZEN_RELEASE_EXECUTION_COMMIT,
    )
    if (
        _require_int(episode_commits.get(execution_commit), "episode commit row count")
        != episode_rows
    ):
        raise FrozenReleaseEvidenceError(
            "collision reconciliation episode commit count does not match episode_rows"
        )
    if preflight.get("status") != "pass" or reconciliation.get("status") != "pass":
        raise FrozenReleaseEvidenceError(
            "tracked frozen-release validation summaries are not passing"
        )
    if (
        _require_int(
            reconciliation.get("violation_count"), "collision_reconciliation.violation_count"
        )
        != 0
    ):
        raise FrozenReleaseEvidenceError("collision reconciliation reports violations")

    schema_properties = _require_mapping(row_schema.get("properties"), "row schema properties")
    row_schema_required = row_schema.get("required")
    if not isinstance(row_schema_required, list) or not all(
        isinstance(field, str) for field in row_schema_required
    ):
        raise FrozenReleaseEvidenceError("frozen row schema required must be a string list")
    missing_required_fields = sorted(
        set(REQUIRED_FROZEN_ROW_SCHEMA_FIELDS) - set(row_schema_required)
    )
    if missing_required_fields:
        raise FrozenReleaseEvidenceError(
            f"frozen row schema is missing required fields: {missing_required_fields}"
        )
    if row_schema.get("properties", {}).get("version", {}).get("const") != "v1":
        raise FrozenReleaseEvidenceError("frozen row schema must declare version v1")

    declared_schema_fields = _declared_schema_fields(row_schema)
    retained_radius_fields = sorted(EFFECTIVE_RADIUS_FIELDS & declared_schema_fields)
    retained_map_fields = sorted(MAP_PROVENANCE_FIELDS & declared_schema_fields)
    metric_parameters = _require_mapping(
        schema_properties.get("metric_parameters"), "row schema metric_parameters"
    )
    metric_parameter_fields = sorted(
        _require_mapping(
            metric_parameters.get("properties"), "row schema metric_parameters.properties"
        )
    )
    campaign_payload = _load_mapping(
        manifest.canonical_campaign_config_path, "frozen release campaign configuration"
    )
    manifest_fields = _mapping_keys(campaign_payload)
    manifest_radius_fields = sorted(EFFECTIVE_RADIUS_FIELDS & manifest_fields)
    manifest_map_fields = sorted(MAP_PROVENANCE_FIELDS & manifest_fields)

    effective_radius_retained = bool(retained_radius_fields or manifest_radius_fields)
    map_asset_bytes_pinned = bool(retained_map_fields or manifest_map_fields)
    provenance_gaps = {
        "effective_robot_and_pedestrian_radius_retained": effective_radius_retained,
        "map_asset_bytes_pinned": map_asset_bytes_pinned,
        "finding": (
            "Tracked row schema/config fields do not retain an effective robot/pedestrian radius "
            "or map-asset byte digest; the external release bundle is not materialized for row "
            "inspection."
            if not effective_radius_retained and not map_asset_bytes_pinned
            else "Tracked provenance fields require further row-level inspection before reclassification."
        ),
    }
    frozen_release = {
        "release_tag": release_tag,
        "episode_rows": episode_rows,
        "arms": arms,
        "rows_per_arm": rows_per_arm,
        "execution_commit": execution_commit,
        "bundle_sha256": bundle_sha256,
        "artifact_pointer": FROZEN_RELEASE_POINTER,
        "collision_reconciliation_pointer": FROZEN_RELEASE_COLLISION_RECONCILIATION,
        "release_manifest": release_manifest_path,
        "row_schema": FROZEN_RELEASE_ROW_SCHEMA,
        "row_location": (
            f"External release asset {pointer_artifact.get('name')!r}; tracked row schema and "
            "release metadata were inspected, but the bundle bytes are not materialized in the "
            "checkout."
        ),
    }
    inspection = {
        "status": "tracked_metadata_inspected_bundle_unavailable",
        "decision_basis": "tracked_release_evidence_and_frozen_row_schema",
        "bundle": {
            "status": "unavailable",
            "name": pointer_artifact.get("name"),
            "sha256": bundle_sha256,
            "reason": "The release bundle is an external asset; no local bundle directory was supplied.",
        },
        "tracked_files": {
            FROZEN_RELEASE_POINTER: {"sha256": _sha256(pointer_path), "verified": True},
            FROZEN_RELEASE_COLLISION_RECONCILIATION: {
                "sha256": _sha256(reconciliation_path),
                "verified": True,
            },
            FROZEN_RELEASE_PUBLICATION_PREFLIGHT: {
                "sha256": _sha256(preflight_path),
                "verified": True,
            },
            FROZEN_RELEASE_CHECKSUMS: {"verified": True},
        },
        "row_schema": {
            "path": FROZEN_RELEASE_ROW_SCHEMA,
            "schema_version": row_schema["properties"]["version"]["const"],
            "required_fields": list(row_schema_required),
            "declared_effective_radius_fields": retained_radius_fields,
            "declared_map_provenance_fields": retained_map_fields,
            "metric_parameter_fields": metric_parameter_fields,
        },
        "release_manifest": {
            "path": release_manifest_path,
            "release_tag": manifest.release_tag,
            "scenario_matrix": manifest.scenario_matrix_path.relative_to(root).as_posix(),
            "scenario_matrix_sha256": manifest.scenario_matrix_sha256,
            "declared_effective_radius_fields": manifest_radius_fields,
            "declared_map_provenance_fields": manifest_map_fields,
            "validation": manifest_validation,
        },
        "cross_checks": {
            "release_tag_matches": True,
            "bundle_sha256_matches": True,
            "episode_rows_matches": True,
            "arms_matches": True,
            "rows_per_arm_matches": True,
            "execution_commit_matches": True,
            "row_schema_required_fields_present": True,
            "reconciliation_status": reconciliation.get("status"),
            "publication_preflight_status": preflight.get("status"),
        },
    }
    return {
        "frozen_release": frozen_release,
        "provenance_gaps": provenance_gaps,
        "inspection": inspection,
    }


def _clearance_family_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Radius-aware clearance-family metrics; all replay-required.

    ``min_clearance``/``mean_clearance`` shift linearly with the radius *for one fixed
    trajectory*, but the radius-arm trajectory itself differs across arms, so the cross-arm value
    is not re-derivable from the retained baseline aggregate. ``human_collisions`` and
    ``near_misses`` are threshold counts over a per-timestep clearance distribution that the
    aggregate frozen rows do not retain, so they require the full replay trajectory (see
    ``robot_sf/benchmark/threshold_sensitivity.py``, which recomputes these counts from
    ``replay_steps``/``replay_peds`` and never from aggregate rows).

    Returns:
        Tuple of replay-required radius-aware clearance-family outcomes.
    """
    common_caveat = (
        "Cross-arm value is trajectory-dependent: the radius changes planner behaviour and "
        "simulator collision geometry, so the 0.5 m / 0.8 m arm trajectory differs from the "
        "retained 1.0 m baseline trajectory."
    )
    return (
        RadiusSensitivityOutcome(
            outcome_id="human_collisions_count",
            outcome="Per-episode pedestrian (human) collision count",
            category="scalar_metric_clearance_family",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Counts timesteps where min clearance < 0. Reclassifying the count at a new radius "
                "needs the per-timestep clearance distribution; only the aggregate count is "
                "retained. Collision outcomes are replay-required by stop condition #3."
            ),
            caveats=(
                common_caveat,
                "threshold_sensitivity.near_miss_count / human_collisions require full replay "
                "trajectories, not aggregate rows.",
            ),
        ),
        RadiusSensitivityOutcome(
            outcome_id="near_misses_count",
            outcome="Per-episode pedestrian near-miss count",
            category="scalar_metric_clearance_family",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Counts timesteps where 0 <= min clearance < NEAR_MISS_DIST. A threshold-count "
                "over a non-retained per-timestep distribution; not re-derivable from the "
                "retained aggregate."
            ),
            caveats=(common_caveat,),
        ),
        RadiusSensitivityOutcome(
            outcome_id="min_clearance_scalar",
            outcome="Per-episode minimum robot-pedestrian surface clearance",
            category="scalar_metric_clearance_family",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=False,
            classification=REPLAY_REQUIRED,
            rationale=(
                "For one fixed trajectory, min_clearance shifts by -(delta_robot_radius) exactly. "
                "Across radius arms the centre-distance trajectory differs, so the cross-arm value "
                "is not re-derivable from the retained baseline aggregate."
            ),
            caveats=(
                common_caveat,
                "The within-trajectory linear shift is a flown-baseline diagnostic only, not a "
                "cross-arm re-derivation.",
            ),
        ),
        RadiusSensitivityOutcome(
            outcome_id="mean_clearance_scalar",
            outcome="Per-episode mean robot-pedestrian surface clearance",
            category="scalar_metric_clearance_family",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=False,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Mean of per-step minimum clearance; cross-arm value is trajectory-dependent and "
                "not re-derivable from the retained baseline aggregate."
            ),
            caveats=(common_caveat,),
        ),
    )


def _fixed_threshold_collision_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Wall/agent collision counts use a fixed threshold but are still trajectory-dependent.

    Returns:
        Tuple of replay-required fixed-threshold collision-count outcomes.
    """
    caveat = (
        "The metric formula uses the fixed COLLISION_DIST centre-distance threshold and does not "
        "subtract the collision-envelope radius, but the radius binds the simulator collision "
        "geometry during trajectory generation, so the count still differs across radius arms."
    )
    outcomes: list[RadiusSensitivityOutcome] = []
    for metric, label in (
        ("wall_collisions", "Per-episode wall/obstacle collision count"),
        ("agent_collisions", "Per-episode other-agent collision count"),
    ):
        outcomes.append(
            RadiusSensitivityOutcome(
                outcome_id=f"{metric}_count",
                outcome=label,
                category="scalar_metric_fixed_threshold",
                radius_binding="fixed_threshold_metric_radius_aware_simulator",
                source_geometry_retained_in_frozen_rows=False,
                is_collision_contact_feasibility_or_planner_outcome=True,
                classification=REPLAY_REQUIRED,
                rationale=(
                    f"{metric} uses the fixed {COLLISION_DIST} m centre-distance threshold against "
                    "point obstacles/agents. The count is trajectory-dependent across radius arms "
                    "and is a collision outcome (stop condition #3)."
                ),
                caveats=(caveat,),
            )
        )
    return tuple(outcomes)


def _radius_independent_geometry_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Geometry metrics with no radius term; cross-arm value still trajectory-dependent.

    Returns:
        Tuple of replay-required radius-independent-geometry metric outcomes.
    """
    rows = (
        ("clearing_distance_min", "Per-episode minimum robot-centre-to-obstacle-point distance"),
        ("clearing_distance_avg", "Per-episode mean robot-centre-to-obstacle-point distance"),
        ("min_distance", "Per-episode minimum robot-pedestrian centre-to-centre distance"),
        ("mean_distance", "Per-episode mean robot-pedestrian centre-to-centre distance"),
        ("robot_ped_within_5m_frac", "Per-episode fraction of steps within 5 m of a pedestrian"),
    )
    outcomes: list[RadiusSensitivityOutcome] = []
    for metric, label in rows:
        cd_caveat = (
            "The retained value is a radius-independent proximity bound on the flown 1.0 m "
            "baseline trajectory only; it does not reconstruct the 0.5 m / 0.8 m arm value."
            if metric.startswith("clearing_distance")
            else (
                "Centre-to-centre distance is radius-independent in formula; the cross-arm value "
                "still differs because the trajectory differs."
            )
        )
        outcomes.append(
            RadiusSensitivityOutcome(
                outcome_id=f"{metric}_scalar",
                outcome=label,
                category="scalar_metric_radius_independent_geometry",
                radius_binding="none_in_formula_trajectory_via_simulator",
                source_geometry_retained_in_frozen_rows=False,
                is_collision_contact_feasibility_or_planner_outcome=False,
                classification=REPLAY_REQUIRED,
                rationale=(
                    f"{metric} has no radius term in its formula, but the value is computed on the "
                    "radius-arm trajectory, which differs across arms; not re-derivable from the "
                    "retained baseline aggregate."
                ),
                caveats=(cd_caveat,),
            )
        )
    return tuple(outcomes)


def _success_and_aggregate_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Binary success and aggregate collision counts; replay-required.

    Returns:
        Tuple of replay-required success and aggregate collision-count outcomes.
    """
    return (
        RadiusSensitivityOutcome(
            outcome_id="binary_success",
            outcome="Per-episode binary success (reached goal AND zero collisions)",
            category="binary_success",
            radius_binding="collision_gate_clearance_plus_timing_gate",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Success gates on total_collision_count == 0 (radius-aware via human_collisions) "
                "AND reached_goal_step < horizon. The collision gate is replay-required; the "
                "frozen collision reconciliation also warns the bundle lacks reached_goal_step / "
                "horizon inputs to recompute the timing gate."
            ),
            caveats=(
                "The frozen release collision reconciliation records one row with both "
                "goal_reached and timeout and warns the timing-gate inputs are not retained.",
            ),
        ),
        RadiusSensitivityOutcome(
            outcome_id="total_collision_count",
            outcome="Per-episode total collision count (pedestrian + wall + agent)",
            category="aggregate_collision_count",
            radius_binding="aggregate_of_clearance_and_fixed_threshold_counts",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Aggregates human_collisions (radius-aware), wall_collisions and agent_collisions "
                "(fixed threshold). All three are trajectory-dependent collision outcomes."
            ),
            caveats=(),
        ),
        RadiusSensitivityOutcome(
            outcome_id="ped_collision_count",
            outcome="Per-episode pedestrian collision count (alias of human_collisions)",
            category="aggregate_collision_count",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Stored on the row as ped_collision_count and equal to human_collisions; a "
                "radius-aware collision outcome (stop condition #3)."
            ),
            caveats=(),
        ),
    )


def _trajectory_simulator_planner_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Trajectory-, simulator-, planner-, and family-level outcomes; all replay-required.

    Returns:
        Tuple of replay-required trajectory/simulator/planner/family outcomes.
    """
    rows = (
        (
            "simulator_obstacle_contact",
            "Simulator-level obstacle (wall) contact during trajectory generation",
            "simulator_contact_outcome",
            "The collision-envelope radius binds simulator collision geometry; contact is a "
            "trajectory-dependent collision/contact outcome (stop condition #3).",
            True,
        ),
        (
            "geometric_body_pedestrian_contact",
            "Physical body pedestrian contact (geometric-body clearance <= contact threshold)",
            "simulator_contact_outcome",
            "Physical contact is a trajectory-dependent contact outcome (stop condition #3).",
            True,
        ),
        (
            "trajectory_feasibility_traversal_executed",
            "Whether a collision-free traversal was actually executed to the goal",
            "trajectory_feasibility_outcome",
            "Executed-traversal feasibility is trajectory-dependent (stop condition #3); the "
            "static-map margin reclassification does not reconstruct it.",
            True,
        ),
        (
            "planner_behaviour_decisions",
            "Planner decisions / control actions along the trajectory",
            "planner_behaviour_outcome",
            "The radius changes planner inputs and behaviour; planner decisions are "
            "trajectory-dependent (stop condition #3).",
            True,
        ),
        (
            "planner_rankings_success_typed_collisions_snqi",
            "Planner rankings on success, typed collisions, and SNQI",
            "planner_ranking_outcome",
            "Rankings aggregate radius-arm episodes; re-deriving them requires the per-arm "
            "trajectories (stop condition #3).",
            True,
        ),
        (
            "scenario_family_conclusions_transitions",
            "Scenario-family conclusions and transitions (e.g. narrow-doorway family)",
            "scenario_family_outcome",
            "Family-level transitions depend on per-arm episode outcomes and are replay-required.",
            True,
        ),
        (
            "snqi_per_episode",
            "Per-episode Social Navigation Quality Index (SNQI)",
            "scalar_metric_trajectory_dependent",
            "SNQI consumes collision/clearance and trajectory metrics; re-deriving the cross-arm "
            "value requires the radius-arm trajectory.",
            True,
        ),
        (
            "kinematic_efficiency_metrics",
            (
                "Kinematic / efficiency metrics (path efficiency, speed, jerk, curvature, energy, "
                "force quantiles, etc.)"
            ),
            "scalar_metric_trajectory_dependent",
            "Computed on the radius-arm trajectory; cross-arm values are trajectory-dependent.",
            False,
        ),
    )
    outcomes: list[RadiusSensitivityOutcome] = []
    for outcome_id, label, category, rationale, is_collision_family in rows:
        outcomes.append(
            RadiusSensitivityOutcome(
                outcome_id=outcome_id,
                outcome=label,
                category=category,
                radius_binding="trajectory_via_simulator_and_planner",
                source_geometry_retained_in_frozen_rows=False,
                is_collision_contact_feasibility_or_planner_outcome=is_collision_family,
                classification=REPLAY_REQUIRED,
                rationale=rationale,
                caveats=(),
            )
        )
    return tuple(outcomes)


def _provenance_blocked_outcomes(
    evidence: dict[str, Any],
) -> tuple[RadiusSensitivityOutcome, ...]:
    """Classify the tempting post-hoc diagnostics as replay-required.

    The prior implementation treated these two diagnostics as re-derivable. The frozen release
    manifest does not record the effective per-row robot/pedestrian radius, and its scenario
    matrix checksum does not pin the referenced map asset bytes. Until those provenance gaps are
    closed, neither diagnostic satisfies the exact-retention rule.

    Returns:
        Tuple of the two provenance-blocked diagnostic outcomes.
    """
    gaps = evidence["provenance_gaps"]
    inspection = evidence["inspection"]
    row_schema = inspection["row_schema"]
    release_manifest = inspection["release_manifest"]
    bundle_status = inspection["bundle"]["status"]
    return (
        RadiusSensitivityOutcome(
            outcome_id="retained_radius_and_threshold_parameters",
            outcome=(
                "Retained radius / threshold parameters (robot_radius, ped_radius, COLLISION_DIST, "
                "NEAR_MISS_DIST) recoverable from the frozen config and metric constants"
            ),
            category="metadata_parameter",
            radius_binding="parameter_value",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=False,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Tracked release evidence and the frozen row schema were inspected, but no effective "
                "per-row robot/pedestrian radius field was found in the retained metadata. The "
                "external bundle is not materialized, so the effective parameter provenance must be "
                "recovered before any post-hoc reclassification; this outcome is replay-required."
            ),
            caveats=(
                f"Evidence inspection status is {inspection['status']}; bundle status is {bundle_status}.",
                "Observed tracked provenance flags: "
                f"effective_radius_retained={gaps['effective_robot_and_pedestrian_radius_retained']}, "
                f"map_asset_bytes_pinned={gaps['map_asset_bytes_pinned']}.",
                f"Row-schema effective-radius fields: {row_schema['declared_effective_radius_fields']!r}; "
                f"release-metadata effective-radius fields: "
                f"{release_manifest['declared_effective_radius_fields']!r}.",
                "The collision-envelope radius default is not uniform across the metric contract "
                f"(metrics.py EpisodeData default robot_radius={METRICS_EPISODE_DATA_DEFAULT_ROBOT_RADIUS_M} m, "
                f"ped_radius={METRICS_EPISODE_DATA_DEFAULT_PED_RADIUS_M} m; runner.py default "
                f"robot_radius={RUNNER_DEFAULT_ROBOT_RADIUS_M} m, ped_radius={RUNNER_DEFAULT_PED_RADIUS_M} m). "
                "Gate 1 must confirm the per-row effective radius before any reclassification.",
                "A metric constant or default value is not evidence of the effective value used by "
                "the frozen release rows.",
            ),
        ),
        RadiusSensitivityOutcome(
            outcome_id="static_map_geometry_feasibility_margin",
            outcome=(
                "Static map-geometry feasibility margin (e.g. doorway gap minus swept diameter "
                "2*radius) on frozen map geometry"
            ),
            category="static_geometry_diagnostic",
            radius_binding="swept_envelope_parameter_on_frozen_geometry",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=False,
            classification=REPLAY_REQUIRED,
            rationale=(
                "A static map margin would be a radius-only reparameterisation if the exact map "
                "asset were retained. Tracked release metadata exposes a scenario-matrix digest but "
                "no map-asset byte digest, and the external bundle is not materialized for row-level "
                "inspection. The exact geometry provenance must be recovered before this margin can "
                "be classified, so it is replay-required."
            ),
            caveats=(
                f"Row-schema map-provenance fields: {row_schema['declared_map_provenance_fields']!r}; "
                f"release-metadata map-provenance fields: "
                f"{release_manifest['declared_map_provenance_fields']!r}.",
                "The scenario matrix checksum is not a checksum of its included map assets.",
                "Threshold reclassification of static geometry does NOT, by itself, infer a full "
                "radius sweep (stop condition #4).",
                "A positive static margin would not reconstruct scripted-traversal or planner "
                "feasibility, which remain replay-required (the #5574 0.5 m probe reclassifies the "
                "doorway as solvable yet its scripted traversal still collided).",
            ),
        ),
    )


def build_outcome_registry(
    evidence: dict[str, Any] | None = None,
    *,
    repo_root: str | Path | None = None,
) -> tuple[RadiusSensitivityOutcome, ...]:
    """Return the full Gate 0 outcome registry bound to inspected release evidence."""
    evidence = evidence or inspect_frozen_release_evidence(repo_root)
    return (
        *_clearance_family_outcomes(),
        *_fixed_threshold_collision_outcomes(),
        *_radius_independent_geometry_outcomes(),
        *_success_and_aggregate_outcomes(),
        *_trajectory_simulator_planner_outcomes(),
        *_provenance_blocked_outcomes(evidence),
    )


def _campaign_block() -> dict[str, Any]:
    return {
        "axis": CAMPAIGN_AXIS,
        "arms_m": list(CAMPAIGN_ARMS_M),
        "baseline_arm_m": CAMPAIGN_BASELINE_ARM_M,
        "scenario_surface": "classic_interactions_francis2023_48_cells",
        "planners": "complete_14_planner_release_roster",
        "seeds": "paper_eval_s30 (111-140)",
        "horizon_steps": 600,
    }


def _frozen_release_block(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return frozen-release identity copied from the inspected evidence packet."""
    return dict(evidence["frozen_release"])


def _metric_contract_block(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return metric-contract metadata plus provenance gaps observed in tracked evidence."""
    inspection = evidence["inspection"]
    return {
        "radius_aware_clearance_metrics": list(RADIUS_AWARE_CLEARANCE_METRICS),
        "clearance_formula": CLEARANCE_FORMULA,
        "fixed_threshold_collision_metrics": {
            metric: {"threshold_m": COLLISION_DIST, "uses_radius_in_formula": False}
            for metric in FIXED_THRESHOLD_COLLISION_METRICS
        },
        "radius_independent_geometry_metrics": list(RADIUS_INDEPENDENT_GEOMETRY_METRICS),
        "collision_constants": {
            "COLLISION_DIST_m": COLLISION_DIST,
            "NEAR_MISS_DIST_m": NEAR_MISS_DIST,
        },
        "radius_default_inconsistency": {
            "metrics_episode_data_default_robot_radius_m": METRICS_EPISODE_DATA_DEFAULT_ROBOT_RADIUS_M,
            "metrics_episode_data_default_ped_radius_m": METRICS_EPISODE_DATA_DEFAULT_PED_RADIUS_M,
            "runner_default_robot_radius_m": RUNNER_DEFAULT_ROBOT_RADIUS_M,
            "runner_default_ped_radius_m": RUNNER_DEFAULT_PED_RADIUS_M,
            "finding": (
                "Collision-envelope radius default is not uniform across the metric contract, and "
                "the inspected release metadata does not bind an effective per-row radius; Gate 1 "
                "must confirm the runtime binding before any post-hoc reclassification."
            ),
        },
        "frozen_provenance_gaps": dict(evidence["provenance_gaps"]),
        "frozen_row_schema": dict(inspection["row_schema"]),
        "frozen_release_manifest": dict(inspection["release_manifest"]),
        "tracked_evidence_cross_checks": dict(inspection["cross_checks"]),
        "threshold_sensitivity_requires_replay": (
            "robot_sf/benchmark/threshold_sensitivity.py recomputes near-miss/comfort counts from "
            "full replay trajectories (replay_steps/replay_peds), never from aggregate frozen rows."
        ),
    }


def _rubric_block() -> dict[str, Any]:
    return {
        "re_derivable": (
            "The outcome at a new radius is an exact deterministic function of fields retained in "
            "the frozen release rows plus radius metadata and pinned source assets, under the "
            "frozen metric semantics, "
            "AND it is not a trajectory-dependent planner/obstacle-contact/feasibility/collision "
            "outcome, AND it is not a threshold-count over a non-retained per-timestep distribution."
        ),
        "replay_required": (
            "The outcome depends on the radius-arm trajectory (which differs across arms because "
            "the collision-envelope radius changes planner behaviour and simulator collision "
            "geometry), on per-timestep geometry the aggregate frozen rows do not retain, or on "
            "effective radius/map provenance that the frozen release does not pin."
        ),
        "stop_conditions": [
            "Do not modify frozen 0.0.3.post1 metric semantics or any release config or manifest.",
            "Do not run any production SLURM compute.",
            "Post-hoc reclassification is allowed only for quantities whose source geometry and "
            "semantics are retained exactly; trajectory-dependent planner behaviour, obstacle "
            "contact, feasibility, and collision outcomes must be classified replay-required.",
            "The output lists every outcome as re-derivable or replay-required.",
            "Do not infer a full sweep from threshold reclassification alone.",
            "Do not classify an outcome as re-derivable when effective radius or map asset bytes "
            "are not retained and pinned by the frozen release provenance.",
        ],
    }


def _summary_block(outcomes: tuple[RadiusSensitivityOutcome, ...]) -> dict[str, Any]:
    re_derivable = [o.outcome_id for o in outcomes if o.classification == RE_DERIVABLE]
    replay_required = [o.outcome_id for o in outcomes if o.classification == REPLAY_REQUIRED]
    return {
        "total_outcomes": len(outcomes),
        "re_derivable_count": len(re_derivable),
        "replay_required_count": len(replay_required),
        "re_derivable_outcome_ids": re_derivable,
        "replay_required_outcome_ids": replay_required,
    }


def build_gate0_decision(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build the deterministic Gate 0 decision payload.

    The payload is validated in place by :func:`validate_gate0_decision` before it is returned, so
    a stop-condition violation raises rather than emitting an inconsistent decision.

    Returns:
        Validated ``radius_sensitivity_gate0_decision.v1`` decision payload.
    """
    evidence = inspect_frozen_release_evidence(repo_root)
    outcomes = build_outcome_registry(evidence)
    decision: dict[str, Any] = {
        "schema_version": GATE0_DECISION_SCHEMA,
        "issue": 6640,
        "parent_issue": 6600,
        "validity_study_issue": 3207,
        "gate": "gate0_post_hoc_feasibility_audit",
        "campaign": _campaign_block(),
        "frozen_release": _frozen_release_block(evidence),
        "metric_contract": _metric_contract_block(evidence),
        "classification_rubric": _rubric_block(),
        "outcomes": [o.to_dict() for o in outcomes],
        "summary": _summary_block(outcomes),
        "evidence_inspection": evidence["inspection"],
        "claim_boundary": CLAIM_BOUNDARY,
        "next_gate": "gate1_binding_canary",
        "review_marker": GATE0_REVIEW_MARKER,
    }
    validate_gate0_decision(decision, repo_root=repo_root)
    return decision


# No outcome currently meets the exact-retention rule. Keep this explicit so a future change cannot
# silently turn an unpinned parameter or map source into benchmark evidence.
ALLOWED_RE_DERIVABLE_IDS = frozenset()


def _validate_outcome_schema_fields(entry: dict[str, Any]) -> None:
    """Validate the required fields and types of one outcome entry."""
    missing_fields = REQUIRED_OUTCOME_FIELDS - entry.keys()
    if missing_fields:
        raise ValueError(f"each outcome is missing required fields: {sorted(missing_fields)}")

    for field in ("outcome_id", "outcome", "category", "radius_binding", "rationale"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            raise ValueError(f"outcome field {field!r} must be a non-empty string")
    for field in (
        "source_geometry_retained_in_frozen_rows",
        "is_collision_contact_feasibility_or_planner_outcome",
    ):
        if not isinstance(entry[field], bool):
            raise ValueError(f"outcome field {field!r} must be a boolean")
    if not isinstance(entry["caveats"], list) or not all(
        isinstance(caveat, str) and caveat.strip() for caveat in entry["caveats"]
    ):
        raise ValueError("outcome field 'caveats' must be a list of non-empty strings")


def _validate_outcome_entry(entry: Any, seen_ids: set[str]) -> str | None:
    """Validate one outcome entry and return its id when it is re-derivable.

    Returns:
        The outcome id when the entry is classified ``re-derivable``, otherwise ``None``.
    """
    if not isinstance(entry, dict):
        raise ValueError("each outcome must be a dict")
    _validate_outcome_schema_fields(entry)

    outcome_id = entry.get("outcome_id")
    if not isinstance(outcome_id, str) or not outcome_id:
        raise ValueError("each outcome needs a non-empty string outcome_id")
    if outcome_id in seen_ids:
        raise ValueError(f"duplicate outcome_id {outcome_id!r}")
    seen_ids.add(outcome_id)

    classification = entry.get("classification")
    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"outcome {outcome_id!r} classification must be one of {sorted(VALID_CLASSIFICATIONS)}"
        )

    is_collision_family = bool(entry.get("is_collision_contact_feasibility_or_planner_outcome"))
    category = entry.get("category")
    _assert_replay_required_for_trajectory_outcome(
        outcome_id, classification, is_collision_family, category
    )

    if classification != RE_DERIVABLE:
        return None
    _assert_re_derivable_constraints(entry, outcome_id, is_collision_family)
    return outcome_id


def _assert_replay_required_for_trajectory_outcome(
    outcome_id: str,
    classification: str,
    is_collision_family: bool,
    category: Any,
) -> None:
    """Stop condition #3: trajectory-dependent outcomes are always replay-required."""
    if is_collision_family and classification != REPLAY_REQUIRED:
        raise ValueError(
            f"outcome {outcome_id!r} is a collision/contact/feasibility/planner outcome and "
            f"must be {REPLAY_REQUIRED!r}, got {classification!r}"
        )
    if category in TRAJECTORY_DEPENDENT_CATEGORIES and classification != REPLAY_REQUIRED:
        raise ValueError(
            f"outcome {outcome_id!r} category {category!r} is trajectory-dependent and must be "
            f"{REPLAY_REQUIRED!r}"
        )


def _assert_re_derivable_constraints(
    entry: dict[str, Any], outcome_id: str, is_collision_family: bool
) -> None:
    """Stop condition #2: re-derivable requires retained source geometry and semantics."""
    if not bool(entry.get("source_geometry_retained_in_frozen_rows")):
        raise ValueError(
            f"re-derivable outcome {outcome_id!r} must retain its source geometry exactly"
        )
    if is_collision_family:
        raise ValueError(
            f"re-derivable outcome {outcome_id!r} must not be a collision/contact/"
            "feasibility/planner outcome"
        )


def _assert_narrow_re_derivable_set(re_derivable_ids: list[str]) -> None:
    """Stop condition #5: only explicitly retained outcomes may be re-derivable."""
    extra = set(re_derivable_ids) - ALLOWED_RE_DERIVABLE_IDS
    if extra:
        raise ValueError(
            "re-derivable set contains outcomes without exact retained provenance and would imply "
            "a sweep: "
            f"{sorted(extra)}"
        )


def _assert_frozen_provenance_gaps(decision: dict[str, Any]) -> None:
    """Enforce the provenance blockers that make this Gate 0 decision fail closed."""
    metric_contract = decision.get("metric_contract")
    if not isinstance(metric_contract, dict):
        raise ValueError("decision.metric_contract must be a dict")
    gaps = metric_contract.get("frozen_provenance_gaps")
    if not isinstance(gaps, dict):
        raise ValueError("metric_contract.frozen_provenance_gaps must be a dict")
    for field in (
        "effective_robot_and_pedestrian_radius_retained",
        "map_asset_bytes_pinned",
    ):
        if gaps.get(field) is not False:
            raise ValueError(
                f"metric_contract.frozen_provenance_gaps.{field} must be false for this Gate 0 "
                "decision"
            )


def _assert_evidence_linkage(decision: dict[str, Any], evidence: dict[str, Any]) -> None:
    """Ensure emitted provenance blocks are copied from the inspected packet."""
    if decision.get("frozen_release") != evidence["frozen_release"]:
        raise ValueError("decision.frozen_release is not linked to tracked release evidence")
    if decision.get("evidence_inspection") != evidence["inspection"]:
        raise ValueError("decision.evidence_inspection is not linked to tracked release evidence")
    metric_contract = decision.get("metric_contract")
    if not isinstance(metric_contract, dict):
        raise ValueError("decision.metric_contract must be a dict")
    expected_blocks = {
        "frozen_provenance_gaps": evidence["provenance_gaps"],
        "frozen_row_schema": evidence["inspection"]["row_schema"],
        "frozen_release_manifest": evidence["inspection"]["release_manifest"],
        "tracked_evidence_cross_checks": evidence["inspection"]["cross_checks"],
    }
    for field, expected in expected_blocks.items():
        if metric_contract.get(field) != expected:
            raise ValueError(f"metric_contract.{field} is not linked to tracked release evidence")


def _assert_summary_consistent(
    decision: dict[str, Any], outcomes: list[dict[str, Any]], re_derivable_ids: list[str]
) -> None:
    """Validate the summary block against the outcome list."""
    summary = decision.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("decision.summary must be a dict")
    if summary.get("re_derivable_count") != len(re_derivable_ids):
        raise ValueError("summary.re_derivable_count does not match the outcomes")
    if summary.get("replay_required_count") != len(outcomes) - len(re_derivable_ids):
        raise ValueError("summary.replay_required_count does not match the outcomes")
    if summary.get("total_outcomes") != len(outcomes):
        raise ValueError("summary.total_outcomes does not match the outcomes")
    replay_required_ids = [
        entry["outcome_id"] for entry in outcomes if entry.get("classification") == REPLAY_REQUIRED
    ]
    if summary.get("re_derivable_outcome_ids") != re_derivable_ids:
        raise ValueError("summary.re_derivable_outcome_ids does not match the outcomes")
    if summary.get("replay_required_outcome_ids") != replay_required_ids:
        raise ValueError("summary.replay_required_outcome_ids does not match the outcomes")


def validate_gate0_decision(
    decision: dict[str, Any], *, repo_root: str | Path | None = None
) -> None:
    """Validate a Gate 0 decision payload against the stop conditions and schema.

    Raises:
        ValueError: if any stop condition or schema invariant is violated.
    """
    if not isinstance(decision, dict):
        raise ValueError("decision must be a dict")
    if decision.get("schema_version") != GATE0_DECISION_SCHEMA:
        raise ValueError(
            f"schema_version must be {GATE0_DECISION_SCHEMA!r}, got {decision.get('schema_version')!r}"
        )
    outcomes = decision.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError("decision.outcomes must be a non-empty list")

    _assert_frozen_provenance_gaps(decision)
    _assert_evidence_linkage(decision, inspect_frozen_release_evidence(repo_root))
    seen_ids: set[str] = set()
    re_derivable_ids: list[str] = []
    for entry in outcomes:
        re_derivable_id = _validate_outcome_entry(entry, seen_ids)
        if re_derivable_id is not None:
            re_derivable_ids.append(re_derivable_id)

    # Stop condition #4: every outcome has exactly one classification (enforced per entry above).
    _assert_narrow_re_derivable_set(re_derivable_ids)
    _assert_summary_consistent(decision, outcomes, re_derivable_ids)


def write_gate0_decision(output_path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Write the deterministic Gate 0 decision JSON and return its path.

    The written file is canonical and reproducible: running this function always emits the same
    bytes for the same metric contract.

    Returns:
        Path to the written decision JSON.
    """
    decision = build_gate0_decision(repo_root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, decision)
    return path


def load_gate0_decision(path: str | Path, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a Gate 0 decision JSON file.

    Returns:
        Validated decision payload.
    """
    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_gate0_decision(decision, repo_root=repo_root)
    return decision


__all__ = [
    "CAMPAIGN_ARMS_M",
    "CAMPAIGN_AXIS",
    "CAMPAIGN_BASELINE_ARM_M",
    "CLAIM_BOUNDARY",
    "FIXED_THRESHOLD_COLLISION_METRICS",
    "FROZEN_RELEASE_TAG",
    "GATE0_DECISION_SCHEMA",
    "RADIUS_AWARE_CLEARANCE_METRICS",
    "RADIUS_INDEPENDENT_GEOMETRY_METRICS",
    "REPLAY_REQUIRED",
    "RE_DERIVABLE",
    "FrozenReleaseEvidenceError",
    "RadiusSensitivityOutcome",
    "build_gate0_decision",
    "build_outcome_registry",
    "inspect_frozen_release_evidence",
    "load_gate0_decision",
    "validate_gate0_decision",
    "write_gate0_decision",
]
