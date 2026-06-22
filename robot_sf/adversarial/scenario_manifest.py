"""Adversarial scenario manifest generation, validation, and classification."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.config import CandidateSpec, Pose2D, SearchSpaceConfig
from robot_sf.adversarial.samplers import RandomCandidateSampler
from robot_sf.benchmark.manifest_lineage import validate_lineage_contract

MANIFEST_SCHEMA_VERSION = "adversarial_scenario_manifest.v1"
VALIDATOR_VERSION = "adversarial_scenario_manifest_validator.v1"
EVIDENCE_TIER = "diagnostic-only"
DENOMINATOR_POLICY = "generated_candidates_not_benchmark_denominator"
NATURALISTIC_PRIOR_SCHEMA_VERSION = "naturalistic_vru_prior.v1"
DEFAULT_NATURALISTIC_PRIOR_PROFILE = "urban_vru_default_v1"

_DEFAULT_NATURALISTIC_PRIOR_BOUNDS: dict[str, tuple[float, float, str]] = {
    "pedestrian_speed_mps": (
        0.4,
        2.2,
        "bounded walking-to-running VRU speed for plausible hard cases",
    ),
    "pedestrian_delay_s": (
        0.0,
        3.0,
        "bounded VRU reaction or release delay before route following",
    ),
    "spawn_time_s": (
        0.0,
        10.0,
        "bounded scenario-entry timing for generated stress candidates",
    ),
}


class ManifestCategory(Enum):
    """Classification for a candidate manifest: valid, invalid, or degenerate."""

    VALID = "valid"
    INVALID = "invalid"
    DEGENERATE = "degenerate"


@dataclass(frozen=True)
class SourceLineage:
    """Provenance for the map, template, and config that produced a manifest."""

    scenario_template: str | None = None
    search_space: str | None = None
    map_id: str | None = None
    scenario_name: str | None = None
    config_path: str | None = None
    search_space_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict, omitting None fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class GeneratorInfo:
    """Generator provenance: family, class, seed, and index within batch."""

    family: str = "random"
    generator_id: str = "RandomCandidateSampler"
    seed: int = 0
    candidate_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "family": self.family,
            "generator_id": self.generator_id,
            "seed": self.seed,
            "candidate_index": self.candidate_index,
        }


@dataclass(frozen=True)
class ValidationRecord:
    """Validation outcome: category, errors, warnings, and duplicate-detection hash."""

    status: ManifestCategory = ManifestCategory.VALID
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    normalized_control_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        result: dict[str, Any] = {
            "status": self.status.value,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }
        if self.normalized_control_hash is not None:
            result["normalized_control_hash"] = self.normalized_control_hash
        return result


@dataclass(frozen=True)
class NaturalisticPriorRecord:
    """Interpretable VRU naturalness-prior result for one generated candidate."""

    schema_version: str = NATURALISTIC_PRIOR_SCHEMA_VERSION
    profile: str = DEFAULT_NATURALISTIC_PRIOR_PROFILE
    constraints: tuple[dict[str, Any], ...] = ()
    passed: bool = True
    violation_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON/YAML-safe dict."""
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "constraints": [dict(constraint) for constraint in self.constraints],
            "passed": bool(self.passed),
            "violation_flags": list(self.violation_flags),
        }


@dataclass(frozen=True)
class AdversarialScenarioManifest:
    """An adversarial_scenario_manifest.v1 candidate with source, controls, and validation."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    source: SourceLineage | None = None
    generator: GeneratorInfo | None = None
    candidate_controls: dict[str, Any] | None = None
    naturalistic_prior: NaturalisticPriorRecord | None = None
    validation: ValidationRecord | None = None
    execution_status: str = "generated_only"
    validator_version: str = VALIDATOR_VERSION
    evidence_tier: str = EVIDENCE_TIER
    denominator_policy: str = DENOMINATOR_POLICY
    evidence_boundary: str = (
        "diagnostic-only: no planner weakness, adversarial coverage, "
        "or benchmark-strength claim is made from this manifest."
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a JSON/YAML-safe dict."""
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "generator_id": (
                self.generator.generator_id
                if self.generator is not None
                else GeneratorInfo().generator_id
            ),
            "validator_version": self.validator_version,
            "execution_status": self.execution_status,
            "execution_gate": self.execution_status,
            "evidence_tier": self.evidence_tier,
            "denominator_policy": self.denominator_policy,
            "claim_boundary": self.evidence_boundary,
            "evidence_boundary": self.evidence_boundary,
        }
        if self.source is not None:
            result["source"] = self.source.to_dict()
        if self.generator is not None:
            result["generator"] = self.generator.to_dict()
        if self.candidate_controls is not None:
            result["candidate_controls"] = dict(self.candidate_controls)
        if self.naturalistic_prior is not None:
            result["naturalistic_prior"] = self.naturalistic_prior.to_dict()
        if self.validation is not None:
            result["validation"] = self.validation.to_dict()
        return result

    def to_yaml(self) -> str:
        """Serialize the manifest to a YAML string."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AdversarialScenarioManifest:
        """Build a manifest from a JSON/YAML dict."""
        return cls(
            schema_version=_optional_string(payload.get("schema_version"), MANIFEST_SCHEMA_VERSION),
            source=SourceLineage(**payload["source"]) if "source" in payload else None,
            generator=GeneratorInfo(**payload["generator"]) if "generator" in payload else None,
            candidate_controls=payload.get("candidate_controls"),
            naturalistic_prior=_naturalistic_prior_from_dict(payload.get("naturalistic_prior")),
            validation=_validation_from_dict(payload.get("validation")),
            execution_status=_optional_string(payload.get("execution_status"), "generated_only"),
            validator_version=_optional_string(payload.get("validator_version"), VALIDATOR_VERSION),
            evidence_tier=_optional_string(payload.get("evidence_tier"), EVIDENCE_TIER),
            denominator_policy=_optional_string(
                payload.get("denominator_policy"), DENOMINATOR_POLICY
            ),
            evidence_boundary=_optional_string(
                payload.get("evidence_boundary"),
                "diagnostic-only: no planner weakness, adversarial coverage, "
                "or benchmark-strength claim is made from this manifest.",
            ),
        )

    @classmethod
    def from_yaml(cls, text: str) -> AdversarialScenarioManifest:
        """Build a manifest from a YAML string."""
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError("manifest YAML must be a mapping")
        return cls.from_dict(payload)


def _optional_string(value: Any, default: str) -> str:
    """Return default for absent YAML fields without coercing None to a string."""
    if value is None:
        return default
    return str(value)


def _validation_from_dict(payload: Any) -> ValidationRecord | None:
    if payload is None or not isinstance(payload, dict):
        return None
    status_raw = payload.get("status", "valid")
    try:
        status = ManifestCategory(status_raw)
    except ValueError:
        status = ManifestCategory.INVALID
    return ValidationRecord(
        status=status,
        errors=tuple(payload.get("errors", [])),
        warnings=tuple(payload.get("warnings", [])),
        normalized_control_hash=payload.get("normalized_control_hash"),
    )


def _naturalistic_prior_from_dict(payload: Any) -> NaturalisticPriorRecord | None:
    if payload is None or not isinstance(payload, dict):
        return None
    constraints_raw = payload.get("constraints", [])
    constraints: list[dict[str, Any]] = []
    if isinstance(constraints_raw, list):
        constraints = [dict(item) for item in constraints_raw if isinstance(item, dict)]
    flags_raw = payload.get("violation_flags", [])
    flags: list[str] = []
    if isinstance(flags_raw, list):
        flags = [str(flag) for flag in flags_raw]
    return NaturalisticPriorRecord(
        schema_version=_optional_string(
            payload.get("schema_version"), NATURALISTIC_PRIOR_SCHEMA_VERSION
        ),
        profile=_optional_string(payload.get("profile"), DEFAULT_NATURALISTIC_PRIOR_PROFILE),
        constraints=tuple(constraints),
        passed=bool(payload.get("passed", not flags)),
        violation_flags=tuple(flags),
    )


def _require_mapping_fields(
    payload: Any,
    *,
    namespace: str,
    requirements: tuple[tuple[str, type, str], ...],
) -> list[str]:
    """Collect type errors for required mapping fields."""
    if not isinstance(payload, dict):
        return [f"{namespace} must be a mapping"]
    errors: list[str] = []
    for field, expected_type, descriptor in requirements:
        if not isinstance(payload.get(field), expected_type):
            errors.append(f"{namespace}.{field} must be {descriptor}")
    return errors


def evaluate_naturalistic_prior(candidate: CandidateSpec) -> NaturalisticPriorRecord:
    """Evaluate the default interpretable VRU naturalness prior for a candidate.

    The v1 adversarial manifest only encodes scalar speed and timing controls, so the default
    profile intentionally starts there. Trajectory acceleration priors stay out of this function
    until generated manifests carry time-indexed trajectory controls.
    """
    observed_values = {
        "pedestrian_speed_mps": candidate.pedestrian_speed_mps,
        "pedestrian_delay_s": candidate.pedestrian_delay_s,
        "spawn_time_s": candidate.spawn_time_s,
    }
    constraints: list[dict[str, Any]] = []
    violation_flags: list[str] = []

    for field, (min_value, max_value, description) in _DEFAULT_NATURALISTIC_PRIOR_BOUNDS.items():
        observed = float(observed_values[field])
        passed = math.isfinite(observed) and min_value <= observed <= max_value
        constraints.append(
            {
                "field": field,
                "min": min_value,
                "max": max_value,
                "observed": observed,
                "passed": passed,
                "description": description,
            }
        )
        if not passed:
            violation_flags.append(f"{field}_outside_{DEFAULT_NATURALISTIC_PRIOR_PROFILE}")

    return NaturalisticPriorRecord(
        constraints=tuple(constraints),
        passed=not violation_flags,
        violation_flags=tuple(violation_flags),
    )


def _naturalistic_prior_warnings(prior: NaturalisticPriorRecord) -> list[str]:
    """Return human-readable validation warnings for failed naturalistic constraints."""
    warnings: list[str] = []
    for constraint in prior.constraints:
        if bool(constraint.get("passed", False)):
            continue
        field = str(constraint.get("field", "unknown"))
        observed = constraint.get("observed")
        min_value = constraint.get("min")
        max_value = constraint.get("max")
        warnings.append(
            "naturalistic prior violation: "
            f"{field}={observed} outside [{min_value}, {max_value}] "
            f"for {prior.profile}"
        )
    return warnings


def _validate_naturalistic_prior_payload(
    payload: Any,
    *,
    computed: NaturalisticPriorRecord,
) -> tuple[list[str], list[str]]:
    """Validate optional serialized prior metadata against computed controls."""
    errors: list[str] = []
    warnings: list[str] = []
    if payload is None:
        return errors, _naturalistic_prior_warnings(computed)
    if not isinstance(payload, dict):
        return ["naturalistic_prior must be a mapping"], warnings

    prior = _naturalistic_prior_from_dict(payload)
    if prior is None:
        return ["naturalistic_prior must be a mapping"], warnings

    errors.extend(_naturalistic_prior_shape_errors(payload, prior))

    if prior.passed != computed.passed:
        errors.append("naturalistic_prior.passed does not match candidate controls")
    if tuple(prior.violation_flags) != tuple(computed.violation_flags):
        errors.append("naturalistic_prior.violation_flags do not match candidate controls")

    warnings.extend(_naturalistic_prior_warnings(computed))
    return errors, warnings


def _naturalistic_prior_shape_errors(
    payload: dict[str, Any],
    prior: NaturalisticPriorRecord,
) -> list[str]:
    """Return schema-shape errors for serialized naturalistic-prior metadata."""
    errors: list[str] = []
    if prior.schema_version != NATURALISTIC_PRIOR_SCHEMA_VERSION:
        errors.append(
            f"naturalistic_prior.schema_version must be {NATURALISTIC_PRIOR_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("profile"), str):
        errors.append("naturalistic_prior.profile must be a string")
    if not isinstance(payload.get("constraints"), list):
        errors.append("naturalistic_prior.constraints must be a list")
    if not isinstance(payload.get("passed"), bool):
        errors.append("naturalistic_prior.passed must be a boolean")
    raw_flags = payload.get("violation_flags", [])
    if not isinstance(raw_flags, list) or not all(isinstance(flag, str) for flag in raw_flags):
        errors.append("naturalistic_prior.violation_flags must be a list of strings")
    return errors


def compute_control_hash(candidate: CandidateSpec, precision: int = 6) -> str:
    """Deterministic hash of normalized candidate controls."""
    values = (
        round(float(candidate.start.x), precision) + 0.0,
        round(float(candidate.start.y), precision) + 0.0,
        round(float(candidate.goal.x), precision) + 0.0,
        round(float(candidate.goal.y), precision) + 0.0,
        round(float(candidate.spawn_time_s), precision) + 0.0,
        round(float(candidate.pedestrian_speed_mps), precision) + 0.0,
        round(float(candidate.pedestrian_delay_s), precision) + 0.0,
        int(candidate.scenario_seed),
    )
    raw = json.dumps(values, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _candidate_controls_dict(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "start": {"x": float(candidate.start.x), "y": float(candidate.start.y)},
        "goal": {"x": float(candidate.goal.x), "y": float(candidate.goal.y)},
        "spawn_time_s": float(candidate.spawn_time_s),
        "pedestrian_speed_mps": float(candidate.pedestrian_speed_mps),
        "pedestrian_delay_s": float(candidate.pedestrian_delay_s),
        "scenario_seed": int(candidate.scenario_seed),
    }


def _candidate_from_controls(controls: dict[str, Any]) -> CandidateSpec:
    """Build a CandidateSpec from serialized manifest controls."""

    def _pose(name: str) -> dict[str, float]:
        raw_pose = controls.get(name)
        if not isinstance(raw_pose, dict):
            raise ValueError(f"candidate_controls.{name} must be a mapping")
        if "x" not in raw_pose or "y" not in raw_pose:
            raise ValueError(f"candidate_controls.{name} must define x and y")
        return {"x": float(raw_pose["x"]), "y": float(raw_pose["y"])}

    def _float(name: str) -> float:
        if name not in controls:
            raise ValueError(f"candidate_controls.{name} is required")
        return float(controls[name])

    if "scenario_seed" not in controls:
        raise ValueError("candidate_controls.scenario_seed is required")
    scenario_seed_value = float(controls["scenario_seed"])
    if not scenario_seed_value.is_integer():
        raise ValueError("candidate_controls.scenario_seed must be an integer")
    start = _pose("start")
    goal = _pose("goal")
    return CandidateSpec(
        start=Pose2D(start["x"], start["y"]),
        goal=Pose2D(goal["x"], goal["y"]),
        spawn_time_s=_float("spawn_time_s"),
        pedestrian_speed_mps=_float("pedestrian_speed_mps"),
        pedestrian_delay_s=_float("pedestrian_delay_s"),
        scenario_seed=int(scenario_seed_value),
    )


def _classify_errors(errors: list[str]) -> ManifestCategory:
    """Classify validation errors into valid/invalid/degenerate.

    Out-of-bounds errors -> INVALID (schema/range issue).
    Non-finite, non-positive speed, negative timing, too-short route -> DEGENERATE.
    """
    invalid_prefixes = (
        "schema_version must be",
        "candidate_controls.",
        "execution_status must be",
        "evidence_boundary must be",
        "source.",
        "generator.",
        "start.x outside",
        "start.y outside",
        "goal.x outside",
        "goal.y outside",
    )
    invalid_exact = (
        "candidate_controls must be a mapping",
        "naturalistic_prior must be a mapping",
        "source must be a mapping",
        "generator must be a mapping",
    )
    degenerate_markers = (
        "non-finite",
        "must be positive",
        "must be non-negative",
        "min_start_goal_distance_m",
        "outside search space",
        "scenario_seed must be non-negative",
    )

    for err in errors:
        if err in invalid_exact or err.startswith(invalid_prefixes):
            return ManifestCategory.INVALID
        if err.startswith("naturalistic_prior."):
            return ManifestCategory.INVALID
        if "outside search space" in err or "scenario_seed must be non-negative" in err:
            return ManifestCategory.INVALID

    for err in errors:
        if any(marker in err for marker in degenerate_markers):
            return ManifestCategory.DEGENERATE

    if errors:
        return ManifestCategory.DEGENERATE
    return ManifestCategory.VALID


def validate_manifest_payload(
    payload: object,
    *,
    search_space: SearchSpaceConfig | None = None,
    existing_hashes: set[str] | None = None,
) -> ValidationRecord:
    """Validate a serialized adversarial_scenario_manifest.v1 payload."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return ValidationRecord(
            status=ManifestCategory.INVALID,
            errors=("manifest payload must be a mapping",),
            warnings=(),
        )
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    if not isinstance(payload.get("execution_status"), str):
        errors.append("execution_status must be a string")
    lineage_errors = validate_lineage_contract(payload)
    errors.extend(lineage_errors)
    if not isinstance(payload.get("evidence_boundary"), str):
        errors.append("evidence_boundary must be a string")
    errors.extend(
        _require_mapping_fields(
            payload.get("source"),
            namespace="source",
            requirements=(
                ("scenario_template", str, "a string"),
                ("search_space", str, "a string"),
                ("map_id", str, "a string"),
                ("scenario_name", str, "a string"),
                ("config_path", str, "a string"),
                ("search_space_path", str, "a string"),
            ),
        )
    )
    errors.extend(
        _require_mapping_fields(
            payload.get("generator"),
            namespace="generator",
            requirements=(
                ("family", str, "a string"),
                ("generator_id", str, "a string"),
                ("seed", int, "an integer"),
                ("candidate_index", int, "an integer"),
            ),
        )
    )
    controls = payload.get("candidate_controls")
    if not isinstance(controls, dict):
        errors.append("candidate_controls must be a mapping")
        return ValidationRecord(status=ManifestCategory.INVALID, errors=tuple(errors))
    try:
        candidate = _candidate_from_controls(controls)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return ValidationRecord(status=ManifestCategory.INVALID, errors=tuple(errors))
    candidate_errors, warnings = validate_candidate_manifest(
        candidate,
        search_space=search_space,
        existing_hashes=existing_hashes,
    )
    errors.extend(candidate_errors)
    computed_prior = evaluate_naturalistic_prior(candidate)
    prior_errors, prior_warnings = _validate_naturalistic_prior_payload(
        payload.get("naturalistic_prior"),
        computed=computed_prior,
    )
    errors.extend(prior_errors)
    warnings.extend(prior_warnings)
    status = _classify_errors(errors)
    has_duplicate_warning = any(
        "duplicate normalized control hash" in warning for warning in warnings
    )
    if status is ManifestCategory.VALID and has_duplicate_warning:
        status = ManifestCategory.DEGENERATE
    return ValidationRecord(
        status=status,
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized_control_hash=compute_control_hash(candidate),
    )


def validate_candidate_manifest(
    candidate: CandidateSpec,
    search_space: SearchSpaceConfig | None = None,
    existing_hashes: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Validate a candidate and return (errors, warnings).

    Uses SearchSpaceConfig.validate_candidate when search_space is provided.
    Always checks degenerate conditions even without a search space.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if search_space is not None:
        errors.extend(search_space.validate_candidate(candidate))
    else:
        values = {
            "start.x": candidate.start.x,
            "start.y": candidate.start.y,
            "goal.x": candidate.goal.x,
            "goal.y": candidate.goal.y,
            "spawn_time_s": candidate.spawn_time_s,
            "pedestrian_speed_mps": candidate.pedestrian_speed_mps,
            "pedestrian_delay_s": candidate.pedestrian_delay_s,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                errors.append(f"{name} must be finite")
        if candidate.spawn_time_s < 0.0:
            errors.append("spawn_time_s must be non-negative")
        if candidate.pedestrian_speed_mps <= 0.0:
            errors.append("pedestrian_speed_mps must be positive")
        if candidate.pedestrian_delay_s < 0.0:
            errors.append("pedestrian_delay_s must be non-negative")
        if candidate.scenario_seed < 0:
            errors.append("scenario_seed must be non-negative")

    if existing_hashes is not None:
        control_hash = compute_control_hash(candidate)
        if control_hash in existing_hashes:
            warnings.append(f"duplicate normalized control hash: {control_hash}")

    return errors, warnings


def build_manifest(
    candidate: CandidateSpec,
    *,
    source: SourceLineage | None = None,
    generator: GeneratorInfo | None = None,
    search_space: SearchSpaceConfig | None = None,
    existing_hashes: set[str] | None = None,
) -> AdversarialScenarioManifest:
    """Build a manifest from a candidate, optionally validating it."""
    errors, warnings = validate_candidate_manifest(
        candidate,
        search_space=search_space,
        existing_hashes=existing_hashes,
    )
    status = _classify_errors(errors)
    has_duplicate_warning = any(
        "duplicate normalized control hash" in warning for warning in warnings
    )
    if status is ManifestCategory.VALID and has_duplicate_warning:
        status = ManifestCategory.DEGENERATE
    control_hash = compute_control_hash(candidate)
    naturalistic_prior = evaluate_naturalistic_prior(candidate)
    warnings.extend(_naturalistic_prior_warnings(naturalistic_prior))

    validation = ValidationRecord(
        status=status,
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized_control_hash=control_hash,
    )

    return AdversarialScenarioManifest(
        source=source,
        generator=generator,
        candidate_controls=_candidate_controls_dict(candidate),
        naturalistic_prior=naturalistic_prior,
        validation=validation,
    )


def generate_manifests(
    search_space: SearchSpaceConfig,
    *,
    seed: int,
    count: int,
    source: SourceLineage | None = None,
    generator_family: str = "random",
) -> tuple[list[AdversarialScenarioManifest], dict[str, Any]]:
    """Generate a batch of manifests with deterministic validation."""
    sampler = RandomCandidateSampler(search_space, seed=seed)
    manifests: list[AdversarialScenarioManifest] = []
    seen_hashes: set[str] = set()
    rejection_reasons: dict[str, int] = {}

    for i in range(count):
        candidate = sampler.sample()
        gen_info = GeneratorInfo(
            family=generator_family,
            generator_id="RandomCandidateSampler",
            seed=seed,
            candidate_index=i,
        )
        manifest = build_manifest(
            candidate,
            source=source,
            generator=gen_info,
            search_space=search_space,
            existing_hashes=seen_hashes,
        )
        control_hash = compute_control_hash(candidate)
        seen_hashes.add(control_hash)

        if manifest.validation is not None:
            for reason in (*manifest.validation.errors, *manifest.validation.warnings):
                key = reason.split(":")[0] if ":" in reason else reason
                rejection_reasons[key] = rejection_reasons.get(key, 0) + 1

        manifests.append(manifest)

    summary = _build_summary(manifests, rejection_reasons)
    return manifests, summary


def _build_summary(
    manifests: list[AdversarialScenarioManifest],
    rejection_reasons: dict[str, int],
) -> dict[str, Any]:
    """Build a compact summary dict from a list of manifests."""
    total = len(manifests)
    counts: dict[str, int] = {}
    for m in manifests:
        if m.validation is not None:
            cat = m.validation.status.value
        else:
            cat = "unknown"
        counts[cat] = counts.get(cat, 0) + 1

    return {
        "total_candidates": total,
        "valid": counts.get("valid", 0),
        "invalid": counts.get("invalid", 0),
        "degenerate": counts.get("degenerate", 0),
        "naturalistic_prior": _build_naturalistic_prior_summary(manifests),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def _build_naturalistic_prior_summary(
    manifests: list[AdversarialScenarioManifest],
) -> dict[str, Any]:
    """Summarize naturalistic-prior pass/fail counts for a generated batch."""
    pass_count = 0
    fail_count = 0
    unavailable_count = 0
    violation_counts: dict[str, int] = {}
    for manifest in manifests:
        prior = manifest.naturalistic_prior
        if prior is None:
            unavailable_count += 1
            continue
        if prior.passed:
            pass_count += 1
        else:
            fail_count += 1
        for flag in prior.violation_flags:
            violation_counts[flag] = violation_counts.get(flag, 0) + 1

    total = len(manifests)
    return {
        "profile": DEFAULT_NATURALISTIC_PRIOR_PROFILE,
        "pass": pass_count,
        "fail": fail_count,
        "unavailable": unavailable_count,
        "pass_rate": round(pass_count / total, 6) if total else 0.0,
        "fail_rate": round(fail_count / total, 6) if total else 0.0,
        "violation_counts": dict(sorted(violation_counts.items())),
    }


def write_manifest_yaml(manifest: AdversarialScenarioManifest, path: Path) -> Path:
    """Write a single manifest to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_yaml(), encoding="utf-8")
    return path
