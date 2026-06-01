"""Scenario perturbation manifest preflight helpers.

The v1 surface is intentionally small: it certifies no-op baselines and bounded route-offset
families before any generated variant can be counted as benchmark evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.scenario_certification.v1 import (
    ScenarioCertificate,
    certificate_to_dict,
    certify_map_definition,
    certify_scenario,
)
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)

if TYPE_CHECKING:
    from robot_sf.nav.global_route import GlobalRoute

PERTURBATION_MANIFEST_SCHEMA_VERSION = "scenario_perturbation_manifest.v1"
PREFLIGHT_SCHEMA_VERSION = "scenario_perturbation_preflight.v1"
PILOT_MATRIX_SCHEMA_VERSION = "scenario_perturbation_pilot_matrix.v1"
_PERTURBATION_MANIFEST_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "schemas"
    / "scenario_perturbation_manifest.v1.json"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OUTPUT_ROOT = _REPOSITORY_ROOT / "output"

_SUCCESS_EVIDENCE_CANDIDATE = "eligible_success_evidence_candidate"
_EXCLUDED_FROM_SUCCESS_EVIDENCE = "excluded_from_success_evidence"
_STRESS_ONLY_NOT_SUCCESS_EVIDENCE = "stress_only_not_success_evidence"
_DEFAULT_SINGLE_PEDESTRIAN_SPEED_M_S = 0.5


@dataclass(frozen=True)
class PerturbationPreflightResult:
    """Preflight status for one scenario perturbation variant."""

    variant_id: str
    scenario_id: str
    family: str
    seeds: tuple[int, ...]
    validity_status: str
    benchmark_evidence_status: str
    reasons: list[str]
    perturbation_summary: dict[str, Any]
    certificate: ScenarioCertificate | None = None


@dataclass(frozen=True)
class PerturbationPreflightReport:
    """Batch preflight report for a perturbation manifest."""

    schema_version: str
    manifest_id: str
    manifest_path: str
    scenario_config: str
    results: list[PerturbationPreflightResult] = field(default_factory=list)


@dataclass(frozen=True)
class PerturbationPilotMaterialization:
    """Local scenario matrix emitted from preflight-eligible perturbation variants."""

    schema_version: str
    manifest_id: str
    manifest_path: str
    scenario_matrix_path: str
    summary_path: str
    included_variants: tuple[str, ...]
    excluded_variants: tuple[str, ...]


def preflight_perturbation_manifest(manifest_path: Path | str) -> PerturbationPreflightReport:
    """Preflight every variant in a scenario perturbation manifest.

    Returns:
        PerturbationPreflightReport: Fail-closed validity and evidence status per variant.
    """
    path = Path(manifest_path)
    manifest = _load_manifest(path)
    scenario_config = _resolve_path(
        manifest["scenario_config"],
        base_dir=path.parent,
        field_name="scenario_config",
    )
    scenarios = load_scenarios(scenario_config)
    results = [
        _preflight_variant(
            variant,
            manifest=manifest,
            scenario_config=scenario_config,
            scenarios=scenarios,
        )
        for variant in manifest["variants"]
    ]
    return PerturbationPreflightReport(
        schema_version=PREFLIGHT_SCHEMA_VERSION,
        manifest_id=str(manifest["manifest_id"]),
        manifest_path=path.as_posix(),
        scenario_config=scenario_config.as_posix(),
        results=results,
    )


def preflight_to_dict(report: PerturbationPreflightReport) -> dict[str, Any]:
    """Convert a preflight report into JSON-safe primitives.

    Returns:
        dict[str, Any]: JSON-safe preflight payload.
    """
    return {
        "schema_version": report.schema_version,
        "manifest_id": report.manifest_id,
        "manifest_path": report.manifest_path,
        "scenario_config": report.scenario_config,
        "results": [
            {
                "variant_id": result.variant_id,
                "scenario_id": result.scenario_id,
                "family": result.family,
                "seeds": list(result.seeds),
                "validity_status": result.validity_status,
                "benchmark_evidence_status": result.benchmark_evidence_status,
                "reasons": list(result.reasons),
                "perturbation_summary": dict(result.perturbation_summary),
                "certificate": certificate_to_dict(result.certificate)
                if result.certificate is not None
                else None,
            }
            for result in report.results
        ],
    }


def materialize_perturbation_pilot_matrix(
    manifest_path: Path | str,
    *,
    output_dir: Path | str,
    seed_limit: int | None = None,
) -> PerturbationPilotMaterialization:
    """Write a local scenario matrix for preflight-eligible perturbation variants.

    The generated files are execution inputs for a later small planner pilot. They are not
    benchmark evidence on their own, and variants excluded by the preflight are omitted.

    Returns:
        PerturbationPilotMaterialization: Paths and included/excluded variant IDs.
    """
    path = Path(manifest_path)
    out_dir = Path(output_dir)
    _ensure_local_output_boundary(out_dir)
    manifest = _load_manifest(path)
    report = preflight_perturbation_manifest(path)
    preflight_by_variant = {result.variant_id: result for result in report.results}
    scenario_config = _resolve_path(
        manifest["scenario_config"],
        base_dir=path.parent,
        field_name="scenario_config",
    )
    scenarios = load_scenarios(scenario_config)
    matrix_path = out_dir / "scenario_matrix.yaml"
    summary_path = out_dir / "materialization_summary.json"
    routes_dir = out_dir / "route_overrides"
    out_dir.mkdir(parents=True, exist_ok=True)
    routes_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_materialization(
        matrix_path=matrix_path,
        summary_path=summary_path,
        routes_dir=routes_dir,
    )

    materialized_scenarios: list[dict[str, Any]] = []
    included: list[str] = []
    excluded: list[str] = []
    for variant in manifest["variants"]:
        normalized = _normalize_variant(variant)
        result = preflight_by_variant[normalized["variant_id"]]
        if result.benchmark_evidence_status != _SUCCESS_EVIDENCE_CANDIDATE:
            excluded.append(result.variant_id)
            continue
        scenario = _materialize_variant_scenario(
            normalized,
            result=result,
            manifest=manifest,
            scenario_config=scenario_config,
            scenarios=scenarios,
            routes_dir=routes_dir,
            matrix_path=matrix_path,
            seed_limit=seed_limit,
        )
        materialized_scenarios.append(scenario)
        included.append(result.variant_id)

    matrix_payload = {
        "schema_version": PILOT_MATRIX_SCHEMA_VERSION,
        "source_manifest": path.as_posix(),
        "source_manifest_id": manifest["manifest_id"],
        "evidence_boundary": (
            "local pilot execution input only; not benchmark or paper-facing evidence"
        ),
        "scenarios": materialized_scenarios,
    }
    matrix_path.write_text(yaml.safe_dump(matrix_payload, sort_keys=False), encoding="utf-8")
    summary_payload = {
        "schema_version": PILOT_MATRIX_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "manifest_path": path.as_posix(),
        "scenario_matrix_path": matrix_path.as_posix(),
        "preflight_schema_version": report.schema_version,
        "included_variants": included,
        "excluded_variants": excluded,
        "seed_limit": seed_limit,
        "variant_count": len(materialized_scenarios),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    return PerturbationPilotMaterialization(
        schema_version=PILOT_MATRIX_SCHEMA_VERSION,
        manifest_id=str(manifest["manifest_id"]),
        manifest_path=path.as_posix(),
        scenario_matrix_path=matrix_path.as_posix(),
        summary_path=summary_path.as_posix(),
        included_variants=tuple(included),
        excluded_variants=tuple(excluded),
    )


def _ensure_local_output_boundary(output_dir: Path) -> None:
    """Keep repository-local generated pilot inputs under the ignored output tree."""
    resolved = output_dir.resolve()
    try:
        resolved.relative_to(_REPOSITORY_ROOT)
    except ValueError:
        return
    try:
        resolved.relative_to(_LOCAL_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(
            "output_dir inside the repository must be under output/ so generated "
            "pilot inputs remain local, ignored artifacts"
        ) from exc


def _clear_previous_materialization(
    *,
    matrix_path: Path,
    summary_path: Path,
    routes_dir: Path,
) -> None:
    """Remove stale materializer-owned files before writing a fresh pilot matrix."""
    for generated_path in (matrix_path, summary_path):
        if generated_path.exists():
            generated_path.unlink()
    if not routes_dir.is_dir():
        raise ValueError(f"route_overrides path exists and is not a directory: {routes_dir}")
    for route_path in routes_dir.glob("*.route_overrides.yaml"):
        route_path.unlink()


def _load_manifest(path: Path) -> Mapping[str, Any]:
    """Load and minimally validate a scenario perturbation manifest.

    Returns:
        Mapping[str, Any]: Parsed manifest mapping.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Perturbation manifest must contain a mapping: {path}")
    schema_version = data.get("schema_version")
    if schema_version != PERTURBATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: expected schema_version {PERTURBATION_MANIFEST_SCHEMA_VERSION!r}, "
            f"found {schema_version!r}"
        )
    for field_name in ("manifest_id", "scenario_config", "seed_controls", "validity", "variants"):
        if field_name not in data:
            raise ValueError(f"{path}: missing required field {field_name!r}")
    if not isinstance(data["validity"], Mapping):
        raise ValueError(f"{path}: validity must be a mapping")
    if not isinstance(data["variants"], list) or not data["variants"]:
        raise ValueError(f"{path}: variants must be a non-empty list")
    _validate_manifest_schema(data, path=path)
    return data


def _validate_manifest_schema(data: Mapping[str, Any], *, path: Path) -> None:
    """Validate a perturbation manifest against the public v1 schema."""
    schema = json.loads(_PERTURBATION_MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise ValueError(f"{path}: manifest schema violation at {location}: {error.message}")


def _resolve_path(raw: Any, *, base_dir: Path, field_name: str) -> Path:
    """Resolve manifest paths from the manifest directory, then the current checkout.

    Returns:
        Path: Resolved path candidate.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} must be a non-empty string path")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    local = (base_dir / candidate).resolve()
    if local.exists():
        return local
    checkout_relative = candidate.resolve()
    if checkout_relative.exists():
        return checkout_relative
    return local


def _preflight_variant(
    variant: Any,
    *,
    manifest: Mapping[str, Any],
    scenario_config: Path,
    scenarios: list[Mapping[str, Any]],
) -> PerturbationPreflightResult:
    """Preflight one variant, recording fail-closed errors as excluded rows.

    Returns:
        PerturbationPreflightResult: Variant validity row.
    """
    try:
        normalized = _normalize_variant(variant)
        seed_tuple = _normalize_seeds(normalized.get("seeds"), manifest=manifest)
        bound_reasons, perturbation_summary = _validate_perturbation_bounds(
            normalized,
            manifest=manifest,
        )
        if bound_reasons:
            return _result(
                normalized,
                seeds=seed_tuple,
                validity_status="invalid",
                benchmark_evidence_status=_EXCLUDED_FROM_SUCCESS_EVIDENCE,
                reasons=bound_reasons,
                perturbation_summary=perturbation_summary,
            )
        certificate = _certify_variant(
            normalized,
            scenario_config=scenario_config,
            scenarios=scenarios,
            perturbation_summary=perturbation_summary,
        )
        validity_status = _validity_status_from_certificate(certificate)
        return _result(
            normalized,
            seeds=seed_tuple,
            validity_status=validity_status,
            benchmark_evidence_status=_evidence_status_from_certificate(certificate),
            reasons=list(certificate.reasons),
            perturbation_summary=perturbation_summary,
            certificate=certificate,
        )
    except Exception as exc:  # noqa: BLE001 - preflight reports must fail closed per variant.
        fallback = variant if isinstance(variant, Mapping) else {}
        normalized = {
            "variant_id": str(fallback.get("variant_id") or "unknown"),
            "scenario_id": str(fallback.get("scenario_id") or "unknown"),
            "family": str(fallback.get("family") or "unknown"),
        }
        return _result(
            normalized,
            seeds=(),
            validity_status="invalid",
            benchmark_evidence_status=_EXCLUDED_FROM_SUCCESS_EVIDENCE,
            reasons=[f"preflight_error: {exc}"],
            perturbation_summary={},
        )


def _normalize_variant(variant: Any) -> Mapping[str, Any]:
    """Validate variant identity fields.

    Returns:
        Mapping[str, Any]: Normalized variant mapping.
    """
    if not isinstance(variant, Mapping):
        raise ValueError("variant must be a mapping")
    for field_name in ("variant_id", "scenario_id", "family"):
        value = variant.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"variant.{field_name} must be a non-empty string")
    family = str(variant["family"])
    if family not in {
        "noop",
        "robot_route_offset",
        "pedestrian_route_offset",
        "single_pedestrian_start_delay_offset",
        "single_pedestrian_speed_offset",
        "single_pedestrian_wait_duration_offset",
    }:
        raise ValueError(f"unsupported perturbation family: {family}")
    return variant


def _normalize_seeds(raw: Any, *, manifest: Mapping[str, Any]) -> tuple[int, ...]:
    """Return explicit replay seeds for a variant, defaulting to the manifest baseline seeds."""
    if raw is None:
        seed_controls = manifest.get("seed_controls", {})
        raw = seed_controls.get("baseline_seeds") if isinstance(seed_controls, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("variant seeds or seed_controls.baseline_seeds must be a non-empty list")
    seeds = tuple(int(seed) for seed in raw)
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    return seeds


def _validate_perturbation_bounds(
    variant: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Validate the bounded perturbation surface before certification.

    Returns:
        tuple[list[str], dict[str, Any]]: Fail-closed reasons and perturbation summary.
    """
    family = str(variant["family"])
    if family == "noop":
        return [], {"family": "noop", "magnitude_m": 0.0}

    parameters = variant.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{family} variants require a parameters mapping")
    if family == "single_pedestrian_start_delay_offset":
        return _validate_start_delay_offset_bounds(variant, manifest=manifest)
    if family == "single_pedestrian_speed_offset":
        return _validate_single_pedestrian_speed_offset_bounds(variant, manifest=manifest)
    if family == "single_pedestrian_wait_duration_offset":
        return _validate_wait_duration_offset_bounds(variant, manifest=manifest)

    dx = _finite_float(parameters.get("dx_m", 0.0), field_name="parameters.dx_m")
    dy = _finite_float(parameters.get("dy_m", 0.0), field_name="parameters.dy_m")
    magnitude = math.hypot(dx, dy)
    variant_max = _positive_float(
        parameters.get("max_magnitude_m"),
        field_name="parameters.max_magnitude_m",
    )
    manifest_max = _positive_float(
        manifest["validity"].get("max_route_offset_m"),
        field_name="validity.max_route_offset_m",
    )
    bound = min(variant_max, manifest_max)
    summary = {
        "family": family,
        "dx_m": dx,
        "dy_m": dy,
        "magnitude_m": magnitude,
        "max_magnitude_m": bound,
        "target": {
            "spawn_id": parameters.get("spawn_id"),
            "goal_id": parameters.get("goal_id"),
            "waypoint_selector": parameters.get("waypoint_selector", "all"),
        },
    }
    if magnitude > bound:
        return [
            f"{family} magnitude {magnitude:.6f} m exceeds variant max_magnitude_m {bound:.6f} m"
        ], summary
    return [], summary


def _optional_positive_float(raw: Any, *, field_name: str) -> float | None:
    """Return a positive float when present."""
    if raw is None:
        return None
    return _positive_float(raw, field_name=field_name)


def _validate_start_delay_offset_bounds(
    variant: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Validate bounded pedestrian start-delay perturbations.

    Returns:
        tuple[list[str], dict[str, Any]]: Fail-closed reasons and perturbation summary.
    """
    family = str(variant["family"])
    parameters = variant.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{family} variants require a parameters mapping")
    dt_s = _finite_float(parameters.get("dt_s"), field_name="parameters.dt_s")
    variant_max = _positive_float(
        parameters.get("max_abs_dt_s"),
        field_name="parameters.max_abs_dt_s",
    )
    manifest_max_raw = manifest["validity"].get("max_start_delay_offset_s")
    manifest_max = _positive_float(
        manifest_max_raw,
        field_name="validity.max_start_delay_offset_s",
    )
    bound = min(variant_max, manifest_max)
    abs_dt_s = abs(dt_s)
    summary = {
        "family": family,
        "dt_s": dt_s,
        "abs_dt_s": abs_dt_s,
        "max_abs_dt_s": bound,
        "target": {
            "pedestrian_id": parameters.get("pedestrian_id"),
            "selector": parameters.get("pedestrian_selector", "all"),
        },
    }
    if abs_dt_s > bound:
        return [
            f"{family} abs_dt_s {abs_dt_s:.6f} s exceeds variant max_abs_dt_s {bound:.6f} s"
        ], summary
    return [], summary


def _validate_single_pedestrian_speed_offset_bounds(
    variant: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Validate bounded single-pedestrian speed perturbations.

    Returns:
        tuple[list[str], dict[str, Any]]: Fail-closed reasons and perturbation summary.
    """
    family = str(variant["family"])
    parameters = variant.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{family} variants require a parameters mapping")
    speed_delta_m_s = _finite_float(
        parameters.get("speed_delta_m_s"),
        field_name="parameters.speed_delta_m_s",
    )
    variant_max = _positive_float(
        parameters.get("max_abs_speed_delta_m_s"),
        field_name="parameters.max_abs_speed_delta_m_s",
    )
    manifest_max = _positive_float(
        manifest["validity"].get("max_single_pedestrian_speed_delta_m_s"),
        field_name="validity.max_single_pedestrian_speed_delta_m_s",
    )
    bound = min(variant_max, manifest_max)
    manifest_speed_cap = _optional_positive_float(
        manifest["validity"].get("max_single_pedestrian_speed_m_s"),
        field_name="validity.max_single_pedestrian_speed_m_s",
    )
    variant_speed_cap = _optional_positive_float(
        parameters.get("max_speed_m_s"),
        field_name="parameters.max_speed_m_s",
    )
    speed_caps = [value for value in (manifest_speed_cap, variant_speed_cap) if value is not None]
    max_speed_m_s = min(speed_caps) if speed_caps else None
    abs_speed_delta_m_s = abs(speed_delta_m_s)
    summary = {
        "family": family,
        "speed_delta_m_s": speed_delta_m_s,
        "abs_speed_delta_m_s": abs_speed_delta_m_s,
        "max_abs_speed_delta_m_s": bound,
        "max_speed_m_s": max_speed_m_s,
        "default_baseline_speed_m_s": _DEFAULT_SINGLE_PEDESTRIAN_SPEED_M_S,
        "target": {
            "pedestrian_id": parameters.get("pedestrian_id"),
            "selector": parameters.get("pedestrian_selector", "all"),
        },
    }
    if abs_speed_delta_m_s > bound:
        return [
            f"{family} abs_speed_delta_m_s {abs_speed_delta_m_s:.6f} m/s exceeds "
            f"variant max_abs_speed_delta_m_s {bound:.6f} m/s"
        ], summary
    return [], summary


def _validate_wait_duration_offset_bounds(
    variant: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Validate bounded single-pedestrian wait-duration perturbations.

    Returns:
        tuple[list[str], dict[str, Any]]: Fail-closed reasons and perturbation summary.
    """
    family = str(variant["family"])
    parameters = variant.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{family} variants require a parameters mapping")
    wait_delta_s = _finite_float(
        parameters.get("wait_delta_s"),
        field_name="parameters.wait_delta_s",
    )
    variant_max = _positive_float(
        parameters.get("max_abs_wait_delta_s"),
        field_name="parameters.max_abs_wait_delta_s",
    )
    manifest_max = _positive_float(
        manifest["validity"].get("max_wait_duration_offset_s"),
        field_name="validity.max_wait_duration_offset_s",
    )
    bound = min(variant_max, manifest_max)
    abs_wait_delta_s = abs(wait_delta_s)
    summary = {
        "family": family,
        "wait_delta_s": wait_delta_s,
        "abs_wait_delta_s": abs_wait_delta_s,
        "max_abs_wait_delta_s": bound,
        "target": {
            "pedestrian_id": parameters.get("pedestrian_id"),
            "selector": parameters.get("pedestrian_selector", "all"),
        },
    }
    if abs_wait_delta_s > bound:
        return [
            f"{family} abs_wait_delta_s {abs_wait_delta_s:.6f} s exceeds "
            f"variant max_abs_wait_delta_s {bound:.6f} s"
        ], summary
    return [], summary


def _certify_variant(
    variant: Mapping[str, Any],
    *,
    scenario_config: Path,
    scenarios: list[Mapping[str, Any]],
    perturbation_summary: Mapping[str, Any],
) -> ScenarioCertificate:
    """Certify a no-op or perturbed scenario variant.

    Returns:
        ScenarioCertificate: Certificate for the variant.
    """
    scenario = dict(select_scenario(scenarios, str(variant["scenario_id"])))
    if variant["family"] == "noop":
        return certify_scenario(scenario, scenario_path=scenario_config)

    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_config)
    if not config.map_pool.map_defs:
        raise ValueError("scenario map pool is empty")
    _map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    perturbed_map = deepcopy(map_def)
    if variant["family"] == "single_pedestrian_start_delay_offset":
        _apply_start_delay_offset(
            perturbed_map.single_pedestrians,
            dt_s=float(perturbation_summary["dt_s"]),
            parameters=variant.get("parameters", {}),
        )
    elif variant["family"] == "single_pedestrian_speed_offset":
        _apply_single_pedestrian_speed_offset(
            perturbed_map.single_pedestrians,
            speed_delta_m_s=float(perturbation_summary["speed_delta_m_s"]),
            max_speed_m_s=perturbation_summary.get("max_speed_m_s"),
            parameters=variant.get("parameters", {}),
        )
    elif variant["family"] == "single_pedestrian_wait_duration_offset":
        _apply_wait_duration_offset(
            perturbed_map.single_pedestrians,
            wait_delta_s=float(perturbation_summary["wait_delta_s"]),
            parameters=variant.get("parameters", {}),
        )
    else:
        _apply_route_offset(
            _routes_for_family(perturbed_map, str(variant["family"])),
            dx=float(perturbation_summary["dx_m"]),
            dy=float(perturbation_summary["dy_m"]),
            parameters=variant.get("parameters", {}),
            family=str(variant["family"]),
        )
    perturbed_map.__post_init__()
    scenario["metadata"] = _variant_metadata(
        scenario.get("metadata"), variant, perturbation_summary
    )
    return certify_map_definition(
        perturbed_map,
        scenario=scenario,
        source=f"{scenario_config.as_posix()}#{variant['scenario_id']}:{variant['variant_id']}",
        robot_config=config.robot_config,
        sim_config=config.sim_config,
    )


def _materialize_variant_scenario(
    variant: Mapping[str, Any],
    *,
    result: PerturbationPreflightResult,
    manifest: Mapping[str, Any],
    scenario_config: Path,
    scenarios: list[Mapping[str, Any]],
    routes_dir: Path,
    matrix_path: Path,
    seed_limit: int | None,
) -> dict[str, Any]:
    """Build one scenario entry suitable for a local paired planner pilot.

    Returns:
        dict[str, Any]: Scenario entry with variant identity, seeds, and optional route override.
    """
    scenario = deepcopy(dict(select_scenario(scenarios, str(variant["scenario_id"]))))
    scenario = _resolve_materialized_scenario_paths(scenario, scenario_config=scenario_config)
    scenario["name"] = str(variant["variant_id"])
    scenario["scenario_id"] = str(variant["variant_id"])
    scenario["seeds"] = _limit_seeds(result.seeds, seed_limit)
    metadata = (
        dict(scenario.get("metadata")) if isinstance(scenario.get("metadata"), Mapping) else {}
    )
    metadata["scenario_perturbation"] = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "source_manifest_id": manifest["manifest_id"],
        "source_scenario_id": variant["scenario_id"],
        "variant_id": variant["variant_id"],
        "family": variant["family"],
        "validity_status": result.validity_status,
        "benchmark_evidence_status": result.benchmark_evidence_status,
        "perturbation_summary": dict(result.perturbation_summary),
        "evidence_boundary": "local_pilot_input_not_benchmark_evidence",
    }
    scenario["metadata"] = metadata
    if variant["family"] in {"robot_route_offset", "pedestrian_route_offset"}:
        route_path = routes_dir / f"{variant['variant_id']}.route_overrides.yaml"
        _write_route_offset_override(
            variant,
            scenario_config=scenario_config,
            scenarios=scenarios,
            perturbation_summary=result.perturbation_summary,
            route_path=route_path,
        )
        scenario["route_overrides_file"] = route_path.relative_to(matrix_path.parent).as_posix()
    elif variant["family"] == "single_pedestrian_start_delay_offset":
        scenario["single_pedestrians"] = _single_pedestrian_overrides_for_start_delay(
            variant,
            scenario=scenario,
            scenario_config=scenario_config,
            perturbation_summary=result.perturbation_summary,
        )
    elif variant["family"] == "single_pedestrian_speed_offset":
        scenario["single_pedestrians"] = _single_pedestrian_overrides_for_speed(
            variant,
            scenario=scenario,
            scenario_config=scenario_config,
            perturbation_summary=result.perturbation_summary,
        )
    elif variant["family"] == "single_pedestrian_wait_duration_offset":
        scenario["single_pedestrians"] = _single_pedestrian_overrides_for_wait_duration(
            variant,
            scenario=scenario,
            scenario_config=scenario_config,
            perturbation_summary=result.perturbation_summary,
        )
    return scenario


def _resolve_materialized_scenario_paths(
    scenario: dict[str, Any],
    *,
    scenario_config: Path,
) -> dict[str, Any]:
    """Resolve source scenario asset paths before writing an out-of-tree matrix.

    Returns:
        dict[str, Any]: Scenario entry with loadable path fields.
    """
    updated = dict(scenario)
    for field_name in ("map_file", "route_overrides_file"):
        value = updated.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            continue
        resolved = (scenario_config.parent / candidate).resolve()
        if resolved.exists():
            updated[field_name] = resolved.as_posix()
    return updated


def _limit_seeds(seeds: tuple[int, ...], seed_limit: int | None) -> list[int]:
    """Apply an optional positive seed limit to a variant seed tuple.

    Returns:
        list[int]: Variant seeds, optionally truncated for a small local pilot.
    """
    if seed_limit is None:
        return list(seeds)
    if seed_limit <= 0:
        raise ValueError("seed_limit must be positive when provided")
    return list(seeds[:seed_limit])


def _write_route_offset_override(
    variant: Mapping[str, Any],
    *,
    scenario_config: Path,
    scenarios: list[Mapping[str, Any]],
    perturbation_summary: Mapping[str, Any],
    route_path: Path,
) -> None:
    """Write a route override artifact for one route-offset variant."""
    scenario = dict(select_scenario(scenarios, str(variant["scenario_id"])))
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_config)
    if not config.map_pool.map_defs:
        raise ValueError("scenario map pool is empty")
    _map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    perturbed_map = deepcopy(map_def)
    _apply_route_offset(
        _routes_for_family(perturbed_map, str(variant["family"])),
        dx=float(perturbation_summary["dx_m"]),
        dy=float(perturbation_summary["dy_m"]),
        parameters=variant.get("parameters", {}),
        family=str(variant["family"]),
    )
    route_payload = {
        "schema_version": "scenario_route_overrides.v1",
        "source": f"{scenario_config.as_posix()}#{variant['scenario_id']}",
        "variant_id": variant["variant_id"],
        "route_payload": {
            "robot_routes": (
                [_route_to_payload(route) for route in perturbed_map.robot_routes]
                if variant["family"] == "robot_route_offset"
                else []
            ),
            "ped_routes": (
                [_route_to_payload(route) for route in perturbed_map.ped_routes]
                if variant["family"] == "pedestrian_route_offset"
                else []
            ),
        },
    }
    route_path.write_text(yaml.safe_dump(route_payload, sort_keys=False), encoding="utf-8")


def _routes_for_family(map_def: Any, family: str) -> list[GlobalRoute]:
    """Return the route collection targeted by a perturbation family."""
    if family == "robot_route_offset":
        return map_def.robot_routes
    if family == "pedestrian_route_offset":
        return map_def.ped_routes
    raise ValueError(f"unsupported route-offset family: {family}")


def _select_single_pedestrians(
    pedestrians: list[Any],
    parameters: Any,
    *,
    family: str = "single_pedestrian_start_delay_offset",
) -> list[Any]:
    """Return the single-pedestrian definitions targeted by a timing perturbation."""
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    selector = str(parameters.get("pedestrian_selector", "all"))
    if selector != "all":
        raise ValueError(f"{family} supports selector='all' only")
    pedestrian_id = parameters.get("pedestrian_id")
    selected = [
        ped for ped in pedestrians if pedestrian_id is None or str(ped.id) == str(pedestrian_id)
    ]
    if not selected:
        raise ValueError(f"{family} selected no single pedestrians")
    return selected


def _apply_start_delay_offset(
    pedestrians: list[Any],
    *,
    dt_s: float,
    parameters: Any,
) -> None:
    """Offset selected single-pedestrian start delays in place."""
    selected = _select_single_pedestrians(pedestrians, parameters)
    for ped in selected:
        updated = float(ped.start_delay_s) + float(dt_s)
        if updated < 0.0:
            raise ValueError(
                "single_pedestrian_start_delay_offset would make "
                f"pedestrian {ped.id!r} start_delay_s negative"
            )
        ped.start_delay_s = updated


def _baseline_single_pedestrian_speed(ped: Any) -> float:
    """Return the speed used by runtime when a single pedestrian has no explicit override."""
    return (
        float(ped.speed_m_s)
        if getattr(ped, "speed_m_s", None) is not None
        else _DEFAULT_SINGLE_PEDESTRIAN_SPEED_M_S
    )


def _updated_single_pedestrian_speed(
    ped: Any,
    *,
    speed_delta_m_s: float,
    max_speed_m_s: Any,
) -> float:
    """Return a bounded updated single-pedestrian speed."""
    updated = _baseline_single_pedestrian_speed(ped) + float(speed_delta_m_s)
    if updated <= 0.0:
        raise ValueError(
            "single_pedestrian_speed_offset would make "
            f"pedestrian {ped.id!r} speed_m_s non-positive"
        )
    if max_speed_m_s is not None and updated > float(max_speed_m_s):
        raise ValueError(
            "single_pedestrian_speed_offset would make "
            f"pedestrian {ped.id!r} speed_m_s exceed max_speed_m_s"
        )
    return updated


def _apply_single_pedestrian_speed_offset(
    pedestrians: list[Any],
    *,
    speed_delta_m_s: float,
    max_speed_m_s: Any,
    parameters: Any,
) -> None:
    """Offset selected single-pedestrian speed overrides in place."""
    selected = _select_single_pedestrians(
        pedestrians,
        parameters,
        family="single_pedestrian_speed_offset",
    )
    for ped in selected:
        ped.speed_m_s = _updated_single_pedestrian_speed(
            ped,
            speed_delta_m_s=speed_delta_m_s,
            max_speed_m_s=max_speed_m_s,
        )


def _updated_wait_rule_payload(
    ped: Any, rule_index: int, rule: Any, *, wait_delta_s: float
) -> dict[str, Any]:
    """Return a YAML-safe wait rule payload after applying a bounded duration offset."""
    updated = float(rule.wait_s) + float(wait_delta_s)
    if updated < 0.0:
        raise ValueError(
            "single_pedestrian_wait_duration_offset would make "
            f"pedestrian {ped.id!r} wait_at[{rule_index}].wait_s negative"
        )
    payload = {
        "waypoint_index": int(rule.waypoint_index),
        "wait_s": updated,
    }
    if getattr(rule, "note", None) is not None:
        payload["note"] = str(rule.note)
    return payload


def _wait_duration_payloads_for_selected_pedestrians(
    pedestrians: list[Any],
    *,
    wait_delta_s: float,
    parameters: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Return updated wait rules for selected pedestrians, failing closed if no waits exist."""
    selected = _select_single_pedestrians(
        pedestrians,
        parameters,
        family="single_pedestrian_wait_duration_offset",
    )
    waits_by_pedestrian: dict[str, list[dict[str, Any]]] = {}
    for ped in selected:
        wait_rules = list(getattr(ped, "wait_at", None) or [])
        if not wait_rules:
            raise ValueError("single_pedestrian_wait_duration_offset selected no wait_at entries")
        waits_by_pedestrian[str(ped.id)] = [
            _updated_wait_rule_payload(ped, idx, rule, wait_delta_s=wait_delta_s)
            for idx, rule in enumerate(wait_rules)
        ]
    return waits_by_pedestrian


def _apply_wait_duration_offset(
    pedestrians: list[Any],
    *,
    wait_delta_s: float,
    parameters: Any,
) -> None:
    """Offset selected single-pedestrian wait durations in place."""
    waits_by_pedestrian = _wait_duration_payloads_for_selected_pedestrians(
        pedestrians,
        wait_delta_s=wait_delta_s,
        parameters=parameters,
    )
    for ped in pedestrians:
        updated_rules = waits_by_pedestrian.get(str(ped.id))
        if updated_rules is None:
            continue
        for rule, payload in zip(ped.wait_at or [], updated_rules, strict=True):
            rule.wait_s = float(payload["wait_s"])


def _single_pedestrian_overrides_for_start_delay(
    variant: Mapping[str, Any],
    *,
    scenario: Mapping[str, Any],
    scenario_config: Path,
    perturbation_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return scenario overrides with adjusted single-pedestrian start delays."""
    config = build_robot_config_from_scenario(dict(scenario), scenario_path=scenario_config)
    if not config.map_pool.map_defs:
        raise ValueError("scenario map pool is empty")
    _map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    selected = _select_single_pedestrians(
        list(map_def.single_pedestrians),
        variant.get("parameters", {}),
    )
    selected_ids = {str(ped.id) for ped in selected}
    delays = {}
    for ped in selected:
        updated = float(ped.start_delay_s) + float(perturbation_summary["dt_s"])
        if updated < 0.0:
            raise ValueError(
                "single_pedestrian_start_delay_offset would make "
                f"pedestrian {ped.id!r} start_delay_s negative"
            )
        delays[str(ped.id)] = updated

    raw_overrides = scenario.get("single_pedestrians")
    overrides = [dict(item) for item in raw_overrides] if isinstance(raw_overrides, list) else []
    seen_ids: set[str] = set()
    for entry in overrides:
        ped_id = str(entry.get("id") or "")
        seen_ids.add(ped_id)
        if ped_id in selected_ids:
            entry["start_delay_s"] = delays[ped_id]
    for ped_id in sorted(selected_ids - seen_ids):
        overrides.append({"id": ped_id, "start_delay_s": delays[ped_id]})
    return overrides


def _single_pedestrian_overrides_for_speed(
    variant: Mapping[str, Any],
    *,
    scenario: Mapping[str, Any],
    scenario_config: Path,
    perturbation_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return scenario overrides with adjusted single-pedestrian speeds."""
    config = build_robot_config_from_scenario(dict(scenario), scenario_path=scenario_config)
    if not config.map_pool.map_defs:
        raise ValueError("scenario map pool is empty")
    _map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    selected = _select_single_pedestrians(
        list(map_def.single_pedestrians),
        variant.get("parameters", {}),
        family="single_pedestrian_speed_offset",
    )
    selected_ids = {str(ped.id) for ped in selected}
    speeds = {}
    for ped in selected:
        speeds[str(ped.id)] = _updated_single_pedestrian_speed(
            ped,
            speed_delta_m_s=float(perturbation_summary["speed_delta_m_s"]),
            max_speed_m_s=perturbation_summary.get("max_speed_m_s"),
        )

    raw_overrides = scenario.get("single_pedestrians")
    overrides = [dict(item) for item in raw_overrides] if isinstance(raw_overrides, list) else []
    seen_ids: set[str] = set()
    for entry in overrides:
        ped_id = str(entry.get("id") or "")
        seen_ids.add(ped_id)
        if ped_id in selected_ids:
            entry["speed_m_s"] = speeds[ped_id]
    for ped_id in sorted(selected_ids - seen_ids):
        overrides.append({"id": ped_id, "speed_m_s": speeds[ped_id]})
    return overrides


def _single_pedestrian_overrides_for_wait_duration(
    variant: Mapping[str, Any],
    *,
    scenario: Mapping[str, Any],
    scenario_config: Path,
    perturbation_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return scenario overrides with adjusted single-pedestrian wait durations."""
    config = build_robot_config_from_scenario(dict(scenario), scenario_path=scenario_config)
    if not config.map_pool.map_defs:
        raise ValueError("scenario map pool is empty")
    _map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    waits_by_pedestrian = _wait_duration_payloads_for_selected_pedestrians(
        list(map_def.single_pedestrians),
        wait_delta_s=float(perturbation_summary["wait_delta_s"]),
        parameters=variant.get("parameters", {}),
    )
    selected_ids = set(waits_by_pedestrian)

    raw_overrides = scenario.get("single_pedestrians")
    overrides = [dict(item) for item in raw_overrides] if isinstance(raw_overrides, list) else []
    seen_ids: set[str] = set()
    for entry in overrides:
        ped_id = str(entry.get("id") or "")
        seen_ids.add(ped_id)
        if ped_id in selected_ids:
            entry["wait_at"] = waits_by_pedestrian[ped_id]
    for ped_id in sorted(selected_ids - seen_ids):
        overrides.append({"id": ped_id, "wait_at": waits_by_pedestrian[ped_id]})
    return overrides


def _route_to_payload(route: GlobalRoute) -> dict[str, Any]:
    """Serialize a global route into the route-override YAML surface.

    Returns:
        dict[str, Any]: Route override entry with spawn, goal, and waypoints.
    """
    return {
        "spawn_id": int(route.spawn_id),
        "goal_id": int(route.goal_id),
        "waypoints": [[float(x), float(y)] for x, y in route.waypoints],
    }


def _apply_route_offset(
    routes: list[GlobalRoute],
    *,
    dx: float,
    dy: float,
    parameters: Any,
    family: str,
) -> None:
    """Offset selected route waypoints in place."""
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    spawn_id = parameters.get("spawn_id")
    goal_id = parameters.get("goal_id")
    selector = str(parameters.get("waypoint_selector", "all"))
    if selector != "all":
        raise ValueError(f"{family} currently supports waypoint_selector='all' only")
    selected = [
        route
        for route in routes
        if (spawn_id is None or route.spawn_id == int(spawn_id))
        and (goal_id is None or route.goal_id == int(goal_id))
    ]
    if not selected:
        route_kind = "pedestrian" if family == "pedestrian_route_offset" else "robot"
        raise ValueError(f"{family} selected no {route_kind} routes")
    for route in selected:
        route.waypoints = [(float(x) + dx, float(y) + dy) for x, y in route.waypoints]
        route.source_label = f"{route.source_label or family}|{family}"


def _variant_metadata(
    metadata: Any,
    variant: Mapping[str, Any],
    perturbation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach perturbation provenance to the scenario payload passed into certification.

    Returns:
        dict[str, Any]: Metadata with scenario perturbation provenance.
    """
    copied = dict(metadata) if isinstance(metadata, Mapping) else {}
    copied["scenario_perturbation"] = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "variant_id": variant["variant_id"],
        "family": variant["family"],
        "summary": dict(perturbation_summary),
        "evidence_policy": "validity_preflight_only_not_execution_evidence",
    }
    return copied


def _finite_float(raw: Any, *, field_name: str) -> float:
    """Coerce a finite float.

    Returns:
        float: Parsed finite value.
    """
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _positive_float(raw: Any, *, field_name: str) -> float:
    """Coerce a positive finite float.

    Returns:
        float: Parsed positive finite value.
    """
    value = _finite_float(raw, field_name=field_name)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def _validity_status_from_certificate(certificate: ScenarioCertificate) -> str:
    """Map certification eligibility to perturbation validity status.

    Returns:
        str: Perturbation validity status.
    """
    if certificate.benchmark_eligibility == "excluded":
        return "invalid"
    if certificate.benchmark_eligibility == "stress_only":
        return "stress_only"
    return "valid"


def _evidence_status_from_certificate(certificate: ScenarioCertificate) -> str:
    """Map certification eligibility to success-evidence policy.

    Returns:
        str: Evidence policy status.
    """
    if certificate.benchmark_eligibility == "eligible":
        return _SUCCESS_EVIDENCE_CANDIDATE
    if certificate.benchmark_eligibility == "stress_only":
        return _STRESS_ONLY_NOT_SUCCESS_EVIDENCE
    return _EXCLUDED_FROM_SUCCESS_EVIDENCE


def _result(
    variant: Mapping[str, Any],
    *,
    seeds: tuple[int, ...],
    validity_status: str,
    benchmark_evidence_status: str,
    reasons: list[str],
    perturbation_summary: dict[str, Any],
    certificate: ScenarioCertificate | None = None,
) -> PerturbationPreflightResult:
    """Build a result row from normalized variant fields.

    Returns:
        PerturbationPreflightResult: Normalized result row.
    """
    return PerturbationPreflightResult(
        variant_id=str(variant["variant_id"]),
        scenario_id=str(variant["scenario_id"]),
        family=str(variant["family"]),
        seeds=seeds,
        validity_status=validity_status,
        benchmark_evidence_status=benchmark_evidence_status,
        reasons=reasons,
        perturbation_summary=perturbation_summary,
        certificate=certificate,
    )
