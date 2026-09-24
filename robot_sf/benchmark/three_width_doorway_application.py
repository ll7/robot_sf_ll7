"""Three-width doorway comparison application for issue #9348.

Consumes the versioned three-level application manifest
(`robot_sf.issue_9348_three_width_doorway.v1`) and reuses the issue #6644
geometry-family owners for computation: variant-map generation, the variant
matrix, doorway-geometry derivation, clearance margins, and the
oracle-first sensitivity sweep. This module adds no second generator; it pins
the narrow/middle/wide application (3.6 m / 4.2 m / 4.8 m at fixed 1.0 m
depth), builds cross-width pair manifests, and checks that generated variant
scenarios differ from the historical baseline only in explained fields.

Evidence boundary: diagnostic within-simulator geometry evidence only. No
physical-footprint validation, realism evidence, sim-to-real evidence,
deployment safety, frozen-release evidence, or general planner ranking.
Planner rows stay ``not_run`` until a separately authorized campaign packet
executes them; the full paired campaign and uncertainty-quantified comparison
report belong to the successor issue, not this application slice.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from robot_sf.benchmark.narrow_doorway_geometry_family import (
    build_variant_matrix,
    generate_variant_map,
)
from robot_sf.benchmark.narrow_doorway_radius_audit import (
    derive_doorway_geometry,
    envelope_clearance_margin_m,
)
from robot_sf.evidence.writers import write_json
from robot_sf.scenario_certification.feasibility_oracle import (
    FeasibilityOracleConfig,
    envelope_sensitivity_verdict_to_dict,
    run_envelope_sensitivity_sweep,
)
from robot_sf.training.scenario_loader import load_scenarios

if TYPE_CHECKING:
    from robot_sf.scenario_certification.v1 import ScenarioCertificate

APPLICATION_SCHEMA = "robot_sf.issue_9348_three_width_doorway.v1"
PREFLIGHT_SCHEMA = "issue_9348_three_width_doorway_preflight.v1"
PAIR_MANIFEST_SCHEMA = "issue_9348_three_width_pair_manifest.v1"
CLAIM_BOUNDARY = (
    "diagnostic within-simulator geometry evidence only; not physical-footprint validation, "
    "realism evidence, sim-to-real evidence, deployment safety, frozen-release evidence, "
    "or a general planner ranking"
)
DEFAULT_MANIFEST_PATH = Path("configs/benchmarks/issue_9348_three_width_doorway_v1.yaml")
EXPECTED_TIER = "geometrically_feasible_candidate"
_TOLERANCE_M = 1e-9

_ALLOWED_TOP_LEVEL_CHANGES = frozenset(
    {"name", "map_file", "seeds", "simulation_config", "robot_config", "metadata"}
)
_ALLOWED_NESTED_CHANGES = frozenset(
    {
        "simulation_config.max_episode_steps",
        "robot_config.radius",
        "metadata.geometry_family_id",
        "metadata.geometry_variant_id",
        "metadata.gap_width_m",
        "metadata.constriction_depth_m",
        "metadata.diagnostic_claim_boundary",
        "metadata.geometry_application_issue",
    }
)


def _finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    """Parse one finite manifest number.

    Returns:
        Parsed finite value.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return parsed


def _positive_int(value: Any, *, field: str) -> int:
    """Parse one positive integer manifest value.

    Returns:
        Parsed positive integer.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0 or float(value) != parsed:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _float_levels(value: Any, *, field: str, minimum: float = 0.0) -> tuple[float, ...]:
    """Parse a non-empty ordered list of unique finite levels.

    Returns:
        Ordered tuple of levels.
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    levels = tuple(_finite_float(item, field=f"{field}[]", minimum=minimum) for item in value)
    if len({round(item, 9) for item in levels}) != len(levels):
        raise ValueError(f"{field} must not contain duplicate levels")
    return levels


def _resolve_reference(manifest_path: Path, raw: str) -> Path:
    """Resolve a repository-relative path recorded in the manifest.

    Returns:
        Existing absolute path.
    """
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = next(
            (
                parent / candidate
                for parent in manifest_path.resolve().parents
                if (parent / candidate).exists()
            ),
            manifest_path.parent / candidate,
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest reference does not exist: {raw}")
    return resolved.resolve()


def _read_manifest_header(path: Path) -> tuple[Path, dict[str, Any]]:
    """Read and check the application manifest envelope.

    Returns:
        Tuple of (resolved source path, payload mapping).
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("three-width manifest must contain a mapping")
    if payload.get("schema_version") != APPLICATION_SCHEMA:
        raise ValueError(f"schema_version must be {APPLICATION_SCHEMA!r}")
    if int(payload.get("issue", 0)) != 9348:
        raise ValueError("issue must be 9348")
    if not str(payload.get("family_id") or "").strip():
        raise ValueError("family_id must be non-empty")
    if str(payload.get("claim_boundary") or "").strip() != CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must retain the diagnostic boundary")
    return source, payload


def _validate_application_geometry(geometry: Any) -> tuple[tuple[float, ...], ...]:
    """Validate the pinned three-width geometry block.

    Returns:
        Tuple of (gap levels, depth levels, baseline gap, baseline depth).
    """
    if not isinstance(geometry, dict) or geometry.get("units") != "m":
        raise ValueError("geometry.units must be 'm'")
    gap_levels = _float_levels(geometry.get("gap_width_m"), field="geometry.gap_width_m")
    depth_levels = _float_levels(
        geometry.get("constriction_depth_m"),
        field="geometry.constriction_depth_m",
        minimum=_TOLERANCE_M,
    )
    if len(gap_levels) != 3:
        raise ValueError("geometry.gap_width_m must pin exactly three application widths")
    if len(depth_levels) != 1:
        raise ValueError("geometry.constriction_depth_m must fix one application depth")
    baseline = geometry.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("geometry.baseline must be a mapping")
    baseline_gap = _finite_float(baseline.get("gap_width_m"), field="geometry.baseline.gap_width_m")
    baseline_depth = _finite_float(
        baseline.get("constriction_depth_m"),
        field="geometry.baseline.constriction_depth_m",
        minimum=_TOLERANCE_M,
    )
    if not all(level > baseline_gap for level in gap_levels):
        raise ValueError("all comparison widths must exceed the zero-clearance baseline")
    if not any(math.isclose(baseline_depth, level, abs_tol=_TOLERANCE_M) for level in depth_levels):
        raise ValueError("geometry.constriction_depth_m must contain the baseline depth")
    return gap_levels, depth_levels, baseline_gap, baseline_depth


def _validate_application_envelope(envelope: Any) -> tuple[float, float]:
    """Validate the authoritative radius envelope.

    Returns:
        Tuple of (nominal radius, reduced probe radius).
    """
    if not isinstance(envelope, dict):
        raise ValueError("envelope must be a mapping")
    nominal_radius = _finite_float(
        envelope.get("nominal_radius_m"), field="envelope.nominal_radius_m", minimum=_TOLERANCE_M
    )
    reduced_radius = _finite_float(
        envelope.get("reduced_probe_radius_m"),
        field="envelope.reduced_probe_radius_m",
        minimum=_TOLERANCE_M,
    )
    if reduced_radius >= nominal_radius:
        raise ValueError("envelope.reduced_probe_radius_m must be below the nominal radius")
    if not str(envelope.get("source") or "").strip():
        raise ValueError("envelope.source must identify the authoritative radius source")
    return nominal_radius, reduced_radius


def _check_width_tiers(gap_levels: tuple[float, ...], nominal_radius: float) -> None:
    """Require three ascending widths with positive collision-envelope clearance."""
    if tuple(sorted(gap_levels)) != gap_levels:
        raise ValueError("geometry.gap_width_m must be strictly ascending")
    if any(envelope_clearance_margin_m(gap, nominal_radius) <= _TOLERANCE_M for gap in gap_levels):
        raise ValueError("all comparison widths must have positive collision-envelope clearance")


def _validate_application_protocol(
    oracle: Any, planner: Any
) -> tuple[int, int, tuple[str, ...], tuple[int, ...]]:
    """Validate the oracle-first protocol and planner roster.

    Returns:
        Tuple of (oracle seed, horizon steps, roster, seeds).
    """
    if not isinstance(oracle, dict) or not bool(oracle.get("run_before_planners")):
        raise ValueError("oracle.run_before_planners must be true")
    oracle_seed = _positive_int(oracle.get("seed"), field="oracle.seed")
    horizon = _positive_int(oracle.get("horizon_steps"), field="oracle.horizon_steps")
    if not isinstance(planner, dict):
        raise ValueError("planner_protocol must be a mapping")
    roster = planner.get("roster")
    if not isinstance(roster, list) or not roster or any(not str(item).strip() for item in roster):
        raise ValueError("planner_protocol.roster must be a non-empty list")
    if len(set(roster)) != len(roster):
        raise ValueError("planner_protocol.roster must not contain duplicates")
    seeds = planner.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("planner_protocol.seeds must be a non-empty list")
    normalized_seeds = [_positive_int(seed, field="planner_protocol.seeds[]") for seed in seeds]
    planner_horizon = _positive_int(
        planner.get("horizon_steps"), field="planner_protocol.horizon_steps"
    )
    if planner_horizon != horizon:
        raise ValueError("oracle and planner protocol horizons must match")
    if str(planner.get("execution_status")) != "not_started":
        raise ValueError("planner_protocol.execution_status must remain not_started")
    if _positive_int(planner.get("expected_rows"), field="planner_protocol.expected_rows") != (
        3 * len(roster) * len(normalized_seeds)
    ):
        raise ValueError("planner_protocol.expected_rows must equal 3 * planners * seeds")
    if planner.get("pair_admission") != (
        "fail_closed_until_initial_state_and_external_rng_hashes_match"
    ):
        raise ValueError("planner_protocol.pair_admission must require verified portable pairing")
    return oracle_seed, horizon, tuple(str(item) for item in roster), tuple(normalized_seeds)


def _validate_application_execution(execution: Any) -> None:
    """Require un-authorized campaign execution gates."""
    if not isinstance(execution, dict):
        raise ValueError("execution must be a mapping")
    if execution.get("production_campaign_authorized") is not False:
        raise ValueError("production campaign authorization must remain false")
    if execution.get("slurm_submission_authorized") is not False:
        raise ValueError("Slurm submission authorization must remain false")


def load_three_width_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the versioned #9348 three-width application manifest.

    Returns:
        Validated manifest with resolved source paths in the private ``_resolved`` block,
        shaped so the #6644 ``build_variant_matrix`` owner consumes it directly.
    """
    source, payload = _read_manifest_header(path)

    base = payload.get("base_scenario")
    if not isinstance(base, dict):
        raise ValueError("base_scenario must be a mapping")
    if not str(base.get("scenario_id") or "").strip():
        raise ValueError("base_scenario.scenario_id must be non-empty")
    scenario_path = _resolve_reference(source, str(base.get("scenario_path") or ""))
    map_path = _resolve_reference(source, str(base.get("map_path") or ""))

    gap_levels, depth_levels, baseline_gap, baseline_depth = _validate_application_geometry(
        payload.get("geometry")
    )
    nominal_radius, reduced_radius = _validate_application_envelope(payload.get("envelope"))
    _check_width_tiers(gap_levels, nominal_radius)
    oracle_seed, horizon, roster, normalized_seeds = _validate_application_protocol(
        payload.get("oracle"), payload.get("planner_protocol")
    )
    _validate_application_execution(payload.get("execution"))

    normalized = copy.deepcopy(payload)
    normalized["_resolved"] = {
        "manifest_path": source,
        "scenario_path": scenario_path,
        "map_path": map_path,
        "gap_levels": gap_levels,
        "depth_levels": depth_levels,
        "baseline_gap_m": baseline_gap,
        "baseline_depth_m": baseline_depth,
        "nominal_radius_m": nominal_radius,
        "reduced_radius_m": reduced_radius,
        "oracle_seed": oracle_seed,
        "horizon_steps": horizon,
        "planner_roster": tuple(str(item) for item in roster),
        "planner_seeds": tuple(normalized_seeds),
    }
    return normalized


def _sha256(path: Path) -> str:
    """Hash one generated or source asset.

    Returns:
        Lower-case SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_receipt_value(value: Any) -> Any:
    """Convert captured reset/RNG values into deterministic JSON primitives.

    Returns:
        A value supported by canonical JSON encoding.
    """
    if isinstance(value, np.ndarray):
        return _canonical_receipt_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_receipt_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _canonical_receipt_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_receipt_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"unsupported or nonfinite receipt value: {type(value).__name__}")


def _receipt_digest(payload: Any) -> str:
    encoded = json.dumps(
        _canonical_receipt_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_pair_receipt(reset: Mapping[str, Any], snapshot: Any) -> dict[str, str]:
    """Hash reset actors and external streams from a pre-command simulator snapshot.

    The caller must capture ``SimulatorCounterfactualModel.snapshot()`` directly
    after ``env.reset(seed=...)`` and before the first policy command. The
    snapshot is not a complete planner checkpoint; this receipt only checks
    the external streams and actor states needed to admit a width pair.

    Returns:
        SHA-256 receipts for actor state and external RNG state.
    """
    robot = reset.get("robot")
    pedestrians = reset.get("pedestrians")
    if not isinstance(robot, Mapping) or not isinstance(pedestrians, list):
        raise ValueError("reset actor state is unavailable")
    if robot.get("position") is None or robot.get("velocity") is None:
        raise ValueError("reset robot pose or velocity is unavailable")
    actors = []
    for actor in pedestrians:
        if not isinstance(actor, Mapping) or actor.get("actor_id") is None:
            raise ValueError("stable reset pedestrian identity is unavailable")
        if actor.get("position") is None or actor.get("velocity") is None:
            raise ValueError("reset pedestrian pose or velocity is unavailable")
        actors.append(
            {
                "id": actor["actor_id"],
                "position": actor["position"],
                "velocity": actor["velocity"],
                "heading": actor.get("heading"),
            }
        )
    actors.sort(key=lambda actor: str(actor["id"]))
    global_rng = getattr(snapshot, "global_rng_state", None)
    python_rng = getattr(snapshot, "python_random_state", None)
    behavior_rng = getattr(snapshot, "behavior_rng_states", None)
    if global_rng is None or python_rng is None or behavior_rng is None:
        raise ValueError("external RNG snapshot is incomplete")
    return {
        "initial_actor_state_sha256": _receipt_digest({"robot": dict(robot), "actors": actors}),
        "external_rng_state_sha256": _receipt_digest(
            {
                "global_numpy": global_rng,
                "python_random": python_rng,
                "behavior_rng": behavior_rng,
                "residual_adversary_rng": getattr(snapshot, "residual_adversary_state", None),
            }
        ),
    }


def _build_application_scenario(
    base_scenario: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    map_file: str,
    nominal_radius_m: float,
    seeds: tuple[int, ...],
    horizon_steps: int,
    family_id: str,
) -> dict[str, Any]:
    """Build an isolated scenario entry for one application variant map.

    Returns:
        Scenario mapping with explicit diagnostic geometry metadata.
    """
    scenario = copy.deepcopy(dict(base_scenario))
    scenario["name"] = str(variant["variant_id"])
    scenario["map_file"] = map_file
    scenario["seeds"] = list(seeds)
    simulation_config = dict(scenario.get("simulation_config") or {})
    simulation_config["max_episode_steps"] = horizon_steps
    scenario["simulation_config"] = simulation_config
    robot_config = dict(scenario.get("robot_config") or {})
    robot_config["radius"] = float(nominal_radius_m)
    scenario["robot_config"] = robot_config
    metadata = dict(scenario.get("metadata") or {})
    metadata.update(
        {
            "geometry_family_id": family_id,
            "geometry_application_issue": 9348,
            "geometry_variant_id": str(variant["variant_id"]),
            "gap_width_m": float(variant["gap_width_m"]),
            "constriction_depth_m": float(variant["constriction_depth_m"]),
            "diagnostic_claim_boundary": CLAIM_BOUNDARY,
        }
    )
    scenario["metadata"] = metadata
    return scenario


def _write_variant_scenario(path: Path, scenario: Mapping[str, Any]) -> Path:
    """Write an isolated one-scenario YAML manifest.

    Returns:
        Path to the written manifest.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump({"scenarios": [dict(scenario)]}, sort_keys=False))
    return target


def _load_base_scenario(manifest: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    """Load the historical base scenario without modifying it.

    Returns:
        Tuple of (scenario path, map path, base scenario mapping).
    """
    resolved = manifest["_resolved"]
    scenario_path = Path(resolved["scenario_path"])
    map_path = Path(resolved["map_path"])
    scenario_id = str(manifest["base_scenario"]["scenario_id"])
    scenarios = [dict(item) for item in load_scenarios(scenario_path)]
    matches = [item for item in scenarios if str(item.get("name")) == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"expected one baseline scenario {scenario_id!r}, found {len(matches)}")
    return scenario_path, map_path, matches[0]


def _leaf_paths(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings/lists to dotted leaf paths with list indices as [*].

    Returns:
        Mapping of leaf path to leaf value.
    """
    leaves: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_leaf_paths(value, path))
    elif isinstance(payload, list):
        for value in payload:
            leaves.update(_leaf_paths(value, f"{prefix}[*]" if prefix else "[*]"))
    else:
        leaves[prefix] = payload
    return leaves


def check_variant_diff(
    base_scenario: Mapping[str, Any], variant_scenario: Mapping[str, Any]
) -> list[str]:
    """Check that a variant scenario differs from baseline only in explained fields.

    Whole-value replacements of ``name``, ``map_file``, and ``seeds`` are
    allowed; every other leaf change must sit on the nested allowlist.

    Returns:
        List of unexplained difference descriptions; empty when the diff is clean.
    """
    base_leaves = _leaf_paths(base_scenario)
    variant_leaves = _leaf_paths(variant_scenario)
    violations: list[str] = []
    for path in sorted(set(base_leaves) | set(variant_leaves)):
        if base_leaves.get(path, "<absent>") == variant_leaves.get(path, "<absent>"):
            continue
        stem = path.replace("[*]", "")
        top = stem.split(".")[0]
        if top in {"name", "map_file", "seeds"}:
            continue
        if top not in _ALLOWED_TOP_LEVEL_CHANGES or stem not in _ALLOWED_NESTED_CHANGES:
            violations.append(f"unexplained change at {path}")
    return violations


def build_pair_manifest(
    variant_assets: list[Mapping[str, Any]],
    seeds: tuple[int, ...],
    manifest_sha256: str,
) -> dict[str, Any]:
    """Build cross-width pair records sharing one seed per pair.

    A matching seed alone is insufficient when geometry changes random draws, so
    every pair carries the configuration hash now and reserves realization-hash
    slots for the separately authorized campaign.

    Returns:
        Pair manifest with one pair per seed spanning all three widths.
    """
    ordered = sorted(variant_assets, key=lambda item: float(item["gap_width_m"]))
    if len(ordered) != 3:
        raise ValueError("pair manifest requires exactly the three application widths")
    pairs = []
    for seed in seeds:
        pairs.append(
            {
                "pair_id": f"pair_{int(seed):05d}",
                "seed": int(seed),
                "cells": [
                    {
                        "variant_id": str(item["variant_id"]),
                        "gap_width_m": float(item["gap_width_m"]),
                        "scenario_sha256": str(item["scenario_sha256"]),
                        "map_sha256": str(item["map_sha256"]),
                        "initial_actor_state_sha256": None,
                        "external_rng_state_sha256": None,
                    }
                    for item in ordered
                ],
            }
        )
    return {
        "schema_version": PAIR_MANIFEST_SCHEMA,
        "issue": 9348,
        "manifest_sha256": manifest_sha256,
        "realization_hash_status": "pending_initial_and_external_rng_state_verification",
        "admission": "blocked_until_all_three_cells_match_initial_and_external_rng_state_hashes",
        "pairs": pairs,
    }


def check_pair_receipts(pair_manifest: Mapping[str, Any]) -> list[str]:
    """Return reasons that any three-width pair lacks portable initial-state custody."""
    failures: list[str] = []
    for pair in pair_manifest.get("pairs", []):
        pair_id = str(pair.get("pair_id"))
        cells = pair.get("cells", [])
        if len(cells) != 3:
            failures.append(f"{pair_id}: expected exactly three width cells")
            continue
        for field in ("initial_actor_state_sha256", "external_rng_state_sha256"):
            values = [cell.get(field) for cell in cells]
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in values
            ):
                failures.append(f"{pair_id}: missing {field} receipt")
            elif len(set(values)) != 1:
                failures.append(f"{pair_id}: {field} differs across widths")
    return failures


def generate_application_assets(
    manifest: Mapping[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    """Generate variant maps and scenarios for the three application widths.

    Returns:
        Asset records with variant identity, geometry, and content hashes.
    """
    resolved = manifest["_resolved"]
    _, map_path, base_scenario = _load_base_scenario(manifest)
    # The reusable #6644 matrix insists its diagnostic baseline be a matrix cell.
    # This application deliberately excludes the historical zero-clearance baseline;
    # supply a matrix-local anchor while retaining the true baseline for all checks.
    matrix_manifest = copy.deepcopy(manifest)
    matrix_manifest["_resolved"]["baseline_gap_m"] = float(resolved["gap_levels"][0])
    variants = build_variant_matrix(matrix_manifest)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = root / str(variant["variant_id"])
        map_output = variant_dir / "variant.svg"
        scenario_output = variant_dir / "scenario.yaml"
        generate_variant_map(
            map_path,
            gap_width_m=float(variant["gap_width_m"]),
            constriction_depth_m=float(variant["constriction_depth_m"]),
            output_path=map_output,
        )
        scenario_payload = _build_application_scenario(
            base_scenario,
            variant,
            map_file="variant.svg",
            nominal_radius_m=float(resolved["nominal_radius_m"]),
            seeds=tuple(resolved["planner_seeds"]),
            horizon_steps=int(resolved["horizon_steps"]),
            family_id=str(manifest["family_id"]),
        )
        _write_variant_scenario(scenario_output, scenario_payload)
        loaded_variant = dict(load_scenarios(scenario_output)[0])
        violations = check_variant_diff(base_scenario, loaded_variant)
        if violations:
            raise ValueError(f"variant diff check failed for {variant['variant_id']}: {violations}")
        assets.append(
            {
                "variant_id": str(variant["variant_id"]),
                "gap_width_m": float(variant["gap_width_m"]),
                "constriction_depth_m": float(variant["constriction_depth_m"]),
                "envelope_radius_m": float(variant["envelope_radius_m"]),
                "envelope_diameter_m": float(variant["envelope_diameter_m"]),
                "derived_clearance_margin_m": float(variant["derived_clearance_margin_m"]),
                "expected_geometry_tier": str(variant["expected_geometry_tier"]),
                "scenario_path": scenario_output.as_posix(),
                "scenario_sha256": _sha256(scenario_output),
                "map_path": map_output.as_posix(),
                "map_sha256": _sha256(map_output),
            }
        )
    return assets


def run_three_width_preflight(
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
    episode_runner: Callable[..., Mapping[str, Any]] | None = None,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate] | None = None,
) -> dict[str, Any]:
    """Generate the three-width matrix and run the planner-free oracle per variant.

    Returns:
        Review-ready preflight report with one oracle-first record per width.
    """
    manifest_source = Path(manifest_path).resolve()
    manifest = load_three_width_manifest(manifest_source)
    scenario_path, map_path, base_scenario = _load_base_scenario(manifest)
    resolved = manifest["_resolved"]
    baseline_geometry = derive_doorway_geometry(scenario_path, base_scenario)
    baseline_depth = float(baseline_geometry.obstacle_rects[0]["width"])
    baseline_checks = {
        "scenario_id_matches": base_scenario.get("name")
        == manifest["base_scenario"]["scenario_id"],
        "gap_width_matches_manifest": math.isclose(
            baseline_geometry.gap_width_m, float(resolved["baseline_gap_m"]), abs_tol=_TOLERANCE_M
        ),
        "constriction_depth_matches_manifest": math.isclose(
            baseline_depth, float(resolved["baseline_depth_m"]), abs_tol=_TOLERANCE_M
        ),
        "route_waypoints_match_manifest": [
            list(point) for point in baseline_geometry.route_waypoints
        ]
        == manifest["geometry"]["route_waypoints"],
        "baseline_map_is_authored_source": map_path == Path(resolved["map_path"]),
    }
    if not all(baseline_checks.values()):
        raise ValueError(f"baseline geometry does not satisfy manifest: {baseline_checks}")

    temp_root = tempfile.TemporaryDirectory(prefix="issue-9348-three-width-")
    try:
        root = Path(output_dir).resolve() if output_dir is not None else Path(temp_root.name)
        assets = generate_application_assets(manifest, root)
        records: list[dict[str, Any]] = []
        for asset in assets:
            scenario_output = Path(asset["scenario_path"])
            loaded_variant = dict(load_scenarios(scenario_output)[0])
            oracle_config = FeasibilityOracleConfig(
                scenario_path=scenario_output,
                envelope_radii_m=(
                    float(resolved["nominal_radius_m"]),
                    float(resolved["reduced_radius_m"]),
                ),
                rollout_algo=str(manifest["oracle"]["algorithm"]),
                rollout_seed=int(resolved["oracle_seed"]),
            )
            try:
                oracle_verdict = run_envelope_sensitivity_sweep(
                    loaded_variant,
                    config=oracle_config,
                    episode_runner=episode_runner,
                    certifier=certifier,
                )
                oracle = envelope_sensitivity_verdict_to_dict(oracle_verdict, issue="9348")
                oracle["execution_status"] = "available"
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                oracle = {
                    "execution_status": "blocked",
                    "blocker": f"oracle_error: {exc}",
                }
            records.append(
                {
                    "variant_id": asset["variant_id"],
                    "geometry": {
                        "gap_width_m": asset["gap_width_m"],
                        "constriction_depth_m": asset["constriction_depth_m"],
                        "envelope_radius_m": asset["envelope_radius_m"],
                        "envelope_diameter_m": asset["envelope_diameter_m"],
                        "derived_clearance_margin_m": asset["derived_clearance_margin_m"],
                        "expected_geometry_tier": asset["expected_geometry_tier"],
                    },
                    "assets": {
                        "scenario_path": asset["scenario_path"],
                        "scenario_sha256": asset["scenario_sha256"],
                        "map_path": asset["map_path"],
                        "map_sha256": asset["map_sha256"],
                    },
                    "oracle": oracle,
                    "planner": {
                        "status": "not_run",
                        "valid_evidence": False,
                        "fallback": None,
                        "degraded": None,
                        "rows": [],
                        "reason": "production campaign requires a separate issue-owned packet and oracle admission",
                    },
                    "disposition": "oracle_only_preflight",
                }
            )
    finally:
        if output_dir is None:
            temp_root.cleanup()

    oracle_available = all(
        item["oracle"].get("execution_status") == "available" for item in records
    )
    geometry_feasible = all(
        item["oracle"]
        .get("nominal_verdict", {})
        .get("geometric", {})
        .get("route_geometrically_feasible")
        is True
        for item in records
    )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "issue": 9348,
        "family_id": manifest["family_id"],
        "manifest": manifest_source.as_posix(),
        "claim_boundary": CLAIM_BOUNDARY,
        "baseline": {
            "scenario_path": scenario_path.as_posix(),
            "map_path": map_path.as_posix(),
            "scenario_sha256": _sha256(scenario_path),
            "map_sha256": _sha256(map_path),
            "geometry": {
                "gap_width_m": baseline_geometry.gap_width_m,
                "constriction_depth_m": baseline_depth,
                "route_waypoints": [list(point) for point in baseline_geometry.route_waypoints],
                "route_min_center_distance_m": baseline_geometry.route_min_center_distance_m,
            },
            "checks": baseline_checks,
        },
        "protocol": {
            "oracle_first": True,
            "nominal_radius_m": resolved["nominal_radius_m"],
            "reduced_probe_radius_m": resolved["reduced_radius_m"],
            "planner_roster": list(resolved["planner_roster"]),
            "planner_seeds": list(resolved["planner_seeds"]),
            "horizon_steps": resolved["horizon_steps"],
            "production_campaign_authorized": False,
            "slurm_submission_authorized": False,
        },
        "checks": {
            "baseline_passes": all(baseline_checks.values()),
            "variant_count": len(records),
            "all_widths_positive_clearance": all(
                item["geometry"]["derived_clearance_margin_m"] > _TOLERANCE_M
                and item["geometry"]["expected_geometry_tier"] == EXPECTED_TIER
                for item in records
            ),
            "oracle_available_for_every_variant": oracle_available,
            "nominal_grid_route_feasible_for_every_variant": geometry_feasible,
            "planner_records_are_not_run": all(
                item["planner"]["status"] == "not_run" for item in records
            ),
            "no_campaign_evidence": True,
        },
        "variants": records,
        "execution": {
            "campaign_submitted": False,
            "confirmation_ready": False,
            "confirmation_blocker": "portable_initial_and_external_rng_pair_receipts_not_recorded",
            "evidence_admission": "not_started",
            "missingness_policy": "blocked or degraded oracle/planner rows remain explicit and are not promoted",
        },
        "go": all(
            (
                all(baseline_checks.values()),
                bool(records),
                oracle_available,
                geometry_feasible,
                all(item["planner"]["status"] == "not_run" for item in records),
            )
        ),
    }


def write_preflight_report(report: Mapping[str, Any], output_path: Path) -> Path:
    """Write one review-marked deterministic preflight report.

    Returns:
        Path to the written report.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, dict(report))
    return target


__all__ = [
    "APPLICATION_SCHEMA",
    "CLAIM_BOUNDARY",
    "DEFAULT_MANIFEST_PATH",
    "EXPECTED_TIER",
    "PAIR_MANIFEST_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "build_pair_manifest",
    "build_pair_receipt",
    "check_pair_receipts",
    "check_variant_diff",
    "generate_application_assets",
    "load_three_width_manifest",
    "run_three_width_preflight",
    "write_preflight_report",
]
