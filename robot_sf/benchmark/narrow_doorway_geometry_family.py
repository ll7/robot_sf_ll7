"""Generate and preflight a controlled narrow-doorway geometry family.

Issue #6644 is a diagnostic geometry package.  It varies the free doorway width and
the wall-lined constriction depth while keeping the authored route, map bounds, spawn,
goal, horizon, and planner protocol explicit.  Variant maps are generated in a caller
provided directory (or a temporary directory), so the frozen baseline map is never
rewritten.

The preflight runs the existing planner-free feasibility oracle before it creates a
planner result record.  Planner execution is deliberately represented as ``not_run``
until a separate campaign packet authorizes it.
"""

from __future__ import annotations

import copy
import hashlib
import math
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

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

# evidence-writer-exempt: generated temporary SVG/YAML scenario assets are protocol inputs,
# not evidence artifacts; JSON reports use robot_sf.evidence.writers.write_json below.

GEOMETRY_FAMILY_SCHEMA = "robot_sf.issue_6644_narrow_doorway_geometry_family.v1"
PREFLIGHT_SCHEMA = "issue_6644_narrow_doorway_geometry_family_preflight.v1"
CLAIM_BOUNDARY = (
    "diagnostic within-simulator geometry evidence only; not physical-footprint validation, "
    "realism evidence, sim-to-real evidence, deployment safety, frozen-release evidence, "
    "or a general planner ranking"
)
DEFAULT_MANIFEST_PATH = Path("configs/benchmarks/issue_6644_narrow_doorway_geometry_family_v1.yaml")
_INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_INKSCAPE_NAMESPACE = "http://www.inkscape.org/namespaces/inkscape"
_TOLERANCE_M = 1e-9


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


def load_geometry_family_manifest(path: Path) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Load and validate the versioned #6644 geometry-family manifest.

    Returns:
        Validated manifest with resolved source paths in the private ``_resolved`` block.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("geometry-family manifest must contain a mapping")
    if payload.get("schema_version") != GEOMETRY_FAMILY_SCHEMA:
        raise ValueError(f"schema_version must be {GEOMETRY_FAMILY_SCHEMA!r}")
    if int(payload.get("issue", 0)) != 6644:
        raise ValueError("issue must be 6644")
    if not str(payload.get("family_id") or "").strip():
        raise ValueError("family_id must be non-empty")
    if str(payload.get("claim_boundary") or "").strip() != CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must retain the #6644 diagnostic boundary")

    base = payload.get("base_scenario")
    if not isinstance(base, dict):
        raise ValueError("base_scenario must be a mapping")
    if not str(base.get("scenario_id") or "").strip():
        raise ValueError("base_scenario.scenario_id must be non-empty")
    scenario_path = _resolve_reference(source, str(base.get("scenario_path") or ""))
    map_path = _resolve_reference(source, str(base.get("map_path") or ""))

    geometry = payload.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("units") != "m":
        raise ValueError("geometry.units must be 'm'")
    gap_levels = _float_levels(geometry.get("gap_width_m"), field="geometry.gap_width_m")
    depth_levels = _float_levels(
        geometry.get("constriction_depth_m"),
        field="geometry.constriction_depth_m",
        minimum=_TOLERANCE_M,
    )
    baseline = geometry.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("geometry.baseline must be a mapping")
    baseline_gap = _finite_float(baseline.get("gap_width_m"), field="geometry.baseline.gap_width_m")
    baseline_depth = _finite_float(
        baseline.get("constriction_depth_m"),
        field="geometry.baseline.constriction_depth_m",
        minimum=_TOLERANCE_M,
    )
    if not any(math.isclose(baseline_gap, level, abs_tol=_TOLERANCE_M) for level in gap_levels):
        raise ValueError("geometry.gap_width_m must contain the baseline gap")
    if not any(math.isclose(baseline_depth, level, abs_tol=_TOLERANCE_M) for level in depth_levels):
        raise ValueError("geometry.constriction_depth_m must contain the baseline depth")
    if not any(
        math.isclose(level, baseline_gap - 0.1, abs_tol=_TOLERANCE_M) for level in gap_levels
    ):
        raise ValueError("geometry.gap_width_m must include a 0.10 m narrower probe")
    if not any(
        math.isclose(level, baseline_gap + 0.1, abs_tol=_TOLERANCE_M) for level in gap_levels
    ):
        raise ValueError("geometry.gap_width_m must include a 0.10 m wider probe")
    _finite_float(geometry.get("route_y_m"), field="geometry.route_y_m")
    waypoints = geometry.get("route_waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError("geometry.route_waypoints must contain at least two points")
    for index, point in enumerate(waypoints):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"geometry.route_waypoints[{index}] must be [x, y]")
        _finite_float(point[0], field=f"geometry.route_waypoints[{index}][0]")
        _finite_float(point[1], field=f"geometry.route_waypoints[{index}][1]")

    envelope = payload.get("envelope")
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

    oracle = payload.get("oracle")
    if not isinstance(oracle, dict) or not bool(oracle.get("run_before_planners")):
        raise ValueError("oracle.run_before_planners must be true")
    oracle_seed = _positive_int(oracle.get("seed"), field="oracle.seed")
    horizon = _positive_int(oracle.get("horizon_steps"), field="oracle.horizon_steps")
    planner = payload.get("planner_protocol")
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
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("execution must be a mapping")
    if execution.get("production_campaign_authorized") is not False:
        raise ValueError("production campaign authorization must remain false")
    if execution.get("slurm_submission_authorized") is not False:
        raise ValueError("Slurm submission authorization must remain false")

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


def _level_token(value: float) -> str:
    """Return a stable identifier token for one metre-valued factor."""
    token = f"{float(value):.2f}"
    return token.replace("-", "m").replace(".", "p")


def build_variant_matrix(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the ordered Cartesian product of width and depth factors.

    Returns:
        Ordered variant records with derived nominal-envelope margins.
    """
    resolved = manifest.get("_resolved")
    if not isinstance(resolved, Mapping):
        raise ValueError("manifest must be loaded with load_geometry_family_manifest")
    nominal_radius = float(resolved["nominal_radius_m"])
    variants: list[dict[str, Any]] = []
    for gap_width_m in resolved["gap_levels"]:
        for constriction_depth_m in resolved["depth_levels"]:
            gap = float(gap_width_m)
            depth = float(constriction_depth_m)
            margin = envelope_clearance_margin_m(gap, nominal_radius)
            if margin < -_TOLERANCE_M:
                expected_tier = "infeasible_by_construction"
            elif abs(margin) <= _TOLERANCE_M:
                expected_tier = "boundary_tangent"
            else:
                expected_tier = "geometrically_feasible_candidate"
            variants.append(
                {
                    "variant_id": f"gap_{_level_token(gap)}__depth_{_level_token(depth)}",
                    "gap_width_m": gap,
                    "constriction_depth_m": depth,
                    "envelope_radius_m": nominal_radius,
                    "envelope_diameter_m": 2.0 * nominal_radius,
                    "derived_clearance_margin_m": margin,
                    "expected_geometry_tier": expected_tier,
                }
            )
    if not variants:
        raise ValueError("variant matrix must not be empty")
    baseline_gap = float(resolved["baseline_gap_m"])
    baseline_depth = float(resolved["baseline_depth_m"])
    if not any(
        math.isclose(item["gap_width_m"], baseline_gap, abs_tol=_TOLERANCE_M)
        and math.isclose(item["constriction_depth_m"], baseline_depth, abs_tol=_TOLERANCE_M)
        for item in variants
    ):
        raise ValueError("variant matrix must contain the baseline cell")
    return variants


def _xml_float(value: float) -> str:
    """Format a geometry number for a generated SVG attribute.

    Returns:
        Stable compact SVG number.
    """
    return f"{float(value):.12g}"


def generate_variant_map(
    source_map_path: Path,
    *,
    gap_width_m: float,
    constriction_depth_m: float,
    output_path: Path,
) -> Path:
    """Generate one temporary variant map without modifying the source map.

    Returns:
        Path to the generated map.
    """
    source = Path(source_map_path).resolve()
    target = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    gap = _finite_float(gap_width_m, field="gap_width_m", minimum=_TOLERANCE_M)
    depth = _finite_float(constriction_depth_m, field="constriction_depth_m", minimum=_TOLERANCE_M)
    if gap >= 8.0:
        raise ValueError("gap_width_m must leave positive wall segments inside the 10 m map")

    ET.register_namespace("", _SVG_NAMESPACE)
    ET.register_namespace("inkscape", _INKSCAPE_NAMESPACE)
    root = ET.parse(source).getroot()
    candidates: list[ET.Element] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "rect":
            continue
        if element.attrib.get(_INKSCAPE_LABEL) != "obstacle":
            continue
        try:
            x = float(element.attrib["x"])
            y = float(element.attrib["y"])
            width = float(element.attrib["width"])
            height = float(element.attrib["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("baseline map contains a non-numeric obstacle rectangle") from exc
        if (
            0.0 < y < 9.0
            and 0.0 < x < 29.0
            and 0.0 < height < 10.0
            and height > width
            and math.isclose(x, 15.0, abs_tol=_TOLERANCE_M)
        ):
            candidates.append(element)
    if len(candidates) != 2:
        raise ValueError(f"expected two internal doorway wall rectangles, found {len(candidates)}")

    lower, upper = sorted(candidates, key=lambda item: float(item.attrib["y"]))
    lower_edge = 5.0 - gap / 2.0
    upper_edge = 5.0 + gap / 2.0
    lower.attrib.update(
        {"x": "15", "y": "1", "width": _xml_float(depth), "height": _xml_float(lower_edge - 1.0)}
    )
    upper.attrib.update(
        {
            "x": "15",
            "y": _xml_float(upper_edge),
            "width": _xml_float(depth),
            "height": _xml_float(9.0 - upper_edge),
        }
    )
    if float(lower.attrib["height"]) <= 0.0 or float(upper.attrib["height"]) <= 0.0:
        raise ValueError("generated gap leaves no positive wall segment")
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    return target


def _sha256(path: Path) -> str:
    """Hash one generated or source asset.

    Returns:
        Lower-case SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_variant_scenario(
    base_scenario: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    map_file: str,
    nominal_radius_m: float,
    seeds: tuple[int, ...],
    horizon_steps: int,
    family_id: str,
) -> dict[str, Any]:
    """Build an isolated scenario entry for one generated map.

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
        Path to the generated scenario manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"scenarios": [dict(scenario)]}, sort_keys=False), encoding="utf-8"
    )
    return path


def _load_base_scenario(manifest: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve the authored baseline scenario, map, and scenario entry.

    Returns:
        Scenario manifest path, map path, and normalized scenario entry.
    """
    resolved = manifest["_resolved"]
    scenario_path = Path(resolved["scenario_path"])
    scenario_id = str(manifest["base_scenario"]["scenario_id"])
    scenarios = [dict(item) for item in load_scenarios(scenario_path)]
    matches = [item for item in scenarios if str(item.get("name")) == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"expected one baseline scenario {scenario_id!r}, found {len(matches)}")
    map_path = Path(resolved["map_path"])
    return scenario_path, map_path, matches[0]


def _fake_blocked_oracle(variant: Mapping[str, Any], blocker: str) -> dict[str, Any]:
    """Return a fail-closed oracle record when an oracle invocation cannot run."""
    return {
        "schema_version": "scenario_feasibility_oracle.v1",
        "issue": "6644",
        "claim_boundary": CLAIM_BOUNDARY,
        "scenario_id": str(variant["variant_id"]),
        "family_id": "francis2023_narrow_doorway_geometry_v1",
        "feasible": None,
        "status": "blocked",
        "blocker": blocker,
        "execution_status": "blocked",
    }


def run_geometry_family_preflight(
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
    episode_runner: Callable[..., Mapping[str, Any]] | None = None,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate] | None = None,
) -> dict[str, Any]:
    """Generate the matrix and run the planner-free oracle before planner execution.

    Returns:
        Review-ready preflight report with one oracle-first record per geometry cell.
    """
    manifest_source = Path(manifest_path).resolve()
    manifest = load_geometry_family_manifest(manifest_source)
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

    variants = build_variant_matrix(manifest)
    temp_root = tempfile.TemporaryDirectory(prefix="issue-6644-geometry-")
    try:
        root = Path(output_dir).resolve() if output_dir is not None else Path(temp_root.name)
        root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
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
            scenario_payload = _build_variant_scenario(
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
                oracle = envelope_sensitivity_verdict_to_dict(oracle_verdict, issue="6644")
                oracle["execution_status"] = "available"
            except Exception as exc:  # noqa: BLE001 - preflight must record missingness.
                oracle = _fake_blocked_oracle(variant, f"oracle_error: {exc}")

            records.append(
                {
                    "variant_id": variant["variant_id"],
                    "geometry": {
                        "gap_width_m": variant["gap_width_m"],
                        "constriction_depth_m": variant["constriction_depth_m"],
                        "envelope_radius_m": variant["envelope_radius_m"],
                        "envelope_diameter_m": variant["envelope_diameter_m"],
                        "derived_clearance_margin_m": variant["derived_clearance_margin_m"],
                        "expected_geometry_tier": variant["expected_geometry_tier"],
                    },
                    "assets": {
                        "scenario_path": scenario_output.as_posix(),
                        "scenario_sha256": _sha256(scenario_output),
                        "map_path": map_output.as_posix(),
                        "map_sha256": _sha256(map_output),
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
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "issue": 6644,
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
            "contains_infeasible_boundary": any(
                item["geometry"]["expected_geometry_tier"] == "boundary_tangent" for item in records
            ),
            "contains_positive_clearance_cell": any(
                item["geometry"]["expected_geometry_tier"] == "geometrically_feasible_candidate"
                for item in records
            ),
            "contains_short_depth": any(
                item["geometry"]["constriction_depth_m"] < 0.5 for item in records
            ),
            "contains_metre_scale_depth": any(
                item["geometry"]["constriction_depth_m"] >= 1.0 for item in records
            ),
            "oracle_available_for_every_variant": oracle_available,
            "planner_records_are_not_run": all(
                item["planner"]["status"] == "not_run" for item in records
            ),
            "no_campaign_evidence": True,
        },
        "variants": records,
        "execution": {
            "campaign_submitted": False,
            "evidence_admission": "not_started",
            "missingness_policy": "blocked or degraded oracle/planner rows remain explicit and are not promoted",
        },
        "go": all(
            (
                all(baseline_checks.values()),
                bool(records),
                oracle_available,
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
    "CLAIM_BOUNDARY",
    "DEFAULT_MANIFEST_PATH",
    "GEOMETRY_FAMILY_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "build_variant_matrix",
    "generate_variant_map",
    "load_geometry_family_manifest",
    "run_geometry_family_preflight",
    "write_preflight_report",
]
