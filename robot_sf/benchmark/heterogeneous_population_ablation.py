"""Mean-matched heterogeneous-population ablation harness for issue #3574.

The harness is intentionally pure analysis/export tooling. It constructs paired
population arms that share the same population-mean speed factor and pedestrian
radius while preserving a heterogeneous mixture arm, and it can attach
per-archetype metric breakdowns when per-pedestrian control traces are present.
It does not run benchmark campaigns or promote heterogeneous-population claims.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.heterogeneous_population_metrics import (
    assess_control_trace_readiness,
    per_archetype_metrics_from_control_trace,
)
from robot_sf.benchmark.pedestrian_control_trace import (
    PEDESTRIAN_CONTROL_TRACE_LABELS_KEY,
    build_generated_population_control_trace_labels,
)
from robot_sf.ped_npc.ped_archetypes import allocate_archetype_counts, assign_archetype_labels
from robot_sf.training.scenario_loader import load_scenarios

HETEROGENEOUS_POPULATION_ABLATION_SCHEMA = "heterogeneous_population_ablation_harness.v1"
MEAN_MATCHED_HETEROGENEITY_HARNESS_SCHEMA = "mean_matched_heterogeneity_harness.v1"
MEAN_MATCHED_EPISODE_READINESS_SCHEMA = "mean_matched_episode_readiness.v1"
EPISODE_CONTROL_TRACE_PATH = "algorithm_metadata.pedestrian_control_trace"
RANK_METRIC_KEY = "mean_clearance"


@dataclass(frozen=True, slots=True)
class ArchetypePopulationSpec:
    """Parameter slice used by the mean-matched population generator."""

    desired_speed_factor: float
    radius_m: float
    response_law: str | None = None


def build_mean_matched_population_pair(
    *,
    population_size: int,
    composition: Mapping[str, float],
    archetypes: Mapping[str, ArchetypePopulationSpec | Mapping[str, Any]],
    seed: int | None = None,
    homogeneous_archetype: str = "mean_matched_homogeneous",
) -> dict[str, Any]:
    """Build heterogeneous and mean-matched homogeneous population arms.

    The heterogeneous arm keeps the requested mixture. The homogeneous arm uses a
    single synthetic archetype whose speed factor and radius equal the mixture's
    weighted means, so later paired runs can isolate heterogeneity from a simple
    population-mean shift.

    Returns:
        Versioned JSON-serializable report with paired population records.
    """

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    composition_values = _normalize_composition(composition)
    specs = {name: _coerce_archetype_spec(name, value) for name, value in archetypes.items()}
    _validate_composition_keys(composition_values, specs)

    mean_speed_factor = _weighted_mean(
        composition_values,
        {name: spec.desired_speed_factor for name, spec in specs.items()},
        "desired_speed_factor",
    )
    mean_radius_m = _weighted_mean(
        composition_values,
        {name: spec.radius_m for name, spec in specs.items()},
        "radius_m",
    )

    heterogeneous_labels = assign_archetype_labels(
        population_size,
        dict(composition_values),
        seed=seed,
    )
    heterogeneous_counts = allocate_archetype_counts(population_size, dict(composition_values))
    heterogeneous_records = [
        _population_record(
            simulator_index=index,
            archetype=label,
            spec=specs[str(label)],
        )
        for index, label in enumerate(heterogeneous_labels)
    ]
    homogeneous_spec = ArchetypePopulationSpec(
        desired_speed_factor=mean_speed_factor,
        radius_m=mean_radius_m,
        response_law="mean_matched",
    )
    homogeneous_records = [
        _population_record(
            simulator_index=index,
            archetype=homogeneous_archetype,
            spec=homogeneous_spec,
        )
        for index in range(population_size)
    ]

    return {
        "schema_version": HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
        "status": "analysis_harness_only",
        "claim_boundary": (
            "Constructs paired population inputs and trace-derived summaries only; "
            "does not run a benchmark campaign or establish a heterogeneity claim."
        ),
        "seed": seed,
        "population_size": population_size,
        "mean_matched_parameters": {
            "desired_speed_factor": mean_speed_factor,
            "radius_m": mean_radius_m,
        },
        "arms": {
            "heterogeneous": {
                "composition": dict(sorted(composition_values.items())),
                "counts": dict(sorted(heterogeneous_counts.items())),
                "records": heterogeneous_records,
                PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: build_generated_population_control_trace_labels(
                    heterogeneous_records,
                    source="mean_matched_harness.heterogeneous_population",
                ),
            },
            "mean_matched_homogeneous": {
                "composition": {homogeneous_archetype: 1.0},
                "counts": {homogeneous_archetype: population_size},
                "records": homogeneous_records,
                PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: build_generated_population_control_trace_labels(
                    homogeneous_records,
                    source="mean_matched_harness.mean_matched_homogeneous_population",
                ),
            },
        },
    }


def build_per_archetype_ablation_report(
    *,
    control_traces_by_arm: Mapping[str, Mapping[str, Any] | None],
    metric_key: str,
    higher_is_safer: bool = True,
    cvar_alpha: float = 0.1,
    reducer: str = "mean",
) -> dict[str, Any]:
    """Build per-archetype metric reports for each available ablation arm.

    Missing or incomplete trace payloads are represented as blocked diagnostics
    instead of being treated as successful evidence.

    Returns:
        Versioned per-arm report with ready metrics or fail-closed blockers.
    """

    metric_key = str(metric_key).strip()
    if not metric_key:
        raise ValueError("metric_key must be non-empty")

    arms: dict[str, Any] = {}
    for arm_name in sorted(control_traces_by_arm):
        control_trace = control_traces_by_arm[arm_name]
        if control_trace is None:
            arms[arm_name] = {
                "status": "blocked",
                "ready": False,
                "blockers": ["pedestrian_control_trace missing"],
            }
            continue

        readiness = assess_control_trace_readiness(control_trace, metric_key)
        if not readiness.ready:
            arms[arm_name] = readiness.to_dict()
            continue

        arms[arm_name] = {
            "status": "ready",
            "ready": True,
            "metrics": per_archetype_metrics_from_control_trace(
                control_trace,
                metric_key,
                higher_is_safer=higher_is_safer,
                cvar_alpha=cvar_alpha,
                reducer=reducer,
            ),
        }

    return {
        "schema_version": HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
        "evidence_kind": "trace_metric_breakdown",
        "metric_key": metric_key,
        "higher_is_safer": higher_is_safer,
        "cvar_alpha": cvar_alpha,
        "pedestrian_metric_reducer": reducer,
        "arms": arms,
    }


def _resolve_harness_scenarios(
    config: Mapping[str, Any],
    *,
    config_path: str | None,
) -> tuple[Sequence[Any], dict[str, Any] | None]:
    """Resolve inline cells or derive cells from the canonical scenario matrix.

    Matrix cells intentionally contribute only their scenario identity, map, and
    pedestrian density. The required scenario_matrix_derivation block supplies
    the fixed population size, composition, and archetype parameters used to create
    the paired treatment arms; this prevents a silent harness-specific population
    default from changing the matrix contract.

    Returns:
        Tuple of normalized scenario rows and optional matrix provenance metadata.
    """

    raw_matrix_path = config.get("scenario_matrix")
    if raw_matrix_path is None:
        return _required_sequence(config, "scenarios"), None
    if "scenarios" in config:
        raise ValueError("configure either scenarios or scenario_matrix, not both")

    derivation = _required_mapping(config, "scenario_matrix_derivation")
    matrix_path = _resolve_input_path(
        raw_matrix_path,
        config_path=config_path,
        field_name="scenario_matrix",
    )
    _validate_raw_scenario_matrix(matrix_path)
    matrix_scenarios = load_scenarios(matrix_path)
    if not matrix_scenarios:
        raise ValueError(f"scenario_matrix produced no scenarios: {matrix_path}")

    derived_scenarios: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, matrix_scenario in enumerate(matrix_scenarios):
        if not isinstance(matrix_scenario, Mapping):
            raise ValueError(f"scenario_matrix scenarios[{index}] must be mapping")
        scenario_id = _matrix_scenario_id(matrix_scenario, index=index)
        if scenario_id in seen_ids:
            raise ValueError(f"scenario_matrix contains duplicate scenario id: {scenario_id}")
        seen_ids.add(scenario_id)
        map_file = _resolve_matrix_map_file(matrix_scenario, matrix_path=matrix_path, index=index)
        density = _matrix_scenario_density(matrix_scenario, index=index)
        derived = dict(derivation)
        derived.update(
            {
                "id": scenario_id,
                "density": density,
                "map_file": _portable_path(map_file),
            }
        )
        derived_scenarios.append(derived)

    return derived_scenarios, {
        "scenario_matrix": _portable_path(matrix_path),
        "scenario_matrix_derivation": dict(derivation),
    }


def _validate_raw_scenario_matrix(path: Path) -> None:
    """Reject malformed raw entries before the canonical loader can skip them."""

    _validate_raw_scenario_file(path, visited=set())


def _validate_raw_scenario_file(path: Path, *, visited: set[Path]) -> None:
    """Validate scenario-entry shapes across one matrix file and its includes."""

    resolved = path.resolve()
    if resolved in visited:
        raise ValueError(f"scenario_matrix include cycle detected at: {resolved}")
    visited.add(resolved)
    try:
        scenarios, includes = _load_raw_scenario_file(resolved)
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, Mapping):
                raise ValueError(f"scenario_matrix scenarios[{index}] must be mapping: {resolved}")
        for include in includes:
            if not include.is_file():
                raise ValueError(f"scenario_matrix include does not resolve to a file: {include}")
            _validate_raw_scenario_file(include, visited=visited)
    finally:
        visited.remove(resolved)


def _load_raw_scenario_file(path: Path) -> tuple[list[Any], list[Path]]:
    """Load raw scenario entries and include paths without normalizing entries.

    Returns:
        Raw scenario entries and resolved include paths in configured order.
    """

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"scenario_matrix could not read {path}: {exc}") from exc

    if isinstance(data, Mapping):
        raw_scenarios = data.get("scenarios")
        if raw_scenarios is None:
            scenarios: list[Any] = []
        elif isinstance(raw_scenarios, list):
            scenarios = raw_scenarios
        else:
            raise ValueError(f"scenario_matrix scenarios must be a list: {path}")
        includes = _raw_scenario_include_paths(
            data.get("includes") or data.get("include") or data.get("scenario_files"),
            source=path,
        )
        return scenarios, includes
    if isinstance(data, list):
        return data, []
    raise ValueError(f"scenario_matrix must contain a mapping or list: {path}")


def _raw_scenario_include_paths(raw_includes: Any, *, source: Path) -> list[Path]:
    """Normalize raw include references for structural validation.

    Returns:
        Resolved include paths in their configured order.
    """

    if raw_includes is None:
        return []
    if isinstance(raw_includes, (str, Path)):
        entries = [raw_includes]
    elif isinstance(raw_includes, list):
        entries = raw_includes
    else:
        raise ValueError(f"scenario_matrix includes must be a list or string: {source}")
    paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, (str, Path)):
            raise ValueError(f"scenario_matrix include must be a path: {source}")
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        paths.append(candidate.resolve())
    return paths


def _resolve_input_path(
    raw_path: Any,
    *,
    config_path: str | None,
    field_name: str,
) -> Path:
    """Resolve a config path from the working tree or alongside its config file.

    Returns:
        Existing absolute path for the configured input file.
    """

    if isinstance(raw_path, bool) or not isinstance(raw_path, (str, Path)):
        raise ValueError(f"{field_name} must be a non-empty path")
    path = Path(raw_path).expanduser()
    if not str(path).strip():
        raise ValueError(f"{field_name} must be a non-empty path")
    if path.is_absolute():
        candidate_paths = [path]
    else:
        candidate_paths = [Path.cwd() / path]
        if config_path is not None:
            candidate_paths.append(Path(config_path).expanduser().parent / path)
    for candidate in _unique_paths(candidate_paths):
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate.resolve()) for candidate in _unique_paths(candidate_paths))
    raise ValueError(f"{field_name} does not resolve to a file; searched: {searched}")


def _matrix_scenario_id(scenario: Mapping[str, Any], *, index: int) -> str:
    """Return the stable name/id used to attribute one matrix cell."""

    for key in ("name", "id", "scenario_id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"scenario_matrix scenarios[{index}].name/id must be non-empty")


def _matrix_scenario_density(scenario: Mapping[str, Any], *, index: int) -> float:
    """Read the matrix cell's declared pedestrian density without inventing one.

    Returns:
        The finite non-negative density declared by the matrix cell.
    """

    top_level = scenario.get("ped_density")
    simulation_config = scenario.get("simulation_config")
    nested = (
        simulation_config.get("ped_density") if isinstance(simulation_config, Mapping) else None
    )
    if top_level is not None and nested is not None:
        try:
            if not math.isclose(float(top_level), float(nested), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"scenario_matrix scenarios[{index}] has conflicting ped_density values"
                )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "conflicting" in str(exc):
                raise
            raise ValueError(
                f"scenario_matrix scenarios[{index}].ped_density must be finite numeric"
            ) from exc
    raw_density = nested if nested is not None else top_level
    if raw_density is None:
        raise ValueError(
            f"scenario_matrix scenarios[{index}] must declare simulation_config.ped_density"
        )
    try:
        density = float(raw_density)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"scenario_matrix scenarios[{index}].ped_density must be finite numeric"
        ) from exc
    if not math.isfinite(density) or density < 0.0:
        raise ValueError(f"scenario_matrix scenarios[{index}].ped_density must be finite and >= 0")
    return density


def _resolve_matrix_map_file(
    scenario: Mapping[str, Any],
    *,
    matrix_path: Path,
    index: int,
) -> Path:
    """Resolve and validate a matrix cell's map reference before row creation.

    Returns:
        Existing absolute map path for the matrix cell.
    """

    raw_map_file = scenario.get("map_file")
    if not isinstance(raw_map_file, str) or not raw_map_file.strip():
        raise ValueError(f"scenario_matrix scenarios[{index}].map_file must be non-empty")
    candidate = Path(raw_map_file).expanduser()
    bases = [Path(".")] if candidate.is_absolute() else [matrix_path.parent, _repo_root()]
    candidates = [candidate] if candidate.is_absolute() else [base / candidate for base in bases]
    for path in _unique_paths(candidates):
        if path.is_file():
            return path.resolve()
    searched = ", ".join(str(path.resolve()) for path in _unique_paths(candidates))
    raise ValueError(
        f"scenario_matrix scenarios[{index}].map_file does not resolve to a file; searched: {searched}"
    )


def _repo_root() -> Path:
    """Return the repository root for portable map/matrix paths in generated manifests."""

    return Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    """Prefer repository-relative paths while retaining valid external absolute paths.

    Returns:
        Portable path string for manifest serialization.
    """

    resolved = path.resolve()
    try:
        return resolved.relative_to(_repo_root()).as_posix()
    except ValueError:
        return str(resolved)


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    """Preserve candidate order while removing equivalent path spellings.

    Returns:
        Candidate paths with duplicate spellings removed.
    """

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def build_mean_matched_harness_manifest(
    config: Mapping[str, Any],
    *,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Build a dry-run manifest for paired heterogeneous mean-matched runs.

    The manifest is a pre-run contract. It proves that future rows are attributable by
    scenario, seed, planner, density, and population arm, while keeping missing episode
    control traces explicit until a benchmark run supplies them.

    Returns:
        Versioned manifest payload with paired rows and trace-readiness diagnostics.
    """

    scenarios, scenario_source = _resolve_harness_scenarios(config, config_path=config_path)
    planner_rows = _planner_rows(_required_sequence(config, "planners"))
    seed_rows = _seed_rows(_required_sequence(config, "seeds"))
    metric_keys = _metric_keys(config.get("trace_metric_keys", ("clearance_m",)))
    response_law_fractions = _response_law_fractions(config.get("response_law_fractions"))

    scenario_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    has_pending_capture = False

    for scenario_index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise ValueError(f"scenarios[{scenario_index}] must be mapping")
        scenario_row, scenario_manifest_rows, scenario_blockers = _manifest_scenario_rows(
            scenario,
            scenario_index=scenario_index,
            planner_rows=planner_rows,
            seed_rows=seed_rows,
            metric_keys=metric_keys,
            response_law_fractions=response_law_fractions,
        )
        scenario_rows.append(scenario_row)
        manifest_rows.extend(scenario_manifest_rows)
        blockers.extend(scenario_blockers)
        for arm_readiness in scenario_row.get("trace_readiness_by_arm", {}).values():
            if (
                isinstance(arm_readiness, Mapping)
                and arm_readiness.get("status") == "pending_runtime_capture"
            ):
                has_pending_capture = True

    if not blockers and not has_pending_capture:
        status = "ready"
    elif blockers:
        status = "blocked_pending_control_trace"
    else:
        status = "pending_runtime_capture"
    manifest = {
        "schema_version": MEAN_MATCHED_HETEROGENEITY_HARNESS_SCHEMA,
        "issue": 3574,
        "status": status,
        "claim_boundary": "harness_only_no_ablation_result",
        "config_path": config_path,
        "paired_arms": ["heterogeneous", "mean_matched_homogeneous"],
        "scenario_rows": scenario_rows,
        "planner_rows": planner_rows,
        "seed_rows": seed_rows,
        "trace_metric_keys": metric_keys,
        "response_law_fractions": response_law_fractions,
        "expected_episode_output_keys": [
            f"scenario.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY}",
            EPISODE_CONTROL_TRACE_PATH,
            f"{EPISODE_CONTROL_TRACE_PATH}.pedestrians[].archetype",
            f"{EPISODE_CONTROL_TRACE_PATH}.pedestrians[].steps[]",
            *(
                [f"{EPISODE_CONTROL_TRACE_PATH}.near_field_clearance_threshold_m"]
                if "near_field_exposure_s" in metric_keys
                else []
            ),
        ],
        "manifest_rows": manifest_rows,
        "row_count": len(manifest_rows),
        "blockers": blockers,
    }
    if scenario_source is not None:
        manifest.update(scenario_source)
    return manifest


def assess_mean_matched_episode_records(
    manifest: Mapping[str, Any],
    episode_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate real episode records against the paired manifest before analysis.

    Every expected scenario/planner/seed/arm row must occur exactly once and carry a
    ready per-pedestrian control trace for every declared metric. Missing, duplicate,
    unexpected, or malformed rows remain explicit blockers; they are never silently
    dropped from the future ablation report.

    Returns:
        Versioned integration-readiness report with row-level blockers.
    """

    manifest_rows = _required_sequence(manifest, "manifest_rows")
    metric_keys = _metric_keys(manifest.get("trace_metric_keys"))
    expected_by_key = _rows_by_campaign_key(manifest_rows, source="manifest_rows")
    observed_by_key, blockers = _index_episode_records(episode_records)

    row_readiness: list[dict[str, Any]] = []
    for key in sorted(expected_by_key):
        record = observed_by_key.get(key)
        row_blockers: list[str] = []
        if record is None:
            row_blockers.append("episode record missing")
        else:
            row_blockers.extend(_episode_trace_blockers(record, expected_by_key[key], metric_keys))
            row_blockers.extend(_episode_rank_metric_blockers(record))
        row_readiness.append(
            {
                "scenario_id": key[0],
                "planner": key[1],
                "seed": key[2],
                "population_arm": key[3],
                "status": "ready" if not row_blockers else "blocked",
                "ready": not row_blockers,
                "blockers": row_blockers,
            }
        )
        blockers.extend(f"{_format_campaign_key(key)}: {blocker}" for blocker in row_blockers)

    for key in sorted(set(observed_by_key) - set(expected_by_key)):
        blockers.append(f"unexpected episode record for {_format_campaign_key(key)}")

    manifest_status = str(manifest.get("status", "")).strip()
    if manifest_status == "blocked_pending_control_trace" and not blockers:
        blockers.append(
            "manifest status is blocked_pending_control_trace but episode records "
            "appear complete; the manifest contract was not satisfied"
        )

    return {
        "schema_version": MEAN_MATCHED_EPISODE_READINESS_SCHEMA,
        "issue": 3574,
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "claim_boundary": "integration_readiness_only_no_ablation_result",
        "trace_metric_keys": metric_keys,
        "rank_metric_key": RANK_METRIC_KEY,
        "expected_row_count": len(expected_by_key),
        "observed_row_count": len(observed_by_key),
        "row_readiness": row_readiness,
        "blockers": blockers,
    }


def _index_episode_records(
    episode_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str, int, str], Mapping[str, Any]], list[str]]:
    observed_by_key: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    blockers: list[str] = []
    for index, record in enumerate(episode_records):
        if not isinstance(record, Mapping):
            blockers.append(f"episode_records[{index}] must be mapping")
            continue
        try:
            key = _campaign_row_key(record, context=f"episode_records[{index}]")
        except ValueError as exc:
            blockers.append(str(exc))
            continue
        if key in observed_by_key:
            blockers.append(f"duplicate episode record for {_format_campaign_key(key)}")
            continue
        observed_by_key[key] = record
    return observed_by_key, blockers


def _episode_trace_blockers(
    record: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    metric_keys: Sequence[str],
) -> list[str]:
    algorithm_metadata = record.get("algorithm_metadata")
    if not isinstance(algorithm_metadata, Mapping):
        return ["algorithm_metadata missing or not mapping"]
    trace = algorithm_metadata.get("pedestrian_control_trace")
    if not isinstance(trace, Mapping):
        return [f"{EPISODE_CONTROL_TRACE_PATH} missing or not mapping"]
    blockers = [
        blocker
        for metric_key in metric_keys
        for blocker in assess_control_trace_readiness(trace, metric_key).blockers
    ]
    blockers.extend(_trace_metric_metadata_blockers(trace, metric_keys))
    blockers.extend(_trace_population_metadata_blockers(trace, manifest_row))
    return blockers


def _episode_rank_metric_blockers(record: Mapping[str, Any]) -> list[str]:
    """Require the finite episode metric consumed by rank sensitivity.

    Returns:
        Field-level blockers when the rank metric is absent or non-finite.
    """

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return ["metrics missing or not mapping"]
    value = metrics.get(RANK_METRIC_KEY)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return [f"metrics.{RANK_METRIC_KEY} missing or not finite number"]
    if not math.isfinite(float(value)):
        return [f"metrics.{RANK_METRIC_KEY} missing or not finite number"]
    return []


def audit_smoke_mean_match(
    smoke_report: Mapping[str, Any],
    *,
    metric_key: str = "avg_speed",
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Audit whether an existing smoke report has equal aggregate population means.

    This is a compatibility bridge for the three-seed issue #3206 smoke artifact:
    it records that the artifact is usable as a mean-matched smoke input while
    preserving its existing per-archetype ``not_computable`` limitation.

    Returns:
        Versioned audit payload describing aggregate mean-match readiness.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    conditions = _smoke_conditions(smoke_report)

    arm_metrics: dict[str, float] = {}
    blockers: list[str] = []
    for condition_name in sorted(conditions):
        metric_value, condition_blockers = _smoke_condition_metric(
            conditions[condition_name],
            condition_name=condition_name,
            metric_key=metric_key,
        )
        blockers.extend(condition_blockers)
        if metric_value is not None:
            arm_metrics[condition_name] = metric_value

    if len(arm_metrics) != 2:
        status = "blocked"
        absolute_delta = None
        mean_matched = False
    else:
        first, second = sorted(arm_metrics)
        absolute_delta = abs(arm_metrics[first] - arm_metrics[second])
        mean_matched = absolute_delta <= tolerance
        status = "ready" if mean_matched else "blocked"
        if not mean_matched:
            blockers.append(
                f"{metric_key} means differ by {absolute_delta:.12g}, tolerance {tolerance:.12g}"
            )

    return {
        "schema_version": HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
        "source_schema_version": smoke_report.get("schema_version"),
        "status": status,
        "metric_key": metric_key,
        "tolerance": tolerance,
        "mean_matched": mean_matched,
        "absolute_delta": absolute_delta,
        "arm_means": dict(sorted(arm_metrics.items())),
        "blockers": blockers,
        "claim_boundary": (
            "Smoke audit only. Existing three-seed artifact can check aggregate "
            "mean matching but cannot establish per-archetype or benchmark claims."
        ),
    }


def _smoke_condition_metric(
    condition: Any,
    *,
    condition_name: str,
    metric_key: str,
) -> tuple[float | None, list[str]]:
    """Return aggregate metric mean and local blockers for one smoke arm."""

    if not isinstance(condition, Mapping):
        raise ValueError(f"smoke_report.conditions[{condition_name!r}] must be mapping")
    metrics = condition.get("metrics") if "metrics" in condition else condition
    if not isinstance(metrics, Mapping):
        raise ValueError(f"smoke_report.conditions[{condition_name!r}].metrics must be mapping")

    blockers: list[str] = []
    metric_payload = metrics.get(metric_key)
    if not isinstance(metric_payload, Mapping) or "mean" not in metric_payload:
        blockers.append(f"{condition_name}: missing metrics.{metric_key}.mean")
        return None, blockers

    value = float(metric_payload["mean"])
    if not math.isfinite(value):
        raise ValueError(f"{condition_name}: metrics.{metric_key}.mean must be finite")

    distributional_status = condition.get("distributional_disruption", {})
    if isinstance(distributional_status, Mapping):
        status = str(distributional_status.get("status", "")).strip()
        if status and status != "ready":
            blockers.append(f"{condition_name}: per-archetype breakdown {status}")
    return value, blockers


def _smoke_conditions(smoke_report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return two condition payloads from detailed or aggregate smoke artifacts."""

    conditions = smoke_report.get("conditions")
    if isinstance(conditions, Mapping):
        if len(conditions) != 2:
            raise ValueError("smoke_report.conditions must contain exactly two arms")
        return conditions

    condition_payloads = {
        name: value
        for name, value in smoke_report.items()
        if isinstance(value, Mapping)
        and name not in {"delta_mixed_minus_homogeneous", "metadata", "config", "environment"}
    }
    if len(condition_payloads) != 2:
        raise ValueError("smoke_report must contain exactly two condition arms")
    return condition_payloads


def _manifest_scenario_rows(
    scenario: Mapping[str, Any],
    *,
    scenario_index: int,
    planner_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    metric_keys: Sequence[str],
    response_law_fractions: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    scenario_id = _required_str(scenario, "id", context=f"scenarios[{scenario_index}]")
    density = _required_float(scenario, "density", context=f"scenarios[{scenario_index}]")
    if density < 0.0:
        raise ValueError(f"scenarios[{scenario_index}].density must be >= 0")
    map_file = _optional_map_file(scenario, scenario_index=scenario_index)
    population_size = _required_int(
        scenario, "population_size", context=f"scenarios[{scenario_index}]"
    )
    composition = _normalize_composition(_required_mapping(scenario, "composition"))
    archetypes = _required_mapping(scenario, "archetypes")
    pair = build_mean_matched_population_pair(
        population_size=population_size,
        composition=composition,
        archetypes=archetypes,
        seed=scenario.get("archetype_seed"),
    )
    base_composition_hash = _stable_hash(
        {
            "population_size": population_size,
            "composition": pair["arms"]["heterogeneous"]["composition"],
            "mean_matched_parameters": pair["mean_matched_parameters"],
        }
    )
    trace_readiness_by_arm, blockers, _has_pending = _trace_readiness_by_arm(
        scenario,
        scenario_id=scenario_id,
        metric_keys=metric_keys,
        arm_populations=pair["arms"],
    )

    scenario_row = {
        "scenario_id": scenario_id,
        "density": density,
        "population_size": population_size,
        "population_composition_hash": base_composition_hash,
        "response_law_fractions": list(response_law_fractions),
        "heterogeneous_counts": pair["arms"]["heterogeneous"]["counts"],
        "mean_matched_parameters": pair["mean_matched_parameters"],
        "trace_readiness_by_arm": trace_readiness_by_arm,
    }
    if map_file is not None:
        scenario_row["map_file"] = map_file

    manifest_rows: list[dict[str, Any]] = []
    fractions: Sequence[float | None] = response_law_fractions or (None,)
    response_law_seed = scenario.get("response_law_seed", scenario.get("archetype_seed"))
    for response_law_fraction in fractions:
        response_pair = (
            pair
            if response_law_fraction is None
            else _with_response_law_fraction(
                pair,
                fraction=response_law_fraction,
                seed=response_law_seed,
            )
        )
        composition_hash = (
            base_composition_hash
            if response_law_fraction is None
            else _stable_hash(
                {
                    "base_population_composition_hash": base_composition_hash,
                    "response_law_fraction": response_law_fraction,
                }
            )
        )
        for planner in planner_rows:
            for seed in seed_rows:
                for arm in ("heterogeneous", "mean_matched_homogeneous"):
                    row = {
                        "scenario_id": scenario_id,
                        "planner": planner["key"],
                        "seed": seed["seed"],
                        "density": density,
                        "population_arm": arm,
                        "paired_arm": (
                            "mean_matched_homogeneous"
                            if arm == "heterogeneous"
                            else "heterogeneous"
                        ),
                        "population_composition_hash": composition_hash,
                        "expected_episode_output_keys": [
                            f"scenario.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY}",
                            EPISODE_CONTROL_TRACE_PATH,
                            f"{EPISODE_CONTROL_TRACE_PATH}.pedestrians[].archetype",
                            *(
                                [f"{EPISODE_CONTROL_TRACE_PATH}.near_field_clearance_threshold_m"]
                                if "near_field_exposure_s" in metric_keys
                                else []
                            ),
                            *[
                                f"{EPISODE_CONTROL_TRACE_PATH}.pedestrians[].steps[].{metric_key}"
                                for metric_key in metric_keys
                            ],
                        ],
                        "trace_readiness": trace_readiness_by_arm[arm],
                        "arm_population": response_pair["arms"][arm],
                    }
                    if map_file is not None:
                        row["map_file"] = map_file
                    if response_law_fraction is not None:
                        row["response_law_fraction"] = response_law_fraction
                    manifest_rows.append(row)

    return scenario_row, manifest_rows, blockers


def _optional_map_file(scenario: Mapping[str, Any], *, scenario_index: int) -> str | None:
    """Return an explicitly configured map without changing legacy row payloads."""

    if "map_file" not in scenario:
        return None
    value = scenario.get("map_file")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"scenarios[{scenario_index}].map_file must be non-empty")
    return value.strip()


def _arm_has_control_trace_labels(
    arm_populations: Mapping[str, Mapping[str, Any]] | None,
    arm: str,
) -> bool:
    """Return True when an arm's population carries trace labels for runtime capture."""
    if arm_populations is None:
        return False
    population = arm_populations.get(arm)
    if not isinstance(population, Mapping):
        return False
    labels = population.get(PEDESTRIAN_CONTROL_TRACE_LABELS_KEY)
    return isinstance(labels, Sequence) and not isinstance(labels, str) and len(labels) > 0


def _trace_readiness_by_arm(
    scenario: Mapping[str, Any],
    *,
    scenario_id: str,
    metric_keys: Sequence[str],
    arm_populations: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    traces = scenario.get("control_traces")
    if traces is not None and not isinstance(traces, Mapping):
        raise ValueError("scenario.control_traces must be mapping when provided")

    readiness_by_arm: dict[str, Any] = {}
    blockers: list[str] = []
    has_pending = False
    for arm in ("heterogeneous", "mean_matched_homogeneous"):
        trace = traces.get(arm) if isinstance(traces, Mapping) else None
        if trace is None:
            has_labels = _arm_has_control_trace_labels(arm_populations, arm)
            if has_labels:
                has_pending = True
                readiness = {
                    "schema_version": HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
                    "source": "pedestrian_control_trace",
                    "status": "pending_runtime_capture",
                    "ready": False,
                    "metric_keys": list(metric_keys),
                    "blockers": [
                        f"{EPISODE_CONTROL_TRACE_PATH} pending runtime capture "
                        "(pedestrian_control_trace_labels present in arm population)",
                    ],
                }
            else:
                readiness = {
                    "schema_version": HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
                    "source": "pedestrian_control_trace",
                    "status": "blocked",
                    "ready": False,
                    "metric_keys": list(metric_keys),
                    "blockers": [
                        f"{EPISODE_CONTROL_TRACE_PATH} missing",
                        *[
                            f"{EPISODE_CONTROL_TRACE_PATH}.pedestrians[].steps[].{metric_key} missing"
                            for metric_key in metric_keys
                        ],
                    ],
                }
        else:
            readiness = {
                "schema_version": HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
                "source": "pedestrian_control_trace",
                "status": "ready",
                "ready": True,
                "metric_keys": list(metric_keys),
                "metrics": {
                    metric_key: assess_control_trace_readiness(trace, metric_key).to_dict()
                    for metric_key in metric_keys
                },
            }
            metric_blockers = [
                blocker
                for metric_readiness in readiness["metrics"].values()
                for blocker in metric_readiness["blockers"]
            ]
            metric_blockers.extend(_trace_metric_metadata_blockers(trace, metric_keys))
            if metric_blockers:
                readiness["status"] = "blocked"
                readiness["ready"] = False
                readiness["blockers"] = metric_blockers

        readiness_by_arm[arm] = readiness
        if not readiness["ready"] and readiness["status"] == "blocked":
            blockers.extend(f"{scenario_id}/{arm}: {blocker}" for blocker in readiness["blockers"])

    return readiness_by_arm, blockers, has_pending


def _trace_metric_metadata_blockers(
    trace: Mapping[str, Any], metric_keys: Sequence[str]
) -> list[str]:
    """Return provenance blockers required by non-self-describing trace metrics."""

    if "near_field_exposure_s" not in metric_keys:
        return []
    threshold = trace.get("near_field_clearance_threshold_m")
    if threshold is None:
        return ["control_trace.near_field_clearance_threshold_m missing"]
    if isinstance(threshold, bool | str):
        return ["control_trace.near_field_clearance_threshold_m must be finite non-negative number"]
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return ["control_trace.near_field_clearance_threshold_m must be finite non-negative number"]
    if not math.isfinite(threshold_value) or threshold_value < 0.0:
        return ["control_trace.near_field_clearance_threshold_m must be finite non-negative number"]
    return []


def _trace_population_metadata_blockers(
    trace: Mapping[str, Any], manifest_row: Mapping[str, Any]
) -> list[str]:
    """Check simulator-indexed trace labels against the manifest population arm.

    Returns:
        Field-level blockers for missing or mismatched population metadata.
    """

    arm_population = manifest_row.get("arm_population")
    if not isinstance(arm_population, Mapping):
        return ["manifest row arm_population missing or not mapping"]
    expected_labels = arm_population.get(PEDESTRIAN_CONTROL_TRACE_LABELS_KEY)
    if not isinstance(expected_labels, Sequence) or isinstance(expected_labels, str):
        return [f"manifest row arm_population.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY} missing"]
    pedestrians = trace.get("pedestrians")
    if not isinstance(pedestrians, Sequence) or isinstance(pedestrians, str):
        return ["control_trace.pedestrians must be a sequence"]
    blockers: list[str] = []
    if len(pedestrians) != len(expected_labels):
        blockers.append(
            "control_trace pedestrian count does not match manifest arm population: "
            f"{len(pedestrians)} != {len(expected_labels)}"
        )
        return blockers

    for index, (pedestrian, expected_label) in enumerate(
        zip(pedestrians, expected_labels, strict=True)
    ):
        blockers.extend(_trace_label_blockers(pedestrian, expected_label, index=index))
    return blockers


def _trace_label_blockers(pedestrian: Any, expected_label: Any, *, index: int) -> list[str]:
    if not isinstance(pedestrian, Mapping) or not isinstance(expected_label, Mapping):
        return [f"control_trace label alignment at index {index} must use mappings"]
    blockers: list[str] = []
    for key in ("simulator_index", "archetype", "desired_speed_factor", "response_law"):
        if key not in expected_label:
            continue
        if key not in pedestrian:
            blockers.append(f"control_trace.pedestrians[{index}].{key} missing")
        elif pedestrian[key] != expected_label[key]:
            blockers.append(
                f"control_trace.pedestrians[{index}].{key} does not match manifest label"
            )
    return blockers


def _planner_rows(planners: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, planner in enumerate(planners):
        if isinstance(planner, str):
            key = planner.strip()
            payload: dict[str, Any] = {"key": key}
        elif isinstance(planner, Mapping):
            key = _required_str(planner, "key", context=f"planners[{index}]")
            payload = {"key": key}
            if planner.get("algo") is not None:
                payload["algo"] = str(planner["algo"])
        else:
            raise ValueError(f"planners[{index}] must be string or mapping")
        if not key:
            raise ValueError(f"planners[{index}].key must be non-empty")
        rows.append(payload)
    return rows


def _seed_rows(seeds: Sequence[Any]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for index, seed in enumerate(seeds):
        if isinstance(seed, bool):
            raise ValueError(f"seeds[{index}] must be integer")
        try:
            rows.append({"seed": int(seed)})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"seeds[{index}] must be integer") from exc
    return rows


def _metric_keys(raw_metric_keys: Any) -> list[str]:
    if not isinstance(raw_metric_keys, Sequence) or isinstance(raw_metric_keys, str):
        raise ValueError("trace_metric_keys must be sequence")
    metric_keys = [str(metric_key).strip() for metric_key in raw_metric_keys]
    if not metric_keys or any(not metric_key for metric_key in metric_keys):
        raise ValueError("trace_metric_keys must contain non-empty metric keys")
    return metric_keys


def _response_law_fractions(raw_fractions: Any) -> list[float]:
    """Validate optional non-reactive fractions for the fixed-density sweep.

    Returns:
        Ordered, finite response-law fractions, or an empty list for legacy configs.
    """

    if raw_fractions is None:
        return []
    if not isinstance(raw_fractions, Sequence) or isinstance(raw_fractions, str):
        raise ValueError("response_law_fractions must be a non-empty sequence")
    fractions: list[float] = []
    for index, raw_fraction in enumerate(raw_fractions):
        if isinstance(raw_fraction, bool):
            raise ValueError(f"response_law_fractions[{index}] must be a finite number in [0, 1]")
        try:
            fraction = float(raw_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"response_law_fractions[{index}] must be a finite number in [0, 1]"
            ) from exc
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"response_law_fractions[{index}] must be a finite number in [0, 1]")
        if fraction in fractions:
            raise ValueError("response_law_fractions must not contain duplicates")
        fractions.append(fraction)
    if not fractions:
        raise ValueError("response_law_fractions must be a non-empty sequence")
    return fractions


def _with_response_law_fraction(
    pair: Mapping[str, Any], *, fraction: float, seed: int | None
) -> dict[str, Any]:
    """Attach deterministic reactive/non-reactive labels to both paired arms.

    Returns:
        A copy of the paired population payload with response-law composition and labels.
    """

    response_law_composition = {}
    if fraction < 1.0:
        response_law_composition["reactive"] = 1.0 - fraction
    if fraction > 0.0:
        response_law_composition["non_reactive"] = fraction

    arms: dict[str, dict[str, Any]] = {}
    for arm_name, arm in pair["arms"].items():
        records = arm["records"]
        response_laws = assign_archetype_labels(len(records), response_law_composition, seed=seed)
        labeled_records = [
            {**record, "response_law": str(response_law)}
            for record, response_law in zip(records, response_laws, strict=True)
        ]
        arms[arm_name] = {
            **arm,
            "records": labeled_records,
            "response_law_composition": dict(response_law_composition),
            "response_law_seed": seed,
            PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: build_generated_population_control_trace_labels(
                labeled_records,
                source=f"mean_matched_harness.{arm_name}.response_law_sweep",
            ),
        }
    return {**pair, "arms": arms}


def _campaign_row_key(row: Mapping[str, Any], *, context: str) -> tuple[str, str, int, str, float]:
    scenario_id = _required_str(row, "scenario_id", context=context)
    planner = _required_str(row, "planner", context=context)
    seed = _required_int(row, "seed", context=context)
    population_arm = _required_str(row, "population_arm", context=context)
    response_law_fraction = _response_law_fraction_from_row(row, context=context)
    return scenario_id, planner, seed, population_arm, response_law_fraction


def _rows_by_campaign_key(
    rows: Sequence[Any], *, source: str
) -> dict[tuple[str, str, int, str, float], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, int, str, float], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}[{index}] must be mapping")
        key = _campaign_row_key(row, context=f"{source}[{index}]")
        if key in indexed:
            raise ValueError(f"{source} contains duplicate {_format_campaign_key(key)}")
        indexed[key] = row
    return indexed


def _format_campaign_key(key: tuple[str, str, int, str, float]) -> str:
    scenario_id, planner, seed, population_arm, response_law_fraction = key
    return (
        f"{scenario_id}/{planner}/seed_{seed}/{population_arm}/"
        f"response_law_fraction_{response_law_fraction:g}"
    )


def _response_law_fraction_from_row(row: Mapping[str, Any], *, context: str) -> float:
    """Read the sweep dimension, retaining ``0.0`` for legacy paired manifests.

    Returns:
        A finite response-law fraction in the inclusive interval ``[0, 1]``.
    """

    value = row.get("response_law_fraction")
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise ValueError(f"{context}.response_law_fraction must be a finite number in [0, 1]")
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}.response_law_fraction must be a finite number in [0, 1]"
        ) from exc
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{context}.response_law_fraction must be a finite number in [0, 1]")
    return fraction


def _required_sequence(config: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = config.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty sequence")
    return value


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{key} must be non-empty mapping")
    return value


def _required_str(config: Mapping[str, Any], key: str, *, context: str) -> str:
    value = config.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{context}.{key} must be non-empty")
    return text


def _required_float(config: Mapping[str, Any], key: str, *, context: str) -> float:
    try:
        value = float(config[key])
    except KeyError as exc:
        raise ValueError(f"{context}.{key} missing") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context}.{key} must be finite")
    return value


def _required_int(config: Mapping[str, Any], key: str, *, context: str) -> int:
    if isinstance(config.get(key), bool):
        raise ValueError(f"{context}.{key} must be integer")
    try:
        value = int(config[key])
    except KeyError as exc:
        raise ValueError(f"{context}.{key} missing") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{key} must be integer") from exc
    if value <= 0:
        raise ValueError(f"{context}.{key} must be positive")
    return value


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _normalize_composition(composition: Mapping[str, float]) -> dict[str, float]:
    if not composition:
        raise ValueError("composition must be non-empty")
    normalized = {str(name).strip(): float(value) for name, value in composition.items()}
    for name, value in normalized.items():
        if not name:
            raise ValueError("composition archetype names must be non-empty")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"composition fraction {name!r} must be finite > 0")
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"composition fractions must sum to 1.0 (got {total})")
    return {name: value / total for name, value in normalized.items()}


def _coerce_archetype_spec(
    name: str,
    value: ArchetypePopulationSpec | Mapping[str, Any],
) -> ArchetypePopulationSpec:
    if isinstance(value, ArchetypePopulationSpec):
        spec = value
    elif isinstance(value, Mapping):
        try:
            desired_speed_factor = float(value["desired_speed_factor"])
            radius_m = float(value["radius_m"])
        except KeyError as exc:
            raise ValueError(f"archetype spec {name!r} missing key: {exc.args[0]}") from exc
        spec = ArchetypePopulationSpec(
            desired_speed_factor=desired_speed_factor,
            radius_m=radius_m,
            response_law=(
                None if value.get("response_law") is None else str(value.get("response_law"))
            ),
        )
    else:
        raise TypeError(f"archetype spec {name!r} must be ArchetypePopulationSpec or mapping")

    if not math.isfinite(spec.desired_speed_factor) or spec.desired_speed_factor <= 0.0:
        raise ValueError(f"archetype {name!r} desired_speed_factor must be finite > 0")
    if not math.isfinite(spec.radius_m) or spec.radius_m <= 0.0:
        raise ValueError(f"archetype {name!r} radius_m must be finite > 0")
    return spec


def _validate_composition_keys(
    composition: Mapping[str, float],
    specs: Mapping[str, ArchetypePopulationSpec],
) -> None:
    missing = sorted(name for name in composition if name not in specs)
    if missing:
        raise ValueError(f"composition references unknown archetypes: {missing}")


def _weighted_mean(
    composition: Mapping[str, float],
    values: Mapping[str, float],
    parameter_name: str,
) -> float:
    weighted_value = sum(composition[name] * values[name] for name in composition)
    if not math.isfinite(weighted_value):
        raise ValueError(f"weighted {parameter_name} mean must be finite")
    return float(weighted_value)


def _population_record(
    *,
    simulator_index: int,
    archetype: str,
    spec: ArchetypePopulationSpec,
) -> dict[str, Any]:
    return {
        "simulator_index": simulator_index,
        "archetype": str(archetype),
        "desired_speed_factor": float(spec.desired_speed_factor),
        "radius_m": float(spec.radius_m),
        "response_law": spec.response_law,
    }


__all__ = [
    "HETEROGENEOUS_POPULATION_ABLATION_SCHEMA",
    "MEAN_MATCHED_HETEROGENEITY_HARNESS_SCHEMA",
    "ArchetypePopulationSpec",
    "audit_smoke_mean_match",
    "build_mean_matched_harness_manifest",
    "build_mean_matched_population_pair",
    "build_per_archetype_ablation_report",
]
