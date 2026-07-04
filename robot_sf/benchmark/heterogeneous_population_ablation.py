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
from typing import Any

from robot_sf.benchmark.heterogeneous_population_metrics import (
    assess_control_trace_readiness,
    per_archetype_metrics_from_control_trace,
)
from robot_sf.ped_npc.ped_archetypes import allocate_archetype_counts, assign_archetype_labels

HETEROGENEOUS_POPULATION_ABLATION_SCHEMA = "heterogeneous_population_ablation_harness.v1"
MEAN_MATCHED_HETEROGENEITY_HARNESS_SCHEMA = "mean_matched_heterogeneity_harness.v1"


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
            },
            "mean_matched_homogeneous": {
                "composition": {homogeneous_archetype: 1.0},
                "counts": {homogeneous_archetype: population_size},
                "records": homogeneous_records,
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

    scenarios = _required_sequence(config, "scenarios")
    planner_rows = _planner_rows(_required_sequence(config, "planners"))
    seed_rows = _seed_rows(_required_sequence(config, "seeds"))
    metric_keys = _metric_keys(config.get("trace_metric_keys", ("clearance_m",)))

    scenario_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    blockers: list[str] = []

    for scenario_index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise ValueError(f"scenarios[{scenario_index}] must be mapping")
        scenario_row, scenario_manifest_rows, scenario_blockers = _manifest_scenario_rows(
            scenario,
            scenario_index=scenario_index,
            planner_rows=planner_rows,
            seed_rows=seed_rows,
            metric_keys=metric_keys,
        )
        scenario_rows.append(scenario_row)
        manifest_rows.extend(scenario_manifest_rows)
        blockers.extend(scenario_blockers)

    status = "ready" if not blockers else "blocked_pending_control_trace"
    return {
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
        "expected_episode_output_keys": [
            "metadata.pedestrian_control_trace",
            "metadata.pedestrian_control_trace.pedestrians[].archetype",
            "metadata.pedestrian_control_trace.pedestrians[].steps[]",
        ],
        "manifest_rows": manifest_rows,
        "row_count": len(manifest_rows),
        "blockers": blockers,
    }


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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    scenario_id = _required_str(scenario, "id", context=f"scenarios[{scenario_index}]")
    density = _required_float(scenario, "density", context=f"scenarios[{scenario_index}]")
    if density < 0.0:
        raise ValueError(f"scenarios[{scenario_index}].density must be >= 0")
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
    composition_hash = _stable_hash(
        {
            "population_size": population_size,
            "composition": pair["arms"]["heterogeneous"]["composition"],
            "mean_matched_parameters": pair["mean_matched_parameters"],
        }
    )
    trace_readiness_by_arm, blockers = _trace_readiness_by_arm(
        scenario,
        scenario_id=scenario_id,
        metric_keys=metric_keys,
    )

    scenario_row = {
        "scenario_id": scenario_id,
        "density": density,
        "population_size": population_size,
        "population_composition_hash": composition_hash,
        "heterogeneous_counts": pair["arms"]["heterogeneous"]["counts"],
        "mean_matched_parameters": pair["mean_matched_parameters"],
        "trace_readiness_by_arm": trace_readiness_by_arm,
    }

    manifest_rows: list[dict[str, Any]] = []
    for planner in planner_rows:
        for seed in seed_rows:
            for arm in ("heterogeneous", "mean_matched_homogeneous"):
                manifest_rows.append(
                    {
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
                            "metadata.pedestrian_control_trace",
                            "metadata.pedestrian_control_trace.pedestrians[].archetype",
                            *[
                                "metadata.pedestrian_control_trace.pedestrians[]."
                                f"steps[].{metric_key}"
                                for metric_key in metric_keys
                            ],
                        ],
                        "trace_readiness": trace_readiness_by_arm[arm],
                        "arm_population": pair["arms"][arm],
                    }
                )

    return scenario_row, manifest_rows, blockers


def _trace_readiness_by_arm(
    scenario: Mapping[str, Any],
    *,
    scenario_id: str,
    metric_keys: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    traces = scenario.get("control_traces")
    if traces is not None and not isinstance(traces, Mapping):
        raise ValueError("scenario.control_traces must be mapping when provided")

    readiness_by_arm: dict[str, Any] = {}
    blockers: list[str] = []
    for arm in ("heterogeneous", "mean_matched_homogeneous"):
        trace = traces.get(arm) if isinstance(traces, Mapping) else None
        if trace is None:
            readiness = {
                "schema_version": HETEROGENEOUS_POPULATION_ABLATION_SCHEMA,
                "source": "pedestrian_control_trace",
                "status": "blocked",
                "ready": False,
                "metric_keys": list(metric_keys),
                "blockers": [
                    "metadata.pedestrian_control_trace missing",
                    *[
                        "metadata.pedestrian_control_trace.pedestrians[]."
                        f"steps[].{metric_key} missing"
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
            if metric_blockers:
                readiness["status"] = "blocked"
                readiness["ready"] = False
                readiness["blockers"] = metric_blockers

        readiness_by_arm[arm] = readiness
        if not readiness["ready"]:
            blockers.extend(f"{scenario_id}/{arm}: {blocker}" for blocker in readiness["blockers"])

    return readiness_by_arm, blockers


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
