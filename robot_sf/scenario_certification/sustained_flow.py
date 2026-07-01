"""Preflight checks for sustained-flow scenario scaffold variants.

Issue #3813 introduces metadata-only continuous-spawn scenario variants. These
checks intentionally do not claim runtime continuous respawn exists; they prove
the opt-in scaffold is complete, enumerable, and fail-closed for benchmark use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.scenario_loader import load_scenarios

SUSTAINED_FLOW_PREFLIGHT_SCHEMA_VERSION = "issue_3813.sustained_flow_preflight.v1"
DEFAULT_SUSTAINED_FLOW_SCENARIO_SET = (
    Path(__file__).resolve().parents[2]
    / "configs/scenarios/sets/issue_3813_sustained_flow_scaffold_v0.yaml"
)
EXPECTED_SUSTAINED_FLOW_TIERS: tuple[tuple[str, float, float, tuple[int, ...]], ...] = (
    ("light", 0.02, 6.0, (381301, 381302, 381303)),
    ("medium", 0.05, 12.0, (381311, 381312, 381313)),
    ("heavy", 0.08, 18.0, (381321, 381322, 381323)),
)
REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE: tuple[str, ...] = (
    "continuous pedestrian respawn runtime support",
    "progress-rate metric implementation",
    "interaction-exposure diagnostic sanity check",
    "scenario_cert.v1 eligibility review",
)
EXPECTED_CONTINUOUS_SPAWN_DEFINITION: dict[str, object] = {
    "demand_model": "non_clearing_poisson_flow",
    "respawn_trigger": "pedestrian_exits_route_corridor",
    "spawn_budget": "time_bounded_episode",
    "minimum_active_pedestrians": 1,
    "clearing_policy": "disallow_empty_scene_wait_success",
}
_SUSTAINED_FLOW_FAMILY = "sustained_flow_t_intersection"
_SUSTAINED_FLOW_MAP_FILE = "../../../maps/svg_maps/classic_t_intersection.svg"
_SUSTAINED_FLOW_MAX_EPISODE_STEPS = 600
SUSTAINED_FLOW_METADATA_ONLY_RUNTIME_SUPPORT = "metadata_only"
SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE = "runtime_continuous_spawn"


@dataclass(frozen=True)
class SustainedFlowVariantSpec:
    """Generated expected sustained-flow scenario variant definition."""

    name: str
    density_tier: str
    ped_density: float
    spawn_rate_per_min: float
    seeds: tuple[int, ...]
    map_file: str
    max_episode_steps: int


def iter_expected_sustained_flow_variant_specs() -> tuple[SustainedFlowVariantSpec, ...]:
    """Generate the expected issue #3813 sustained-flow variant definitions.

    This is a deterministic preflight surface, not a runtime continuous-spawn
    implementation or benchmark-evidence generator.

    Returns:
        Expected sustained-flow scaffold variants in deterministic matrix order.
    """

    return tuple(
        SustainedFlowVariantSpec(
            name=f"issue_3813_{_SUSTAINED_FLOW_FAMILY}_{tier}",
            density_tier=tier,
            ped_density=ped_density,
            spawn_rate_per_min=spawn_rate_per_min,
            seeds=seeds,
            map_file=_SUSTAINED_FLOW_MAP_FILE,
            max_episode_steps=_SUSTAINED_FLOW_MAX_EPISODE_STEPS,
        )
        for tier, ped_density, spawn_rate_per_min, seeds in EXPECTED_SUSTAINED_FLOW_TIERS
    )


def generate_expected_sustained_flow_scenarios(
    *,
    current_runtime_support: str = SUSTAINED_FLOW_METADATA_ONLY_RUNTIME_SUPPORT,
) -> list[dict[str, Any]]:
    """Generate full scenario rows for the sustained-flow scaffold.

    This helper deliberately materializes only the pre-benchmark scenario-matrix
    rows. It does not implement continuous pedestrian respawn or promote the
    family to benchmark evidence.

    Returns:
        Scenario-matrix rows in deterministic light, medium, heavy order.
    """

    scenarios: list[dict[str, Any]] = []
    for spec in iter_expected_sustained_flow_variant_specs():
        scenarios.append(
            {
                "name": spec.name,
                "map_file": spec.map_file,
                "simulation_config": {
                    "max_episode_steps": spec.max_episode_steps,
                    "ped_density": spec.ped_density,
                    "robot_config": {},
                },
                "metadata": {
                    "archetype": _SUSTAINED_FLOW_FAMILY,
                    "density": spec.density_tier,
                    "flow": "continuous_crossing",
                    "pack_id": "issue_3813_sustained_flow_scaffold_v0",
                    "status": "pre_benchmark_scaffold",
                    "enabled_by_default": False,
                    "benchmark_evidence": False,
                    "claim_boundary": (
                        "Opt-in scenario-matrix scaffold only. Do not cite as sustained-flow "
                        "benchmark evidence until runtime continuous spawn, progress metrics, "
                        "and interaction-exposure diagnostics are implemented and validated."
                    ),
                    "continuous_spawn": {
                        "required_before_benchmark_use": True,
                        "intended_process": "poisson_respawn",
                        "definition": dict(EXPECTED_CONTINUOUS_SPAWN_DEFINITION),
                        "spawn_rate_per_min": spec.spawn_rate_per_min,
                        "target_density_tier": spec.density_tier,
                        "current_runtime_support": current_runtime_support,
                    },
                    "success_metric": {
                        "id": "sustained_progress_rate_m_per_s",
                        "definition": (
                            "Net robot path progress toward the route goal divided by simulated "
                            "episode seconds under non-clearing pedestrian demand."
                        ),
                        "wait_policy_expectation": "zero_or_near_zero_progress",
                    },
                    "termination": {
                        "mode": "time_bounded",
                        "max_episode_steps": spec.max_episode_steps,
                        "goal_reach_is_not_primary_success": True,
                    },
                    "requires_before_benchmark_use": list(REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE),
                },
                "seeds": list(spec.seeds),
            }
        )
    return scenarios


def generate_runtime_supported_sustained_flow_scenarios() -> list[dict[str, Any]]:
    """Generate sustained-flow rows that advertise continuous-spawn runtime support.

    This is preflight/checker input, not benchmark evidence. It separates
    generator drift checks from the later runner proof that continuous pedestrian
    respawn is actually implemented.

    Returns:
        Scenario-matrix rows in deterministic light, medium, heavy order.
    """

    return generate_expected_sustained_flow_scenarios(
        current_runtime_support=SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE
    )


@dataclass(frozen=True)
class SustainedFlowVariant:
    """Validated sustained-flow scaffold variant summary."""

    name: str
    density_tier: str
    ped_density: float
    spawn_rate_per_min: float
    spawn_definition: dict[str, object]
    seeds: tuple[int, ...]
    map_file: str


@dataclass(frozen=True)
class SustainedFlowPreflightReport:
    """Structured result for the issue #3813 scenario scaffold preflight."""

    schema_version: str
    scenario_set: str
    conforms: bool
    variants: tuple[SustainedFlowVariant, ...]
    errors: tuple[str, ...]
    benchmark_evidence: bool = False
    runtime_support: str = "metadata_only"


def _repo_relative(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_raw_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Scenario set must be a mapping: {path}")
    return data


def _require_equal(errors: list[str], name: str, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{name}: {field} must be {expected}")


def _check_basic_metadata(
    metadata: dict[str, Any],
    *,
    name: str,
    expected_tier: str,
) -> list[str]:
    errors: list[str] = []
    _require_equal(
        errors,
        name,
        "metadata.pack_id",
        metadata.get("pack_id"),
        "issue_3813_sustained_flow_scaffold_v0",
    )
    _require_equal(
        errors,
        name,
        "metadata.status",
        metadata.get("status"),
        "pre_benchmark_scaffold",
    )
    _require_equal(
        errors, name, "metadata.enabled_by_default", metadata.get("enabled_by_default"), False
    )
    _require_equal(
        errors, name, "metadata.benchmark_evidence", metadata.get("benchmark_evidence"), False
    )
    _require_equal(errors, name, "metadata.density", metadata.get("density"), expected_tier)
    return errors


def _check_continuous_spawn(
    metadata: dict[str, Any],
    *,
    name: str,
    expected_tier: str,
    expected_spawn_rate: float,
) -> list[str]:
    continuous_spawn = metadata.get("continuous_spawn")
    if not isinstance(continuous_spawn, dict):
        return [f"{name}: metadata.continuous_spawn block is required"]

    errors: list[str] = []
    _require_equal(
        errors,
        name,
        "continuous_spawn.required_before_benchmark_use",
        continuous_spawn.get("required_before_benchmark_use"),
        True,
    )
    _require_equal(
        errors,
        name,
        "continuous_spawn.intended_process",
        continuous_spawn.get("intended_process"),
        "poisson_respawn",
    )
    _require_equal(
        errors,
        name,
        "continuous_spawn.definition",
        continuous_spawn.get("definition"),
        EXPECTED_CONTINUOUS_SPAWN_DEFINITION,
    )
    _require_equal(
        errors,
        name,
        "continuous_spawn.current_runtime_support",
        continuous_spawn.get("current_runtime_support"),
        "metadata_only",
    )
    _require_equal(
        errors,
        name,
        "continuous_spawn.target_density_tier",
        continuous_spawn.get("target_density_tier"),
        expected_tier,
    )
    _require_equal(
        errors,
        name,
        "continuous_spawn.spawn_rate_per_min",
        float(continuous_spawn.get("spawn_rate_per_min", -1.0)),
        expected_spawn_rate,
    )
    return errors


def _check_termination_and_metric(
    scenario: dict[str, Any],
    metadata: dict[str, Any],
    *,
    name: str,
) -> list[str]:
    errors: list[str] = []
    termination = metadata.get("termination")
    if not isinstance(termination, dict):
        errors.append(f"{name}: metadata.termination block is required")
    else:
        _require_equal(errors, name, "termination.mode", termination.get("mode"), "time_bounded")
        _require_equal(
            errors,
            name,
            "termination.goal_reach_is_not_primary_success",
            termination.get("goal_reach_is_not_primary_success"),
            True,
        )
        sim_steps = scenario.get("simulation_config", {}).get("max_episode_steps")
        _require_equal(
            errors,
            name,
            "termination.max_episode_steps",
            termination.get("max_episode_steps"),
            sim_steps,
        )

    success_metric = metadata.get("success_metric")
    if not isinstance(success_metric, dict):
        errors.append(f"{name}: metadata.success_metric block is required")
    else:
        _require_equal(
            errors,
            name,
            "success_metric.id",
            success_metric.get("id"),
            "sustained_progress_rate_m_per_s",
        )

    blockers = tuple(metadata.get("requires_before_benchmark_use", ()))
    if blockers != REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE:
        errors.append(f"{name}: requires_before_benchmark_use must match the fail-closed list")
    return errors


def _check_metadata_fail_closed(
    scenario: dict[str, Any],
    *,
    expected_tier: str,
    expected_spawn_rate: float,
) -> list[str]:
    name = str(scenario.get("name", "<unnamed>"))
    metadata = scenario.get("metadata")
    if not isinstance(metadata, dict):
        return [f"{name}: metadata block is required"]

    errors: list[str] = []
    errors.extend(_check_basic_metadata(metadata, name=name, expected_tier=expected_tier))
    errors.extend(
        _check_continuous_spawn(
            metadata,
            name=name,
            expected_tier=expected_tier,
            expected_spawn_rate=expected_spawn_rate,
        )
    )
    errors.extend(_check_termination_and_metric(scenario, metadata, name=name))
    return errors


def _validate_variant(
    scenario: dict[str, Any],
    *,
    scenario_set: Path,
    expected_by_tier: dict[str, tuple[float, float, tuple[int, ...]]],
    seen_seeds: set[int],
) -> tuple[SustainedFlowVariant | None, list[str]]:
    """Validate one scenario variant against the sustained-flow scaffold contract.

    Returns:
        Optional variant summary and any validation errors for the scenario.
    """

    errors: list[str] = []
    name = str(scenario.get("name", "<unnamed>"))
    metadata = scenario.get("metadata", {})
    tier = str(metadata.get("density", ""))
    if tier not in expected_by_tier:
        return None, [f"{name}: unexpected sustained-flow density tier {tier!r}"]

    expected_ped_density, expected_spawn_rate, expected_seeds = expected_by_tier[tier]
    simulation_config = scenario.get("simulation_config", {})
    ped_density = float(simulation_config.get("ped_density", -1.0))
    seeds = tuple(scenario.get("seeds", ()))
    map_file = str(scenario.get("map_file", ""))
    map_path = (scenario_set.parent / map_file).resolve()

    if not name.startswith(f"issue_3813_sustained_flow_t_intersection_{tier}"):
        errors.append(f"{name}: scenario name must include issue, family, and tier")
    _require_equal(errors, name, "ped_density", ped_density, expected_ped_density)
    _require_equal(errors, name, "seeds", seeds, expected_seeds)
    for seed in seeds:
        if seed in seen_seeds:
            errors.append(f"{name}: duplicate seed {seed}")
        seen_seeds.add(seed)
    if not map_path.exists():
        errors.append(f"{name}: map_file does not resolve to an existing file: {map_file}")

    errors.extend(
        _check_metadata_fail_closed(
            scenario,
            expected_tier=tier,
            expected_spawn_rate=expected_spawn_rate,
        )
    )
    return (
        SustainedFlowVariant(
            name=name,
            density_tier=tier,
            ped_density=ped_density,
            spawn_rate_per_min=expected_spawn_rate,
            spawn_definition=dict(EXPECTED_CONTINUOUS_SPAWN_DEFINITION),
            seeds=seeds,
            map_file=map_file,
        ),
        errors,
    )


def preflight_sustained_flow_scenario_set(
    path: str | Path = DEFAULT_SUSTAINED_FLOW_SCENARIO_SET,
) -> SustainedFlowPreflightReport:
    """Validate and enumerate the issue #3813 sustained-flow scaffold.

    Returns:
        Sustained-flow preflight report with variants and fail-closed errors.
    """

    scenario_set = Path(path).resolve()
    errors: list[str] = []
    if not scenario_set.exists():
        return SustainedFlowPreflightReport(
            schema_version=SUSTAINED_FLOW_PREFLIGHT_SCHEMA_VERSION,
            scenario_set=_repo_relative(scenario_set),
            conforms=False,
            variants=(),
            errors=(f"scenario set does not exist: {scenario_set}",),
        )

    raw = _load_raw_yaml(scenario_set)
    if raw.get("schema_version") != "robot_sf.scenario_matrix.v1":
        errors.append("schema_version must be robot_sf.scenario_matrix.v1")

    loaded = load_scenarios(scenario_set)
    if len(loaded) != len(EXPECTED_SUSTAINED_FLOW_TIERS):
        errors.append(
            f"expected {len(EXPECTED_SUSTAINED_FLOW_TIERS)} sustained-flow variants, "
            f"found {len(loaded)}"
        )

    variants: list[SustainedFlowVariant] = []
    seen_seeds: set[int] = set()
    expected_by_tier = {
        tier: (ped_density, spawn_rate, seeds)
        for tier, ped_density, spawn_rate, seeds in EXPECTED_SUSTAINED_FLOW_TIERS
    }
    expected_order = [tier for tier, *_ in EXPECTED_SUSTAINED_FLOW_TIERS]
    observed_order: list[str] = []

    for scenario in loaded:
        metadata = scenario.get("metadata", {})
        tier = str(metadata.get("density", ""))
        observed_order.append(tier)
        variant, variant_errors = _validate_variant(
            scenario,
            scenario_set=scenario_set,
            expected_by_tier=expected_by_tier,
            seen_seeds=seen_seeds,
        )
        errors.extend(variant_errors)
        if variant is not None:
            variants.append(variant)

    if observed_order != expected_order:
        errors.append(
            "density tiers must enumerate light, medium, heavy in deterministic order; "
            f"observed {observed_order}"
        )

    return SustainedFlowPreflightReport(
        schema_version=SUSTAINED_FLOW_PREFLIGHT_SCHEMA_VERSION,
        scenario_set=_repo_relative(scenario_set),
        conforms=not errors,
        variants=tuple(variants),
        errors=tuple(errors),
    )


def preflight_generated_sustained_flow_scenarios() -> SustainedFlowPreflightReport:
    """Validate the canonical generator before rows are materialized to YAML.

    This is a generator preflight only: it proves the deterministic
    light/medium/heavy continuous-spawn definitions remain internally
    enumerable and fail-closed, without promoting them to benchmark evidence.
    Returns:
        Sustained-flow preflight report for canonical generated rows.
    """

    scenario_set = DEFAULT_SUSTAINED_FLOW_SCENARIO_SET.resolve()
    loaded = generate_expected_sustained_flow_scenarios()
    errors: list[str] = []
    if len(loaded) != len(EXPECTED_SUSTAINED_FLOW_TIERS):
        errors.append(
            f"expected {len(EXPECTED_SUSTAINED_FLOW_TIERS)} sustained-flow variants, "
            f"found {len(loaded)}"
        )

    variants: list[SustainedFlowVariant] = []
    seen_seeds: set[int] = set()
    expected_by_tier = {
        tier: (ped_density, spawn_rate, seeds)
        for tier, ped_density, spawn_rate, seeds in EXPECTED_SUSTAINED_FLOW_TIERS
    }
    expected_order = [tier for tier, *_ in EXPECTED_SUSTAINED_FLOW_TIERS]
    observed_order: list[str] = []
    for scenario in loaded:
        metadata = scenario.get("metadata", {})
        tier = str(metadata.get("density", ""))
        observed_order.append(tier)
        variant, variant_errors = _validate_variant(
            scenario,
            scenario_set=scenario_set,
            expected_by_tier=expected_by_tier,
            seen_seeds=seen_seeds,
        )
        errors.extend(variant_errors)
        if variant is not None:
            variants.append(variant)

    if observed_order != expected_order:
        errors.append(
            "density tiers must enumerate light, medium, heavy in deterministic order; "
            f"observed {observed_order}"
        )

    return SustainedFlowPreflightReport(
        schema_version=SUSTAINED_FLOW_PREFLIGHT_SCHEMA_VERSION,
        scenario_set="generated:issue_3813_sustained_flow_scaffold_v0",
        conforms=not errors,
        variants=tuple(variants),
        errors=tuple(errors),
    )


def sustained_flow_preflight_to_dict(report: SustainedFlowPreflightReport) -> dict[str, Any]:
    """Serialize a sustained-flow preflight report.

    Returns:
        JSON-serializable sustained-flow preflight payload.
    """

    return {
        "schema_version": report.schema_version,
        "scenario_set": report.scenario_set,
        "conforms": report.conforms,
        "benchmark_evidence": report.benchmark_evidence,
        "runtime_support": report.runtime_support,
        "variant_count": len(report.variants),
        "variants": [
            {
                "name": variant.name,
                "density_tier": variant.density_tier,
                "ped_density": variant.ped_density,
                "spawn_rate_per_min": variant.spawn_rate_per_min,
                "spawn_definition": dict(variant.spawn_definition),
                "seeds": list(variant.seeds),
                "map_file": variant.map_file,
            }
            for variant in report.variants
        ],
        "errors": list(report.errors),
    }
