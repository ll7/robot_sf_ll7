"""Read-only declared-vs-runtime parameter consistency checks for pinned archetypes.

This checker answers, per pinned archetype scenario, whether each parameter
declared in the scenario's ``simulation_config`` actually reaches the runtime
configuration that ``build_robot_config_from_scenario`` produces:

1. direct passthrough attributes are compared field-for-field;
2. derived parameters (``max_episode_steps`` -> sim horizon, ``ped_density`` ->
   per-difficulty density) are resolved through the same transformation the
   loader applies;
3. declared keys with no runtime driver at all are reported as
   ``unresolved_driver`` rows so silent configuration drift surfaces instead of
   being dropped.

The checker never mutates scenario or map files and performs no simulation
execution beyond scenario loading. Its report is deterministic for a given
scenario set.

Exit-code policy: findings are informational by default so the first landing
records existing known deviations without reddening CI; ``--fail-on-violation``
promotes any finding to exit code 1 for later gate enforcement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
)

# The four pinned archetype scenario definitions (same set as the geometry
# checker's DEFAULT_MAPS, sourced from their scenario/config surface).
DEFAULT_SCENARIOS = (
    "configs/scenarios/archetypes/classic_doorway.yaml",
    "configs/scenarios/archetypes/classic_head_on_corridor.yaml",
    "configs/scenarios/archetypes/classic_group_crossing.yaml",
    "configs/scenarios/archetypes/classic_crossing.yaml",
)

# Declared simulation_config keys that pass straight through to the runtime
# SimulationSettings attribute of the same name.
_DIRECT_ATTRIBUTES = (
    "difficulty",
    "max_peds_per_group",
    "peds_speed_mult",
    "action_latency_steps",
    "action_latency_ms",
    "pedestrian_integration_scheme",
    "ped_radius",
    "pedestrian_uncertainty_envelope_enabled",
    "pedestrian_uncertainty_alpha_mps",
    "goal_radius",
    "pedestrian_model",
    "route_spawn_distribution",
    "route_spawn_jitter_frac",
    "route_spawn_seed",
    "archetype_composition",
    "archetype_speed_factors",
    "archetype_seed",
    "response_law_composition",
    "response_law_seed",
    "population_size",
    "non_reactive_response_multiplier",
    "hesitating_response_multiplier",
)


@dataclass
class ParameterCheck:
    """One declared-vs-runtime parameter comparison for a scenario."""

    parameter: str
    declared_value: object
    runtime_value: object
    driver: str  # "direct" | "derived" | "unresolved"
    match: bool


@dataclass
class ScenarioParameterReport:
    """Aggregated parameter-consistency findings for one scenario."""

    scenario: str
    source: str
    checks: list[ParameterCheck] = field(default_factory=list)

    @property
    def findings(self) -> int:
        """Count mismatch and unresolved-driver rows."""
        return sum(1 for c in self.checks if not c.match or c.driver == "unresolved")


def _resolve_declared(
    parameter: str,
    declared_value: object,
    runtime,
) -> ParameterCheck:
    """Resolve one declared simulation_config key against the runtime config.

    ``runtime`` is the ``SimulationSettings`` instance the loader built from
    the scenario (already mutated by ``_apply_simulation_overrides``).
    """
    if parameter in _DIRECT_ATTRIBUTES:
        runtime_value = getattr(runtime, parameter, None)
        return ParameterCheck(
            parameter=parameter,
            declared_value=declared_value,
            runtime_value=runtime_value,
            driver="direct",
            match=_values_match(declared_value, runtime_value),
        )
    if parameter == "max_episode_steps":
        steps = max(1, int(declared_value))
        runtime_value = round(runtime.sim_time_in_secs / runtime.time_per_step_in_secs)
        return ParameterCheck(
            parameter=parameter,
            declared_value=steps,
            runtime_value=runtime_value,
            driver="derived",
            match=runtime_value == steps,
        )
    if parameter == "ped_density":
        depth = len(runtime.ped_density_by_difficulty)
        runtime_value = runtime.ped_density_by_difficulty[min(runtime.difficulty, depth - 1)]
        declared_float = float(declared_value)
        return ParameterCheck(
            parameter=parameter,
            declared_value=declared_float,
            runtime_value=runtime_value,
            driver="derived",
            match=_values_match(declared_float, runtime_value),
        )
    return ParameterCheck(
        parameter=parameter,
        declared_value=declared_value,
        runtime_value=None,
        driver="unresolved",
        match=False,
    )


def _values_match(declared: object, runtime: object) -> bool:
    """Compare declared and runtime values, tolerating numeric float form."""
    if declared is runtime:
        return True
    if isinstance(declared, (int, float)) and isinstance(runtime, (int, float)):
        left = round(float(declared), 6)
        right = round(float(runtime), 6)
        return left == right
    if isinstance(declared, dict) and isinstance(runtime, dict):
        if declared.keys() != runtime.keys():
            return False
        return all(_values_match(declared[k], runtime[k]) for k in declared)
    if isinstance(declared, (list, tuple)) and isinstance(runtime, (list, tuple)):
        if len(declared) != len(runtime):
            return False
        return all(_values_match(a, b) for a, b in zip(declared, runtime, strict=True))
    return declared == runtime


def inspect_scenario_parameters(
    scenario_path: Path,
) -> list[ScenarioParameterReport]:
    """Run the declared-vs-runtime parameter check for one scenario YAML."""
    reports: list[ScenarioParameterReport] = []
    scenarios = load_scenarios(scenario_path)
    for scenario in scenarios:
        name = str(scenario.get("name") or scenario.get("scenario_id") or "unknown")
        config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
        runtime = config.sim_config
        report = ScenarioParameterReport(scenario=name, source=str(scenario_path))
        for parameter, declared_value in scenario.get("simulation_config", {}).items():
            report.checks.append(_resolve_declared(parameter, declared_value, runtime))
        reports.append(report)
    return reports


def format_console_table(report: ScenarioParameterReport) -> str:
    """Render a compact human-readable table for one scenario report."""
    lines = [f"scenario: {report.scenario}  ({report.source})"]
    for check in report.checks:
        flag = "ok  " if check.match and check.driver != "unresolved" else "MISS"
        lines.append(
            f"  [{flag}] {check.parameter:<32} {check.driver:<11}"
            f" declared={check.declared_value!r:<18} runtime={check.runtime_value!r}"
        )
    if report.findings == 0:
        lines.append("  all declared parameters resolve to a runtime driver")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning a process exit code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario YAML path; repeatable. Defaults to the four pinned archetypes.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report only.")
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 when any finding exists (for later CI enforcement).",
    )
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.scenario] or [Path(p) for p in DEFAULT_SCENARIOS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(
            f"ERROR: scenario files not found: {', '.join(str(p) for p in missing)}",
            file=sys.stderr,
        )
        return 2

    reports = [r for p in paths for r in inspect_scenario_parameters(p)]
    total = sum(r.findings for r in reports)

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=1))
    else:
        for report in reports:
            print(format_console_table(report))
        print(f"\ntotal findings: {total}")

    if args.fail_on_violation and total > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
