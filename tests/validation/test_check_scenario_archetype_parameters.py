"""Focused contract tests for the archetype parameter-consistency checker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.validation.check_scenario_archetype_parameters import (
    DEFAULT_SCENARIOS,
    ParameterCheck,
    ScenarioParameterReport,
    _resolve_declared,
    _values_match,
    format_console_table,
    inspect_scenario_parameters,
    main,
)


@dataclass
class _FakeRuntime:
    """Minimal duck-typed stand-in for the resolved SimulationSettings."""

    difficulty: int = 0
    max_peds_per_group: int = 3
    route_spawn_distribution: str = "cluster"
    route_spawn_jitter_frac: float = 0.2
    ped_density_by_difficulty: list[float] | None = None
    sim_time_in_secs: float = 50.0
    time_per_step_in_secs: float = 0.1

    def __post_init__(self) -> None:
        if self.ped_density_by_difficulty is None:
            self.ped_density_by_difficulty = [0.02, 0.05, 0.08, 0.12]


def test_direct_attribute_match() -> None:
    check = _resolve_declared("max_peds_per_group", 3, _FakeRuntime(max_peds_per_group=3))
    assert check.driver == "direct"
    assert check.match is True


def test_direct_attribute_mismatch_surfaces() -> None:
    check = _resolve_declared("max_peds_per_group", 1, _FakeRuntime(max_peds_per_group=3))
    assert check.driver == "direct"
    assert check.match is False
    assert check.runtime_value == 3


def test_max_episode_steps_derived_from_sim_horizon() -> None:
    runtime = _FakeRuntime(sim_time_in_secs=50.0, time_per_step_in_secs=0.1)
    check = _resolve_declared("max_episode_steps", 500, runtime)
    assert check.driver == "derived"
    assert check.match is True
    assert check.runtime_value == 500


def test_ped_density_derived_from_difficulty_slot() -> None:
    runtime = _FakeRuntime(difficulty=0, ped_density_by_difficulty=[0.05, 0.02, 0.04, 0.08])
    check = _resolve_declared("ped_density", 0.05, runtime)
    assert check.driver == "derived"
    assert check.match is True


@pytest.mark.parametrize(
    ("declared", "runtime", "match"),
    [
        (0.5, 0.5, True),
        (0.5, 0.5000001, True),
        ("spread", "spread", True),
        ("spread", "cluster", False),
        ({"a": 1}, {"a": 1.0}, True),
        ({"a": 1}, {"a": 2}, False),
        ([1, 2], [1, 2.0], True),
        ([1, 2], [2, 1], False),
    ],
)
def test_values_match(declared, runtime, match) -> None:
    assert _values_match(declared, runtime) is match


def test_unknown_key_reports_unresolved_driver() -> None:
    check = _resolve_declared("groups", 0.5, _FakeRuntime())
    assert check.driver == "unresolved"
    assert check.match is False
    assert check.runtime_value is None


def test_report_finding_counting() -> None:
    report = ScenarioParameterReport(scenario="classic_group_crossing_low", source="s.yaml")
    report.checks.append(ParameterCheck("max_episode_steps", 500, 500, "derived", True))
    report.checks.append(ParameterCheck("groups", 0.5, None, "unresolved", False))
    assert report.findings == 1


def test_console_table_marks_findings() -> None:
    report = ScenarioParameterReport(scenario="classic_group_crossing_low", source="s.yaml")
    report.checks.append(ParameterCheck("groups", 0.5, None, "unresolved", False))
    text = format_console_table(report)
    assert "[MISS]" in text
    assert "groups" in text


def test_integration_pinned_group_crossing_surfaces_groups_driver_gap() -> None:
    """The pinned group-crossing archetype declares groups=0.5 with no runtime driver.

    This is the deterministic finding the checker exists to surface: the declared
    simulation_config.groups key is read by the benchmark validator (cli.py) but
    has no SimulationSettings attribute after build_robot_config_from_scenario.
    """
    repo_root = Path(__file__).resolve().parents[2]
    scenario = repo_root / "configs/scenarios/archetypes/classic_group_crossing.yaml"
    if not scenario.exists():
        pytest.skip("pinned archetype scenarios not present")
    reports = inspect_scenario_parameters(scenario)
    assert reports
    groups_findings = [
        check for report in reports for check in report.checks if check.parameter == "groups"
    ]
    assert groups_findings
    assert all(check.driver == "unresolved" for check in groups_findings)


def test_cli_default_exit_zero_and_fail_flag(tmp_path: Path, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scenario = repo_root / "configs/scenarios/archetypes/classic_doorway.yaml"
    if not scenario.exists():
        pytest.skip("pinned archetype scenarios not present")
    # The doorway archetype has no known unresolved keys today; informational mode
    # still exits 0 even with the group-crossing finding present in the default set.
    assert main(["--scenario", str(scenario)]) == 0
    out = capsys.readouterr().out
    assert "all declared parameters resolve to a runtime driver" in out or "[MISS]" in out


def test_cli_fail_flag_requires_an_explicit_waiver_file(capsys) -> None:
    assert main(["--fail-on-violation"]) == 2
    assert "requires --waiver-file" in capsys.readouterr().err


def test_default_scenarios_are_the_four_pinned_archetypes() -> None:
    assert len(DEFAULT_SCENARIOS) == 4
    for scenario in DEFAULT_SCENARIOS:
        assert Path(scenario).name.startswith("classic_")
