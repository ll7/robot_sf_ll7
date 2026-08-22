"""Fail-closed contract tests for exact scenario-validation waivers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation.check_scenario_archetype_geometry import (
    EndpointCheck,
    MapGeometryReport,
    enforce_geometry_waivers,
    inspect_map_geometry,
)
from scripts.validation.check_scenario_archetype_parameters import (
    DEFAULT_SCENARIOS,
    enforce_parameter_waivers,
    inspect_scenario_parameters,
)
from scripts.validation.scenario_validation_waivers import (
    WaiverValidationError,
    validate_exact_waivers,
)

ROOT = Path(__file__).resolve().parents[2]
WAIVER_FILE = ROOT / "configs/scenarios/archetype_validation_waivers.yaml"


def _evidence_matches(actual: dict[str, object], waiver: dict[str, object]) -> bool:
    """Compare the deliberately small synthetic evidence field."""

    return actual["evidence"] == waiver["evidence"]


def _validate(rows: list[dict[str, object]], waivers: list[dict[str, object]]) -> None:
    """Run the shared matcher with a compact synthetic identity contract."""

    validate_exact_waivers(
        rows,
        waivers,
        identity_fields=("kind", "name"),
        evidence_matches=_evidence_matches,
        label="fixture",
    )


def test_pinned_geometry_and_parameter_baselines_pass_exact_waivers() -> None:
    """The accepted current findings pass only against the checked-in rows."""

    geometry_reports = [
        inspect_map_geometry(ROOT / map_path)
        for map_path in (
            "maps/svg_maps/classic_doorway.svg",
            "maps/svg_maps/classic_head_on_corridor.svg",
            "maps/svg_maps/classic_group_crossing.svg",
            "maps/svg_maps/classic_crossing.svg",
        )
    ]
    enforce_geometry_waivers(geometry_reports, WAIVER_FILE)

    parameter_reports = [
        report
        for scenario in DEFAULT_SCENARIOS
        for report in inspect_scenario_parameters(ROOT / scenario)
    ]
    enforce_parameter_waivers(parameter_reports, WAIVER_FILE)


def test_missing_waiver_is_rejected() -> None:
    rows = [{"kind": "endpoint", "name": "new", "evidence": 1}]
    waivers: list[dict[str, object]] = []
    with pytest.raises(WaiverValidationError, match="missing fixture waivers"):
        _validate(rows, waivers)


def test_stale_waiver_is_rejected() -> None:
    rows: list[dict[str, object]] = []
    waivers = [{"kind": "endpoint", "name": "old", "evidence": 1}]
    with pytest.raises(WaiverValidationError, match="stale fixture waivers"):
        _validate(rows, waivers)


def test_duplicate_waiver_is_rejected_as_ambiguous() -> None:
    rows = [{"kind": "endpoint", "name": "one", "evidence": 1}]
    waivers = [
        {"kind": "endpoint", "name": "one", "evidence": 1},
        {"kind": "endpoint", "name": "one", "evidence": 1},
    ]
    with pytest.raises(WaiverValidationError, match="duplicate waiver fixture identities"):
        _validate(rows, waivers)


def test_changed_expected_evidence_is_rejected() -> None:
    rows = [{"kind": "endpoint", "name": "one", "evidence": 2}]
    waivers = [{"kind": "endpoint", "name": "one", "evidence": 1}]
    with pytest.raises(WaiverValidationError, match="changed expected evidence"):
        _validate(rows, waivers)


def test_deliberately_new_geometry_finding_cannot_be_absorbed(tmp_path: Path) -> None:
    """A new fixture finding fails even with a valid but empty waiver document."""

    report = MapGeometryReport(map_path="fixture.svg")
    report.endpoints.append(
        EndpointCheck(
            route_kind="robot",
            label="new_fixture_route",
            end="start",
            zone_kind="robot_spawn_zone",
            zone_index=0,
            inside_zone=False,
            offset_to_centre_m=9.0,
        )
    )
    waiver_file = tmp_path / "empty-waivers.yaml"
    waiver_file.write_text(
        yaml.safe_dump(
            {"schema": "scenario_validation_waivers.v1", "geometry": [], "parameters": []}
        ),
        encoding="utf-8",
    )
    with pytest.raises(WaiverValidationError, match="missing geometry waivers"):
        enforce_geometry_waivers([report], waiver_file)
