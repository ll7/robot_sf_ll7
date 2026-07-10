"""Focused tests for issue #4969: ``ped_density`` unit documentation and validation.

Issue #4969 asks to (1) document the ``simulation_config.ped_density`` unit
(pedestrians per square meter of SPAWNABLE route/zone sidewalk area) and the
marker-controlled ``0.0`` convention in the scenario schema docs and archetype
YAML comments, (2) make the CLI ``validate-config`` warning self-explanatory by
stating the unit, and (3) add schema-level validation that rejects negative
values and flags very high values (``> 0.15``) as likely unit confusion.

These tests cover the behavioral contract: schema rejection of negatives, the
self-explanatory CLI warning text, the ``> 0.15`` unit-confusion flag, and the
preserved marker-mode ``0.0`` behavior (no false "empty scene" warning).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.scenario_schema import validate_scenario_list

if TYPE_CHECKING:
    from pathlib import Path


def _write_matrix(tmp_path: Path, scenarios: list[dict]) -> Path:
    """Write a scenario matrix YAML file and return its path."""
    matrix_path = tmp_path / "matrix.yaml"
    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"scenarios": scenarios}, f)
    return matrix_path


def _validate_config_warnings(matrix_path: Path, capsys) -> list[str]:
    """Run ``validate-config`` and return the list of warning message strings."""
    rc = cli_main(["validate-config", "--matrix", str(matrix_path)])
    captured = capsys.readouterr()
    assert rc == 0, f"validate-config failed: {captured.out}"
    report = json.loads(captured.out)
    return [w["warning"] for w in report["warnings"]]


# --- Schema-level validation (part 3: reject negative ped_density) ---


def test_schema_rejects_negative_ped_density() -> None:
    """A negative ``ped_density`` must fail JSON-Schema validation."""
    scenarios = [
        {
            "name": "neg-density",
            "map_file": "x.svg",
            "simulation_config": {"ped_density": -0.1},
            "metadata": {"archetype": "crossing", "density": "low"},
        }
    ]
    errors = validate_scenario_list(scenarios)
    ped_density_errors = [e for e in errors if "ped_density" in e.get("path", "")]
    assert ped_density_errors, f"Expected a schema error for negative ped_density, got: {errors}"


def test_schema_accepts_zero_and_positive_ped_density() -> None:
    """``0.0`` and positive ``ped_density`` must pass JSON-Schema validation."""
    scenarios = [
        {
            "name": "zero-density",
            "map_file": "x.svg",
            "simulation_config": {"ped_density": 0.0},
            "metadata": {"archetype": "bottleneck", "density": "low"},
        },
        {
            "name": "pos-density",
            "map_file": "x.svg",
            "simulation_config": {"ped_density": 0.08},
            "metadata": {"archetype": "crossing", "density": "high"},
        },
    ]
    errors = validate_scenario_list(scenarios)
    assert errors == [], f"Expected no schema errors for valid densities, got: {errors}"


# --- CLI self-explanatory warning (part 2: state the unit) ---


def test_cli_warning_states_spawnable_area_unit(tmp_path: Path, capsys) -> None:
    """Out-of-range density warning must state the unit (peds/m^2 of spawnable area)."""
    scenarios = [
        {
            "name": "sparse",
            "map_file": "x.svg",
            "simulation_config": {"ped_density": 0.001},
            "metadata": {"archetype": "crossing", "density": "low"},
        }
    ]
    matrix_path = _write_matrix(tmp_path, scenarios)
    warnings = _validate_config_warnings(matrix_path, capsys)
    range_warnings = [w for w in warnings if "ped_density outside recommended" in w]
    assert range_warnings, f"Expected an out-of-range warning, got: {warnings}"
    # The warning must now be self-explanatory about the unit.
    assert any("peds per m^2 of spawnable area" in w for w in range_warnings), (
        f"Warning must state the spawnable-area unit; got: {range_warnings}"
    )


def test_cli_flags_high_density_as_likely_unit_confusion(tmp_path: Path, capsys) -> None:
    """A ``ped_density`` above 0.15 must be flagged as likely unit confusion."""
    scenarios = [
        {
            "name": "very-dense",
            "map_file": "x.svg",
            "simulation_config": {"ped_density": 0.2},
            "metadata": {"archetype": "crowd", "density": "high"},
        }
    ]
    matrix_path = _write_matrix(tmp_path, scenarios)
    warnings = _validate_config_warnings(matrix_path, capsys)
    assert any("likely a unit confusion" in w for w in warnings), (
        f"Expected a unit-confusion flag for ped_density>0.15; got: {warnings}"
    )


def test_cli_does_not_flag_band_edge_0_15_as_unit_confusion(tmp_path: Path, capsys) -> None:
    """``ped_density == 0.15`` is at the edge and must NOT trigger the >0.15 flag."""
    scenarios = [
        {
            "name": "edge",
            "map_file": "x.svg",
            "simulation_config": {"ped_density": 0.15},
            "metadata": {"archetype": "crowd", "density": "high"},
        }
    ]
    matrix_path = _write_matrix(tmp_path, scenarios)
    warnings = _validate_config_warnings(matrix_path, capsys)
    assert not any("likely a unit confusion" in w for w in warnings), (
        f"0.15 must not trigger the >0.15 flag; got: {warnings}"
    )


# --- Preserved default behavior (marker-mode 0.0 convention) ---


def test_cli_marker_mode_zero_density_is_not_warned_as_empty(tmp_path: Path, capsys) -> None:
    """Marker-spawned ``0.0`` must NOT warn as an empty scene (regression guard)."""
    scenarios = [
        {
            "name": "marker-spawn",
            "map_file": "x.svg",
            "simulation_config": {"ped_density": 0.0},
            "metadata": {
                "archetype": "bottleneck",
                "density": "low",
                "spawn_mode": "markers",
                "density_advisory": "zero_baseline_route_spawn",
            },
        }
    ]
    matrix_path = _write_matrix(tmp_path, scenarios)
    warnings = _validate_config_warnings(matrix_path, capsys)
    assert "ped_density=0.0 means no pedestrians spawn" not in warnings
    assert not any("ped_density outside recommended" in w for w in warnings)
    assert not any("likely a unit confusion" in w for w in warnings)


def test_cli_recommended_band_density_emits_no_ped_density_warnings(tmp_path: Path, capsys) -> None:
    """A density inside the recommended band must emit no ped_density warnings."""
    scenarios = [
        {
            "name": "in-band",
            "map_file": "x.svg",
            "simulation_config": {"ped_density": 0.05},
            "metadata": {"archetype": "crossing", "density": "medium"},
        }
    ]
    matrix_path = _write_matrix(tmp_path, scenarios)
    warnings = _validate_config_warnings(matrix_path, capsys)
    assert not any("ped_density" in w for w in warnings), (
        f"Expected no ped_density warnings for in-band density; got: {warnings}"
    )
