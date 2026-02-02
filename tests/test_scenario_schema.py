"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.scenario_schema import validate_scenario_list

if TYPE_CHECKING:
    from pathlib import Path


def test_validate_scenario_list_success():
    """TODO docstring. Document this function."""
    scenarios = [
        {"id": "s1", "density": "low", "flow": "uni", "obstacle": "open", "repeats": 1},
        {"id": "s2", "density": "med", "flow": "bi", "obstacle": "bottleneck", "repeats": 2},
    ]
    errs = validate_scenario_list(scenarios)
    assert errs == []


def test_validate_scenario_list_errors():
    """TODO docstring. Document this function."""
    scenarios = [
        {"id": "a", "density": "invalid", "flow": "uni", "obstacle": "open", "repeats": 1},
        {"id": "a", "density": "low", "flow": "bi", "obstacle": "maze", "repeats": 0},
        {"density": "low", "flow": "uni", "obstacle": "open", "repeats": 1},
    ]
    errs = validate_scenario_list(scenarios)
    assert len(errs) >= 3
    # Expect at least one duplicate id and one repeats-related error
    assert any(e.get("error") == "duplicate id" for e in errs)
    assert any("repeats" in e.get("path", "") for e in errs)


def test_cli_validate_config_with_schema(tmp_path: Path, capsys):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {"id": "ok", "density": "low", "flow": "uni", "obstacle": "open", "repeats": 1},
        {"id": "dup", "density": "low", "flow": "uni", "obstacle": "open", "repeats": 1},
        {"id": "dup", "density": "med", "flow": "bi", "obstacle": "open", "repeats": 1},
    ]
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    rc = cli_main(["validate-config", "--matrix", str(matrix_path)])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert rc != 0
    assert report["num_scenarios"] == 3
    assert any(e.get("error") == "duplicate id" for e in report["errors"])
    assert report["source"]["format"] == "list"
    assert report["summary"]["missing"]["metadata"] == 3


def test_cli_validate_config_manifest_source(tmp_path: Path, capsys):
    """Ensure validate-config reports manifest includes in the summary."""
    include_path = tmp_path / "include.yaml"
    include_path.write_text(
        "---\nscenarios:\n  - name: sc_a\n    map_file: map.svg\n    simulation_config:\n      max_episode_steps: 50\n    metadata:\n      archetype: crossing\n      density: low\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("includes:\n  - include.yaml\n", encoding="utf-8")

    rc = cli_main(["validate-config", "--matrix", str(manifest_path)])
    out = capsys.readouterr().out
    report = json.loads(out)

    assert rc == 0
    assert report["num_scenarios"] == 1
    assert report["source"]["format"] == "manifest"
    assert "include.yaml" in report["source"]["includes"]
