from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.scenario_schema import validate_scenario_list


def test_validate_scenario_list_success():
    scenarios = [
        {"id": "s1", "density": "low", "flow": "uni", "obstacle": "open", "repeats": 1},
        {"id": "s2", "density": "med", "flow": "bi", "obstacle": "bottleneck", "repeats": 2},
    ]
    errs = validate_scenario_list(scenarios)
    assert errs == []


def test_validate_scenario_list_errors():
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
