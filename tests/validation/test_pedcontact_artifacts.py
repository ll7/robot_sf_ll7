"""Actual evidence-writer and admission contracts for the bounded fit driver."""

import hashlib
import json

import pytest

from robot_sf.evidence.writers import write_json
from scripts.validation.pedcontact_fit_10101 import freeze


@pytest.mark.parametrize("admitted", [True, False])
def test_freeze_writes_replayable_grid_only_after_physical_admission(tmp_path, admitted):
    """Real JSON writers must persist the grid, and failed physics cannot admit it."""
    comparison = tmp_path / "step4_comparison.json"
    robot = tmp_path / "robot_gate_summary.json"
    write_json(comparison, {"fit_admitted": admitted})
    write_json(robot, {"pairs": 8550})
    root = tmp_path / "fit"
    root.mkdir()
    if not admitted:
        with pytest.raises(ValueError, match="qualification"):
            freeze(root)
        assert not (root / "grid.json").exists()
        return
    freeze(root)
    grid = json.loads((root / "grid.json").read_bytes())
    assert len(grid["points"]) == len({p["id"] for p in grid["points"]}) == 108
    assert grid["seeds"] == [1001, 1002, 1003]
    assert (
        grid["admission_sha256"][comparison.name]
        == hashlib.sha256(comparison.read_bytes()).hexdigest()
    )
    assert all(p["pedestrian_contact_rule"] == "projection_v1" for p in grid["points"])


def test_freeze_refuses_incomplete_robot_gate(tmp_path):
    """A partial pilot cannot stand in for the requested paired robot gate."""
    write_json(tmp_path / "step4_comparison.json", {"fit_admitted": True})
    write_json(tmp_path / "robot_gate_summary.json", {"pairs": 54})
    root = tmp_path / "fit"
    root.mkdir()
    with pytest.raises(ValueError, match="robot gate"):
        freeze(root)
    assert not (root / "grid.json").exists()
