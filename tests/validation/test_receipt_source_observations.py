"""Portable comparison-input tests using the existing producer join on both trees."""

import json

import pytest

from scripts.ci import make_behaviour_receipt as producer


def write_sweep(root, field=None, value=None, *, missing=False, contradiction=False):
    raw = {
        "scenario_id": "open",
        "seed": 1001,
        "steps": 1,
        "metrics": {"success": False, "collisions": 0, "total_collision_count": 0},
    }
    summary = {
        "arm": "goal",
        "scenario": "open",
        "seed": 1001,
        "steps": 1,
        "success": False,
        "collisions": 0,
        "execution_status": "written",
        "kinematics": "differential_drive",
        "source_file": "runs/goal__differential_drive/episodes.jsonl",
    }
    if field:
        if missing:
            summary.pop(field)
            if field == "steps":
                raw.pop(field)
            elif field == "collisions":
                raw["metrics"].pop("collisions")
                raw["metrics"].pop("total_collision_count")
            else:
                raw["metrics"].pop(field)
        else:
            summary[field] = value
            if field == "steps":
                raw[field] = value
            elif field == "collisions":
                raw["metrics"]["collisions"] = raw["metrics"]["total_collision_count"] = value
            else:
                raw["metrics"][field] = value
    if contradiction:
        raw["success"] = summary["success"] = True
    path = root / "campaigns/empty_world_main" / summary["source_file"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(raw) + "\n")
    (root / "episodes_main.jsonl").write_text(json.dumps(summary) + "\n")


@pytest.mark.parametrize("field", ["success", "collisions", "steps"])
def test_join_refuses_coherently_missing_observations(tmp_path, field):
    write_sweep(tmp_path, field, missing=True)
    with pytest.raises(ValueError, match=field):
        producer._load_sweep_rows(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success", None),
        ("success", "false"),
        ("success", 0),
        ("success", []),
        ("collisions", "0"),
        ("collisions", False),
        ("collisions", -1),
        ("collisions", 0.5),
        ("steps", "1"),
        ("steps", True),
        ("steps", 0),
        ("steps", 0.5),
    ],
)
def test_join_refuses_coherently_malformed_observations(tmp_path, field, value):
    write_sweep(tmp_path, field, value)
    with pytest.raises(ValueError, match=field):
        producer._load_sweep_rows(tmp_path)


def test_join_refuses_contradictory_raw_success_alias(tmp_path):
    write_sweep(tmp_path, contradiction=True)
    with pytest.raises(ValueError, match="success"):
        producer._load_sweep_rows(tmp_path)


def test_join_accepts_explicit_false_zero_and_positive_steps(tmp_path):
    write_sweep(tmp_path)
    assert len(producer._load_sweep_rows(tmp_path)) == 1
