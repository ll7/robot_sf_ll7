"""Receipt comparison must refuse incomplete or malformed source observations."""

import pytest

from scripts.ci import make_behaviour_receipt as producer


def _row():
    return {
        "metrics": {"success": False, "collisions": 0, "total_collision_count": 0},
        "steps": 1,
    }


@pytest.mark.parametrize("field", ["success", "collisions", "steps"])
def test_comparison_requires_every_outcome_observation(field):
    """Missing observations must never become false or zero comparison outcomes."""
    row = _row()
    if field == "steps":
        row.pop("steps")
    elif field == "collisions":
        row["metrics"].pop("collisions")
        row["metrics"].pop("total_collision_count")
    else:
        row["metrics"].pop("success")
    with pytest.raises(ValueError, match=field):
        producer._row_success(row)


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
def test_comparison_refuses_untyped_outcome_observations(field, value):
    """Present fields require their declared types without truthiness or coercion."""
    row = _row()
    if field == "steps":
        row["steps"] = value
    elif field == "collisions":
        row["metrics"]["collisions"] = row["metrics"]["total_collision_count"] = value
    else:
        row["metrics"]["success"] = value
    with pytest.raises(ValueError, match=field):
        producer._row_success(row)


def test_comparison_refuses_contradictory_success_aliases():
    """A second outcome declaration cannot hide contradictory raw observations."""
    row = _row()
    row["success"] = True
    with pytest.raises(ValueError, match="success"):
        producer._row_success(row)


def test_comparison_preserves_explicit_false_and_zero():
    """A valid failure and zero collisions are observations, not missing fields."""
    row = _row()
    assert producer._row_success(row) is False
    assert producer._row_collisions(row) == 0
