"""Tests for issue #4016 discrete unicycle action lattice primitives."""

from __future__ import annotations

import pytest

from robot_sf.training.discrete_action_lattice import DiscreteUnicycleActionLattice


def test_lattice_size_and_index_to_command_mapping_are_deterministic() -> None:
    """Cartesian lattice order is stable for checkpoint/action-index contracts."""

    lattice = DiscreteUnicycleActionLattice(
        linear_values=(0.0, 0.5),
        angular_values=(-1.0, 0.0, 1.0),
        max_linear_speed=0.5,
        max_angular_speed=1.0,
    )

    assert lattice.action_count == 6
    assert lattice.command_at(0).as_tuple() == (0.0, -1.0)
    assert lattice.command_at(2).as_tuple() == (0.0, 1.0)
    assert lattice.command_at(3).as_tuple() == (0.5, -1.0)
    assert [command.as_tuple() for command in lattice.commands()] == [
        (0.0, -1.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (0.5, -1.0),
        (0.5, 0.0),
        (0.5, 1.0),
    ]


def test_lattice_invalid_indices_fail_closed() -> None:
    """Out-of-range action ids should not silently wrap."""

    lattice = DiscreteUnicycleActionLattice(
        linear_values=(0.0,),
        angular_values=(0.0,),
        max_linear_speed=1.0,
        max_angular_speed=1.0,
    )

    with pytest.raises(IndexError):
        lattice.command_at(-1)
    with pytest.raises(IndexError):
        lattice.command_at(1)


def test_lattice_rejects_commands_outside_configured_bounds() -> None:
    """Speed bounds are part of the action-space contract."""

    with pytest.raises(ValueError, match="max_linear_speed"):
        DiscreteUnicycleActionLattice(
            linear_values=(0.0, 1.5),
            angular_values=(0.0,),
            max_linear_speed=1.0,
            max_angular_speed=1.0,
        )

    with pytest.raises(ValueError, match="max_angular_speed"):
        DiscreteUnicycleActionLattice(
            linear_values=(0.0,),
            angular_values=(-2.0, 0.0),
            max_linear_speed=1.0,
            max_angular_speed=1.0,
        )


def test_lattice_json_round_trip_preserves_ordering(tmp_path) -> None:
    """Lattice files should preserve action-index ordering exactly."""

    lattice = DiscreteUnicycleActionLattice(
        linear_values=(-0.1, 0.2),
        angular_values=(-0.4, 0.4),
        max_linear_speed=0.2,
        max_angular_speed=0.4,
    )

    path = tmp_path / "action_lattice.json"
    lattice.to_json_file(path)
    loaded = DiscreteUnicycleActionLattice.from_json_file(path)

    assert loaded == lattice
    assert [command.as_tuple() for command in loaded.commands()] == [
        command.as_tuple() for command in lattice.commands()
    ]


def test_lattice_rejects_unknown_json_keys() -> None:
    """Malformed lattice contracts should fail before training starts."""

    with pytest.raises(ValueError, match="unknown lattice keys"):
        DiscreteUnicycleActionLattice.from_dict(
            {
                "linear_values": [0.0],
                "angular_values": [0.0],
                "max_linear_speed": 1.0,
                "max_angular_speed": 1.0,
                "extra": True,
            }
        )
