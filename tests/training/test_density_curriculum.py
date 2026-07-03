"""Tests for issue #4018 deterministic density curriculum schedules."""

from __future__ import annotations

import pytest

from robot_sf.training.density_curriculum import (
    DENSITY_CURRICULUM_SCHEMA_VERSION,
    apply_density_curriculum_stage_to_scenario,
    build_density_curriculum_schedule,
    stage_metadata,
)


def _enabled_payload() -> dict[str, object]:
    return {
        "enabled": True,
        "stages": [
            {"id": "sparse", "until_timesteps": 100, "density_m2": 0.04, "difficulty": 0},
            {"id": "dense", "until_timesteps": None, "density_m2": 0.12, "difficulty": 0},
        ],
    }


def test_disabled_schedule_has_no_active_stage() -> None:
    """Disabled or empty curriculum keeps existing training behavior."""
    schedule = build_density_curriculum_schedule({"enabled": False})

    assert schedule.enabled is False
    assert schedule.stage_for_timestep(0) is None


def test_enabled_schedule_requires_valid_stages() -> None:
    """Invalid enabled schedules fail closed during config validation."""
    with pytest.raises(ValueError, match="requires at least one stage"):
        build_density_curriculum_schedule({"enabled": True, "stages": []})
    with pytest.raises(ValueError, match="Duplicate"):
        build_density_curriculum_schedule(
            {
                "enabled": True,
                "stages": [
                    {"id": "x", "until_timesteps": 10},
                    {"id": "x", "until_timesteps": None},
                ],
            }
        )
    with pytest.raises(ValueError, match="strictly increase"):
        build_density_curriculum_schedule(
            {
                "enabled": True,
                "stages": [
                    {"id": "a", "until_timesteps": 10, "density_m2": 0.1},
                    {"id": "b", "until_timesteps": 10, "density_m2": 0.2},
                ],
            }
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        build_density_curriculum_schedule(
            {
                "enabled": True,
                "stages": [
                    {"id": "a", "until_timesteps": 10, "density_m2": 0.2},
                    {"id": "b", "until_timesteps": None, "density_m2": 0.1},
                ],
            }
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"id": "bad", "until_timesteps": None, "density_m2": -0.1}, "non-negative"),
        ({"id": "bad", "until_timesteps": None, "difficulty": -1}, "non-negative"),
    ],
)
def test_stage_numeric_fields_must_be_non_negative(payload: dict[str, object], match: str) -> None:
    """Density and difficulty fields reject negative values."""
    with pytest.raises(ValueError, match=match):
        build_density_curriculum_schedule({"enabled": True, "stages": [payload]})


def test_stage_for_timestep_uses_sparse_boundaries() -> None:
    """Stage selection is deterministic by global timestep."""
    schedule = build_density_curriculum_schedule(_enabled_payload())

    assert schedule.stage_for_timestep(0).id == "sparse"
    assert schedule.stage_for_timestep(99).id == "sparse"
    assert schedule.stage_for_timestep(100).id == "dense"
    assert schedule.stage_for_timestep(10_000).id == "dense"


def test_apply_stage_updates_existing_simulation_fields_only() -> None:
    """Stage transform uses existing sim config fields and preserves unrelated scenario data."""
    schedule = build_density_curriculum_schedule(_enabled_payload())
    stage = schedule.stage_for_timestep(100)
    scenario = {
        "name": "demo",
        "map": "maps/svg_maps/demo.svg",
        "simulation_config": {"dt": 0.1, "difficulty": 3},
        "unrelated": {"kept": True},
    }

    updated = apply_density_curriculum_stage_to_scenario(scenario, stage)

    assert updated["simulation_config"] == {
        "dt": 0.1,
        "difficulty": 0,
        "ped_density_by_difficulty": [0.12],
    }
    assert updated["unrelated"] == {"kept": True}
    assert scenario["simulation_config"] == {"dt": 0.1, "difficulty": 3}


def test_stage_metadata_declares_schema_and_claim_boundary() -> None:
    """Metadata is explicit that this is mechanism evidence, not benchmark evidence."""
    schedule = build_density_curriculum_schedule(_enabled_payload())
    metadata = stage_metadata(schedule.stage_for_timestep(0))

    assert metadata["schema_version"] == DENSITY_CURRICULUM_SCHEMA_VERSION
    assert "not benchmark evidence" in metadata["claim_boundary"]
