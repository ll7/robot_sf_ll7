"""Regression tests for fail-closed campaign scenario staging."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.scenario_staging import ScenarioStagingError, select_unique_scenario


def test_select_unique_scenario_returns_the_exact_match() -> None:
    """An exact unique name must resolve without changing the scenario row."""
    expected = {"name": "classic_head_on_corridor_medium", "seeds": [1, 2]}

    selected = select_unique_scenario(
        [{"name": "classic_head_on_corridor_low"}, expected],
        "classic_head_on_corridor_medium",
        source="configs/scenarios/archetypes/classic_head_on_corridor.yaml",
    )

    assert selected is expected


def test_select_unique_scenario_reports_source_and_zero_matches() -> None:
    """A drifted preregistration must explain the source and zero-match count."""
    source = "configs/scenarios/archetypes/classic_head_on_corridor.yaml"

    with pytest.raises(ScenarioStagingError) as exc_info:
        select_unique_scenario(
            [
                {"name": "classic_head_on_corridor_low"},
                {"name": "classic_head_on_corridor_medium"},
            ],
            "classic_head_on_corridor_high",
            source=source,
        )

    message = str(exc_info.value)
    assert "classic_head_on_corridor_high" in message
    assert source in message
    assert "found 0 matches" in message


def test_select_unique_scenario_reports_duplicate_matches() -> None:
    """Duplicate source rows must fail closed with their observed match count."""
    with pytest.raises(ScenarioStagingError, match=r"found 2 matches"):
        select_unique_scenario(
            [{"name": "duplicate"}, {"name": "duplicate"}],
            "duplicate",
            source="duplicate-scenarios.yaml",
        )


@pytest.mark.parametrize("stored_name", [None, 123])
def test_select_unique_scenario_does_not_coerce_non_string_names(stored_name: object) -> None:
    """Exact matching must not coerce malformed source names into valid strings."""
    with pytest.raises(ScenarioStagingError, match=r"found 0 matches"):
        select_unique_scenario(
            [{"name": stored_name}],
            str(stored_name),
            source="malformed-scenarios.yaml",
        )
