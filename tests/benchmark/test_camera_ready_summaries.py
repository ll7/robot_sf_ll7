"""Tests for robot_sf.benchmark.camera_ready._summaries — pure summary builders."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.camera_ready._summaries import (
    _extract_amv_taxonomy,
    _scenario_family_from_scenario,
    _validate_family_map,
    _validate_metric_map,
    _validate_planner_key_map,
)


class TestScenarioFamilyFromScenario:
    """Tests for _scenario_family_from_scenario resolution."""

    def test_archetype_in_metadata(self) -> None:
        """Archetype from metadata must be used as the family label."""
        scenario = {"metadata": {"archetype": "crossing"}}
        assert _scenario_family_from_scenario(scenario) == "crossing"

    def test_scenario_family_in_metadata(self) -> None:
        """scenario_family from metadata must be used when archetype is absent."""
        scenario = {"metadata": {"scenario_family": "overtaking"}}
        assert _scenario_family_from_scenario(scenario) == "overtaking"

    def test_family_key_in_metadata(self) -> None:
        """family key from metadata must be used as a fallback."""
        scenario = {"metadata": {"family": "head_on"}}
        assert _scenario_family_from_scenario(scenario) == "head_on"

    def test_top_level_archetype(self) -> None:
        """Top-level archetype must be used when metadata is absent."""
        scenario = {"archetype": "crossing"}
        assert _scenario_family_from_scenario(scenario) == "crossing"

    def test_scenario_id_fallback(self) -> None:
        """scenario_id must be used when no family/archetype keys exist."""
        scenario = {"scenario_id": "sc-42"}
        assert _scenario_family_from_scenario(scenario) == "sc-42"

    def test_name_fallback(self) -> None:
        """name must be used when scenario_id is absent."""
        scenario = {"name": "my_scenario"}
        assert _scenario_family_from_scenario(scenario) == "my_scenario"

    def test_unknown_fallback(self) -> None:
        """An empty scenario must return 'unknown'."""
        assert _scenario_family_from_scenario({}) == "unknown"

    def test_whitespace_stripped(self) -> None:
        """Whitespace around family labels must be stripped."""
        scenario = {"metadata": {"archetype": "  crossing  "}}
        assert _scenario_family_from_scenario(scenario) == "crossing"

    def test_empty_string_skipped(self) -> None:
        """Empty-string archetype must be skipped in favor of fallback."""
        scenario = {"metadata": {"archetype": ""}, "scenario_id": "sc-1"}
        assert _scenario_family_from_scenario(scenario) == "sc-1"


class TestExtractAmvTaxonomy:
    """Tests for _extract_amv_taxonomy extraction."""

    def test_amv_at_top_level(self) -> None:
        """AMV taxonomy at the top level must be extracted."""
        scenario = {"amv": {"use_case": "delivery", "speed_regime": "high"}}
        result = _extract_amv_taxonomy(scenario)
        assert result["use_case"] == "delivery"
        assert result["speed_regime"] == "high"

    def test_amv_in_metadata(self) -> None:
        """AMV taxonomy nested in metadata must be extracted."""
        scenario = {"metadata": {"amv": {"context": "urban"}}}
        result = _extract_amv_taxonomy(scenario)
        assert result["context"] == "urban"

    def test_no_amv_returns_empty(self) -> None:
        """A scenario without AMV data must return an empty dict."""
        assert _extract_amv_taxonomy({}) == {}

    def test_non_string_values_skipped(self) -> None:
        """Non-string AMV values must be skipped."""
        scenario = {"amv": {"use_case": 42}}
        result = _extract_amv_taxonomy(scenario)
        assert "use_case" not in result

    def test_whitespace_stripped(self) -> None:
        """AMV values must have whitespace stripped."""
        scenario = {"amv": {"use_case": "  delivery  "}}
        result = _extract_amv_taxonomy(scenario)
        assert result["use_case"] == "delivery"


class TestValidateFamilyMap:
    """Tests for _validate_family_map validation."""

    def test_valid_family_map(self) -> None:
        """A valid string->string family map must not raise."""
        _validate_family_map({"crossing": "pedestrian_crossing", "overtaking": "overtake"})

    def test_empty_key_rejected(self) -> None:
        """An empty key must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_family_map({"": "value"})

    def test_empty_value_rejected(self) -> None:
        """An empty value must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_family_map({"key": ""})

    def test_non_string_key_rejected(self) -> None:
        """A non-string key must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_family_map({123: "value"})  # type: ignore[dict-item]


class TestValidateMetricMap:
    """Tests for _validate_metric_map validation."""

    def test_valid_metric_map(self) -> None:
        """A valid metric map must not raise."""
        _validate_metric_map({"success": {"classification": "comparable", "alyassi_metric": "sr"}})

    def test_invalid_classification_rejected(self) -> None:
        """An invalid classification must raise ValueError."""
        with pytest.raises(ValueError, match="classification"):
            _validate_metric_map({"success": {"classification": "invalid"}})

    def test_missing_classification_rejected(self) -> None:
        """A missing classification must raise ValueError."""
        with pytest.raises(ValueError, match="classification"):
            _validate_metric_map({"success": {"alyassi_metric": "sr"}})

    def test_non_dict_config_rejected(self) -> None:
        """A non-dict config must raise ValueError."""
        with pytest.raises(ValueError, match="mapping"):
            _validate_metric_map({"success": "not_a_dict"})  # type: ignore[dict-item]

    def test_empty_metric_key_rejected(self) -> None:
        """An empty metric key must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_metric_map({"": {"classification": "comparable"}})


class TestValidatePlannerKeyMap:
    """Tests for _validate_planner_key_map validation."""

    def test_valid_planner_key_map(self) -> None:
        """A valid planner key map must not raise."""
        _validate_planner_key_map({"sf_planner": "social_force", "mpc_planner": "model_predictive"})

    def test_empty_key_rejected(self) -> None:
        """An empty key must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_planner_key_map({"": "value"})

    def test_empty_value_rejected(self) -> None:
        """An empty value must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_planner_key_map({"key": "  "})
