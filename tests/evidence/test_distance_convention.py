"""Tests for the explicit distance-convention vocabulary (issue #5141).

Covers:
- the ``DistanceConvention`` enum and metadata field name,
- ``validate_distance_convention`` / ``require_distance_convention`` fail-closed paths,
- ``write_distance_series_csv`` annotates the CSV with the convention header,
- filename/column distance-like detection used by the lint.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import pytest

from robot_sf.evidence import writers
from robot_sf.evidence.distance_convention import (
    CONVENTION_DESCRIPTIONS,
    DISTANCE_CONVENTION_FIELD,
    DistanceConvention,
    describe_distance_convention,
    has_distance_like_columns,
    is_distance_like_filename,
    require_distance_convention,
    validate_distance_convention,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestDistanceConventionEnum:
    """The enum carries exactly the three conventions named in issue #5141."""

    def test_has_three_expected_values(self) -> None:
        values = {c.value for c in DistanceConvention}
        assert values == {"center_center", "surface_clearance", "center_segment"}

    def test_str_enum_serializes_to_plain_string(self) -> None:
        # str-Enum: the value round-trips as a plain JSON-serializable string.
        assert DistanceConvention.CENTER_CENTER.value == "center_center"
        assert DistanceConvention("center_center") is DistanceConvention.CENTER_CENTER


class TestValidateDistanceConvention:
    """Validation accepts enum/string and rejects unknown or mistyped values."""

    def test_accepts_enum_and_string(self) -> None:
        assert validate_distance_convention("center_center") is DistanceConvention.CENTER_CENTER
        assert (
            validate_distance_convention(DistanceConvention.SURFACE_CLEARANCE)
            is DistanceConvention.SURFACE_CLEARANCE
        )

    @pytest.mark.parametrize("bad", ["center", "center-to-center", "clearance", ""])
    def test_rejects_unknown_string_value(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Unknown distance_convention"):
            validate_distance_convention(bad)

    @pytest.mark.parametrize("bad", [1, 1.5, None, ["center_center"]])
    def test_rejects_non_string_non_enum_value(self, bad: object) -> None:
        with pytest.raises(ValueError, match="distance_convention"):
            validate_distance_convention(bad)


class TestRequireDistanceConvention:
    """The fail-closed helper demands a valid convention on distance-like metadata."""

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="missing the required 'distance_convention'"):
            require_distance_convention({}, "min_distance_series.csv")

    def test_returns_resolved_convention_when_present(self) -> None:
        resolved = require_distance_convention(
            {DISTANCE_CONVENTION_FIELD: "center_center"}, "min_distance_series.csv"
        )
        assert resolved is DistanceConvention.CENTER_CENTER

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            require_distance_convention(
                {DISTANCE_CONVENTION_FIELD: "nope"}, "min_distance_series.csv"
            )


class TestConventionDescriptions:
    """Every convention has a human-readable description for docs/lint messages."""

    def test_all_conventions_described(self) -> None:
        for convention in DistanceConvention:
            assert convention in CONVENTION_DESCRIPTIONS
            assert len(describe_distance_convention(convention)) > 0

    def test_describe_accepts_string(self) -> None:
        assert "surface" in describe_distance_convention("surface_clearance").lower()


class TestDistanceLikeDetection:
    """Filename/column heuristics identify distance-like series for the lint."""

    @pytest.mark.parametrize(
        "name",
        [
            "min_distance_series.csv",
            "robot_clearance_series.csv",
            "MIN_DISTANCE_SERIES.CSV",
            "head_on_clearance.csv",
        ],
    )
    def test_distance_like_filenames(self, name: str) -> None:
        assert is_distance_like_filename(name) is True

    @pytest.mark.parametrize("name", ["metadata.json", "trace_timeseries.csv", "README.md"])
    def test_non_distance_filenames(self, name: str) -> None:
        assert is_distance_like_filename(name) is False

    def test_distance_like_columns(self) -> None:
        assert has_distance_like_columns("step,time_s,min_robot_ped_distance_m") is True
        assert has_distance_like_columns("step,min_clearance") is True

    def test_non_distance_columns(self) -> None:
        assert has_distance_like_columns("step,time_s,speed_m_s") is False


class TestWriteDistanceSeriesCsv:
    """The shared writer annotates the CSV with the convention header."""

    """The shared writer annotates the CSV with the convention header."""

    def test_writes_convention_header_line(self, tmp_path: Path) -> None:
        path = tmp_path / "min_distance_series.csv"
        writers.write_distance_series_csv(
            path,
            [{"step": 0, "min_robot_ped_distance_m": 1.37}],
            convention="center_center",
        )
        content = path.read_text(encoding="utf-8")
        assert "# AI-GENERATED NEEDS-REVIEW" in content
        assert "# distance_convention: center_center" in content

    def test_header_line_precedes_csv_header(self, tmp_path: Path) -> None:
        path = tmp_path / "min_distance_series.csv"
        writers.write_distance_series_csv(
            path,
            [{"step": 0, "min_robot_ped_distance_m": 1.37}],
            convention=DistanceConvention.CENTER_CENTER,
        )
        with path.open(encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        # Lines 1-2 are marker + convention; line 3 is the CSV header.
        assert lines[0] == "# AI-GENERATED NEEDS-REVIEW"
        assert lines[1] == "# distance_convention: center_center"
        assert "step" in lines[2]

    def test_rejects_invalid_convention(self, tmp_path: Path) -> None:
        path = tmp_path / "min_distance_series.csv"
        with pytest.raises(ValueError, match="Unknown distance_convention"):
            writers.write_distance_series_csv(
                path,
                [{"step": 0, "min_robot_ped_distance_m": 1.37}],
                convention="clearance",
            )

    def test_rows_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "min_distance_series.csv"
        writers.write_distance_series_csv(
            path,
            [{"step": 0, "min_robot_ped_distance_m": 1.37}],
            convention="center_center",
        )
        # Skip the marker/convention comment lines before CSV parsing.
        with path.open(encoding="utf-8") as handle:
            data_lines = [line for line in handle if not line.lstrip().startswith("#")]
        rows = list(csv.DictReader(data_lines))
        assert rows == [{"step": "0", "min_robot_ped_distance_m": "1.37"}]
