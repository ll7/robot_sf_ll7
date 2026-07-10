"""Compatibility tests for the research exception hierarchy."""

from __future__ import annotations

import pytest

from robot_sf.errors import RobotSfError
from robot_sf.research.exceptions import (
    AggregationError,
    FigureGenerationError,
    ReportGenerationError,
    ResearchError,
    StatisticalTestError,
    ValidationError,
)

RESEARCH_ERROR_TYPES = (
    ResearchError,
    ReportGenerationError,
    ValidationError,
    AggregationError,
    StatisticalTestError,
    FigureGenerationError,
)


@pytest.mark.parametrize("error_type", RESEARCH_ERROR_TYPES)
def test_research_errors_keep_legacy_and_gain_shared_catches(
    error_type: type[ResearchError],
) -> None:
    """Every public research error remains catchable through old and new bases."""
    error = error_type("test error")

    assert isinstance(error, error_type)
    assert isinstance(error, ResearchError)
    assert isinstance(error, Exception)
    assert isinstance(error, RobotSfError)

    with pytest.raises(error_type):
        raise error_type("specific catch")
    with pytest.raises(ResearchError):
        raise error_type("legacy family catch")
    with pytest.raises(Exception):
        raise error_type("legacy builtin catch")
    with pytest.raises(RobotSfError):
        raise error_type("shared package catch")


@pytest.mark.parametrize("error_type", RESEARCH_ERROR_TYPES)
def test_research_error_mro_preserves_legacy_ancestry(
    error_type: type[ResearchError],
) -> None:
    """The shared base is inserted without removing or reordering legacy bases."""
    mro = error_type.__mro__

    assert mro.index(ResearchError) < mro.index(RobotSfError) < mro.index(Exception)
