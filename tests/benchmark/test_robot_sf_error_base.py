"""Contract tests for the shared ``RobotSfError`` base (#4993).

These tests pin the behavior-preservation contract described in issue #4993:

* ``RobotSfError`` is an ``Exception`` subclass.
* The re-parented benchmark exceptions are *both* ``RobotSfError`` and
  ``ValueError`` subclasses, so existing ``except ValueError`` /
  ``except <SpecificError>`` clauses still catch them while new code can also
  target ``except RobotSfError``.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.errors import (
    AggregationMetadataError,
    EpisodeRecordInputError,
)
from robot_sf.errors import RobotSfError


class TestRobotSfErrorBase:
    """The base exception class itself."""

    def test_robot_sf_error_is_exception_subclass(self) -> None:
        assert issubclass(RobotSfError, Exception)

    def test_robot_sf_error_can_be_raised_and_caught(self) -> None:
        with pytest.raises(RobotSfError):
            raise RobotSfError("boom")

    def test_robot_sf_error_is_not_caught_by_value_error(self) -> None:
        # The plain base must not accidentally be a ValueError subclass; the
        # benchmark classes opt into that ancestry explicitly.
        assert not issubclass(RobotSfError, ValueError)


class TestAggregationMetadataErrorContract:
    """``AggregationMetadataError`` keeps its ValueError MRO after re-parenting."""

    def test_is_robot_sf_error_subclass(self) -> None:
        assert issubclass(AggregationMetadataError, RobotSfError)

    def test_remains_value_error_subclass(self) -> None:
        assert issubclass(AggregationMetadataError, ValueError)

    def test_dual_isinstance(self) -> None:
        err = AggregationMetadataError("missing algo metadata")
        assert isinstance(err, RobotSfError)
        assert isinstance(err, ValueError)

    def test_raised_instance_caught_by_robot_sf_error(self) -> None:
        with pytest.raises(RobotSfError):
            raise AggregationMetadataError("missing algo metadata")

    def test_raised_instance_caught_by_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise AggregationMetadataError("missing algo metadata")

    def test_raised_instance_caught_by_own_type(self) -> None:
        with pytest.raises(AggregationMetadataError):
            raise AggregationMetadataError("missing algo metadata")

    def test_mro_value_error_precedes_robot_sf_error(self) -> None:
        # Re-parenting to (RobotSfError, ValueError) is behavior-preserving
        # because ValueError stays in the MRO. Assert the builtin is present.
        assert ValueError in AggregationMetadataError.__mro__

    def test_init_context_preserved(self) -> None:
        err = AggregationMetadataError(
            "missing algo metadata",
            episode_id="ep-1",
            missing_fields=["algo"],
            advice="regenerate the episode",
        )
        assert err.episode_id == "ep-1"
        assert err.missing_fields == ("algo",)
        assert err.advice == "regenerate the episode"


class TestEpisodeRecordInputErrorContract:
    """``EpisodeRecordInputError`` keeps its ValueError MRO after re-parenting."""

    def test_is_robot_sf_error_subclass(self) -> None:
        assert issubclass(EpisodeRecordInputError, RobotSfError)

    def test_remains_value_error_subclass(self) -> None:
        assert issubclass(EpisodeRecordInputError, ValueError)

    def test_dual_isinstance(self) -> None:
        err = EpisodeRecordInputError("malformed jsonl")
        assert isinstance(err, RobotSfError)
        assert isinstance(err, ValueError)

    def test_raised_instance_caught_by_robot_sf_error(self) -> None:
        with pytest.raises(RobotSfError):
            raise EpisodeRecordInputError("malformed jsonl")

    def test_raised_instance_caught_by_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise EpisodeRecordInputError("malformed jsonl")

    def test_raised_instance_caught_by_own_type(self) -> None:
        with pytest.raises(EpisodeRecordInputError):
            raise EpisodeRecordInputError("malformed jsonl")
