"""Tests for robot_sf.benchmark.errors — custom benchmark exception types."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.errors import AggregationMetadataError, EpisodeRecordInputError
from robot_sf.errors import RobotSfError


class TestAggregationMetadataError:
    """Behavioral tests for AggregationMetadataError."""

    def test_inherits_robot_sf_error_and_value_error(self) -> None:
        """Must be catchable as both RobotSfError and ValueError."""
        exc = AggregationMetadataError("test")
        assert isinstance(exc, RobotSfError)
        assert isinstance(exc, ValueError)

    def test_message_preserved(self) -> None:
        """The message string must be preserved in str(exc)."""
        exc = AggregationMetadataError("missing algo field")
        assert str(exc) == "missing algo field"

    def test_episode_id_stored(self) -> None:
        """episode_id keyword must be stored and accessible."""
        exc = AggregationMetadataError("bad", episode_id="ep-42")
        assert exc.episode_id == "ep-42"

    def test_episode_id_defaults_to_none(self) -> None:
        """episode_id defaults to None when not provided."""
        exc = AggregationMetadataError("bad")
        assert exc.episode_id is None

    def test_missing_fields_stored_as_tuple(self) -> None:
        """missing_fields must be stored as a tuple."""
        exc = AggregationMetadataError("bad", missing_fields=["algo", "seed"])
        assert exc.missing_fields == ("algo", "seed")

    def test_missing_fields_defaults_to_empty_tuple(self) -> None:
        """missing_fields defaults to empty tuple when not provided."""
        exc = AggregationMetadataError("bad")
        assert exc.missing_fields == ()

    def test_advice_stored(self) -> None:
        """advice keyword must be stored and accessible."""
        exc = AggregationMetadataError("bad", advice="re-run with --algo")
        assert exc.advice == "re-run with --algo"

    def test_to_dict_minimal(self) -> None:
        """to_dict with only message returns just the message key."""
        exc = AggregationMetadataError("something broke")
        payload = exc.to_dict()
        assert payload == {"message": "something broke"}

    def test_to_dict_full(self) -> None:
        """to_dict with all fields returns the complete structured payload."""
        exc = AggregationMetadataError(
            "missing fields",
            episode_id="ep-7",
            missing_fields=["algo", "git_hash"],
            advice="check episode record",
        )
        payload = exc.to_dict()
        assert payload["message"] == "missing fields"
        assert payload["episode_id"] == "ep-7"
        assert payload["missing_fields"] == ["algo", "git_hash"]
        assert payload["advice"] == "check episode record"

    def test_to_dict_omits_none_episode_id(self) -> None:
        """to_dict must not include episode_id when it is None."""
        exc = AggregationMetadataError("msg", missing_fields=["x"])
        payload = exc.to_dict()
        assert "episode_id" not in payload

    def test_to_dict_omits_empty_missing_fields(self) -> None:
        """to_dict must not include missing_fields when empty."""
        exc = AggregationMetadataError("msg", episode_id="ep-1")
        payload = exc.to_dict()
        assert "missing_fields" not in payload

    def test_catchable_as_value_error(self) -> None:
        """Existing except ValueError clauses must still catch this error."""
        with pytest.raises(ValueError):
            raise AggregationMetadataError("legacy catch")


class TestEpisodeRecordInputError:
    """Behavioral tests for EpisodeRecordInputError."""

    def test_inherits_robot_sf_error_and_value_error(self) -> None:
        """Must be catchable as both RobotSfError and ValueError."""
        exc = EpisodeRecordInputError("bad input")
        assert isinstance(exc, RobotSfError)
        assert isinstance(exc, ValueError)

    def test_message_preserved(self) -> None:
        """The message string must be preserved."""
        exc = EpisodeRecordInputError("malformed JSONL")
        assert str(exc) == "malformed JSONL"

    def test_catchable_as_value_error(self) -> None:
        """Existing except ValueError clauses must still catch this error."""
        with pytest.raises(ValueError):
            raise EpisodeRecordInputError("legacy catch")
