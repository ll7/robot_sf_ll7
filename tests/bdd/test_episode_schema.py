"""Step definitions for episode schema validation feature."""

from __future__ import annotations

from robot_sf.benchmark.schema_loader import load_schema
from robot_sf.benchmark.schema_validator import validate_episode

SCHEMA = load_schema("episode.schema.v1.json")


def _valid_episode() -> dict:
    return {
        "version": "v1",
        "episode_id": "test-episode-001",
        "scenario_id": "test-scenario",
        "seed": 42,
        "termination_reason": "success",
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
        "metrics": {"collisions": 0, "success_rate": 1.0},
        "integrity": {"contradictions": []},
    }


from pytest_bdd import given, scenarios, then, when  # noqa: E402

scenarios("episode_schema.feature")


@given("a valid minimal episode record", target_fixture="episode_record")
def given_valid_episode() -> dict:
    return _valid_episode()


@given("an episode record missing a required field", target_fixture="episode_record")
def given_invalid_episode() -> dict:
    rec = _valid_episode()
    del rec["termination_reason"]
    return rec


@when(
    "the record is validated against the episode schema",
    target_fixture="validation_error",
)
def when_validate(episode_record: dict) -> Exception | None:
    try:
        validate_episode(episode_record, SCHEMA)
        return None
    except Exception as exc:
        return exc


@then("no validation error should be raised")
def then_no_error(validation_error: Exception | None) -> None:
    assert validation_error is None


@then("a validation error should be raised")
def then_error_raised(validation_error: Exception | None) -> None:
    assert validation_error is not None
