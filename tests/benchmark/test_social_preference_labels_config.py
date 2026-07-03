"""Tests for the diagnostic social preference label config contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.social_preference_labels import (
    REQUIRED_LABEL_IDS,
    SocialPreferenceLabelConfigError,
    label_availability,
    load_social_preference_label_config,
    validate_social_preference_label_config,
)

CONFIG_PATH = Path("configs/diagnostics/social_preference_labels.yaml")


@pytest.fixture
def valid_payload() -> dict:
    """Return the repository social preference label config."""

    return load_social_preference_label_config(CONFIG_PATH)


def test_social_preference_label_config_loads_with_required_labels(valid_payload: dict) -> None:
    """The checked-in config includes the seven issue #4228 labels."""

    label_ids = {label["id"] for label in valid_payload["labels"]}

    assert REQUIRED_LABEL_IDS.issubset(label_ids)
    assert len(label_ids) == len(valid_payload["labels"])
    assert "diagnostic" in valid_payload["claim_boundary"].casefold()
    assert "not a reward" in valid_payload["claim_boundary"].casefold()


def test_duplicate_label_ids_fail_closed(valid_payload: dict) -> None:
    """Duplicate label IDs are rejected."""

    payload = deepcopy(valid_payload)
    payload["labels"].append(deepcopy(payload["labels"][0]))

    with pytest.raises(SocialPreferenceLabelConfigError, match="duplicate label ids"):
        validate_social_preference_label_config(payload)


def test_invalid_preferred_direction_fails_closed(valid_payload: dict) -> None:
    """Preferred directions must come from the config enum."""

    payload = deepcopy(valid_payload)
    payload["labels"][0]["preferred_direction"] = "increase"

    with pytest.raises(SocialPreferenceLabelConfigError, match="preferred_direction"):
        validate_social_preference_label_config(payload)


def test_missing_claim_boundary_fails_closed(valid_payload: dict) -> None:
    """The config must keep an explicit diagnostic claim boundary."""

    payload = deepcopy(valid_payload)
    payload["claim_boundary"] = ""

    with pytest.raises(SocialPreferenceLabelConfigError, match="claim_boundary"):
        validate_social_preference_label_config(payload)


def test_missing_required_label_field_fails_closed(valid_payload: dict) -> None:
    """Labels must keep the fields needed by downstream checkers."""

    payload = deepcopy(valid_payload)
    del payload["labels"][0]["required_trace_fields"]

    with pytest.raises(SocialPreferenceLabelConfigError, match="missing required fields"):
        validate_social_preference_label_config(payload)


def test_threshold_status_must_remain_diagnostic(valid_payload: dict) -> None:
    """Thresholds cannot silently become calibrated preference claims."""

    payload = deepcopy(valid_payload)
    payload["labels"][0]["diagnostic_thresholds"]["status"] = "calibrated"

    with pytest.raises(SocialPreferenceLabelConfigError, match="threshold status"):
        validate_social_preference_label_config(payload)


def test_empty_candidate_metric_keys_require_not_available(valid_payload: dict) -> None:
    """Labels without metric keys must expose not_available status."""

    payload = deepcopy(valid_payload)
    path_blocking = next(label for label in payload["labels"] if label["id"] == "path_blocking")
    path_blocking["computation_status"] = "diagnostic_config_only"

    with pytest.raises(SocialPreferenceLabelConfigError, match="without candidate_metric_keys"):
        validate_social_preference_label_config(payload)


def test_label_availability_uses_not_available_for_missing_trace_fields(
    valid_payload: dict,
) -> None:
    """Missing trace fields produce not_available instead of a zero-like value."""

    clearance = next(label for label in valid_payload["labels"] if label["id"] == "clearance")

    assert label_availability(clearance, {"robot_trajectory"}) == "not_available"
    assert (
        label_availability(
            clearance,
            {
                "robot_trajectory",
                "pedestrian_trajectories",
                "metric_parameters.threshold_profile",
            },
        )
        == "diagnostic_available"
    )


def test_not_available_labels_stay_unavailable_even_with_fields(valid_payload: dict) -> None:
    """Labels marked unavailable do not become available from fields alone."""

    path_blocking = next(
        label for label in valid_payload["labels"] if label["id"] == "path_blocking"
    )

    assert (
        label_availability(path_blocking, set(path_blocking["required_trace_fields"]))
        == "not_available"
    )


def test_splc_source_is_motivation_only(valid_payload: dict) -> None:
    """SPLC literature references cannot claim implemented-method status."""

    roles = {entry["role"] for entry in valid_payload["source_literature"]}

    assert roles == {"motivation_only"}
    assert "implemented_method" not in roles
