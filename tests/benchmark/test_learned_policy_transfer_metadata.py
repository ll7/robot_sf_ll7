"""Tests for learned-policy transfer benchmark metadata validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.learned_policy_transfer_metadata import (
    load_transfer_metadata_schema,
    validate_transfer_metadata,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "learned_policy_transfer_benchmark" / "v1"
)


def _load_fixture(name: str) -> dict[str, object]:
    """Load a JSON fixture by basename."""
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_transfer_metadata_schema_loads() -> None:
    """The v1 transfer metadata schema should be a JSON Schema object."""
    schema = load_transfer_metadata_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == (
        "learned_policy_transfer_benchmark.v1"
    )


def test_native_ppo_fixture_validates_as_success_capable() -> None:
    """The native PPO fixture should validate only with success-capable statuses."""
    payload = _load_fixture("ppo_issue791_best_v1.json")

    validate_transfer_metadata(payload)

    payload["availability_status"] = "not_available"
    with pytest.raises(ValueError, match="benchmark_success=true"):
        validate_transfer_metadata(payload)


def test_crowdnav_height_fixture_validates_as_blocked_non_success() -> None:
    """The CrowdNav HEIGHT source fixture should validate as blocked, not successful."""
    payload = _load_fixture("crowdnav_height_blocked_source.json")

    validate_transfer_metadata(payload)

    assert payload["availability_status"] == "not_available"
    assert payload["benchmark_success"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_mode", "mixed"),
        ("execution_mode", "unknown"),
        ("readiness_status", "fallback"),
        ("readiness_status", "degraded"),
        ("availability_status", "not_available"),
        ("availability_status", "failed"),
        ("availability_status", "partial-failure"),
    ],
)
def test_fail_closed_statuses_cannot_claim_benchmark_success(field: str, value: str) -> None:
    """Fallback, degraded, failed, partial, and unavailable rows cannot be success rows."""
    payload = _load_fixture("ppo_issue791_best_v1.json")
    payload[field] = value
    payload["benchmark_success"] = True

    with pytest.raises(ValueError, match="benchmark_success=true"):
        validate_transfer_metadata(payload)


def test_required_transfer_fields_are_enforced() -> None:
    """Missing required transfer fields should fail with an actionable path."""
    payload = _load_fixture("ppo_issue791_best_v1.json")
    del payload["artifact_provenance"]["artifact_manifest_status"]

    with pytest.raises(ValueError, match="artifact_provenance.artifact_manifest_status"):
        validate_transfer_metadata(payload)


def test_episode_algorithm_metadata_attachment_is_validated() -> None:
    """The validator should accept an object attached under algorithm_metadata.transfer_benchmark."""
    episode_row = {
        "algorithm_metadata": {
            "transfer_benchmark": _load_fixture("crowdnav_height_blocked_source.json"),
        }
    }

    validate_transfer_metadata(episode_row["algorithm_metadata"]["transfer_benchmark"])


def test_success_requires_smoke_or_benchmark_stage() -> None:
    """Source-side metadata cannot become benchmark success without Robot SF execution stage."""
    payload = copy.deepcopy(_load_fixture("ppo_issue791_best_v1.json"))
    payload["transfer_stage"] = "source_reproduction"

    with pytest.raises(ValueError, match="transfer_stage"):
        validate_transfer_metadata(payload)


def test_success_rejects_unresolved_contract_placeholders() -> None:
    """Success rows should not hide unresolved action or observation contracts behind enums."""
    payload = copy.deepcopy(_load_fixture("ppo_issue791_best_v1.json"))
    payload["action_contract"]["action_bounds"] = "unknown"

    with pytest.raises(ValueError, match="action_contract.action_bounds"):
        validate_transfer_metadata(payload)
