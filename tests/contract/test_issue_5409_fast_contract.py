"""Fast-lane contract coverage for the issue #5409 identity and preflight guards."""

from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec, SeedPolicy
from robot_sf.benchmark.campaign.campaign_checkpoint_preflight import (
    check_campaign_arm_checkpoints_preflight,
)
from robot_sf.benchmark.issue_5409_campaign_identity import (
    CAMPAIGN_FAMILY,
    CAMPAIGN_IDENTITY_SCHEMA,
    DEFAULT_CAMPAIGN_ID_PAIR,
    CampaignIdentityError,
    CampaignIdPair,
    campaign_identity_from_packet,
)


def test_identity_rejects_whitespace_wrong_family_and_duplicates() -> None:
    """Malformed IDs fail before any launch/report identity can be inferred."""
    with pytest.raises(CampaignIdentityError, match="non-empty"):
        CampaignIdPair("", DEFAULT_CAMPAIGN_ID_PAIR.h600)
    with pytest.raises(CampaignIdentityError, match="whitespace"):
        CampaignIdPair("issue5409_horizon_ablation_h500 ", "issue5409_horizon_ablation_h600")
    with pytest.raises(CampaignIdentityError, match="belong"):
        CampaignIdPair("other_horizon_ablation_h500", "other_horizon_ablation_h600")
    with pytest.raises(CampaignIdentityError, match="unsupported identity"):
        CampaignIdPair(
            "issue5409_horizon_ablation_h500!",
            DEFAULT_CAMPAIGN_ID_PAIR.h600,
        )
    with pytest.raises(CampaignIdentityError, match="own role marker"):
        CampaignIdPair(
            "issue5409_horizon_ablation_h500_h600",
            DEFAULT_CAMPAIGN_ID_PAIR.h600,
        )
    with pytest.raises(CampaignIdentityError, match="distinct"):
        CampaignIdPair(DEFAULT_CAMPAIGN_ID_PAIR.h500, DEFAULT_CAMPAIGN_ID_PAIR.h500)

    with pytest.raises(CampaignIdentityError, match="same declared rerun"):
        CampaignIdPair(
            "issue5409_horizon_ablation_h500_variant-a",
            DEFAULT_CAMPAIGN_ID_PAIR.h600,
        )


def test_identity_rejects_wrong_value_count_and_payload_shape() -> None:
    """Pair length, schema family, and declared key shape are strict."""
    with pytest.raises(CampaignIdentityError, match="exactly"):
        CampaignIdPair.from_values((DEFAULT_CAMPAIGN_ID_PAIR.h500,))

    payload = DEFAULT_CAMPAIGN_ID_PAIR.to_payload()
    payload["schema_version"] = "issue-5409-campaign-id-pair.v2"
    with pytest.raises(CampaignIdentityError, match="schema"):
        CampaignIdPair.from_payload(payload)

    payload = DEFAULT_CAMPAIGN_ID_PAIR.to_payload()
    payload["campaign_family"] = "other_family"
    with pytest.raises(CampaignIdentityError, match=CAMPAIGN_FAMILY):
        CampaignIdPair.from_payload(payload)

    payload = DEFAULT_CAMPAIGN_ID_PAIR.to_payload()
    payload["ids"] = {"h500": DEFAULT_CAMPAIGN_ID_PAIR.h500}
    with pytest.raises(CampaignIdentityError, match="only h500 and h600"):
        CampaignIdPair.from_payload(payload)


def test_identity_rejects_unknown_role_and_legacy_packet_without_opt_in() -> None:
    """Role lookup and legacy packet handling remain explicit and fail closed."""
    with pytest.raises(CampaignIdentityError, match="unsupported"):
        DEFAULT_CAMPAIGN_ID_PAIR.for_role("h700")
    assert DEFAULT_CAMPAIGN_ID_PAIR.for_role("h600") == DEFAULT_CAMPAIGN_ID_PAIR.h600
    assert campaign_identity_from_packet({}, allow_legacy_default=True) == DEFAULT_CAMPAIGN_ID_PAIR
    with pytest.raises(CampaignIdentityError, match="must declare"):
        campaign_identity_from_packet({})
    with pytest.raises(CampaignIdentityError, match="must be a mapping"):
        campaign_identity_from_packet({"campaign_identity": []})


def test_identity_payload_round_trip_uses_versioned_schema() -> None:
    """The canonical pair retains its versioned payload contract."""
    payload = DEFAULT_CAMPAIGN_ID_PAIR.to_payload()
    assert payload["schema_version"] == CAMPAIGN_IDENTITY_SCHEMA
    assert CampaignIdPair.from_payload(payload) == DEFAULT_CAMPAIGN_ID_PAIR
    assert CampaignIdPair.from_values(DEFAULT_CAMPAIGN_ID_PAIR.as_tuple()) == (
        DEFAULT_CAMPAIGN_ID_PAIR
    )
    assert DEFAULT_CAMPAIGN_ID_PAIR.as_tuple() == (
        DEFAULT_CAMPAIGN_ID_PAIR.h500,
        DEFAULT_CAMPAIGN_ID_PAIR.h600,
    )
    assert campaign_identity_from_packet({"campaign_identity": payload}) == (
        DEFAULT_CAMPAIGN_ID_PAIR
    )


def test_empty_campaign_preflight_is_not_submit_safe() -> None:
    """A campaign with no checkpoint-bearing arms cannot authorize submission."""
    config = CampaignConfig(
        name="issue_5409_empty_preflight",
        scenario_matrix_path=Path("unused-scenarios.yaml"),
        planners=(),
        seed_policy=SeedPolicy(),
    )

    summary = check_campaign_arm_checkpoints_preflight(config)

    assert summary["checked"] == 0
    assert summary["resolved"] == 0
    assert summary["submit_safe"] is False


def test_present_local_checkpoint_is_submit_safe(tmp_path: Path) -> None:
    """A non-empty preflight exercises the explicit submit-safe resolution guard."""
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"fixture")
    algo_config = tmp_path / "algo.yaml"
    algo_config.write_text(f"model_path: {checkpoint}\n", encoding="utf-8")
    config = CampaignConfig(
        name="issue_5409_present_local_preflight",
        scenario_matrix_path=tmp_path / "unused-scenarios.yaml",
        planners=(PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        seed_policy=SeedPolicy(),
    )

    summary = check_campaign_arm_checkpoints_preflight(config)

    assert summary["checked"] == 1
    assert summary["resolved"] == 1
    assert summary["submit_safe"] is True
