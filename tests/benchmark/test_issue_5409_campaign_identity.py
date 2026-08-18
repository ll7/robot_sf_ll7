"""Tests for the shared issue #5409 campaign-identity contract."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.issue_5409_campaign_identity import (
    DEFAULT_CAMPAIGN_ID_PAIR,
    CampaignIdentityError,
    CampaignIdPair,
)


def test_default_pair_round_trips_through_provenance_payload() -> None:
    """The canonical pair has one stable, versioned representation."""
    assert CampaignIdPair.from_payload(DEFAULT_CAMPAIGN_ID_PAIR.to_payload()) == (
        DEFAULT_CAMPAIGN_ID_PAIR
    )


def test_reviewed_pair_binds_both_roles_to_one_rerun() -> None:
    """A reviewed suffix is accepted only when both role IDs share its identity shape."""
    pair = CampaignIdPair.from_values(
        (
            "issue5409_horizon_ablation_rerun1_h500_20260818",
            "issue5409_horizon_ablation_rerun1_h600_20260818",
        )
    )

    assert pair.for_role("h500").endswith("h500_20260818")
    assert pair.for_role("h600").endswith("h600_20260818")


@pytest.mark.parametrize(
    "values",
    [
        ("", "issue5409_horizon_ablation_h600"),
        ("issue5409_horizon_ablation_h500", "issue5409_horizon_ablation_h500"),
        ("issue5409_horizon_ablation_h600", "issue5409_horizon_ablation_h500"),
        (
            "issue5409_horizon_ablation_rerun1_h500",
            "issue5409_horizon_ablation_rerun2_h600",
        ),
        ("issue5409_horizon_ablation_h500/extra", "issue5409_horizon_ablation_h600"),
    ],
)
def test_malformed_or_mismatched_pairs_fail_closed(values: tuple[str, str]) -> None:
    """Empty, duplicated, swapped, unrelated, and malformed IDs are not inferred."""
    with pytest.raises(CampaignIdentityError):
        CampaignIdPair.from_values(values)


def test_unknown_packet_schema_fails_closed() -> None:
    """A packet cannot opt into identity semantics without the versioned schema."""
    payload = DEFAULT_CAMPAIGN_ID_PAIR.to_payload()
    payload["schema_version"] = "issue-5409-campaign-id-pair.v2"

    with pytest.raises(CampaignIdentityError, match="schema"):
        CampaignIdPair.from_payload(payload)
