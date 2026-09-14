"""Round-trip contracts for the canonical lane markers (issue #9254).

Every known marker must survive a format -> parse round trip through the single
implementation both producers and consumers share, and malformed markers must
fail closed.
"""

from __future__ import annotations

import pytest

from scripts.dev import lane_markers
from scripts.dev.lane_markers import (
    ReviewClaim,
    ShaCarrier,
    extract_metadata_digests,
    extract_sha_carriers,
    format_base_policy,
    format_gate_verdict,
    format_pr_metadata,
    format_review_claim,
    format_review_claim_release,
    has_merge_ready_label,
    invalid_sha_carriers,
    parse_review_claim,
    review_claim_released_shas,
)

_HEAD = "9" * 40
_OTHER_HEAD = "8" * 40


def test_review_claim_round_trip() -> None:
    """A formatted claim must parse back to its lane, SHA, and expiry."""
    marker = format_review_claim("my-lane", _HEAD, "2026-09-14T12:00:00Z")
    assert marker == f"review-claim: my-lane @ {_HEAD} until 2026-09-14T12:00:00Z"
    claim = parse_review_claim(marker)
    assert isinstance(claim, ReviewClaim)
    assert claim.lane == "my-lane"
    assert claim.sha == _HEAD
    assert claim.expires_at is not None
    assert claim.expires_at.isoformat() == "2026-09-14T12:00:00+00:00"


def test_review_claim_release_round_trip() -> None:
    """A formatted release must surface its SHA in the released set."""
    marker = format_review_claim_release(_HEAD)
    assert marker == f"review-claim: released @ {_HEAD}"
    assert review_claim_released_shas(marker) == {_HEAD}
    assert review_claim_released_shas("no markers here") == set()


def test_review_claim_rejects_blank_lane_and_bad_sha() -> None:
    """Malformed producers must fail fast instead of emitting skew (issue #9254)."""
    with pytest.raises(ValueError):
        format_review_claim("", _HEAD, "2026-09-14T12:00:00Z")
    with pytest.raises(ValueError):
        format_review_claim("lane with spaces", _HEAD, "2026-09-14T12:00:00Z")
    with pytest.raises(ValueError):
        format_review_claim("lane", "not-hex", "2026-09-14T12:00:00Z")
    with pytest.raises(ValueError):
        format_review_claim_release("xyz")
    assert parse_review_claim("review-claim: lane @ abc until not-a-date") is None
    assert parse_review_claim("") is None


def test_gate_verdict_round_trip() -> None:
    """Both verdicts must format and extract with full-SHA admission."""
    for verdict in ("accepted", "hold"):
        marker = format_gate_verdict(verdict, _HEAD)
        carriers = extract_sha_carriers(marker)
        assert len(carriers) == 1
        carrier = carriers[0]
        assert isinstance(carrier, ShaCarrier)
        assert (carrier.kind, carrier.sha, carrier.full) == ("gate-verdict", _HEAD, True)
        assert invalid_sha_carriers(carriers, _HEAD) == []
        assert len(invalid_sha_carriers(carriers, _OTHER_HEAD)) == 1
    with pytest.raises(ValueError):
        format_gate_verdict("maybe", _HEAD)


def test_base_policy_round_trip() -> None:
    """Both policies must format and extract (issue #9254)."""
    for policy in ("ordinary-cas", "current-base"):
        marker = format_base_policy(policy, _HEAD)
        carriers = extract_sha_carriers(marker)
        assert [(c.kind, c.sha) for c in carriers] == [("base-policy", _HEAD)]
    with pytest.raises(ValueError):
        format_base_policy("stale-base", _HEAD)


def test_pr_metadata_round_trip() -> None:
    """The metadata trailer must format and extract its digest (issue #9254)."""
    digest = "a" * 64
    marker = format_pr_metadata(digest)
    assert marker == f"pr-metadata: reconciled @ {digest}"
    assert extract_metadata_digests(marker) == [digest]
    assert lane_markers.metadata_trailer(digest) == marker
    with pytest.raises(ValueError):
        format_pr_metadata("too-short")


def test_merge_ready_label_helper() -> None:
    """The merge-ready label check must match the canonical spelling only."""
    assert has_merge_ready_label(["review-bot-auto", "merge-ready"])
    assert not has_merge_ready_label(["review-bot-auto"])
    assert not has_merge_ready_label("merge-ready")
    assert not has_merge_ready_label([])


def test_legacy_import_paths_still_resolve() -> None:
    """The consolidation must not move consumer imports (issue #9254)."""
    from scripts.dev import pr_loop_policy, pr_metadata

    assert pr_loop_policy.extract_sha_carriers is lane_markers.extract_sha_carriers
    assert pr_loop_policy.REVIEW_CLAIM_RE is lane_markers.REVIEW_CLAIM_RE
    assert pr_metadata.metadata_trailer is lane_markers.metadata_trailer
    assert pr_metadata.extract_metadata_digests is lane_markers.extract_metadata_digests
