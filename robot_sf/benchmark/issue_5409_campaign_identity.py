"""Shared, versioned campaign-identity contract for issue #5409."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CAMPAIGN_IDENTITY_SCHEMA = "issue-5409-campaign-id-pair.v1"
CAMPAIGN_FAMILY = "issue5409_horizon_ablation"
CAMPAIGN_ROLES = ("h500", "h600")


class CampaignIdentityError(ValueError):
    """Raised when a #5409 campaign-ID pair is malformed or inconsistent."""


def _validate_id(value: object, *, role: str) -> str:
    """Validate one explicit campaign ID without inferring an accepted suffix.

    Returns:
        The validated campaign ID.
    """
    if not isinstance(value, str) or not value:
        raise CampaignIdentityError(f"{role} campaign ID must be a non-empty string")
    if value != value.strip() or any(character.isspace() for character in value):
        raise CampaignIdentityError(f"{role} campaign ID must not contain whitespace")

    parts = value.split("_")
    if len(parts) < 4 or tuple(parts[:3]) != ("issue5409", "horizon", "ablation"):
        raise CampaignIdentityError(f"{role} campaign ID must belong to {CAMPAIGN_FAMILY!r}")
    if any(
        not part or any(not (character.isalnum() or character in ".-") for character in part)
        for part in parts
    ):
        raise CampaignIdentityError(
            f"{role} campaign ID contains an empty or unsupported identity segment"
        )

    other_role = "h600" if role == "h500" else "h500"
    if parts.count(role) != 1 or other_role in parts:
        raise CampaignIdentityError(f"{role} campaign ID must contain exactly its own role marker")
    return value


def _pair_shape(value: str, *, role: str) -> tuple[str, ...]:
    """Return a role-neutral shape used to bind both IDs to one rerun."""
    return tuple("{role}" if part == role else part for part in value.split("_"))


@dataclass(frozen=True)
class CampaignIdPair:
    """An explicit h500/h600 identity pair declared by a launch packet."""

    h500: str
    h600: str

    def __post_init__(self) -> None:
        """Reject malformed, duplicated, swapped, or unrelated arm identities."""
        if self.h500 == self.h600:
            raise CampaignIdentityError("h500 and h600 campaign IDs must be distinct")
        h500 = _validate_id(self.h500, role="h500")
        h600 = _validate_id(self.h600, role="h600")
        if _pair_shape(h500, role="h500") != _pair_shape(h600, role="h600"):
            raise CampaignIdentityError(
                "h500 and h600 campaign IDs must describe the same declared rerun"
            )

    @classmethod
    def from_values(cls, values: Sequence[object]) -> CampaignIdPair:
        """Build a pair from exactly two ordered h500/h600 values.

        Returns:
            The validated campaign-ID pair.
        """
        if isinstance(values, (str, bytes)) or len(values) != len(CAMPAIGN_ROLES):
            raise CampaignIdentityError("campaign ID pair must contain exactly h500 and h600")
        return cls(values[0], values[1])  # type: ignore[arg-type]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CampaignIdPair:
        """Parse the versioned pair payload stored in a launch packet.

        Returns:
            The validated campaign-ID pair.
        """
        if payload.get("schema_version") != CAMPAIGN_IDENTITY_SCHEMA:
            raise CampaignIdentityError(
                f"campaign identity schema must be {CAMPAIGN_IDENTITY_SCHEMA!r}"
            )
        if payload.get("campaign_family") != CAMPAIGN_FAMILY:
            raise CampaignIdentityError(f"campaign identity family must be {CAMPAIGN_FAMILY!r}")
        ids = payload.get("ids")
        if not isinstance(ids, Mapping) or set(ids) != set(CAMPAIGN_ROLES):
            raise CampaignIdentityError("campaign identity ids must declare only h500 and h600")
        return cls(ids["h500"], ids["h600"])  # type: ignore[arg-type]

    def for_role(self, role: str) -> str:
        """Return the declared ID for one known arm role."""
        if role not in CAMPAIGN_ROLES:
            raise CampaignIdentityError(f"unsupported #5409 campaign role: {role!r}")
        return self.h500 if role == "h500" else self.h600

    def as_tuple(self) -> tuple[str, str]:
        """Return the ordered h500/h600 IDs used by the report builder."""
        return self.h500, self.h600

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical provenance payload for packet and report artifacts."""
        return {
            "schema_version": CAMPAIGN_IDENTITY_SCHEMA,
            "campaign_family": CAMPAIGN_FAMILY,
            "ids": {"h500": self.h500, "h600": self.h600},
        }


DEFAULT_CAMPAIGN_ID_PAIR = CampaignIdPair(
    "issue5409_horizon_ablation_h500",
    "issue5409_horizon_ablation_h600",
)


def campaign_identity_from_packet(
    packet: Mapping[str, Any], *, allow_legacy_default: bool = False
) -> CampaignIdPair:
    """Read a packet pair, preserving v1's fixed canonical-ID semantics.

    Returns:
        The packet-declared pair, or the canonical pair for a legacy packet.
    """
    payload = packet.get("campaign_identity")
    if payload is None:
        if allow_legacy_default:
            return DEFAULT_CAMPAIGN_ID_PAIR
        raise CampaignIdentityError("launch packet must declare campaign_identity")
    if not isinstance(payload, Mapping):
        raise CampaignIdentityError("launch packet campaign_identity must be a mapping")
    return CampaignIdPair.from_payload(payload)
