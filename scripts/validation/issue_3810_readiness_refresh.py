"""Shared validation for issue #3810 readiness-refresh packet fields."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

EXPECTED_CURRENT_PR = None
EXPECTED_HEAD_REF = None
EXPECTED_LATEST_MERGED_PACKET_PR = 4072
EXPECTED_READINESS_DATE = "2026-07-01"

Require = Callable[[bool, str], None]
RequireMapping = Callable[[Mapping[str, Any], str], Mapping[str, Any]]


def validate_readiness_refresh(
    launch_packet: Mapping[str, Any],
    expected_target_host: str,
    *,
    require: Require,
    require_mapping: RequireMapping,
    require_commands: bool = False,
    require_head_ref: bool = False,
) -> Mapping[str, Any]:
    """Validate the shared #3810 public readiness-refresh contract."""
    readiness_refresh = require_mapping(launch_packet, "readiness_refresh")
    require(
        readiness_refresh.get("checked_date") == EXPECTED_READINESS_DATE,
        f"readiness refresh date must be {EXPECTED_READINESS_DATE}",
    )
    require(
        readiness_refresh.get("target_host") == expected_target_host,
        "readiness refresh target host mismatch",
    )
    if require_commands:
        require(
            "gh issue view 3810" in str(readiness_refresh.get("live_issue_command", "")),
            "readiness refresh must record live issue command",
        )
        require(
            "gh pr list" in str(readiness_refresh.get("open_pr_dedupe_command", "")),
            "readiness refresh must record open PR dedupe command",
        )

    require(
        readiness_refresh.get("current_pr") is EXPECTED_CURRENT_PR,
        "readiness refresh must record no current review PR before branch publication",
    )
    open_pr_matches = readiness_refresh.get("open_pr_matches")
    require(
        open_pr_matches == [],
        "readiness refresh open_pr_matches must be empty before branch publication",
    )
    require(
        readiness_refresh.get("blocking_open_pr_matches") == [],
        "readiness refresh must record no blocking open PR matches beyond current PR",
    )
    require(
        readiness_refresh.get("latest_merged_packet_pr") == EXPECTED_LATEST_MERGED_PACKET_PR,
        "readiness refresh must record latest merged packet PR",
    )
    return readiness_refresh
