"""Validate constitution metadata consistency for governance edits."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTITUTION = ROOT / ".specify" / "memory" / "constitution.md"


def test_constitution_has_one_footer_matching_sync_report() -> None:
    """Guard the governance footer against stale duplicate version records.

    The constitution is a policy source of truth, so a stale footer can make
    contributors cite the wrong active version after amendments.
    """

    text = CONSTITUTION.read_text(encoding="utf-8")
    sync_match = re.search(
        r"Previous Version:\s*[0-9]+\.[0-9]+\.[0-9]+\s*->\s*"
        r"New Version:\s*([0-9]+\.[0-9]+\.[0-9]+)",
        text,
    )
    footer_matches = re.findall(
        r"\*\*Version\*\*:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*\|\s*"
        r"\*\*Ratified\*\*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*\|\s*"
        r"\*\*Last Amended\*\*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        text,
    )

    assert sync_match is not None, "constitution Sync Impact Report must name the new version"
    assert len(footer_matches) == 1, "constitution must have exactly one active version footer"
    assert footer_matches[0][0] == sync_match.group(1)
