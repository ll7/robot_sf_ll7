"""Test deterministic episode selection ordering (T013).

We simulate records and ensure first N ordering preserved when videos disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts

if TYPE_CHECKING:
    from pathlib import Path


class DummyCfg:
    """Config stub with videos disabled and a maximum of three video entries.

    Video writing is off, so the test observes selection and ordering only.
    """

    smoke = False
    disable_videos = True
    max_videos = 3


def _record(ep_id: int) -> dict:
    """Build a minimal record with a zero-based episode identifier and a fixed scenario.

    Args:
        ep_id: Zero-based integer appended to the episode identifier.

    Returns:
        Record mapping consumed by generate_visual_artifacts.
    """
    return {"episode_id": f"ep{ep_id}", "scenario_id": "scA"}


def test_selection_order(tmp_path: Path):
    """Assert selection preserves input order and stops at max_videos.

    Ten records are passed with videos disabled; exactly the first three are
    selected and their episode identifiers are ep0, ep1, ep2.

    Args:
        tmp_path: Directory receiving the generated visual artifacts.
    """
    records = [_record(i) for i in range(10)]
    out = generate_visual_artifacts(tmp_path, DummyCfg(), groups=[], records=records)
    vids = out["videos"]
    assert len(vids) == 3
    assert [v.episode_id for v in vids] == ["ep0", "ep1", "ep2"]
