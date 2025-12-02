"""Test deterministic episode selection ordering (T013).

We simulate records and ensure first N ordering preserved when videos disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts

if TYPE_CHECKING:
    from pathlib import Path


class DummyCfg:
    """DummyCfg class."""

    smoke = False
    disable_videos = True
    max_videos = 3


def _record(ep_id: int) -> dict:
    """Record.

    Args:
        ep_id: Auto-generated placeholder description.

    Returns:
        dict: Auto-generated placeholder description.
    """
    return {"episode_id": f"ep{ep_id}", "scenario_id": "scA"}


def test_selection_order(tmp_path: Path):
    """Test selection order.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    records = [_record(i) for i in range(10)]
    out = generate_visual_artifacts(tmp_path, DummyCfg(), groups=[], records=records)
    vids = out["videos"]
    assert len(vids) == 3
    assert [v.episode_id for v in vids] == ["ep0", "ep1", "ep2"]
