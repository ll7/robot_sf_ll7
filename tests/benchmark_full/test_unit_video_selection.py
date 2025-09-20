"""Polish Phase T052: Video selection deterministic behavior.

Ensures generate_videos respects smoke mode skipping & deterministic ordering.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.videos import generate_videos


class _Cfg:
    smoke = True
    disable_videos = False
    max_videos = 2


def test_video_selection_smoke_skip():
    records = []
    for i in range(5):
        records.append(
            {
                "episode_id": f"ep{i}",
                "scenario_id": "scenario_a",
                "seed": i,
            }
        )
    artifacts = generate_videos(records, out_dir="/tmp/videos_test", cfg=_Cfg())
    # Expect first 2 episodes only due to max_videos=2; all skipped with note
    assert len(artifacts) == 2
    assert [a.episode_id for a in artifacts] == ["ep0", "ep1"]
    assert all(a.status == "skipped" for a in artifacts)
    assert all(a.note == "smoke mode" for a in artifacts)
