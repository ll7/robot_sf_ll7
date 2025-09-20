"""Contract test T013 for `generate_videos`.

Expectation (final):
  - In smoke mode or when ffmpeg missing, videos are skipped gracefully and artifacts list records status.

Implements T037 basic stub: should return a list of artifacts with status 'skipped' in smoke mode.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.videos import generate_videos


def test_generate_videos_smoke_skip(temp_results_dir, synthetic_episode_record):
    records = [
        synthetic_episode_record(
            episode_id="ep1",
            scenario_id="scenario_a",
            seed=1,
        )
    ]

    class _Cfg:
        smoke = True
        output_root = str(temp_results_dir)

    artifacts = generate_videos(records, str(temp_results_dir / "videos"), _Cfg())
    assert len(artifacts) == 1
    a = artifacts[0]
    assert a.status == "skipped"
    assert a.scenario_id == "scenario_a"
    assert a.episode_id == "ep1"
    assert a.path_mp4.endswith(".mp4")
