"""Tests for issue_4904_latency_decel_profile.py."""

import json
from pathlib import Path

from scripts.analysis.issue_4904_latency_decel_profile import (
    compute_percentiles,
    extract_profiles,
    load_episodes,
)


def _make_episode(
    planner: str,
    exposure_steps: int = 0,
    late_evasive: bool = False,
    response_latency_s: float | None = None,
    required_deceleration_m_s2: float | None = None,
    seed: int = 1,
) -> dict:
    """Build a minimal episode record."""
    return {
        "_planner": planner,
        "algo": planner,
        "seed": seed,
        "interaction_exposure": {"interaction_exposure_steps": exposure_steps},
        "safety_predicates": {
            "late_evasive_predicate": {
                "late_evasive": late_evasive,
                "fields": {
                    "response_latency_s": response_latency_s,
                    "required_deceleration_m_s2": required_deceleration_m_s2,
                    "minimum_distance_m": 1.0,
                },
            }
        },
    }


def _write_episodes_jsonl(tmpdir: Path, planner: str, episodes: list[dict]) -> Path:
    """Write episodes to a planner dir."""
    pdir = tmpdir / f"{planner}__differential_drive"
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return tmpdir


class TestComputePercentiles:
    """Tests for compute_percentiles helper."""

    def test_empty(self):
        result = compute_percentiles([])
        assert result["N"] == 0
        assert result["p50"] is None

    def test_single_value(self):
        result = compute_percentiles([5.0])
        assert result["N"] == 1
        assert result["p50"] == 5.0
        assert result["max"] == 5.0

    def test_known_distribution(self):
        values = list(range(1, 101))  # 1..100
        result = compute_percentiles(values)
        assert result["N"] == 100
        assert abs(result["p50"] - 50.5) < 0.1
        assert result["max"] == 100


class TestExtractProfiles:
    """Tests for extract_profiles helper."""

    def test_skips_zero_exposure(self):
        episodes = [_make_episode("test_planner", exposure_steps=0)]
        profiles = extract_profiles(episodes)
        assert "test_planner" not in profiles

    def test_extracts_with_exposure(self):
        episodes = [
            _make_episode(
                "p1",
                exposure_steps=10,
                late_evasive=True,
                response_latency_s=3.0,
                required_deceleration_m_s2=0.5,
            ),
            _make_episode(
                "p1",
                exposure_steps=5,
                late_evasive=False,
                response_latency_s=None,
                required_deceleration_m_s2=0.1,
            ),
        ]
        profiles = extract_profiles(episodes)
        assert "p1" in profiles
        p = profiles["p1"]
        assert p["total_exposure"] == 2
        assert p["late_evasive_true"] == 1
        assert p["latency_populated"] == 1
        assert len(p["latencies"]) == 1
        assert len(p["decels"]) == 2  # decel is 0.1 even when late_evasive=False

    def test_goal_anomaly_pattern(self):
        episodes = [
            _make_episode(
                "goal",
                exposure_steps=10,
                late_evasive=True,
                response_latency_s=None,
                required_deceleration_m_s2=0.001,
                seed=i,
            )
            for i in range(5)
        ]
        profiles = extract_profiles(episodes)
        p = profiles["goal"]
        assert p["late_evasive_true"] == 5
        assert p["latency_populated"] == 0
        assert len(p["latencies"]) == 0


class TestLoadEpisodes:
    """Tests for load_episodes helper."""

    def test_reads_all_planners(self, tmp_path):
        for planner in ["planner_a", "planner_b"]:
            episodes = [_make_episode(planner, exposure_steps=10, seed=i) for i in range(3)]
            _write_episodes_jsonl(tmp_path, planner, episodes)
        loaded = load_episodes(tmp_path)
        assert len(loaded) == 6
        planners = {ep["_planner"] for ep in loaded}
        assert planners == {"planner_a", "planner_b"}

    def test_skips_missing_jsonl(self, tmp_path):
        pdir = tmp_path / "empty_planner__differential_drive"
        pdir.mkdir(parents=True, exist_ok=True)
        loaded = load_episodes(tmp_path)
        assert len(loaded) == 0
