"""Tests for issue_4904_latency_decel_profile.py."""

import csv
import json
from pathlib import Path

from scripts.analysis.issue_4904_latency_decel_profile import (
    compute_percentiles,
    extract_profiles,
    load_episodes,
    write_percentile_csv,
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
        # Fail closed (issue #5000): late_evasive events with no finite latency are tallied by
        # reason. Legacy records without a producer reason fall back to "unspecified".
        assert sum(p["latency_unavailable_reasons"].values()) == 5
        assert p["latency_unavailable_reasons"]["unspecified"] == 5

    def test_latency_unavailable_reason_is_surfaced(self):
        episodes = [
            _make_episode(
                "goal",
                exposure_steps=10,
                late_evasive=True,
                response_latency_s=None,
                required_deceleration_m_s2=0.001,
                seed=i,
            )
            for i in range(3)
        ]
        for ep in episodes:
            ep["safety_predicates"]["late_evasive_predicate"]["fields"][
                "latency_unavailable_reason"
            ] = "no_clearance_restoring_action"
        profiles = extract_profiles(episodes)
        reasons = profiles["goal"]["latency_unavailable_reasons"]
        assert reasons["no_clearance_restoring_action"] == 3
        assert "unspecified" not in reasons

    def test_finite_latency_records_no_reason(self):
        episodes = [
            _make_episode(
                "p",
                exposure_steps=10,
                late_evasive=True,
                response_latency_s=2.0,
                required_deceleration_m_s2=0.5,
            )
        ]
        profiles = extract_profiles(episodes)
        assert sum(profiles["p"]["latency_unavailable_reasons"].values()) == 0

    def test_filters_non_finite_values(self):
        # json.loads silently accepts NaN/Infinity tokens; they must be treated as
        # missing so they cannot poison the percentile inputs downstream.
        raw = [
            '{"_planner":"p","interaction_exposure":{"interaction_exposure_steps":1},'
            '"safety_predicates":{"late_evasive_predicate":{"late_evasive":true,"fields":'
            '{"response_latency_s":NaN,"required_deceleration_m_s2":Infinity}}}}',
            '{"_planner":"p","interaction_exposure":{"interaction_exposure_steps":1},'
            '"safety_predicates":{"late_evasive_predicate":{"late_evasive":true,"fields":'
            '{"response_latency_s":2.0,"required_deceleration_m_s2":0.5}}}}',
        ]
        episodes = [json.loads(line) for line in raw]
        profiles = extract_profiles(episodes)
        p = profiles["p"]
        assert p["latency_populated"] == 1  # NaN excluded; only the 2.0 counts
        assert p["latencies"] == [2.0]
        assert p["decels"] == [0.5]  # Infinity excluded


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

    def test_skips_blank_trailing_line(self, tmp_path):
        # A trailing blank line (common from editors) must not raise.
        pdir = tmp_path / "pp__differential_drive"
        pdir.mkdir(parents=True, exist_ok=True)
        episode = _make_episode("pp", exposure_steps=1, seed=0)
        (pdir / "episodes.jsonl").write_text(json.dumps(episode) + "\n\n")
        loaded = load_episodes(tmp_path)
        assert len(loaded) == 1


class TestWritePercentileCsv:
    """Tests for the CSV deliverable and the anomaly_gap/rate formulas."""

    def test_anomaly_gap_rates_and_n(self, tmp_path):
        episodes = [
            # goal: late_evasive on every exposure episode, latency never populated
            *_make_episodes(
                "goal",
                count=10,
                exposure_steps=5,
                late_evasive=True,
                response_latency_s=None,
                required_deceleration_m_s2=0.001,
            ),
            # alpha: 2 exposure episodes, 1 late_evasive, both latency populated
            _make_episode(
                "alpha",
                exposure_steps=5,
                late_evasive=True,
                response_latency_s=2.0,
                required_deceleration_m_s2=0.5,
                seed=1,
            ),
            _make_episode(
                "alpha",
                exposure_steps=5,
                late_evasive=False,
                response_latency_s=4.0,
                required_deceleration_m_s2=0.2,
                seed=2,
            ),
        ]
        profiles = extract_profiles(episodes)
        out = tmp_path / "pct.csv"
        write_percentile_csv(profiles, out)

        rows = out.read_text().splitlines()
        header = rows[0].split(",")
        for col in (
            "planner",
            "lat_N",
            "anomaly_gap",
            "late_evasive_rate",
            "latency_populated_rate",
            "lat_p50",
        ):
            assert col in header

        by_planner = {r["planner"]: r for r in csv.DictReader(rows)}

        goal = by_planner["goal"]
        assert goal["exposure_episodes"] == "10"
        assert goal["late_evasive_true"] == "10"
        assert goal["latency_populated"] == "0"
        assert goal["lat_N"] == "0"
        assert goal["late_evasive_rate"] == "1.0"
        assert goal["latency_populated_rate"] == "0.0"
        assert goal["anomaly_gap"] == "1.0"  # 1.0 - 0.0
        # Fail-closed columns (issue #5000): the silent 0-latency gap is now machine-readable.
        assert goal["late_evasive_no_latency"] == "10"
        assert goal["dominant_latency_unavailable_reason"] == "unspecified"
        # alpha's single late_evasive event carried a finite latency, so nothing is unexplained.
        assert by_planner["alpha"]["late_evasive_no_latency"] == "0"
        assert by_planner["alpha"]["dominant_latency_unavailable_reason"] == ""

        alpha = by_planner["alpha"]
        assert alpha["exposure_episodes"] == "2"
        assert alpha["lat_N"] == "2"
        assert alpha["decel_N"] == "2"
        assert abs(float(alpha["lat_p50"]) - 3.0) < 1e-6  # median of [2.0, 4.0]
        assert alpha["late_evasive_rate"] == "0.5"
        assert alpha["latency_populated_rate"] == "1.0"
        assert abs(float(alpha["anomaly_gap"]) - (-0.5)) < 1e-6  # 0.5 - 1.0

    def test_csv_is_deterministic(self, tmp_path):
        episodes = _make_episodes(
            "p",
            count=20,
            exposure_steps=5,
            response_latency_s=0.0,  # overridden per-episode below
            required_deceleration_m_s2=0.1,
        )
        for i, ep in enumerate(episodes):
            ep["safety_predicates"]["late_evasive_predicate"]["fields"]["response_latency_s"] = (
                float(i)
            )
        profiles = extract_profiles(episodes)
        out_a = tmp_path / "a.csv"
        out_b = tmp_path / "b.csv"
        write_percentile_csv(profiles, out_a)
        write_percentile_csv(profiles, out_b)
        assert out_a.read_text() == out_b.read_text()


def _make_episodes(
    planner: str,
    count: int,
    exposure_steps: int = 0,
    late_evasive: bool = False,
    response_latency_s: float | None = None,
    required_deceleration_m_s2: float | None = None,
) -> list[dict]:
    """Build N minimal episode records with distinct seeds."""
    return [
        _make_episode(
            planner,
            exposure_steps=exposure_steps,
            late_evasive=late_evasive,
            response_latency_s=response_latency_s,
            required_deceleration_m_s2=required_deceleration_m_s2,
            seed=i,
        )
        for i in range(count)
    ]
