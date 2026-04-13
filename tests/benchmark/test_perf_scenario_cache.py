"""Tests for the SAC multi-scenario map-cache profiling module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from robot_sf.benchmark import perf_scenario_cache
from robot_sf.benchmark.perf_scenario_cache import (
    CacheProfileReport,
    ScenarioTiming,
    profile_scenario_cache,
    render_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _make_report(
    *,
    unique_maps: int = 3,
    cache_maxsize: int = 64,
    cache_hits: int = 12,
    cache_misses: int = 3,
    cache_currsize: int = 3,
    evictions: int = 0,
    total_scenarios: int = 3,
) -> CacheProfileReport:
    timings = [
        ScenarioTiming(
            scenario_id=f"sc_{i}",
            config_build_sec=0.01 * (i + 1),
            map_file=f"maps/map_{i}.svg",
        )
        for i in range(total_scenarios)
    ]
    return CacheProfileReport(
        scenario_config="configs/test.yaml",
        total_scenarios=total_scenarios,
        unique_maps=unique_maps,
        cache_maxsize=cache_maxsize,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_currsize=cache_currsize,
        evictions_estimated=evictions,
        timings=timings,
    )


class TestCacheProfileReport:
    """Unit tests for CacheProfileReport properties and serialisation."""

    def test_hit_rate_zero_when_no_loads(self) -> None:
        """Hit rate returns 0.0 when neither hits nor misses have occurred."""
        report = _make_report(cache_hits=0, cache_misses=0)
        assert report.hit_rate == 0.0

    def test_hit_rate_full(self) -> None:
        """Hit rate returns 1.0 when all accesses are cache hits."""
        report = _make_report(cache_hits=10, cache_misses=0)
        assert report.hit_rate == pytest.approx(1.0)

    def test_hit_rate_partial(self) -> None:
        """Hit rate is computed as hits / (hits + misses)."""
        report = _make_report(cache_hits=3, cache_misses=1)
        assert report.hit_rate == pytest.approx(0.75)

    def test_to_dict_schema_version(self) -> None:
        """to_dict embeds the expected schema version and suite name."""
        report = _make_report()
        payload = report.to_dict()
        assert payload["schema_version"] == "1.0"
        assert payload["suite"] == "sac-scenario-cache-profile-v1"

    def test_to_dict_hit_rate_rounded(self) -> None:
        """to_dict rounds hit_rate to 4 decimal places."""
        report = _make_report(cache_hits=1, cache_misses=3)
        payload = report.to_dict()
        assert payload["hit_rate"] == pytest.approx(0.25)

    def test_to_dict_timings_length(self) -> None:
        """to_dict includes one timing entry per scenario."""
        report = _make_report(total_scenarios=5)
        payload = report.to_dict()
        assert len(payload["timings"]) == 5

    def test_to_dict_serialisable_as_json(self) -> None:
        """to_dict output must be JSON-serialisable without errors."""
        report = _make_report()
        payload = report.to_dict()
        json.dumps(payload)  # must not raise


class TestRenderMarkdown:
    """Unit tests for Markdown report rendering."""

    def test_ok_line_when_no_evictions(self) -> None:
        """Report includes an OK notice when zero cache evictions occurred."""
        report = _make_report(evictions=0, unique_maps=3, cache_maxsize=64)
        md = render_markdown(report)
        assert "OK" in md

    def test_warning_when_maps_exceed_maxsize(self) -> None:
        """Report warns when unique maps exceed the cache capacity."""
        report = _make_report(unique_maps=10, cache_maxsize=4, evictions=6)
        md = render_markdown(report)
        assert "Warning" in md
        assert "exceed cache maxsize" in md

    def test_warning_when_evictions_but_capacity_ok(self) -> None:
        """Report warns about path normalisation when evictions occur despite sufficient capacity."""
        report = _make_report(unique_maps=3, cache_maxsize=64, evictions=2)
        md = render_markdown(report)
        assert "Warning" in md
        assert "path normalisation" in md

    def test_timing_table_included(self) -> None:
        """Per-scenario timing table is present in the Markdown output."""
        report = _make_report(total_scenarios=2)
        md = render_markdown(report)
        assert "sc_0" in md
        assert "sc_1" in md

    def test_warm_average_included_for_multi_pass(self) -> None:
        """Average warm-pass config-build time is shown when multiple passes are recorded."""
        timings = [
            ScenarioTiming(scenario_id=f"sc_{i}", config_build_sec=0.01, map_file=None)
            for i in range(6)
        ]
        report = CacheProfileReport(
            scenario_config="c.yaml",
            total_scenarios=3,
            unique_maps=2,
            cache_maxsize=64,
            cache_hits=3,
            cache_misses=3,
            cache_currsize=2,
            evictions_estimated=0,
            timings=timings,
        )
        md = render_markdown(report)
        assert "warm passes" in md


class TestProfileScenarioCache:
    """Integration tests for profile_scenario_cache."""

    def test_profile_returns_correct_scenario_count(self, tmp_path: Path) -> None:
        """profile_scenario_cache counts scenarios and rounds cache stats correctly."""
        config = tmp_path / "scenarios.yaml"
        _write_yaml(
            config,
            """
scenarios:
  - name: sc_a
    map_file: a.svg
  - name: sc_b
    map_file: b.svg
""",
        )
        map_a = tmp_path / "a.svg"
        map_b = tmp_path / "b.svg"
        map_a.write_text("<svg/>", encoding="utf-8")
        map_b.write_text("<svg/>", encoding="utf-8")

        # Patch _load_map_definition so we don't need real SVG parsing.
        with patch(
            "robot_sf.training.scenario_loader._load_map_definition", return_value=None
        ) as mock_load:
            mock_load.cache_info.return_value = type(
                "CI", (), {"hits": 4, "misses": 2, "maxsize": 64, "currsize": 2}
            )()
            mock_load.cache_clear = lambda: None

            with patch("robot_sf.benchmark.perf_scenario_cache.map_cache_info") as mock_ci:
                mock_ci.side_effect = [
                    {"hits": 0, "misses": 0, "maxsize": 64, "currsize": 0},
                    {"hits": 4, "misses": 2, "maxsize": 64, "currsize": 2},
                ]
                with patch(
                    "robot_sf.benchmark.perf_scenario_cache.build_robot_config_from_scenario"
                ):
                    report = profile_scenario_cache(config, repetitions=1)

        assert report.total_scenarios == 2
        assert report.unique_maps == 2
        assert report.cache_hits == 4
        assert report.cache_misses == 2

    def test_profile_raises_for_empty_scenario_set(self, tmp_path: Path) -> None:
        """profile_scenario_cache raises when no scenarios are loadable."""
        config = tmp_path / "empty.yaml"
        _write_yaml(config, "scenarios: []")
        with pytest.raises(
            ValueError, match="Scenario config missing scenarios|No scenarios found"
        ):
            profile_scenario_cache(config)


class TestMainCLI:
    """End-to-end CLI entry-point tests."""

    def test_main_exits_zero_no_evictions(self, tmp_path: Path) -> None:
        """main() returns 0 when no cache evictions are detected."""
        config = tmp_path / "scenarios.yaml"
        _write_yaml(
            config,
            """
scenarios:
  - name: sc_a
    map_file: a.svg
""",
        )
        report = _make_report(evictions=0, total_scenarios=1)
        with (
            patch.object(perf_scenario_cache, "profile_scenario_cache", return_value=report),
            patch.object(perf_scenario_cache, "ensure_canonical_tree"),
        ):
            rc = perf_scenario_cache.main(
                [
                    "--scenario-config",
                    str(config),
                    "--output-json",
                    str(tmp_path / "out.json"),
                    "--output-markdown",
                    str(tmp_path / "out.md"),
                ]
            )
        assert rc == 0

    def test_main_exits_one_with_evictions(self, tmp_path: Path) -> None:
        """main() returns 1 when cache evictions are detected."""
        config = tmp_path / "scenarios.yaml"
        _write_yaml(
            config,
            """
scenarios:
  - name: sc_a
    map_file: a.svg
""",
        )
        report = _make_report(evictions=3, total_scenarios=1)
        with (
            patch.object(perf_scenario_cache, "profile_scenario_cache", return_value=report),
            patch.object(perf_scenario_cache, "ensure_canonical_tree"),
        ):
            rc = perf_scenario_cache.main(
                [
                    "--scenario-config",
                    str(config),
                    "--output-json",
                    str(tmp_path / "out.json"),
                    "--output-markdown",
                    str(tmp_path / "out.md"),
                ]
            )
        assert rc == 1


class TestMapCacheInfo:
    """Tests for the map_cache_info() instrumentation helper."""

    def test_map_cache_info_returns_expected_keys(self) -> None:
        """map_cache_info() must return the four standard LRU cache info fields."""
        from robot_sf.training.scenario_loader import map_cache_info

        info = map_cache_info()
        assert set(info.keys()) == {"hits", "misses", "maxsize", "currsize"}

    def test_map_cache_info_maxsize_is_64(self) -> None:
        """The map definition cache must have maxsize=64 to hold all scenario maps."""
        from robot_sf.training.scenario_loader import map_cache_info

        info = map_cache_info()
        assert info["maxsize"] == 64
