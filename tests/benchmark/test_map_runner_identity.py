"""Tests for robot_sf.benchmark.map_runner_identity — seed and resume identity."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.benchmark.map_runner_identity import (
    _compute_map_episode_id,
    _resolve_seed_list,
    _scenario_identity_payload,
    _scenario_with_episode_seed_defaults,
    _select_seeds,
    _suite_key,
)


class TestResolveSeedList:
    """Tests for _resolve_seed_list YAML loading."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """A missing seed file must return an empty dict."""
        assert _resolve_seed_list(tmp_path / "nonexistent.yaml") == {}

    def test_valid_seed_file(self, tmp_path: Path) -> None:
        """A valid YAML seed file must be parsed into named seed lists."""
        seed_file = tmp_path / "seeds.yaml"
        seed_file.write_text(
            yaml.dump({"default": [1, 2, 3], "classic": [10, 20]}),
            encoding="utf-8",
        )
        result = _resolve_seed_list(seed_file)
        assert result["default"] == [1, 2, 3]
        assert result["classic"] == [10, 20]

    def test_non_dict_yaml_returns_empty(self, tmp_path: Path) -> None:
        """A non-mapping YAML file must return an empty dict."""
        seed_file = tmp_path / "seeds.yaml"
        seed_file.write_text("- item1\n- item2\n", encoding="utf-8")
        assert _resolve_seed_list(seed_file) == {}

    def test_non_list_values_skipped(self, tmp_path: Path) -> None:
        """Non-list values in the YAML must be skipped."""
        seed_file = tmp_path / "seeds.yaml"
        seed_file.write_text(
            yaml.dump({"default": [1, 2], "bad": "not_a_list"}),
            encoding="utf-8",
        )
        result = _resolve_seed_list(seed_file)
        assert "default" in result
        assert "bad" not in result


class TestSuiteKey:
    """Tests for _suite_key inference from scenario path."""

    def test_classic_in_stem(self) -> None:
        """A filename containing 'classic' must map to classic_interactions."""
        assert _suite_key(Path("classic_interactions.yaml")) == "classic_interactions"

    def test_francis_in_stem(self) -> None:
        """A filename containing 'francis' must map to francis2023."""
        assert _suite_key(Path("francis2023_scenarios.yaml")) == "francis2023"

    def test_default_fallback(self) -> None:
        """A filename without known patterns must map to default."""
        assert _suite_key(Path("my_scenarios.yaml")) == "default"

    def test_case_insensitive(self) -> None:
        """Suite key detection must be case-insensitive."""
        assert _suite_key(Path("CLASSIC_test.yaml")) == "classic_interactions"


class TestSelectSeeds:
    """Tests for _select_seeds resolution with fallbacks."""

    def test_scenario_seeds_take_priority(self) -> None:
        """Per-scenario seeds must override suite seeds."""
        scenario = {"seeds": [100, 200]}
        result = _select_seeds(scenario, suite_seeds={"default": [1]}, suite_key="default")
        assert result == [100, 200]

    def test_suite_key_seeds_used(self) -> None:
        """Suite-key seeds must be used when scenario has no seeds."""
        scenario = {"name": "sc1"}
        result = _select_seeds(
            scenario,
            suite_seeds={"classic_interactions": [10, 20]},
            suite_key="classic_interactions",
        )
        assert result == [10, 20]

    def test_default_suite_fallback(self) -> None:
        """Default suite seeds must be used when the suite key has no seeds."""
        scenario = {"name": "sc1"}
        result = _select_seeds(
            scenario, suite_seeds={"default": [42]}, suite_key="classic_interactions"
        )
        assert result == [42]

    def test_fallback_to_seed_zero(self) -> None:
        """Without any seeds, the fallback must be [0]."""
        scenario = {"name": "sc1"}
        result = _select_seeds(scenario, suite_seeds={}, suite_key="default")
        assert result == [0]

    def test_empty_scenario_seeds_fall_through(self) -> None:
        """An empty seeds list must fall through to suite seeds."""
        scenario = {"seeds": []}
        result = _select_seeds(scenario, suite_seeds={"default": [5]}, suite_key="default")
        assert result == [5]


class TestScenarioIdentityPayload:
    """Tests for _scenario_identity_payload construction."""

    def test_basic_payload(self) -> None:
        """A basic scenario must produce an identity payload with algo and hash."""
        scenario = {"name": "sc-1", "map_file": "test.svg"}
        payload = _scenario_identity_payload(
            scenario,
            algo="social_force",
            algo_config={"param": 1},
            horizon=500,
            dt=0.1,
            record_forces=False,
        )
        assert payload["id"] == "sc-1"
        assert payload["algo"] == "social_force"
        assert "algo_config_hash" in payload
        assert payload["record_forces"] is False
        assert payload["run_horizon"] == 500
        assert payload["run_dt"] == 0.1

    def test_seed_stripped_from_payload(self) -> None:
        """seed and seeds keys must be stripped from the identity payload."""
        scenario = {"name": "sc-1", "seed": 42, "seeds": [1, 2, 3]}
        payload = _scenario_identity_payload(
            scenario,
            algo="goal",
            algo_config={},
            horizon=None,
            dt=None,
            record_forces=False,
        )
        assert "seed" not in payload
        assert "seeds" not in payload

    def test_observation_mode_included(self) -> None:
        """observation_mode must be included when provided."""
        scenario = {"name": "sc-1"}
        payload = _scenario_identity_payload(
            scenario,
            algo="goal",
            algo_config={},
            horizon=None,
            dt=None,
            record_forces=False,
            observation_mode="lidar_only",
        )
        assert payload["observation_mode"] == "lidar_only"

    def test_horizon_zero_not_included(self) -> None:
        """A zero horizon must not be included in the payload."""
        scenario = {"name": "sc-1"}
        payload = _scenario_identity_payload(
            scenario,
            algo="goal",
            algo_config={},
            horizon=0,
            dt=None,
            record_forces=False,
        )
        assert "run_horizon" not in payload

    def test_unknown_scenario_id_fallback(self) -> None:
        """A scenario without name/id must use 'unknown' as the id."""
        scenario = {"map_file": "test.svg"}
        payload = _scenario_identity_payload(
            scenario,
            algo="goal",
            algo_config={},
            horizon=None,
            dt=None,
            record_forces=False,
        )
        assert payload["id"] == "unknown"


class TestComputeMapEpisodeId:
    """Tests for _compute_map_episode_id."""

    def test_format(self) -> None:
        """Episode id must follow the scenario--seed--hash format."""
        payload = {"id": "sc-1", "algo": "goal"}
        episode_id = _compute_map_episode_id(payload, seed=42)
        parts = episode_id.split("--")
        assert len(parts) == 3
        assert parts[0] == "sc-1"
        assert parts[1] == "42"

    def test_deterministic(self) -> None:
        """The same payload and seed must produce the same episode id."""
        payload = {"id": "sc-1", "algo": "goal", "algo_config_hash": "abc"}
        id1 = _compute_map_episode_id(payload, seed=7)
        id2 = _compute_map_episode_id(payload, seed=7)
        assert id1 == id2

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different episode ids."""
        payload = {"id": "sc-1", "algo": "goal"}
        id1 = _compute_map_episode_id(payload, seed=1)
        id2 = _compute_map_episode_id(payload, seed=2)
        assert id1 != id2

    def test_fallback_to_name_key(self) -> None:
        """The name key must be used when id is absent."""
        payload = {"name": "my_scenario", "algo": "goal"}
        episode_id = _compute_map_episode_id(payload, seed=0)
        assert episode_id.startswith("my_scenario--")


class TestScenarioWithEpisodeSeedDefaults:
    """Tests for _scenario_with_episode_seed_defaults."""

    def test_route_spawn_seed_filled(self) -> None:
        """Missing route_spawn_seed must be filled from the episode seed."""
        scenario = {"name": "sc-1", "simulation_config": {}}
        result = _scenario_with_episode_seed_defaults(scenario, seed=42)
        assert result["simulation_config"]["route_spawn_seed"] == 42

    def test_existing_route_spawn_seed_preserved(self) -> None:
        """An existing route_spawn_seed must not be overwritten."""
        scenario = {"name": "sc-1", "simulation_config": {"route_spawn_seed": 99}}
        result = _scenario_with_episode_seed_defaults(scenario, seed=42)
        assert result["simulation_config"]["route_spawn_seed"] == 99

    def test_original_not_mutated(self) -> None:
        """The original scenario dict must not be mutated."""
        scenario = {"name": "sc-1", "simulation_config": {}}
        _scenario_with_episode_seed_defaults(scenario, seed=42)
        assert "route_spawn_seed" not in scenario["simulation_config"]

    def test_missing_simulation_config_created(self) -> None:
        """A missing simulation_config must be created."""
        scenario = {"name": "sc-1"}
        result = _scenario_with_episode_seed_defaults(scenario, seed=7)
        assert result["simulation_config"]["route_spawn_seed"] == 7
