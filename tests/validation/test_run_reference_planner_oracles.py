"""VV-2 source-matrix and pedestrian-removal contract tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from scripts.validation.run_reference_planner_oracles import (
    EPISODE_SCHEMA,
    ROOT,
    _scenario_arm,
    load_contract,
)

CONFIG = ROOT / "configs/validation/issue_9732_reference_oracles.yaml"

if TYPE_CHECKING:
    from pathlib import Path


def test_contract_pins_full_release_matrix_and_seed_schedule() -> None:
    """The oracle cannot silently shrink the release matrix or seed set."""
    config, scenarios = load_contract(CONFIG)

    assert len(scenarios) == 48
    assert config["_resolved"]["seeds"] == list(range(111, 141))
    assert config["_resolved"]["expected_episode_rows"] == 48 * 30 * 4


def test_pedestrian_free_overlay_removes_explicit_svg_actors_without_mutating_source() -> None:
    """Population zero also removes SVG fixed pedestrians in a copied map."""
    config, scenarios = load_contract(CONFIG)
    source = next(s for s in scenarios if s["name"] == "francis2023_blind_corner")
    matrix_path = ROOT / config["_resolved"]["matrix_path"]
    original = build_env_config(source, scenario_path=matrix_path)
    original_count = sum(len(m.single_pedestrians) for m in original.map_pool.map_defs.values())
    assert original_count > 0

    derived = _scenario_arm([source], [111], "pedestrian_free_v1")[0]
    ped_free = build_env_config(derived, scenario_path=matrix_path)
    assert ped_free.sim_config.population_size == 0
    assert all(not m.single_pedestrians for m in ped_free.map_pool.map_defs.values())
    assert all(not m.social_groups for m in ped_free.map_pool.map_defs.values())
    assert source["seeds"] != [111]
    assert "reference_population_mode" not in source


def test_pedestrian_free_overlay_rejects_nonzero_population() -> None:
    """A malformed variant cannot claim that pedestrians were removed."""
    config, scenarios = load_contract(CONFIG)
    derived = _scenario_arm([scenarios[0]], [111], "pedestrian_free_v1")[0]
    derived["simulation_config"]["population_size"] = 1
    with pytest.raises(ValueError, match="requires simulation_config.population_size: 0"):
        build_env_config(derived, scenario_path=ROOT / config["_resolved"]["matrix_path"])


def test_stand_still_batch_preserves_policy_contract_when_resumed(tmp_path: Path) -> None:
    """A one-step source cell keeps zero-command metadata and stable resume identity."""
    config, scenarios = load_contract(CONFIG)
    source = next(s for s in scenarios if s["name"] == "classic_bottleneck_low")
    scenario = _scenario_arm([source], [111], "original")
    matrix_path = ROOT / config["_resolved"]["matrix_path"]
    output = tmp_path / "stand_still.jsonl"
    kwargs = {
        "scenario_path": matrix_path,
        "provenance_scenario_path": matrix_path,
        "horizon": 1,
        "dt": 0.1,
        "record_forces": False,
        "algo": "stand_still",
        "workers": 1,
        "resume": True,
    }

    first = run_map_batch(scenario, output, EPISODE_SCHEMA, **kwargs)
    second = run_map_batch(scenario, output, EPISODE_SCHEMA, **kwargs)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert first["written"] == 1
    assert second["total_jobs"] == 0
    assert second["written"] == 0
    assert len(rows) == 1
    assert rows[0]["algo"] == "stand_still"
    assert (
        rows[0]["algorithm_metadata"]["planner_contract"]["observation_contract"]["required_inputs"]
        == []
    )
