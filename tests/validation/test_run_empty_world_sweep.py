"""Tests for the empty-world sweep seed guard and pedestrian removal (#9978)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validation import run_empty_world_sweep as sweep

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("seeds", [[1001], [1001, 1002], [1030], [1001, 1030]])
def test_seed_guard_accepts_dev_seeds(seeds: list[int]) -> None:
    """Development seeds 1001-1030 pass unchanged."""
    assert sweep.assert_dev_seeds(seeds) == seeds


@pytest.mark.parametrize(
    "seeds",
    [[111], [140], [125], [1000], [1031], [1001, 111], [], [True], ["x"], [1001.5]],
)
def test_seed_guard_rejects_everything_else(seeds: list[object]) -> None:
    """Holdout 111-140, out-of-range, empty and non-integer seeds abort."""
    with pytest.raises(sweep.SeedGuardError):
        sweep.assert_dev_seeds(seeds)


def test_main_aborts_on_holdout_seed_before_touching_anything(tmp_path: Path) -> None:
    """The CLI rejects a holdout seed before verifying the head or running a campaign."""
    with pytest.raises(sweep.SeedGuardError):
        sweep.main(["--head-sha", "HEAD", "--seeds", "111", "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_remove_pedestrians_clears_every_source_and_sets_dev_seeds() -> None:
    """Density, single pedestrians and groups are removed; the rest is kept."""
    scenario = {
        "name": "fixture",
        "map_id": "classic_cross_trap",
        "simulation_config": {
            "max_episode_steps": 600,
            "ped_density": 0.08,
            "population_size": 4,
            "goal_completion_policy": "goal_zone_entry_v1",
        },
        "single_pedestrians": [{"id": "p1"}],
        "social_groups": [{"id": "g1"}],
        "seeds": [131, 132],  # seed-holdout: synthetic-fixture
    }
    out = sweep.remove_pedestrians(scenario, [1001, 1002])
    assert sweep.pedestrian_residue(out) == []
    assert out["simulation_config"]["ped_density"] == 0.0
    assert out["simulation_config"]["max_episode_steps"] == 600
    assert out["simulation_config"]["goal_completion_policy"] == "goal_zone_entry_v1"
    assert out["single_pedestrians"] == []
    assert out["social_groups"] == []
    assert out["seeds"] == [1001, 1002]
    assert scenario["single_pedestrians"] == [{"id": "p1"}]  # input untouched
    assert scenario["seeds"] == [131, 132]  # seed-holdout: synthetic-fixture


def test_remove_pedestrians_refuses_holdout_seeds() -> None:
    """Pedestrian removal cannot smuggle in evaluation seeds."""
    with pytest.raises(sweep.SeedGuardError):
        sweep.remove_pedestrians({"name": "x", "simulation_config": {}}, [111, 112])


def test_residue_detects_a_scenario_that_still_has_pedestrians() -> None:
    """A scenario that was not passed through the removal is reported."""
    residue = sweep.pedestrian_residue(
        {"simulation_config": {"ped_density": 0.05}, "single_pedestrians": [{"id": "a"}]}
    )
    assert "ped_density != 0" in residue
    assert "single_pedestrians present" in residue
    assert "runtime map-pedestrian removal flag missing" in residue


def test_classify_outcome_prefers_flags_over_ambiguous_reason() -> None:
    """A horizon run with reason 'terminated' is a timeout, not a generic termination."""
    row = {"termination_reason": "terminated", "outcome": {"timeout_event": True}, "metrics": {}}
    assert sweep.classify_outcome(row) == "timeout"
    assert sweep.classify_outcome({"metrics": {"success": True}}) == "success"
    assert sweep.classify_outcome({"outcome": {"collision_event": True}}) == "collision"


def test_width_diagnostic_does_not_inherit_historical_snqi_anchors(tmp_path):
    """Actor-free v2 diagnostics must reach metrics without incompatible v1 anchors."""
    import yaml

    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    cfg_path, _ = sweep.build_derived_inputs(
        "width",
        seeds=[1001],
        arms=["goal"],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    payload = yaml.safe_load(cfg_path.read_text())
    assert payload["snqi_weights"] is None
    assert payload["snqi_baseline"] is None
    cfg = load_campaign_config(cfg_path)
    assert cfg.snqi_weights_path is None
    assert cfg.snqi_baseline_path is None
    source = yaml.safe_load((REPO_ROOT / sweep.SUITES["width"]).read_text())
    assert source["snqi_weights"] is not None
    assert source["snqi_baseline"] is not None
