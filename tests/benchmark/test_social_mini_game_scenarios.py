"""Tests for the exploratory Social Mini-Game scenario suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_sf.benchmark.cli import cli_main
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SMG_SET = REPO_ROOT / "configs/scenarios/sets/social_mini_games_v0.yaml"

REQUIRED_SMG_FIELDS = {
    "schema_version",
    "case",
    "capacity_one_resource",
    "preferred_trajectory_conflict",
    "occlusion",
    "symmetry",
    "expected_failure_modes",
}
EXPECTED_CASES = {
    "doorway",
    "narrow_sidewalk",
    "blind_corner",
    "perpendicular_crossing",
    "bidirectional_hallway",
    "curb_ramp_conflict",
    "bottleneck_with_static_obstacle",
}


def test_social_mini_game_matrix_validates_through_existing_cli(capsys) -> None:
    """SMG suite should load through the repository matrix validator."""
    rc = cli_main(["validate-config", "--matrix", str(SMG_SET)])
    captured = capsys.readouterr()

    assert rc == 0, captured.out + captured.err


def test_social_mini_game_metadata_contract() -> None:
    """Every suite row exposes descriptive SMG metadata."""
    scenarios = load_scenarios(SMG_SET)

    cases: set[str] = set()
    assert len(scenarios) == len(EXPECTED_CASES)
    for scenario in scenarios:
        smg = (scenario.get("metadata") or {}).get("smg")
        assert isinstance(smg, dict), scenario.get("name")
        assert REQUIRED_SMG_FIELDS <= set(smg), scenario.get("name")
        assert smg["schema_version"] == "robot_sf.social_mini_game.v1"
        assert isinstance(smg["capacity_one_resource"], bool)
        assert isinstance(smg["preferred_trajectory_conflict"], bool)
        assert isinstance(smg["occlusion"], bool)
        assert smg["symmetry"] in {"symmetric", "asymmetric", "none"}
        assert isinstance(smg["expected_failure_modes"], list)
        assert smg["expected_failure_modes"], scenario.get("name")
        assert all(isinstance(mode, str) and mode for mode in smg["expected_failure_modes"])
        cases.add(smg["case"])

    assert cases == EXPECTED_CASES


def test_social_mini_game_scenarios_construct_and_step_headlessly() -> None:
    """Each selected SMG scenario can reset and step without benchmark execution."""
    for scenario in load_scenarios(SMG_SET):
        config = build_robot_config_from_scenario(scenario, scenario_path=SMG_SET)
        env = make_robot_env(
            config=config,
            seed=3968,
            suite_name="social_mini_games_v0",
            scenario_name=str(
                scenario.get("name")
                if scenario.get("name") is not None
                else scenario.get("scenario_id")
            ),
            algorithm_name="headless_smoke",
        )
        try:
            env.reset()
            action = np.zeros_like(env.action_space.sample())
            env.step(action)
        finally:
            env.close()
