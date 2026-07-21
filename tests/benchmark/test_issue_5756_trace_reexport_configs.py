"""Exact-population contract tests for the issue #5756 trace re-export configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config import (
    _load_campaign_scenarios,
    load_campaign_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "benchmarks"
CONFIG_CASES = {
    "issue_5756_trace90_ppo_canary.yaml": {
        "planner": "ppo",
        "scenarios": {"classic_doorway_medium"},
        "seeds": {113},
        "episodes": 1,
    },
    "issue_5756_trace90_ppo.yaml": {
        "planner": "ppo",
        "scenarios": {
            "classic_doorway_medium",
            "classic_realworld_double_bottleneck_high",
        },
        "seeds": set(range(111, 141)),
        "episodes": 60,
    },
    "issue_5756_trace90_goal.yaml": {
        "planner": "goal",
        "scenarios": {"classic_realworld_double_bottleneck_high"},
        "seeds": set(range(111, 141)),
        "episodes": 30,
    },
}


@pytest.mark.parametrize(("filename", "expected"), CONFIG_CASES.items())
def test_trace_reexport_config_exact_matrix(filename: str, expected: dict[str, object]) -> None:
    """The execution-commit-compatible loader must resolve exactly 1/60/30 episodes."""
    cfg = load_campaign_config(CONFIG_DIR / filename)
    scenarios = _load_campaign_scenarios(cfg)

    assert [planner.key for planner in cfg.planners] == [expected["planner"]]
    assert {str(row["name"]) for row in scenarios} == expected["scenarios"]
    for scenario in scenarios:
        assert set(scenario["seeds"]) == expected["seeds"]
    cardinality = (
        len(cfg.planners) * len(cfg.kinematics_matrix) * sum(len(row["seeds"]) for row in scenarios)
    )
    assert cardinality == expected["episodes"]


@pytest.mark.parametrize("filename", CONFIG_CASES)
def test_trace_reexport_config_preserves_execution_contract(filename: str) -> None:
    """Every rerun config keeps the pinned release and trace-critical execution settings."""
    cfg = load_campaign_config(CONFIG_DIR / filename)

    assert cfg.scenario_matrix_path == (
        REPO_ROOT / "configs/scenarios/classic_interactions_francis2023.yaml"
    )
    assert cfg.workers == 1
    assert cfg.horizon == 600
    assert cfg.dt == pytest.approx(0.1)
    assert cfg.record_forces is True
    assert cfg.record_planner_decision_trace is True
    assert cfg.record_simulation_step_trace is True
    assert cfg.stop_on_failure is True
    assert cfg.kinematics_matrix == ("differential_drive",)
    assert cfg.export_publication_bundle is False
    assert cfg.include_videos_in_publication is False
    assert cfg.overwrite_publication_bundle is False
    assert cfg.paper_facing is False
    assert cfg.paper_profile_version == "paper-matrix-v1"
    assert cfg.bootstrap_samples == 300
    assert cfg.bootstrap_confidence == pytest.approx(0.95)
    assert cfg.bootstrap_seed == 123


def test_all_trace_reexport_configs_disable_resume() -> None:
    """Fresh exact-population exports must never append into an existing output root."""
    loaded = [load_campaign_config(CONFIG_DIR / filename) for filename in CONFIG_CASES]

    assert loaded
    assert all(cfg.resume is False for cfg in loaded)
