"""Contract tests for the bounded BRNE corridor diagnostic (#6464)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.baselines.brne import BRNE_PINNED_SHA
from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.map_runner_observations import obs_to_brne_format
from scripts.benchmark.run_brne_corridor_diagnostic_issue_6464 import (
    classify_record,
    summarize_records,
    validate_campaign_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml"


def _config() -> dict[str, Any]:
    """Load and validate the committed diagnostic contract."""
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return validate_campaign_config(payload)


def _record(
    *,
    metadata: dict[str, Any] | None = None,
    pedestrians: list[dict[str, Any]] | None = None,
    positions: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Build a small trace-backed episode fixture."""
    positions = positions or [[0.0, 4.0], [0.0, 5.0]]
    pedestrians = pedestrians or []
    steps = [
        {
            "robot": {"position": position},
            "pedestrians": pedestrians,
        }
        for position in positions
    ]
    return {
        "episode_id": "fixture",
        "scenario_id": "classic_head_on_corridor_low",
        "seed": 111,
        "status": "success",
        "metrics": {"success": True},
        "algorithm_metadata": {
            "status": "ok",
            "simulation_step_trace": {"steps": steps},
            **(metadata or {}),
        },
    }


def test_issue_6464_config_is_frozen_and_opt_in() -> None:
    """The campaign is exact, bounded, and not silently benchmark-safe."""
    config = _config()

    assert config["scenario_ids"] == ["classic_head_on_corridor_low"]
    assert config["seeds"] == [111, 112, 113]
    assert config["max_pedestrians"] == 7
    assert [planner["key"] for planner in config["planners"]] == [
        "brne",
        "orca",
        "social_force",
    ]

    readiness = get_algorithm_readiness("brne")
    assert readiness is not None
    assert readiness.tier == "experimental"
    assert readiness.requires_explicit_opt_in is True


def test_obs_to_brne_format_reconstructs_world_velocity() -> None:
    """Map observations retain the BRNE robot/agent contract and heading speed."""
    converted = obs_to_brne_format(
        {
            "robot": {
                "position": [1.0, 2.0],
                "heading": [1.5707963267948966],
                "speed": [2.0],
            },
            "goal": {"current": [1.0, 8.0]},
            "pedestrians": {
                "positions": [[4.0, 5.0]],
                "velocities": [[0.0, -1.0]],
                "count": [1],
            },
            "sim": {"timestep": 0.1},
            "obstacles": [{"kind": "corridor_boundary"}],
        }
    )

    assert converted["dt"] == pytest.approx(0.1)
    assert converted["robot"]["velocity"] == pytest.approx([0.0, 2.0])
    assert converted["agents"][0]["velocity"] == pytest.approx([1.0, 0.0])
    assert converted["obstacles"] == [{"kind": "corridor_boundary"}]


def test_obs_to_brne_format_unflattens_grid_enabled_map_observations() -> None:
    """The occupancy-grid map path must not collapse flat state to zero defaults."""
    converted = obs_to_brne_format(
        {
            "robot_position": [3.0, 4.0],
            "robot_velocity_xy": [0.0, 0.5],
            "robot_heading": [1.5707963267948966],
            "robot_speed": [0.5],
            "robot_radius": [1.0],
            "goal_current": [3.0, 20.0],
            "pedestrians_positions": [[3.0, 12.0], [7.0, 10.0]],
            "pedestrians_velocities": [[0.0, -0.3], [0.0, -0.2]],
            "pedestrians_count": [2],
            "pedestrians_radius": [0.35],
            "sim_timestep": [0.1],
        }
    )

    assert converted["robot"]["position"] == [3.0, 4.0]
    assert converted["robot"]["goal"] == [3.0, 20.0]
    assert converted["robot"]["velocity"] == [0.0, 0.5]
    assert len(converted["agents"]) == 2


def test_comparators_do_not_require_brne_dependency_metadata() -> None:
    """Standard comparator adapters remain usable without BRNE-only metadata."""
    classified = classify_record(_record(), _config(), planner_key="social_force")

    assert classified["execution_ok"] is True
    assert classified["native"] is False
    assert classified["status"] == "available_comparator"


def test_brne_requires_native_dependency_status() -> None:
    """BRNE rows without a successful staged-core status are unavailable."""
    config = _config()
    missing = classify_record(_record(), config, planner_key="brne")
    assert missing["execution_ok"] is False
    assert missing["native"] is False
    assert missing["status"] == "unavailable"

    native_metadata = {
        "brne_diagnostic": {
            "status": "native_core_via_adapter",
            "execution_semantics": "native_upstream_core_through_robot_sf_adapter",
        },
        "planner_metadata": {"status": "ok"},
        "planner_kinematics": {
            "execution_mode": "adapter",
            "adapter_active": True,
            "adapter_name": "BRNEPlanner",
            "supports_native_commands": True,
            "supports_adapter_commands": True,
            "planner_command_space": "unicycle_vw",
        },
        "planner_runtime": {
            "planner_metadata": {
                "status": "ok",
                "runtime_status": "ok",
                "failure_count": 0,
                "source_commit": BRNE_PINNED_SHA,
                "source_pin": BRNE_PINNED_SHA,
                "source_integrity": "clean_pinned_worktree",
                "effective_num_samples": 49,
                "step_count": 1,
            }
        },
    }
    native = classify_record(
        _record(metadata=native_metadata),
        config,
        planner_key="brne",
    )
    assert native["native"] is True
    assert native["status"] == "available_native"

    wrong_sample_count = classify_record(
        _record(
            metadata={
                **native_metadata,
                "planner_runtime": {
                    "planner_metadata": {
                        "status": "ok",
                        "runtime_status": "ok",
                        "failure_count": 0,
                        "source_commit": BRNE_PINNED_SHA,
                        "source_pin": BRNE_PINNED_SHA,
                        "source_integrity": "clean_pinned_worktree",
                        "effective_num_samples": 42,
                    }
                },
            }
        ),
        config,
        planner_key="brne",
    )
    assert wrong_sample_count["status"] == "unavailable"


def test_brne_runtime_failure_is_unavailable_even_with_motion() -> None:
    """Fail-closed zero-motion/runtime failures cannot become native evidence."""
    config = _config()
    failed = classify_record(
        _record(
            metadata={
                "brne_diagnostic": {
                    "status": "native_core_via_adapter",
                    "execution_semantics": "native_upstream_core_through_robot_sf_adapter",
                },
                "planner_metadata": {"status": "ok"},
                "planner_kinematics": {
                    "execution_mode": "adapter",
                    "adapter_active": True,
                    "adapter_name": "BRNEPlanner",
                    "supports_native_commands": True,
                    "supports_adapter_commands": True,
                    "planner_command_space": "unicycle_vw",
                },
                "planner_runtime": {
                    "planner_metadata": {
                        "status": "ok",
                        "runtime_status": "failed",
                        "failure_count": 1,
                        "effective_num_samples": 42,
                    }
                },
            }
        ),
        config,
        planner_key="brne",
    )
    assert failed["execution_ok"] is False
    assert failed["status"] == "unavailable"


def test_summary_requires_unique_exact_pairs_and_excludes_unavailable_goals() -> None:
    """Coverage and outcomes must use exact pairs and eligible rows only."""
    config = _config()
    first = _record()
    first["seed"] = 111
    unavailable_goal = _record(metadata={"fallback_triggered": True})
    unavailable_goal["seed"] = 112
    duplicate = _record()
    duplicate["seed"] = 111

    summary = summarize_records(
        planner_key="social_force",
        records=[first, unavailable_goal, duplicate],
        config=config,
    )

    assert summary["pair_coverage_exact"] is False
    assert summary["duplicate_pairs"] == [["classic_head_on_corridor_low", 111]]
    assert summary["missing_pairs"] == [["classic_head_on_corridor_low", 113]]
    assert summary["goal_reached_rows"] == 2
    assert summary["goal_reached_unavailable_rows"] == 1


def test_campaign_rejects_mutated_frozen_inputs() -> None:
    """The diagnostic cannot silently change its predeclared horizon or seed set."""
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["seeds"] = [111, 112, 114]
    with pytest.raises(ValueError, match="seeds are frozen"):
        validate_campaign_config(payload)


def test_fallback_and_over_cap_rows_are_unavailable() -> None:
    """Fallback/degraded execution and unsupported crowds cannot count as evidence."""
    config = _config()
    fallback = classify_record(
        _record(metadata={"fallback_triggered": True}), config, planner_key="social_force"
    )
    assert fallback["status"] == "unavailable"
    assert fallback["fallback_or_degraded"] is True

    crowd = [{"position": [float(idx), 4.0]} for idx in range(8)]
    over_cap = classify_record(_record(pedestrians=crowd), config, planner_key="social_force")
    assert over_cap["crowd_within_budget"] is False
    assert over_cap["status"] == "unavailable"
