"""Regression tests for the issue #1614 LiDAR planner-compatibility audit."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.algorithm_metadata import planner_contract_for_algorithm
from robot_sf.benchmark.map_runner import _validate_sensor_fusion_adapter_config
from robot_sf.benchmark.planner_command_contract import (
    PlannerContractValidationError,
    validate_planner_contract,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATRIX_PATH = _REPO_ROOT / "configs/benchmarks/lidar/planner_compatibility_issue_1614.yaml"


@lru_cache(maxsize=1)
def _load_matrix() -> dict[str, object]:
    payload = yaml.safe_load(_MATRIX_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _planner_row(planner: str) -> dict[str, object]:
    matrix = _load_matrix()
    rows = matrix["planner_classifications"]
    for row in rows:
        if row["planner"] == planner:
            return row
    raise AssertionError(f"Missing planner row: {planner}")


def _planners_with_status(status: str) -> list[str]:
    rows = _load_matrix()["planner_classifications"]
    return [str(row["planner"]) for row in rows if row["current_contract_status"] == status]


_NATIVE_CANDIDATES = _planners_with_status("passes_lidar_2d_gate")
_BLOCKED_PLANNERS = _planners_with_status("blocked_by_current_contract")
_ADAPTER_CONFIG_REQUIRED = _planners_with_status("requires_lidar_occupancy_adapter_config")


def test_lidar_matrix_records_required_runtime_boundary() -> None:
    """The audit should make privileged runtime inputs explicit."""
    contract = _load_matrix()["observation_contract"]

    assert contract["benchmark_observation_level"] == "lidar_2d"
    assert contract["required_runtime_inputs"] == ["robot_state", "goal", "lidar_rays"]
    assert "socnav_state_pedestrian_positions" in contract["excluded_runtime_inputs"]
    assert "full_map_occupancy" in contract["excluded_runtime_inputs"]


@pytest.mark.parametrize("planner", _NATIVE_CANDIDATES)
def test_current_lidar_native_candidates_match_contract_gate(planner: str) -> None:
    """Only current sensor-fusion learned-policy contracts should pass the LiDAR gate."""
    row = _planner_row(planner)
    contract = planner_contract_for_algorithm(planner, observation_level="lidar_2d")

    assert row["current_contract_status"] == "passes_lidar_2d_gate"
    assert contract.observation_contract.observation_level == "lidar_2d"
    assert contract.observation_contract.active_mode == "sensor_fusion_state"
    assert "lidar_rays" in contract.observation_contract.required_inputs


@pytest.mark.parametrize("planner", _BLOCKED_PLANNERS)
def test_current_structured_or_grid_planners_fail_closed_for_lidar_level(planner: str) -> None:
    """Existing structured/grid planners should not silently run as LiDAR evidence."""
    row = _planner_row(planner)

    with pytest.raises(PlannerContractValidationError):
        validate_planner_contract(
            algo=planner,
            robot_kinematics="differential_drive",
            algo_config={},
            observation_level="lidar_2d",
        )

    assert row["current_contract_status"] == "blocked_by_current_contract"


@pytest.mark.parametrize("planner", _ADAPTER_CONFIG_REQUIRED)
def test_lidar_adapter_config_required_planners_fail_closed_without_adapter(
    planner: str,
) -> None:
    """Adapter-capable planners should still reject LiDAR rows without adapter config."""
    row = _planner_row(planner)
    contract = planner_contract_for_algorithm(planner, observation_level="lidar_2d")

    with pytest.raises(PlannerContractValidationError):
        validate_planner_contract(
            algo=planner,
            robot_kinematics="differential_drive",
            algo_config={},
            observation_level="lidar_2d",
        )
    with pytest.raises(ValueError, match="requires .*lidar_occupancy_adapter"):
        _validate_sensor_fusion_adapter_config(
            algo=planner,
            active_observation_mode=contract.observation_contract.active_mode,
            algo_config={},
        )

    assert row["classification"] == "adapter_required"
    assert row["current_contract_status"] == "requires_lidar_occupancy_adapter_config"
    assert contract.observation_contract.active_mode == "sensor_fusion_state"


def test_crowdnav_height_lidar_gate_keeps_human_field_caveat_visible() -> None:
    """HEIGHT may pass the named LiDAR level but still requires human-state provenance."""
    row = _planner_row("crowdnav_height")
    contract = planner_contract_for_algorithm("crowdnav_height", observation_level="lidar_2d")

    assert row["current_contract_status"] == "passes_lidar_2d_gate_but_requires_human_fields"
    assert contract.observation_contract.active_mode == "lidar_human_state"
    assert set(contract.observation_contract.required_inputs) == {
        "robot_state",
        "goal",
        "lidar_rays",
        "humans",
    }
    assert "Do not count as LiDAR-only" in row["claim_boundary"]


def test_selected_adapter_followups_are_recorded() -> None:
    """The audit should route implementation work to concrete follow-up issues."""
    adapter_paths = {path["issue"]: path for path in _load_matrix()["selected_adapter_paths"]}

    assert set(adapter_paths) == {1659, 1660, 1662}
    assert adapter_paths[1659]["adapter_contract"] == "lidar_rays_to_ego_occupancy_grid"
    assert adapter_paths[1660]["adapter_contract"] == "lidar_rays_to_tracked_agents"
    assert adapter_paths[1662]["adapter_contract"] == "train_lidar_learned_policy"
