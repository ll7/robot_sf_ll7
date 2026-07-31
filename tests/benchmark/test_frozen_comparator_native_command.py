"""Focused contract tests for the four frozen #5416 native comparator commands."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.benchmark.frozen_comparator_native_command import _build_planner
from scripts.benchmark.sipp_native_command import (
    RequestError,
    _geometry_consumption,
    _occupancy_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/benchmark/frozen_comparator_native_command.py"
COMPARATORS = {
    "hybrid_rule_v0_minimal": "configs/algos/hybrid_rule_v0_minimal.yaml",
    "teb": "configs/algos/teb_commitment_camera_ready.yaml",
    "nmpc_social": "configs/algos/nmpc_social_exploratory.yaml",
    "dwa": "configs/algos/dwa_classic.yaml",
}


def _request(
    *,
    include_geometry: bool = True,
    obstacle_segments: list[list[list[float]]] | None = None,
    pedestrian_positions: list[list[float]] | None = None,
) -> dict[str, object]:
    pedestrian_positions = pedestrian_positions or []
    geometry: dict[str, object] = {
        "schema_version": "native-command-static-geometry.v1",
        "scenario_id": "classic_head_on_corridor_low",
        "obstacle_segments": obstacle_segments or [],
        "boundary_segments": [
            [[-5.0, -5.0], [5.0, -5.0]],
            [[5.0, -5.0], [5.0, 5.0]],
            [[5.0, 5.0], [-5.0, 5.0]],
            [[-5.0, 5.0], [-5.0, -5.0]],
        ],
    }
    geometry["sha256"] = hashlib.sha256(
        json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    request: dict[str, object] = {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [0.0],
            "angular_velocity": [0.0],
        },
        "goal": {"current": [2.0, 0.0], "next": [2.0, 0.0]},
        "pedestrians": {
            "positions": pedestrian_positions,
            "velocities": [[0.0, 0.0] for _ in pedestrian_positions],
            "count": [len(pedestrian_positions)],
        },
        "sim": {"timestep": [0.1]},
    }
    if include_geometry:
        request["static_geometry"] = geometry
    return request


@pytest.mark.parametrize(("planner_id", "config_rel"), COMPARATORS.items())
def test_frozen_comparator_config_declares_matching_native_command(
    planner_id: str, config_rel: str
) -> None:
    """Every frozen config retains its registered ID and a geometry-bound command."""
    config = yaml.safe_load((REPO_ROOT / config_rel).read_text(encoding="utf-8"))
    assert config["planner_variant"] == planner_id
    assert config["native_command"]["require_static_geometry"] is True
    assert config["native_command"]["entrypoint_path"] == str(SCRIPT.relative_to(REPO_ROOT))
    assert _build_planner(planner_id, config).__class__.__name__.endswith("Adapter")


@pytest.mark.parametrize(("planner_id", "config_rel"), COMPARATORS.items())
def test_frozen_comparator_native_command_runs_with_verified_geometry(
    planner_id: str, config_rel: str
) -> None:
    """The real native process returns a finite command for the shared request contract."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--planner-id", planner_id, "--config", config_rel],
        input=json.dumps(_request()) + "\n",
        text=True,
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload["linear_velocity"], float)
    assert isinstance(payload["angular_velocity"], float)
    assert payload["geometry_consumption"]["obstacle_occupied_cells"] > 0
    assert payload["geometry_consumption"]["combined_matches_union"] is True


def test_native_occupancy_populates_canonical_geometry_channels() -> None:
    """Static and dynamic geometry occupy their channels and exact union channel."""
    config = yaml.safe_load((REPO_ROOT / COMPARATORS["dwa"]).read_text(encoding="utf-8"))
    observation = _occupancy_observation(
        _request(
            obstacle_segments=[[[0.25, -1.0], [0.25, 1.0]]],
            pedestrian_positions=[[1.0, 1.0]],
        ),
        config,
    )
    grid = np.asarray(observation["occupancy_grid"])
    indices = np.asarray(observation["occupancy_grid_meta"]["channel_indices"], dtype=int)
    obstacles = grid[indices[0]]
    pedestrians = grid[indices[1]]
    combined = grid[indices[3]]
    assert np.count_nonzero(obstacles) > 0
    assert np.count_nonzero(pedestrians) > 0
    assert np.array_equal(combined, np.maximum(obstacles, pedestrians))
    proof = _geometry_consumption(observation)
    assert proof["combined_matches_union"] is True
    assert proof["combined_occupied_cells"] >= proof["obstacle_occupied_cells"]
    assert proof["combined_occupied_cells"] >= proof["pedestrian_occupied_cells"]


@pytest.mark.parametrize(("planner_id", "config_rel"), COMPARATORS.items())
def test_blocking_static_segment_changes_comparator_behavior(
    planner_id: str,
    config_rel: str,
) -> None:
    """Each canonical comparator consumes geometry in its actual planning behavior."""
    config = yaml.safe_load((REPO_ROOT / config_rel).read_text(encoding="utf-8"))
    clear_observation = _occupancy_observation(_request(), config)
    blocked_observation = _occupancy_observation(
        _request(obstacle_segments=[[[0.25, -1.0], [0.25, 1.0]]]),
        config,
    )
    clear_planner = _build_planner(planner_id, config)
    blocked_planner = _build_planner(planner_id, config)
    clear_command = clear_planner.plan(clear_observation)
    blocked_command = blocked_planner.plan(blocked_observation)

    assert not np.allclose(blocked_command, clear_command)
    if planner_id == "hybrid_rule_v0_minimal":
        clear_diagnostics = clear_planner.diagnostics()
        blocked_diagnostics = blocked_planner.diagnostics()
        assert clear_diagnostics["last_decision"]["planner_mode"] == "NORMAL"
        assert blocked_diagnostics["last_decision"]["planner_mode"] == "EMERGENCY_STOP"
        assert blocked_diagnostics["fallback_count"] > clear_diagnostics["fallback_count"]
    elif planner_id == "dwa":
        clear_decision = clear_planner.diagnostics()["last_decision"]
        blocked_decision = blocked_planner.diagnostics()["last_decision"]
        assert clear_decision["candidate_feasible"] > 0
        assert blocked_decision["candidate_feasible"] == 0
        assert blocked_decision["constraint_reason"] == "all_candidates_infeasible_zero_command"
    elif planner_id == "nmpc_social":
        clear_diagnostics = clear_planner.diagnostics()
        blocked_diagnostics = blocked_planner.diagnostics()
        assert blocked_command[0] < clear_command[0]
        assert blocked_diagnostics["mean_abs_linear"] < clear_diagnostics["mean_abs_linear"]
        assert blocked_diagnostics["solver_successes"] == 1
    else:
        assert planner_id == "teb"
        assert blocked_command[0] < clear_command[0]


def test_frozen_comparator_rejects_wrong_identity_and_missing_geometry() -> None:
    """Native execution fails closed rather than silently changing planner or map semantics."""
    teb_config = yaml.safe_load((REPO_ROOT / COMPARATORS["teb"]).read_text(encoding="utf-8"))
    with pytest.raises(RequestError, match="planner_variant"):
        _build_planner("dwa", teb_config)

    dwa_config = yaml.safe_load((REPO_ROOT / COMPARATORS["dwa"]).read_text(encoding="utf-8"))
    with pytest.raises(RequestError, match="static_geometry"):
        _occupancy_observation(_request(include_geometry=False), dwa_config)
