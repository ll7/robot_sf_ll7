"""Tests for the issue #9545 narrow-doorway crash-vs-wait diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.analysis.narrow_doorway_crash_vs_wait_issue_9545 import (
    FINAL_STAGE_WEIGHTS,
    _discounted_return,
    _min_obstacle_clearance,
    _physical_pedestrian_ttc,
    _serialize_env_action,
    _step_trace_state,
    build_binding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = (
    REPO_ROOT / "docs" / "context" / "evidence" / "issue_9545_narrow_doorway_crash_vs_wait"
)


def test_bound_weights_match_training_base_config() -> None:
    """Bound eval weights equal the training base final-stage weights."""
    base = yaml.safe_load(
        (
            REPO_ROOT
            / "configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
        ).read_text(encoding="utf-8")
    )
    final_stage = base["env_factory_kwargs"]["reward_curriculum"]["stages"][-1]["reward_kwargs"][
        "weights"
    ]
    assert dict(final_stage) == dict(FINAL_STAGE_WEIGHTS)
    assert base["env_factory_kwargs"]["reward_name"] == "route_completion_v3"


def test_scenario_cap_is_400_not_600() -> None:
    """The canonical scenario declares a 400-step cap."""
    scenarios = yaml.safe_load(
        (REPO_ROOT / "configs/scenarios/single/francis2023_narrow_doorway.yaml").read_text(
            encoding="utf-8"
        )
    )["scenarios"]
    scenario = next(s for s in scenarios if s["name"] == "francis2023_narrow_doorway")
    assert int(scenario["simulation_config"]["max_episode_steps"]) == 400
    assert sorted(scenario["seeds"]) == [225, 226, 227]


def test_discounted_return_orders_crash_below_wait() -> None:
    """Crash-vs-wait arithmetic is monotone in the collision penalty."""
    wait = [-0.015] * 11
    crash = [0.05] * 10 + [-15.0]
    assert _discounted_return(crash, 0.99) < _discounted_return(wait, 0.99)


def test_obstacle_clearance_parses_legacy_flat_endpoints() -> None:
    """MapDefinition ``[x1, x2, y1, y2]`` rows must not become diagonals."""
    simulator = SimpleNamespace(
        robot_pos=np.array([[1.0, 2.2]], dtype=float),
        map_def=SimpleNamespace(obstacles_pysf=[(0.0, 2.0, 1.0, 1.0)]),
    )
    env = SimpleNamespace(
        simulator=simulator,
        env_config=SimpleNamespace(robot_config=SimpleNamespace(radius=1.0)),
    )
    assert _min_obstacle_clearance(env) == pytest.approx(0.2)


def test_physical_pedestrian_ttc_estimates_head_on_contact() -> None:
    """TTC is geometric relative-motion contact time, not a reward component."""
    ttc = _physical_pedestrian_ttc(
        robot_position=[0.0, 0.0],
        robot_velocity=[1.0, 0.0],
        robot_radius=0.5,
        pedestrian_positions=[[5.0, 0.0]],
        pedestrian_velocities=[[-1.0, 0.0]],
        pedestrian_radius=0.5,
    )
    assert ttc["schema_version"] == "physical-pedestrian-ttc.v1"
    assert "constant observed velocities" in ttc["definition"]
    assert ttc["estimate_s"] == pytest.approx(2.0)
    assert ttc["status"] == "estimated"
    assert ttc["pedestrian_index"] == 0
    assert ttc["combined_radius_m"] == pytest.approx(1.0)


def test_physical_pedestrian_ttc_accepts_grazing_intersection() -> None:
    """A tangent relative path counts as contact with the combined-radius disc."""
    ttc = _physical_pedestrian_ttc(
        robot_position=[0.0, 0.0],
        robot_velocity=[0.0, 0.0],
        robot_radius=0.5,
        pedestrian_positions=[[5.0, 1.0]],
        pedestrian_velocities=[[-1.0, 0.0]],
        pedestrian_radius=0.5,
    )
    assert ttc["estimate_s"] == pytest.approx(5.0)
    assert ttc["status"] == "estimated"


@pytest.mark.parametrize(
    ("pedestrian_position", "pedestrian_velocity"),
    [
        ([5.0, 0.0], [1.0, 0.0]),  # receding on the same line
        ([5.0, 2.0], [-1.0, 0.0]),  # approaching but misses the combined-radius disc
    ],
)
def test_physical_pedestrian_ttc_reports_no_intercept(
    pedestrian_position: list[float], pedestrian_velocity: list[float]
) -> None:
    """Receding and miss trajectories are null/no-intercept, never infinity."""
    ttc = _physical_pedestrian_ttc(
        robot_position=[0.0, 0.0],
        robot_velocity=[0.0, 0.0],
        robot_radius=0.5,
        pedestrian_positions=[pedestrian_position],
        pedestrian_velocities=[pedestrian_velocity],
        pedestrian_radius=0.5,
    )
    assert ttc["estimate_s"] is None
    assert ttc["status"] == "no_intercept"
    assert "constant-velocity" in ttc["reason"]


def test_physical_pedestrian_ttc_distinguishes_no_pedestrians() -> None:
    """A valid empty actor set has its own status, not an unavailable estimate."""
    ttc = _physical_pedestrian_ttc(
        robot_position=[0.0, 0.0],
        robot_velocity=[0.0, 0.0],
        robot_radius=0.5,
        pedestrian_positions=np.empty((0, 2)),
        pedestrian_velocities=np.empty((0, 2)),
        pedestrian_radius=0.5,
    )
    assert ttc["estimate_s"] is None
    assert ttc["status"] == "no_pedestrians"
    assert ttc["reason"]


def test_physical_pedestrian_ttc_fails_closed_on_unavailable_inputs() -> None:
    """Missing actor kinematics produce null/unavailable rather than a proxy value."""
    ttc = _physical_pedestrian_ttc(
        robot_position=[0.0, 0.0],
        robot_velocity=[0.0, 0.0],
        robot_radius=0.5,
        pedestrian_positions=[[5.0, 0.0]],
        pedestrian_velocities=None,
        pedestrian_radius=0.5,
    )
    assert ttc["estimate_s"] is None
    assert ttc["status"] == "unavailable"
    assert "velocities" in ttc["reason"]


def test_trace_state_preserves_executed_action_and_ttc_inputs() -> None:
    """Measured/counterfactual row helpers retain action identity and TTC schema."""
    action = np.asarray([0.125, -0.75], dtype=np.float64)
    serialized_action = _serialize_env_action(action)
    simulator = SimpleNamespace(
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=(1.0, 0.0),
                config=SimpleNamespace(radius=0.5),
            )
        ],
        config=SimpleNamespace(ped_radius=0.5),
        ped_pos=np.asarray([[5.0, 0.0]], dtype=float),
        ped_vel=np.asarray([[-1.0, 0.0]], dtype=float),
    )

    fields = _step_trace_state(simulator)

    assert serialized_action == [0.125, -0.75]
    assert fields["robot_velocity_mps"] == [1.0, 0.0]
    assert fields["ped_positions"] == [[5.0, 0.0]]
    assert fields["ped_velocities_mps"] == [[-1.0, 0.0]]
    assert fields["pedestrian_ttc"]["schema_version"] == "physical-pedestrian-ttc.v1"
    assert fields["pedestrian_ttc"]["estimate_s"] == pytest.approx(2.0)


def test_env_action_trace_rejects_non_vector_or_non_finite_values() -> None:
    """The recorded action cannot silently flatten or sanitize malformed inputs."""
    with pytest.raises(ValueError, match="one-dimensional"):
        _serialize_env_action(np.asarray([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="finite"):
        _serialize_env_action(np.asarray([np.nan, 0.0]))


def test_binding_records_gamma_provenance_gap(tmp_path: Path) -> None:
    """Binding states the training-gamma provenance explicitly."""
    binding = build_binding(tmp_path, 0.99)
    assert binding["scenario_cap_steps"] == 400
    assert binding["checkpoint_gamma_embedded"] == 0.99
    assert binding["gamma_training_config_declared"] is None
    assert "not proof of the training-time objective" in binding["gamma_provenance_note"]


def test_committed_return_table_prefers_wait() -> None:
    """Committed counterfactual table shows wait dominating crash on all seeds."""
    with (EVIDENCE_DIR / "return_table.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    for row in rows:
        if row["mode"] == "hold_stop":
            assert row["ends_in_contact"] == "False"
    crashes = {r["seed"]: float(r["discounted_return"]) for r in rows if "policy" in r["mode"]}
    waits = {r["seed"]: float(r["discounted_return"]) for r in rows if "hold" in r["mode"]}
    assert set(crashes) == {"225", "226", "227"} == set(waits)
    for seed in crashes:
        assert crashes[seed] < waits[seed]


def test_committed_sensitivity_never_prefers_crash() -> None:
    """Committed sensitivity sweep has no crash-preferring cell."""
    with (EVIDENCE_DIR / "sensitivity.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 27
    assert all(row["prefers_crash"] == "False" for row in rows)


def test_committed_traces_end_in_wall_contact() -> None:
    """Committed per-step traces record obstacle contact on all seeds."""
    for seed in (225, 226, 227):
        rows = [
            json.loads(line)
            for line in (EVIDENCE_DIR / f"trace_seed{seed}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert rows[-1]["is_obstacle_collision"] is True
        assert rows[-1]["terminated"] is True
        assert len(rows) < 400
        contact_terms = rows[-1]["reward_terms"]
        assert contact_terms["collision"] == -15.0
