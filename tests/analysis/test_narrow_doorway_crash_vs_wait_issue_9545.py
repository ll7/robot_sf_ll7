"""Tests for the issue #9545 narrow-doorway crash-vs-wait diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import scripts.analysis.narrow_doorway_crash_vs_wait_issue_9545 as producer
from scripts.analysis.narrow_doorway_crash_vs_wait_issue_9545 import (
    BASELINE_PREFLIGHT_CONFIG,
    FINAL_STAGE_WEIGHTS,
    SCENARIO_YAML,
    TRAINING_BASE_CONFIG,
    TRAINING_CONFIG,
    _assert_producer_source_commit_matches_current_source,
    _discounted_return,
    _min_obstacle_clearance,
    _physical_pedestrian_ttc,
    _replay_branch_from_prefix,
    _rollout_policy,
    _serialize_env_action,
    _serialize_trace_row,
    _sha256_file,
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


def test_physical_pedestrian_ttc_estimates_finite_slow_closing_motion() -> None:
    """The unbounded t >= 0 estimate has no undocumented minimum-speed cutoff."""
    ttc = _physical_pedestrian_ttc(
        robot_position=[0.0, 0.0],
        robot_velocity=[0.0, 0.0],
        robot_radius=0.5,
        pedestrian_positions=[[5.0, 0.0]],
        pedestrian_velocities=[[-1e-7, 0.0]],
        pedestrian_radius=0.5,
    )
    assert ttc["estimate_s"] == pytest.approx(40_000_000.0)
    assert ttc["status"] == "estimated"


def test_physical_pedestrian_ttc_fails_closed_on_non_finite_contact_time() -> None:
    """Unrepresentable finite-speed contact times stay null and standards-compliant."""
    ttc = _physical_pedestrian_ttc(
        robot_position=[0.0, 0.0],
        robot_velocity=[0.0, 0.0],
        robot_radius=0.5,
        pedestrian_positions=[[5.0, 0.0]],
        pedestrian_velocities=[[-1e-308, 0.0]],
        pedestrian_radius=0.5,
    )
    assert ttc["estimate_s"] is None
    assert ttc["status"] == "unavailable"
    assert ttc["reason"] == "non_finite_computed_ttc"
    assert json.loads(_serialize_trace_row({"pedestrian_ttc": ttc})) == {"pedestrian_ttc": ttc}


def test_physical_pedestrian_ttc_keeps_finite_first_root() -> None:
    """A finite first contact remains estimated when a later root overflows."""
    ttc = _physical_pedestrian_ttc(
        robot_position=[0.0, 0.0],
        robot_velocity=[0.0, 0.0],
        robot_radius=0.5,
        pedestrian_positions=[[1.0, 0.0]],
        pedestrian_velocities=[[-1e-308, 0.0]],
        pedestrian_radius=0.499,
    )
    assert ttc["estimate_s"] == pytest.approx(1e305)
    assert np.isfinite(ttc["estimate_s"])
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


class _FakeTraceEnv:
    """One-step producer environment that exposes distinct post-step telemetry."""

    def __init__(
        self,
        *,
        robot_x: float,
        robot_speed: float,
        pedestrian_x: float,
        pedestrian_speed: float,
    ) -> None:
        self.env_config = SimpleNamespace(robot_config=SimpleNamespace(radius=0.5))
        self.post_step_state = (robot_x, robot_speed, pedestrian_x, pedestrian_speed)
        self.executed_actions: list[np.ndarray] = []
        self.closed = False
        self.simulator = self._simulator_state(0.0, 0.0, 10.0, 0.0)

    @staticmethod
    def _simulator_state(
        robot_x: float,
        robot_speed: float,
        pedestrian_x: float,
        pedestrian_speed: float,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            robot_pos=np.asarray([[robot_x, 0.0]]),
            robot_poses=np.asarray([[robot_x, 0.0]]),
            robots=[
                SimpleNamespace(
                    pose=((robot_x, 0.0), 0.0),
                    current_speed=(robot_speed, 0.0),
                    config=SimpleNamespace(radius=0.5),
                )
            ],
            ped_pos=np.asarray([[pedestrian_x, 0.0]]),
            ped_vel=np.asarray([[pedestrian_speed, 0.0]]),
            config=SimpleNamespace(ped_radius=0.5),
            iter_obstacle_segments=lambda: [((0.0, 0.0), (0.0, 10.0))],
        )

    def reset(self, *, seed: int):
        return {"seed": seed}, {}

    def step(self, action):
        self.executed_actions.append(action)
        self.simulator = self._simulator_state(*self.post_step_state)
        return (
            {"post_step": True},
            1.25,
            True,
            False,
            {"meta": {"reward_terms": {"progress": 1.25}, "distance_to_goal": 2.0}},
        )

    def close(self) -> None:
        self.closed = True


def test_measured_trace_row_binds_executed_action_to_post_step_telemetry(monkeypatch) -> None:
    """Measured rows pair the exact env action with that step's resulting simulator state."""
    from robot_sf.benchmark.map_runner_policies import map_runner_actions

    env = _FakeTraceEnv(robot_x=2.0, robot_speed=0.25, pedestrian_x=4.0, pedestrian_speed=-0.25)
    expected_action = np.asarray([0.33, -0.44])

    def fake_action_converter(*, env, config, command):
        assert config is env.env_config
        assert command == (0.7, -0.2)
        return expected_action.copy()

    class FakePlanner:
        def step(self, observation):
            assert observation == {"initial": True}
            return {"v": 0.7, "omega": -0.2}

        @staticmethod
        def foresight_diagnostics():
            return {
                "foresight_prediction": {
                    "load_status": "loaded",
                    "effective_prediction_mode": "predictive_foresight",
                    "fallback_used": False,
                }
            }

    monkeypatch.setattr(map_runner_actions, "policy_command_to_env_action", fake_action_converter)
    monkeypatch.setattr(
        "scripts.analysis.narrow_doorway_crash_vs_wait_issue_9545._normalize_runner_obs",
        lambda observation: observation,
    )

    rows = _rollout_policy(env, FakePlanner(), max_steps=2, obs={"initial": True})

    assert len(rows) == len(env.executed_actions) == 1
    row = rows[0]
    assert row["step"] == 0
    np.testing.assert_array_equal(env.executed_actions[0], expected_action)
    assert row["env_action"] == env.executed_actions[row["step"]].tolist()
    assert row["robot_x"] == pytest.approx(2.0)
    assert row["robot_velocity_mps"] == pytest.approx([0.25, 0.0])
    assert row["ped_positions"] == [[4.0, 0.0]]
    assert row["ped_velocities_mps"] == [[-0.25, 0.0]]
    assert row["pedestrian_ttc"]["estimate_s"] == pytest.approx(2.0)
    assert row["pedestrian_ttc"]["status"] == "estimated"
    assert json.loads(_serialize_trace_row(row)) == row


def test_counterfactual_trace_row_binds_executed_action_to_post_step_telemetry(monkeypatch) -> None:
    """Counterfactual rows pair the branch action with the resulting branch state."""
    from robot_sf.benchmark.map_runner_policies import map_runner_actions

    env = _FakeTraceEnv(
        robot_x=10.0,
        robot_speed=0.05,
        pedestrian_x=12.0,
        pedestrian_speed=-0.05,
    )
    expected_action = np.asarray([0.08, -0.04])

    def fake_action_converter(*, env, config, command):
        assert config is env.env_config
        assert command == (0.0, 0.0)
        return expected_action.copy()

    monkeypatch.setattr(map_runner_actions, "policy_command_to_env_action", fake_action_converter)
    monkeypatch.setattr(
        "scripts.analysis.narrow_doorway_crash_vs_wait_issue_9545._build_diagnostic_env",
        lambda seed, reward_weights=None: (env, None, env.env_config),
    )

    rows = _replay_branch_from_prefix(
        225,
        [],
        0,
        mode="hold_stop",
        horizon=2,
        gamma=0.99,
    )

    assert len(rows) == len(env.executed_actions) == 1
    row = rows[0]
    assert row["branch_step"] == 0
    assert row["fork_step"] == 0
    np.testing.assert_array_equal(env.executed_actions[0], expected_action)
    assert row["env_action"] == env.executed_actions[row["branch_step"]].tolist()
    assert row["robot_x"] == pytest.approx(10.0)
    assert row["robot_velocity_mps"] == pytest.approx([0.05, 0.0])
    assert row["ped_positions"] == [[12.0, 0.0]]
    assert row["ped_velocities_mps"] == [[-0.05, 0.0]]
    assert row["pedestrian_ttc"]["estimate_s"] == pytest.approx(10.0)
    assert row["pedestrian_ttc"]["status"] == "estimated"
    assert json.loads(_serialize_trace_row(row)) == row
    assert env.closed


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


def test_binding_records_resolved_cli_overrides_and_hash_scopes(tmp_path: Path) -> None:
    """Binding identity follows a subset/fork override and names each hash scope."""
    binding = build_binding(
        tmp_path,
        0.99,
        resolved_seeds=(225,),
        hold_start_offset=7,
        hold_steps=11,
    )
    assert binding["resolved_cli"] == {
        "seeds": [225],
        "hold_start_offset": 7,
        "hold_steps": 11,
    }
    assert binding["counterfactual_fork"]["hold_start_offset_steps"] == 7
    assert binding["counterfactual_fork"]["hold_horizon_steps"] == 11
    assert "base_config_env_factory_kwargs_sha256" in binding
    assert "base_config_final_stage_reward_kwargs_sha256" in binding
    assert "base_config_reward_block_sha256" not in binding


def test_producer_source_commit_guard_rejects_unfrozen_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay must fail when the source commit does not contain current producer bytes."""
    monkeypatch.setattr(producer, "_git_head", lambda: "deadbeef" * 5)
    monkeypatch.setattr(producer, "_git_producer_blob_sha256", lambda *_args: "0" * 64)
    with pytest.raises(RuntimeError, match="frozen producer source commit"):
        _assert_producer_source_commit_matches_current_source(
            REPO_ROOT / "scripts/analysis/narrow_doorway_crash_vs_wait_issue_9545.py"
        )


def _assert_full_source_provenance(binding: dict) -> None:
    """Require every declared replay input and the producer source to be bound."""
    source_files = {
        "scenario_file": SCENARIO_YAML,
        "training_config": TRAINING_CONFIG,
        "training_base_config": TRAINING_BASE_CONFIG,
        "baseline_preflight_config": BASELINE_PREFLIGHT_CONFIG,
    }
    for binding_key, relative_path in source_files.items():
        assert binding[binding_key] == relative_path
        assert binding[f"{binding_key}_sha256"] == _sha256_file(REPO_ROOT / relative_path)

    producer_path = REPO_ROOT / "scripts/analysis/narrow_doorway_crash_vs_wait_issue_9545.py"
    assert binding["producer_script"] == producer_path.relative_to(REPO_ROOT).as_posix()
    assert binding["producer_script_sha256"] == _sha256_file(producer_path)


def test_binding_records_full_source_provenance(tmp_path: Path) -> None:
    """Generated binding identity covers all declared replay inputs and producer source."""
    _assert_full_source_provenance(build_binding(tmp_path, 0.99))


def test_committed_binding_records_full_source_provenance() -> None:
    """The committed evidence binding must match the current source identities."""
    binding = json.loads((EVIDENCE_DIR / "binding.json").read_text(encoding="utf-8"))
    _assert_full_source_provenance(binding)


def test_committed_traces_use_final_action_velocity_ttc_schema() -> None:
    """Every durable measured/counterfactual row uses the final producer schema."""
    trace_paths = sorted(EVIDENCE_DIR.glob("trace_seed*.jsonl")) + sorted(
        EVIDENCE_DIR.glob("counterfactual_seed*.jsonl")
    )
    assert len(trace_paths) == 9
    required_fields = {
        "env_action",
        "robot_velocity_mps",
        "ped_velocities_mps",
        "pedestrian_ttc",
    }
    required_ttc_fields = {
        "schema_version",
        "definition",
        "estimate_s",
        "status",
        "reason",
        "pedestrian_index",
        "robot_radius_m",
        "pedestrian_radius_m",
        "combined_radius_m",
    }
    for path in trace_paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows, path.name
        for row in rows:
            assert required_fields <= row.keys(), (path.name, sorted(required_fields - row.keys()))
            assert isinstance(row["pedestrian_ttc"], dict)
            assert required_ttc_fields <= row["pedestrian_ttc"].keys()


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
