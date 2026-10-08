"""Full stationary-window diagnostics after pedestrian contact (issue #10212)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark import spawn_preflight as sp

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"


@pytest.mark.parametrize("seed", [1108, 1160, 1166])
def test_circular_contact_completes_stationary_window(seed):
    """Real contact at step 12/13 must not hide the remaining respawn observation."""
    scenario = next(
        row for row in sp._load_matrix(MATRIX) if row["name"] == "francis2023_circular_crossing"
    )
    result = sp._check_release_scenario((scenario, str(MATRIX), (seed,), 0.1, 20, 0.1, False))
    row = result["rows"][0]
    assert row["overall_status"] == "valid", row
    window = row["respawn_safety"]
    assert window["status"] == "pass"
    assert window["steps_checked"] == 20
    assert window["contact_events"][0]["step"] == (13 if seed == 1166 else 12)
    assert all(event["is_pedestrian_collision"] for event in window["contact_events"])
    assert window["diagnostic_continuation_steps"] == (7 if seed == 1166 else 8)
    assert row["step1_collision"] is False


class WindowEnv:
    """Model Gym's terminal boundary and public simulator/state diagnostic APIs."""

    action_space = SimpleNamespace(shape=(2,), dtype=np.dtype(np.float32))

    def __init__(self, *, contact_step=2, later="none"):
        """Set a contact boundary and an optional later diagnostic hazard."""
        self.steps = 0
        self.contact_step = contact_step
        self.later = later
        self.ended = False
        self.closed = False
        robot = SimpleNamespace(
            pose=((0.0, 0.0), 0.0),
            config=SimpleNamespace(radius=0.3),
            parse_action=tuple,
        )
        self.simulator = SimpleNamespace(
            robots=[robot],
            ped_pos=[(1.0, 0.0)],
            peds_behaviors=[SimpleNamespace(respawn_overlap_events=[])],
            step_once=self.advance,
            config=SimpleNamespace(ped_radius=0.2),
        )
        self.state = SimpleNamespace(
            step=self.update_state, meta_dict=self.metadata, is_terminal=False
        )

    def advance(self, actions):
        assert actions == [(0.0, 0.0)]
        self.steps += 1
        if self.steps == 3 and self.later == "overlap":
            self.simulator.peds_behaviors[0].respawn_overlap_events.append(
                {"ped_rows": [0], "step": 3}
            )
        if self.steps == 3 and self.later == "motion":
            self.simulator.robots[0].pose = ((0.01, 0.0), 0.0)

    def metadata(self):
        return {
            "is_pedestrian_collision": self.steps == self.contact_step,
            "is_obstacle_collision": False,
            "is_robot_collision": False,
            "is_route_complete": self.steps == 3 and self.later == "goal",
            "is_timesteps_exceeded": self.steps == 3 and self.later == "timeout",
        }

    def update_state(self):
        self.state.is_terminal = any(self.metadata().values()) or (
            self.steps == 3 and self.later == "unknown"
        )

    def step(self, action):
        assert not self.ended, "Gym episode stepping after terminal is unsupported"
        self.advance([tuple(action)])
        self.update_state()
        self.ended = self.state.is_terminal
        return None, 0.0, self.ended, False, {"meta": self.metadata()}

    def reset(self, seed):
        assert 1001 <= seed <= 1200

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("later", "status", "reason", "steps"),
    [
        ("none", "pass", "no_respawn_inside_robot_exclusion_radius", 5),
        ("overlap", "fail", "pedestrian_respawn_inside_robot_exclusion_radius", 5),
        ("motion", "invalid", "robot_did_not_remain_stationary", 3),
        ("unknown", "invalid", "episode_ended_before_respawn_window", 3),
        ("timeout", "invalid", "episode_ended_before_respawn_window", 3),
        ("goal", "invalid", "episode_ended_before_respawn_window", 3),
    ],
)
def test_contact_continuation_preserves_negative_controls(later, status, reason, steps):
    """Contact never masks a later overlap, motion, timeout or other episode end."""
    env = WindowEnv(later=later)
    result = sp._check_respawn_window(env, window_steps=5)
    assert result["reason"] == reason
    assert result["status"] == status
    assert result["steps_checked"] == steps
    assert result["contact_events"] == [
        {
            "step": 2,
            "is_pedestrian_collision": True,
            "is_obstacle_collision": False,
            "is_robot_collision": False,
            "terminated": True,
            "truncated": False,
        }
    ]
    if later == "overlap":
        assert result["first_overlap_event"] == {"ped_rows": [0], "step": 3}


@pytest.mark.parametrize("contact_step", [1, 2, 99])
def test_release_step1_collision_is_measured(monkeypatch, contact_step):
    """The row reports first-step contact, independent of later termination/overlap."""
    env = WindowEnv(contact_step=contact_step, later="unknown" if contact_step == 99 else "none")
    monkeypatch.setattr(sp, "build_env_config", lambda *_a, **_kw: object())
    monkeypatch.setattr(sp, "make_robot_env", lambda **_kw: env)
    monkeypatch.setattr(
        sp,
        "reset_spawn_clearance",
        lambda _sim: {
            "overlap": False,
            "robot_obstacle_min_surface_clearance_m": 1.0,
            "robot_pedestrian_min_surface_clearance_m": 1.0,
        },
    )
    monkeypatch.setattr(sp, "_static_map_warnings", lambda _sim: [])
    monkeypatch.setattr(sp, "_build_occupancy_analysis", lambda *_a, **_kw: {})
    monkeypatch.setattr(
        sp,
        "_check_footprint_path",
        lambda *_a, **_kw: (
            {"status": "pass"},
            {"status": "pass"},
        ),
    )
    result = sp._check_release_scenario(
        ({"name": "fixture"}, "dev.yaml", (1001,), 0.1, 5, 0.1, False)
    )
    row = result["rows"][0]
    assert row["step1_collision"] is (contact_step == 1)
    assert row["overall_status"] == ("blocked" if contact_step == 99 else "valid")
    assert env.closed
