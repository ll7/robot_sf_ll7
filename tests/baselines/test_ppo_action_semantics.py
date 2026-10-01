"""Release checkpoint action-contract checks using the shipped algorithm configs.

On base 46809b6b, the two explicit-override nodes fail with TypeError for the
unsupported action_semantics keyword, rather than the intended conflict rejection.
The two release-limit nodes pass on base: they are shipped-limit invariants,
not evidence that the delta interpretation changed.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.baselines.interface import Observation
from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.benchmark.map_runner import map_runner
from robot_sf.models import get_registry_entry

CONFIGS = {
    "ppo": "configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml",
    "guarded_ppo": "configs/algos/guarded_ppo_camera_ready_cpu.yaml",
}


def release_config(arm):
    """Read the real CPU release profile without reconstructing a fixture config."""
    return yaml.safe_load(Path(CONFIGS[arm]).read_text())


def stub_planner(arm, raw, *, fallback_to_goal=None):
    """Stub inference only; keep the production config, semantics and action wrapper."""
    config = map_runner._ppo_planner_config(release_config(arm))
    if fallback_to_goal is not None:
        config["fallback_to_goal"] = fallback_to_goal
    planner = PPOPlanner(config, defer_model_loading=True)
    planner._initialized = True
    planner._model = SimpleNamespace(predict=lambda *args, **kwargs: (np.asarray(raw), None))
    return planner


@pytest.mark.parametrize("arm", CONFIGS)
@pytest.mark.parametrize("form", ["flat", "nested", "typed"])
def test_signed_delta_precedes_release_clipping(arm, form):
    """A negative output brakes from current speed; it is not an absolute reverse command."""
    planner = stub_planner(arm, [-0.25, -0.1])
    obs = {"robot_speed": [0.6, 0.2]}
    if form == "nested":
        obs = {"robot": {"speed": [0.6, 0.2]}}
    if form == "typed":
        obs = Observation(0.1, {"speed": [0.6, 0.2]}, [])
    assert planner.step(obs) == {"v": pytest.approx(0.35), "omega": pytest.approx(0.1)}
    assert planner.get_metadata()["action_semantics"] == "velocity_delta"


@pytest.mark.parametrize("arm", CONFIGS)
@pytest.mark.parametrize("declaration", [None, "absolute_velocity", "unknown"])
def test_release_checkpoints_fail_closed_without_delta_declaration(monkeypatch, arm, declaration):
    """Missing/wrong registry semantics must fail even with defer-loading and fallback enabled."""
    config = map_runner._ppo_planner_config(release_config(arm))
    config["fallback_to_goal"] = True
    entry = dict(get_registry_entry(config["model_id"]))
    entry.pop("action_semantics", None)
    if declaration is not None:
        entry["action_semantics"] = declaration
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.get_registry_entry", lambda *args: entry, raising=False
    )
    with pytest.raises(ValueError, match="requires registry action_semantics"):
        PPOPlanner(config, defer_model_loading=True)


@pytest.mark.parametrize("arm", CONFIGS)
def test_checkpoint_declaration_cannot_be_overridden(arm):
    """An algo config must not reinterpret either pinned checkpoint as absolute velocity."""
    config = map_runner._ppo_planner_config(release_config(arm))
    config["action_semantics"] = "absolute_velocity"
    with pytest.raises(ValueError, match="conflicts"):
        PPOPlanner(config, defer_model_loading=True)


@pytest.mark.parametrize("arm", CONFIGS)
def test_delta_requires_current_speed_even_when_fallback_enabled(arm):
    """Unavailable state must not silently turn deltas into targets or goal fallback."""
    planner = stub_planner(arm, [0.3, 0.2], fallback_to_goal=True)
    assert planner.config.fallback_to_goal is True
    with pytest.raises(ValueError, match="finite current robot_speed"):
        planner.step({"robot_position": [0, 0], "goal_current": [1, 0]})


@pytest.mark.parametrize("arm", CONFIGS)
def test_delta_target_keeps_shared_release_limits(arm):
    """Shipped-limit invariant: targets stay at 2 m/s, no reverse and +/-1 rad/s."""
    planner = stub_planner(arm, [3.0, 1.0])
    assert planner.step({"robot_speed": [0.6, 0.2]}) == {"v": 2.0, "omega": 1.0}
    planner._model.predict = lambda *args, **kwargs: (np.array([-3.0, -1.0]), None)
    assert planner.step({"robot_speed": [0.6, -0.2]}) == {"v": 0.0, "omega": -1.0}


def test_guard_sees_corrected_primary_proposal(monkeypatch):
    """Guard arbitration receives the summed proposal before its existing shield/projection."""
    planner = stub_planner("guarded_ppo", [-0.25, -0.1])
    monkeypatch.setattr(map_runner, "PPOPlanner", lambda *args, **kwargs: planner)
    captured = []

    def choose(_guard, _obs, proposal):
        captured.append(proposal)
        return map_runner.ShieldDecision(
            proposed_action=proposal,
            filtered_action=proposal,
            decision_label="ppo_clear",
            intervention_reason="clear",
        )

    monkeypatch.setattr(map_runner, "_choose_shield_command", choose)
    policy, _ = map_runner._build_policy("guarded_ppo", release_config("guarded_ppo"))
    command = policy({"robot": {"speed": [0.6, 0.2], "heading": [0.0]}})
    assert captured == [pytest.approx((0.35, 0.1))]
    assert command == pytest.approx((0.35, 0.1))


@pytest.mark.parametrize("arm", CONFIGS)
def test_reconfigure_cannot_bypass_checkpoint_declaration(monkeypatch, arm):
    """Runtime reconfiguration must reject missing semantics before replacing a valid policy."""
    planner = stub_planner(arm, [-0.25, -0.1])
    previous_model = planner._model
    config = map_runner._ppo_planner_config(release_config(arm))
    entry = dict(get_registry_entry(config["model_id"]))
    entry.pop("action_semantics", None)
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.get_registry_entry", lambda *args: entry, raising=False
    )
    with pytest.raises(ValueError, match="requires registry action_semantics"):
        planner.configure(config)
    assert planner._model is previous_model


@pytest.mark.parametrize(
    ("semantics", "space", "message"),
    [
        ("unknown", "unicycle", "Unsupported PPO action_semantics"),
        ("velocity_delta", "velocity", "requires unicycle action_space"),
    ],
)
def test_invalid_delta_contracts_rejected_before_loading(semantics, space, message):
    """Unknown semantics and delta-on-holonomic configs cannot create a usable planner."""
    with pytest.raises(ValueError, match=message):
        PPOPlanner(
            PPOPlannerConfig(action_semantics=semantics, action_space=space),
            defer_model_loading=True,
        )


@pytest.mark.parametrize("raw", [[float("nan"), 0.0], [0.0, float("inf")], [0.1]])
def test_nonfinite_or_incomplete_delta_output_is_rejected(raw):
    """Inference must not turn malformed deltas into clipped, apparently valid commands."""
    planner = stub_planner("ppo", raw, fallback_to_goal=False)
    with pytest.raises(ValueError, match="two finite policy outputs"):
        planner.step({"robot_speed": [0.6, 0.2]})


@pytest.mark.parametrize("form", ["mapping", "typed"])
def test_vector_delta_policy_keeps_typed_current_speed(form):
    """Vector and typed observations preserve signed speed before decoding the raw output."""
    config = map_runner._ppo_planner_config(release_config("ppo"))
    config["obs_mode"] = "vector"
    planner = PPOPlanner(config, defer_model_loading=True)
    planner._initialized = True
    planner._model = SimpleNamespace(predict=lambda *a, **k: (np.array([-0.25, -0.1]), None))
    robot = {
        "position": [0.0, 0.0],
        "velocity": [0.6, 0.0],
        "goal": [1.0, 0.0],
        "speed": [0.6, 0.2],
    }
    obs = (
        Observation(0.1, robot, [])
        if form == "typed"
        else {"dt": 0.1, "robot": robot, "agents": []}
    )
    assert planner.step(obs) == {"v": pytest.approx(0.35), "omega": pytest.approx(0.1)}


def test_reconfigure_refreshes_action_decoding_contract():
    """Switching an absolute policy to delta semantics changes decoding, not just metadata."""
    planner = PPOPlanner(
        PPOPlannerConfig(action_space="unicycle", obs_mode="dict"), defer_model_loading=True
    )
    planner.configure(
        PPOPlannerConfig(
            action_space="unicycle", obs_mode="dict", action_semantics="velocity_delta"
        )
    )
    planner._initialized = True
    planner._model = SimpleNamespace(predict=lambda *a, **k: (np.array([-0.25, -0.1]), None))
    assert planner.step({"robot_speed": [0.6, 0.2]}) == {
        "v": pytest.approx(0.35),
        "omega": pytest.approx(0.1),
    }


def test_delta_decoder_requires_explicit_physical_state():
    """The decoder cannot reinterpret a delta as an absolute target when state is absent."""
    planner = stub_planner("ppo", [0.3, 0.2], fallback_to_goal=False)
    with pytest.raises(ValueError, match="requires current robot_speed"):
        planner._action_vec_to_dict_from_array(np.array([0.3, 0.2]))


def test_new_registry_checkpoint_decodes_signed_velocity_delta(monkeypatch):
    """A new registry ID must brake from current speed without a config override."""
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.get_registry_entry",
        lambda *_: {"action_semantics": "velocity_delta"},
    )
    planner = PPOPlanner(
        PPOPlannerConfig(model_id="release_robot_new", action_space="unicycle", obs_mode="dict"),
        defer_model_loading=True,
    )
    planner._initialized = True
    planner._model = SimpleNamespace(predict=lambda *a, **k: (np.array([-0.25, -0.1]), None))
    assert planner.step({"robot_speed": [0.6, 0.2]}) == {
        "v": pytest.approx(0.35),
        "omega": pytest.approx(0.1),
    }

    # Generalizing registry lookup must preserve both historical failure contracts:
    # optional unknown IDs defer hydration; required legacy entries fail closed.
    def unavailable_registry(_model_id):
        raise KeyError("registry entry unavailable")

    monkeypatch.setattr("robot_sf.baselines.ppo.get_registry_entry", unavailable_registry)
    deferred = PPOPlanner(
        PPOPlannerConfig(model_id="unregistered", action_space="unicycle"),
        defer_model_loading=True,
    )
    assert deferred.get_metadata()["action_semantics"] == "absolute_velocity"
    assert deferred._model is None
    assert deferred._initialized is False
    with pytest.raises(KeyError, match="registry entry unavailable"):
        PPOPlanner(
            PPOPlannerConfig(
                model_id="ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
                action_space="unicycle",
            ),
            defer_model_loading=True,
        )


@pytest.mark.parametrize("declared", ["velocity_delta", "absolute_velocity"])
def test_new_registry_checkpoint_rejects_conflicting_override(monkeypatch, declared):
    """An arbitrary registered checkpoint cannot be reinterpreted by an algo override."""
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.get_registry_entry", lambda *_: {"action_semantics": declared}
    )
    override = "absolute_velocity" if declared == "velocity_delta" else "velocity_delta"
    with pytest.raises(ValueError, match="conflicts with checkpoint registry"):
        PPOPlanner(
            PPOPlannerConfig(
                model_id="release_robot_new", action_space="unicycle", action_semantics=override
            ),
            defer_model_loading=True,
        )


def test_new_registry_checkpoint_rejects_unknown_semantics(monkeypatch):
    """An invalid registry declaration must fail before loading or fallback."""
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.get_registry_entry", lambda *_: {"action_semantics": "unknown"}
    )
    with pytest.raises(ValueError, match="Unsupported PPO action_semantics"):
        PPOPlanner(
            PPOPlannerConfig(model_id="release_robot_new", fallback_to_goal=True),
            defer_model_loading=True,
        )
