"""Construction preflight must certify the defaults used by the episode entry point."""

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.map_runner import map_runner as runner
from robot_sf.benchmark.map_runner import map_runner_episode as episode
from robot_sf.benchmark.receipt_contract import episode_inputs, inspect_construction
from robot_sf.common.hybrid_defaults import active_default_policy
from scripts.validation.run_empty_world_sweep import REPO_ROOT, build_derived_inputs


def test_registered_risk_dwa_preflight_matches_production_defaults(tmp_path, monkeypatch):
    """Observe real constructors and the production decorator without executing a step."""
    path, _ = build_derived_inputs(
        "main",
        seeds=[1001],
        arms=["risk_dwa"],
        scenarios_filter=["classic_cross_trap_low"],
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    cfg = load_campaign_config(path)
    scenario = _load_campaign_scenarios(cfg)[0]
    scoped, kwargs = episode_inputs(cfg, cfg.planners[0], scenario, "differential_drive", REPO_ROOT)
    observed = {}
    make = episode.make_robot_env

    def observe(*args, **kw):
        observed["preflight"] = {
            "policy": active_default_policy()["default_set"],
            "include_goal_next_valid": kw["config"].include_goal_next_valid,
        }
        return make(*args, **kw)

    monkeypatch.setattr(episode, "make_robot_env", observe)
    snapshot = inspect_construction(scoped, 1001, **kwargs)
    resolve = episode._resolve_episode_run_context

    class StopBeforeExecution(Exception):
        pass

    def stop(**kw):
        ctx = resolve(**kw)
        observed["production"] = {
            "policy": active_default_policy()["default_set"],
            "include_goal_next_valid": ctx.config.include_goal_next_valid,
        }
        raise StopBeforeExecution

    monkeypatch.setattr(episode, "_resolve_episode_run_context", stop)
    with pytest.raises(StopBeforeExecution):
        episode.run_map_episode(
            scoped,
            1001,
            policy_builder=runner._build_policy,
            **kwargs,
        )
    expected = {"policy": "legacy-0.0.8", "include_goal_next_valid": False}
    assert observed["production"] == expected
    assert observed["preflight"] == expected
    assert snapshot["hybrid_default_policy"]["default_set"] == "legacy-0.0.8"
    assert snapshot["environment"]["include_goal_next_valid"] is False
    assert active_default_policy() == {"default_set": "current"}
