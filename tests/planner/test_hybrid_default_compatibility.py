"""Default migration and complete pre-migration typed-config parity."""

import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    _resolve_policy_search_candidate_runtime,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.hybrid_rule_local_planner import build_hybrid_rule_local_planner_config
from scripts.validation.run_policy_search_step_diagnostics import _json_ready

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = json.loads(
    gzip.decompress((ROOT / "tests/fixtures/hybrid_defaults/base_dataclasses.json.gz").read_bytes())
)
PLANNER_SWITCHES = ("physical_static_exclusion_enabled", "goal_next_validity_enabled")


def test_current_defaults_enable_all_three_switches():
    """An unregistered config enables both planner repairs and the validity sensor."""
    cfg = build_hybrid_rule_local_planner_config({})
    assert cfg.physical_static_exclusion_enabled is True
    assert cfg.goal_next_validity_enabled is True
    assert RobotSimulationConfig().include_goal_next_valid is True


@pytest.mark.parametrize("switch", (*PLANNER_SWITCHES, "include_goal_next_valid"))
@pytest.mark.parametrize("value", (False, True))
def test_each_explicit_switch_overrides_the_selected_defaults(switch, value):
    """An explicit flag wins while missing independent flags keep current defaults."""
    if switch == "include_goal_next_valid":
        assert RobotSimulationConfig().include_goal_next_valid is True
        assert RobotSimulationConfig(include_goal_next_valid=value).include_goal_next_valid is value
    else:
        other = next(k for k in PLANNER_SWITCHES if k != switch)
        cfg = build_hybrid_rule_local_planner_config({switch: value})
        assert getattr(cfg, switch) is value
        assert getattr(cfg, other) is True


@pytest.mark.parametrize("source", sorted(k for k in SNAPSHOTS if k != "environment"))
def test_registered_release_full_dataclasses_and_mapping_match_base(source):
    """Current defaults differ, but registered release dumps and raw identities do not."""
    assert build_hybrid_rule_local_planner_config({}).physical_static_exclusion_enabled is True
    from robot_sf.common.hybrid_defaults import defaults_for_source, source_default_policy

    _, raw = _resolve_policy_search_candidate_runtime(
        default_algo="hybrid_rule_local_planner",
        algo_config_path=source,
        scenario={"name": "__default__"},
        config_root=ROOT,
    )
    before = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with defaults_for_source(ROOT / source):
        planner = build_hybrid_rule_local_planner_config(raw)
        env = RobotSimulationConfig()
    assert _json_ready(asdict(planner)) == SNAPSHOTS[source]
    assert (
        _json_ready(asdict(build_hybrid_rule_local_planner_config(raw, source_path=ROOT / source)))
        == SNAPSHOTS[source]
    )
    env_dump = json.dumps(_json_ready(asdict(env)), sort_keys=True).replace(str(ROOT), "<repo>")
    assert json.loads(env_dump) == SNAPSHOTS["environment"]
    assert json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False) == before
    registry = json.loads((ROOT / "robot_sf/common/legacy_hybrid_defaults.json").read_text())
    assert (
        hashlib.sha256(before.encode()).hexdigest()
        == registry["sources"][source]["effective_config_sha256"]
    )
    assert source_default_policy(ROOT / source)["default_set"] == "legacy-0.0.8"
    # Explicit on values must also win in the legacy context.
    with defaults_for_source(ROOT / source):
        explicit = build_hybrid_rule_local_planner_config(
            dict(raw, **dict.fromkeys(PLANNER_SWITCHES, True))
        )
        assert all(getattr(explicit, k) is True for k in PLANNER_SWITCHES)
        assert RobotSimulationConfig(include_goal_next_valid=True).include_goal_next_valid is True
    assert RobotSimulationConfig().include_goal_next_valid is True


def test_registry_requires_known_source_and_matching_bytes(tmp_path, monkeypatch):
    """A plausible frozen filename cannot grant legacy defaults outside the registry."""
    assert RobotSimulationConfig().include_goal_next_valid is True
    from robot_sf.common.hybrid_defaults import defaults_for_source, source_default_policy

    source = tmp_path / "fake_0_0_8_frozen.yaml"
    source.write_text("planner_variant: hybrid_rule_v4_clearance_braking\n")
    with defaults_for_source(source):
        cfg = build_hybrid_rule_local_planner_config({})
        assert all(getattr(cfg, k) is True for k in PLANNER_SWITCHES)
        assert RobotSimulationConfig().include_goal_next_valid is True
    assert source_default_policy(source)["default_set"] == "current"
    from robot_sf.common import hybrid_defaults

    known = "configs/baselines/ppo_release_robot_0_0_8_cpu.yaml"
    registry = dict(hybrid_defaults.legacy_default_registry())
    registry[known] = dict(registry[known], sha256="0" * 64)
    monkeypatch.setattr(hybrid_defaults, "legacy_default_registry", lambda: registry)
    with pytest.raises(ValueError, match="Legacy default source identity changed"):
        source_default_policy(ROOT / known)


def test_release_registry_covers_learned_observation_contract():
    """Released PPO and guarded PPO retain the exact base structured space."""
    assert RobotSimulationConfig().include_goal_next_valid is True
    from robot_sf.common.hybrid_defaults import defaults_for_source
    from robot_sf.sensor.socnav_observation import socnav_observation_space

    current = RobotSimulationConfig()
    map_def = next(iter(current.map_pool.map_defs.values()))
    current_space = socnav_observation_space(map_def, current, 4)
    assert "next_valid" in current_space["goal"].spaces
    for source in (
        "configs/baselines/ppo_release_robot_0_0_8_cpu.yaml",
        "configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml",
        "configs/algos/guarded_ppo_release_v0_0_8.yaml",
        "configs/algos/guarded_ppo_camera_ready_cpu.yaml",
    ):
        with defaults_for_source(ROOT / source):
            legacy = RobotSimulationConfig()
        space = socnav_observation_space(map_def, legacy, 4)
        assert "next_valid" not in space["goal"].spaces
        # Equality covers every bound, dtype, shape, key and subspace.
        assert space == socnav_observation_space(
            map_def, RobotSimulationConfig(include_goal_next_valid=False), 4
        )


def test_release_scenario_environment_dataclasses_match_full_base_dumps():
    """All 48 registered release scenarios retain every environment field from base."""
    assert RobotSimulationConfig().include_goal_next_valid is True
    from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
    from robot_sf.training.scenario_loader import load_scenarios

    matrix = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    baseline = json.loads(
        gzip.decompress(
            (ROOT / "tests/fixtures/hybrid_defaults/base_scenario_envs.json.gz").read_bytes()
        )
    )
    scenarios = load_scenarios(matrix)
    assert {s["name"] for s in scenarios} == set(baseline)
    for scenario in scenarios:
        config = build_env_config(scenario, scenario_path=matrix)
        full_dump = json.dumps(_json_ready(asdict(config)), sort_keys=True).replace(
            str(ROOT), "<repo>"
        )
        assert json.loads(full_dump) == baseline[scenario["name"]]


def test_native_episode_records_legacy_and_current_builder_default_sets():
    """Source identity reaches the real environment/planner builders and row provenance."""
    assert RobotSimulationConfig().include_goal_next_valid is True
    from robot_sf.benchmark.map_runner.map_runner import _build_policy
    from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode
    from robot_sf.training.scenario_loader import load_scenarios

    matrix = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    scenario = load_scenarios(matrix)[0]
    source = "configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release_0_0_8_frozen.yaml"
    _, raw = _resolve_policy_search_candidate_runtime(
        default_algo="hybrid_rule_local_planner",
        algo_config_path=source,
        scenario=scenario,
        config_root=ROOT,
    )
    for expected, inputs in (
        ("legacy-0.0.8", {"algo_config_path": source}),
        ("current", {"algo_config": raw}),
    ):
        record = run_map_episode(
            scenario,
            1001,
            horizon=1,
            dt=0.1,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo="hybrid_rule_local_planner",
            scenario_path=matrix,
            policy_builder=_build_policy,
            **inputs,
        )
        assert record["algorithm_metadata"]["hybrid_default_policy"]["default_set"] == expected
