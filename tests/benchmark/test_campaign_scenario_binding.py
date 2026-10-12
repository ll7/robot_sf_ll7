"""Signed scenario substitutions must fail campaign and resume admission."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config import _apply_fixed_campaign_horizon
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec
from robot_sf.benchmark.camera_ready._resume_plan import (
    ResumeMismatchError,
    _validate_resume_runtime_identities,
    build_resume_plan,
)
from robot_sf.benchmark.camera_ready._run_state import validate_campaign_integrity
from robot_sf.benchmark.camera_ready._runtime_identity import expected_arm_scenarios
from robot_sf.benchmark.utils import _config_hash
from tests.benchmark.campaign_identity_support import fixture_campaign, producer_params

SCENARIO = {
    "name": "scenario-binding",
    "map_file": "maps/fixture.svg",
    "seeds": [1001],
    "simulation_config": {"max_episode_steps": 600, "peds_have_robot_repulsion": True},
}
PLANNER = {"key": "social_force", "algo": "social_force"}


def _row(scenario, *, horizon=600):
    cfg = fixture_campaign([PLANNER], horizon=horizon, dt=0.1)
    scoped = expected_arm_scenarios(
        [scenario], cfg=cfg, planner=cfg.planners[0], kinematics="differential_drive"
    )[0]
    params = producer_params(scoped, 1001, algo="social_force", horizon=horizon, dt=0.1)
    return {
        "scenario_id": SCENARIO["name"],
        "seed": 1001,
        "algo": "social_force",
        "scenario_params": params,
        "config_hash": _config_hash(params),
        "git_hash": "commit-a",
        "result_provenance": {
            "scenario_id": SCENARIO["name"],
            "seed": 1001,
            "config_hash": _config_hash(params),
            "repo_commit": "commit-a",
        },
    }


def _write(tmp_path, row):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _integrity(tmp_path, row, *, scenario=SCENARIO, cfg=None):
    path = _write(tmp_path, row)
    return validate_campaign_integrity(
        [
            {
                "status": "ok",
                "planner": {**PLANNER, "kinematics": "differential_drive"},
                "episodes_path": str(path),
                "summary": {"episodes_total": 1},
            }
        ],
        scenarios=[scenario],
        resolved_seeds=[1001],
        campaign_root=tmp_path,
        campaign_manifest={"git": {"commit": "commit-a"}},
        cfg=cfg or fixture_campaign([PLANNER], horizon=600, dt=0.1),
    )


@pytest.mark.parametrize(
    "field", ["map_file", "max_episode_steps", "peds_have_robot_repulsion", "route_spawn_seed"]
)
def test_signed_scenario_substitution_blocks_integrity_and_resume(tmp_path, field):
    """Re-signing a wrong scenario cannot authorize its release slot or reuse."""
    clean = _row(SCENARIO)
    assert _integrity(tmp_path, clean)["benchmark_success_allowed"] is True
    _validate_resume_runtime_identities(
        _write(tmp_path, clean),
        {**PLANNER, "kinematics": "differential_drive"},
        expected_arm_scenarios(
            [SCENARIO],
            cfg=(cfg := fixture_campaign([PLANNER], horizon=600, dt=0.1)),
            planner=cfg.planners[0],
            kinematics="differential_drive",
        ),
        cfg=cfg,
    )
    altered = deepcopy(SCENARIO)
    if field == "map_file":
        altered[field] = "maps/other.svg"
    else:
        altered["simulation_config"][field] = {
            "max_episode_steps": 60,
            "peds_have_robot_repulsion": False,
            "route_spawn_seed": 1002,
        }[field]
    mutant = _row(altered)
    result = _integrity(tmp_path, mutant)
    assert result["benchmark_success_allowed"] is False
    assert any(b["invariant"] == "runtime_identity_mismatch" for b in result["blockers"])
    with pytest.raises(ResumeMismatchError, match="runtime identity mismatch"):
        _resume(tmp_path, mutant, SCENARIO)


def test_missing_saved_seed_default_blocks_integrity(tmp_path):
    """Validation must not manufacture an omitted saved stochastic input."""
    row = _row(SCENARIO)
    del row["scenario_params"]["simulation_config"]["route_spawn_seed"]
    row["config_hash"] = row["result_provenance"]["config_hash"] = _config_hash(
        row["scenario_params"]
    )
    assert _integrity(tmp_path, row)["benchmark_success_allowed"] is False


def test_trusted_legacy_scenario_preserves_authored_row_identity(tmp_path):
    """The producer's authored row still binds to an admitted historical scenario."""
    scenario = _apply_fixed_campaign_horizon(
        [deepcopy(SCENARIO)],
        horizon=60,
        horizon_policy="legacy_runner_cap",
        protocol_version="0.0.7",
    )[0]
    row = _row(scenario, horizon=60)
    assert row["scenario_params"]["simulation_config"]["max_episode_steps"] == 600
    assert row["scenario_params"]["metadata"] == {}
    cfg = replace(
        fixture_campaign([PLANNER], horizon=60, dt=0.1),
        horizon_policy="legacy_runner_cap",
        protocol_version="0.0.7",
    )
    assert (
        _integrity(tmp_path, row, scenario=scenario, cfg=cfg)["benchmark_success_allowed"] is True
    )
    assert _resume(tmp_path, row, scenario, cfg=cfg)[0].verdict == "skip-complete"


def _resume(tmp_path, row, scenario, *, cfg=None):
    runs_dir = tmp_path / "runs"
    arm_dir = runs_dir / "social_force__differential_drive"
    arm_dir.mkdir(parents=True, exist_ok=True)
    _write(arm_dir, row)
    return build_resume_plan(
        runs_dir,
        planners=[{**PLANNER, "kinematics": "differential_drive"}],
        kinematics_matrix=["differential_drive"],
        scenarios=[scenario],
        cfg=cfg or fixture_campaign([PLANNER], horizon=600, dt=0.1),
    )


@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_forged_legacy_horizon_cannot_admit_short_row(tmp_path, boundary):
    """A saved annotation cannot restore a forged short run to the selected budget."""
    scenario = {**SCENARIO, "metadata": {}}
    clean = _row(scenario)
    if boundary == "admission":
        assert _integrity(tmp_path, clean, scenario=scenario)["benchmark_success_allowed"] is True
    else:
        assert _resume(tmp_path, clean, scenario)[0].verdict == "skip-complete"
    mutant = deepcopy(clean)
    mutant["scenario_params"]["simulation_config"]["max_episode_steps"] = 60
    mutant["scenario_params"]["metadata"] = {
        "scenario_horizon": {"policy": "legacy_runner_cap", "authored_max_episode_steps": 600},
        "campaign_horizon": {"runner_horizon": 60},
    }
    mutant["config_hash"] = mutant["result_provenance"]["config_hash"] = _config_hash(
        mutant["scenario_params"]
    )
    if boundary == "admission":
        report = _integrity(tmp_path, mutant, scenario=scenario)
        assert report["benchmark_success_allowed"] is False
        assert any(b["invariant"] == "runtime_identity_mismatch" for b in report["blockers"])
    else:
        with pytest.raises(ResumeMismatchError, match="runtime identity mismatch"):
            _resume(tmp_path, mutant, scenario)


@pytest.mark.parametrize("metadata", [None, [], "malformed"])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_malformed_saved_metadata_blocks_without_crashing(tmp_path, metadata, boundary):
    """Malformed saved metadata produces a blocker or resume refusal, never AttributeError."""
    clean = _row(SCENARIO)
    if boundary == "admission":
        assert _integrity(tmp_path, clean)["benchmark_success_allowed"] is True
    else:
        assert _resume(tmp_path, clean, SCENARIO)[0].verdict == "skip-complete"
    mutant = deepcopy(clean)
    mutant["scenario_params"]["metadata"] = metadata
    mutant["config_hash"] = mutant["result_provenance"]["config_hash"] = _config_hash(
        mutant["scenario_params"]
    )
    if boundary == "admission":
        report = _integrity(tmp_path, mutant)
        assert report["benchmark_success_allowed"] is False
        assert any(b["invariant"] == "runtime_identity_mismatch" for b in report["blockers"])
    else:
        with pytest.raises(ResumeMismatchError, match="runtime identity mismatch"):
            _resume(tmp_path, mutant, SCENARIO)


def _runtime_case(tmp_path):
    cfg = CampaignConfig(
        name="runtime-dev",
        scenario_matrix_path=Path("scenarios.yaml"),
        planners=(PlannerSpec(key="social_force", algo="social_force"),),
        horizon=600,
        horizon_policy="legacy_runner_cap",
        protocol_version="0.0.7",
        dt=0.1,
        record_forces=False,
        observation_mode="socnav_state",
    )
    scenario = deepcopy(SCENARIO)
    scenario["robot_config"] = {"radius": 0.3}
    scenario["simulation_config"]["robot_goal_sampling_policy"] = "route_end"
    scoped = expected_arm_scenarios(
        [scenario], cfg=cfg, planner=cfg.planners[0], kinematics="differential_drive"
    )[0]
    row = _row(scoped)
    row["scenario_params"]["observation_mode"] = "socnav_state"
    row["scenario_params"]["observation_level"] = "tracked_agents_no_noise"
    _resign(row)
    return cfg, scenario, row


def _resign(row):
    row["config_hash"] = row["result_provenance"]["config_hash"] = _config_hash(
        row["scenario_params"]
    )


def _configured_check(tmp_path, cfg, scenario, row, boundary):
    planner = {**PLANNER, "kinematics": "differential_drive"}
    if boundary == "admission":
        return validate_campaign_integrity(
            [
                {
                    "status": "ok",
                    "planner": planner,
                    "episodes_path": str(_write(tmp_path, row)),
                    "summary": {"episodes_total": 1},
                }
            ],
            scenarios=[scenario],
            resolved_seeds=[1001],
            campaign_root=tmp_path,
            campaign_manifest={"git": {"commit": "commit-a"}},
            cfg=cfg,
        )
    runs_dir = tmp_path / "runs"
    arm_dir = runs_dir / "social_force__differential_drive"
    arm_dir.mkdir(parents=True, exist_ok=True)
    _write(arm_dir, row)
    return build_resume_plan(
        runs_dir,
        planners=[planner],
        kinematics_matrix=["differential_drive"],
        scenarios=[scenario],
        cfg=cfg,
    )


@pytest.mark.parametrize("field,value", [("run_horizon", 60), ("run_dt", 0.2)])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_campaign_runtime_overrides_cannot_be_resigned(tmp_path, field, value, boundary):
    """A legitimate alternate runtime cannot satisfy the selected campaign contract."""
    cfg, scenario, row = _runtime_case(tmp_path)
    clean = _configured_check(tmp_path, cfg, scenario, row, boundary)
    if boundary == "admission":
        assert clean["benchmark_success_allowed"] is True
    else:
        assert clean[0].verdict == "skip-complete"
    if field == "run_horizon":
        alternate = replace(cfg, horizon=60)
        scoped = expected_arm_scenarios(
            [scenario],
            cfg=alternate,
            planner=alternate.planners[0],
            kinematics="differential_drive",
        )[0]
        mutant = _row(scoped, horizon=60)
        mutant["scenario_params"]["observation_mode"] = "socnav_state"
        mutant["scenario_params"]["observation_level"] = "tracked_agents_no_noise"
    else:
        mutant = deepcopy(row)
        mutant["scenario_params"][field] = value
    _resign(mutant)
    if boundary == "admission":
        report = _configured_check(tmp_path, cfg, scenario, mutant, boundary)
        assert report["benchmark_success_allowed"] is False
        assert any(b["invariant"] == "runtime_identity_mismatch" for b in report["blockers"])
    else:
        with pytest.raises(ResumeMismatchError, match="resume.*identity"):
            _configured_check(tmp_path, cfg, scenario, mutant, boundary)


_RUNTIME_MUTATIONS = [
    ("run_horizon", 60),
    ("run_dt", 0.2),
    ("algo", "goal"),
    ("algo_config_hash", "wrong-policy"),
    ("record_forces", True),
    ("observation_mode", "vector"),
    ("observation_level", "lidar"),
    ("benchmark_track", "alternate"),
    ("track_schema_version", "v2"),
    ("observation_noise_profile", "alternate"),
    ("observation_noise_hash", "wrong-noise"),
    ("tracking_precision", {"enabled": True}),
    ("tracking_precision_hash", "wrong-tracking"),
    ("synthetic_actuation_profile", {"name": "alternate"}),
    ("latency_stress_profile", {"name": "alternate"}),
    ("safety_wrapper", {"enabled": True}),
    ("cbf_safety_filter", {"enabled": True}),
    ("record_planner_decision_trace", True),
    ("record_simulation_step_trace", True),
    ("simulation_config.max_episode_steps", 60),
    ("simulation_config.peds_have_robot_repulsion", False),
    ("simulation_config.robot_goal_sampling_policy", "alternate"),
    ("simulation_config.route_spawn_seed", 1002),
    ("simulation_config.time_per_step_in_secs", 0.2),
    ("robot_config.radius", 0.6),
    ("robot_config.kinematics", "holonomic"),
    ("map_file", "maps/other.svg"),
    ("name", "other-case"),
    ("id", "other-case"),
    ("metadata", {"forged": True}),
    ("unexpected_runtime_option", True),
]


def _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary):
    if boundary == "admission":
        report = _configured_check(tmp_path, cfg, scenario, row, boundary)
        assert report["benchmark_success_allowed"] is False
        assert any(b["invariant"] == "runtime_identity_mismatch" for b in report["blockers"])
    else:
        with pytest.raises(ResumeMismatchError, match="resume.*identity"):
            _configured_check(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("field,value", _RUNTIME_MUTATIONS, ids=[x[0] for x in _RUNTIME_MUTATIONS])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_every_runtime_field_mutation_is_refused(tmp_path, field, value, boundary):
    """Re-signing any changed/extra runtime field cannot satisfy either boundary."""
    cfg, scenario, clean = _runtime_case(tmp_path)
    result = _configured_check(tmp_path, cfg, scenario, clean, boundary)
    if boundary == "admission":
        assert result["benchmark_success_allowed"] is True
    else:
        assert result[0].verdict == "skip-complete"
    mutant = deepcopy(clean)
    target = mutant["scenario_params"]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    _resign(mutant)
    _assert_configured_rejection(tmp_path, cfg, scenario, mutant, boundary)


@pytest.mark.parametrize(
    "field",
    [
        "run_horizon",
        "run_dt",
        "algo",
        "algo_config_hash",
        "record_forces",
        "observation_mode",
        "observation_level",
        "record_planner_decision_trace",
        "record_simulation_step_trace",
        "simulation_config",
        "robot_config",
        "map_file",
        "simulation_config.route_spawn_seed",
        "simulation_config.max_episode_steps",
    ],
)
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_missing_runtime_field_is_refused(tmp_path, field, boundary):
    """A re-signed omission is unavailable, never an implicit default or wildcard."""
    cfg, scenario, mutant = _runtime_case(tmp_path)
    target = mutant["scenario_params"]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    del target[parts[-1]]
    _resign(mutant)
    _assert_configured_rejection(tmp_path, cfg, scenario, mutant, boundary)


@pytest.mark.parametrize("field", ["seed", "algo", "scenario_id"])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_record_identity_field_is_required(tmp_path, field, boundary):
    """Runtime identity includes the integer seed and every required row-level copy."""
    cfg, scenario, mutant = _runtime_case(tmp_path)
    del mutant[field]
    _resign(mutant)
    _assert_configured_rejection(tmp_path, cfg, scenario, mutant, boundary)


def test_runtime_envelope_allowlist_covers_the_producer_contract():
    """A producer envelope addition needs an explicit comparison decision."""
    from robot_sf.benchmark.camera_ready._runtime_identity import RUNTIME_IDENTITY_FIELDS
    from robot_sf.benchmark.map_runner.map_runner_identity import (
        _MAP_RUNNER_SCENARIO_IDENTITY_FIELDS,
    )

    assert RUNTIME_IDENTITY_FIELDS == _MAP_RUNNER_SCENARIO_IDENTITY_FIELDS


@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_missing_campaign_context_cannot_authorize_reuse(tmp_path, boundary):
    """An under-specified caller cannot turn a partial check into admission evidence."""
    cfg, scenario, row = _runtime_case(tmp_path)
    scoped = expected_arm_scenarios(
        [scenario], cfg=cfg, planner=cfg.planners[0], kinematics="differential_drive"
    )[0]
    _assert_configured_rejection(tmp_path, None, scoped, row, boundary)


@pytest.mark.parametrize("seed", [1002, "1001", 1001.0, True])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_seed_cannot_be_resigned_into_a_selected_slot(tmp_path, seed, boundary):
    """A different or retyped seed cannot authorize the selected campaign job."""
    cfg, scenario, row = _runtime_case(tmp_path)
    row["seed"] = row["result_provenance"]["seed"] = seed
    _resign(row)
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_expected_policy_comes_from_campaign_not_saved_arm(tmp_path, boundary):
    """A saved arm descriptor cannot authorize a different selected algorithm."""
    cfg, scenario, row = _runtime_case(tmp_path)
    cfg = replace(cfg, planners=(replace(cfg.planners[0], algo="goal"),))
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_declared_arm_overrides_remain_admissible(tmp_path, boundary):
    """The complete identity follows trusted per-arm caps/timesteps, not just defaults."""
    cfg, scenario, row = _runtime_case(tmp_path)
    cfg = replace(cfg, planners=(replace(cfg.planners[0], horizon_override=60, dt_override=0.05),))
    scoped = expected_arm_scenarios(
        [scenario], cfg=cfg, planner=cfg.planners[0], kinematics="differential_drive"
    )[0]
    row["scenario_params"] = producer_params(
        scoped, 1001, algo="social_force", horizon=60, dt=0.05, observation_mode="socnav_state"
    )
    _resign(row)
    result = _configured_check(tmp_path, cfg, scenario, row, boundary)
    if boundary == "admission":
        assert result["benchmark_success_allowed"] is True
    else:
        assert result[0].verdict == "skip-complete"


@pytest.mark.parametrize(
    "field,value",
    [
        ("horizon", 60),
        ("run_dt", 0.2),
        ("observation_mode", "vector"),
        ("observation_level", "lidar"),
        ("observation_noise_hash", "different"),
        ("tracking_precision_hash", "different"),
        ("record_forces", True),
    ],
)
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_unsigned_runtime_copy_cannot_contradict_signed_identity(tmp_path, field, value, boundary):
    """A conflicting redundant runtime label cannot silently mislabel a release row."""
    cfg, scenario, row = _runtime_case(tmp_path)
    row[field] = value
    _resign(row)
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)
