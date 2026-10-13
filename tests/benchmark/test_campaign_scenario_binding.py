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
from robot_sf.benchmark.provenance_identity import IDENTITY_FIELDS
from robot_sf.benchmark.utils import _config_hash
from tests.benchmark.campaign_identity_support import (
    fixture_campaign,
    producer_params,
    producer_record_identity,
)

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
    row = {
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

    return producer_record_identity(row)


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
        campaign_commit="commit-a",
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
    value = _config_hash(row["scenario_params"])
    row["config_hash"] = row["result_provenance"]["config_hash"] = value
    if isinstance(row.get("provenance"), dict):
        row["provenance"]["config_hash"] = value
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
    (tmp_path / "campaign_manifest.json").write_text(json.dumps({"git": {"commit": "commit-a"}}))
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
        campaign_commit="commit-a",
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
    producer_record_identity(row)
    return cfg, scenario, row


def _resign(row):
    value = _config_hash(row["scenario_params"])
    row["config_hash"] = row["result_provenance"]["config_hash"] = value
    if isinstance(row.get("provenance"), dict):
        row["provenance"]["config_hash"] = value


def _configured_check(tmp_path, cfg, scenario, row, boundary, *, campaign_commit="commit-a"):
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
            campaign_manifest={"git": {"commit": campaign_commit}},
            cfg=cfg,
        )
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps({"git": {"commit": campaign_commit}})
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
        campaign_commit=campaign_commit,
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
    producer_record_identity(row)
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("horizon", 60),
        ("dt", 0.2),
        ("record_forces", True),
        ("observation_mode", "vector"),
        ("observation_level", "lidar"),
        ("observation_noise_hash", "different"),
        ("tracking_precision_hash", "different"),
    ],
)
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_provenance_runtime_copy_cannot_contradict_campaign(tmp_path, field, value, boundary):
    """Even a re-signed row cannot hide contradictory producer provenance settings."""
    cfg, scenario, row = _runtime_case(tmp_path)
    _attach_producer_provenance_settings(row)
    clean = _configured_check(tmp_path, cfg, scenario, row, boundary)
    if boundary == "admission":
        assert clean["benchmark_success_allowed"] is True
    else:
        assert clean[0].verdict == "skip-complete"
    row["result_provenance"]["simulator_settings"][field] = value
    _resign(row)
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


def _attach_producer_provenance_settings(row, *, horizon=600, dt=0.1):
    """Use the real producer builder, independently of the admission oracle."""
    from robot_sf.benchmark.observation_noise import (
        normalize_observation_noise_spec,
        observation_noise_hash,
    )
    from robot_sf.benchmark.result_provenance import build_simulator_settings_provenance
    from robot_sf.benchmark.tracking_precision_contract import (
        normalize_tracking_precision_spec,
        tracking_precision_hash,
    )

    row["result_provenance"]["simulator_settings"] = build_simulator_settings_provenance(
        horizon=horizon,
        dt=dt,
        record_forces=False,
        active_observation_mode="socnav_state",
        active_observation_level="tracked_agents_no_noise",
        noise_hash=observation_noise_hash(normalize_observation_noise_spec(None)),
        tracking_precision_hash=tracking_precision_hash(normalize_tracking_precision_spec(None)),
    )


@pytest.mark.parametrize(
    "field",
    [
        "horizon",
        "dt",
        "record_forces",
        "observation_mode",
        "observation_level",
        "observation_noise_hash",
        "tracking_precision_hash",
    ],
)
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_provenance_runtime_copy_requires_complete_settings(tmp_path, field, boundary):
    """A present provenance settings copy must carry the complete producer contract."""
    cfg, scenario, row = _runtime_case(tmp_path)
    _attach_producer_provenance_settings(row)
    del row["result_provenance"]["simulator_settings"][field]
    _resign(row)
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("settings", [None, [], "invalid", "extra"])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_provenance_runtime_copy_refuses_malformed_or_extra_settings(tmp_path, settings, boundary):
    """Malformed and unknown settings cannot bypass the closed runtime comparison."""
    cfg, scenario, row = _runtime_case(tmp_path)
    if settings == "extra":
        _attach_producer_provenance_settings(row)
        row["result_provenance"]["simulator_settings"]["unknown_setting"] = True
    else:
        row["result_provenance"]["simulator_settings"] = settings
    _resign(row)
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize(
    "field,value",
    [
        ("horizon", 60),
        ("dt", 0.2),
        ("record_forces", True),
        ("observation_mode", "vector"),
        ("observation_level", "lidar"),
        ("observation_noise_hash", "different"),
        ("tracking_precision_hash", "different"),
    ],
)
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_direct_provenance_runtime_copy_cannot_contradict_campaign(
    tmp_path, field, value, boundary
):
    """Provenance labels outside the nested block are still redundant runtime copies."""
    cfg, scenario, row = _runtime_case(tmp_path)
    _attach_producer_provenance_settings(row)
    row["result_provenance"][field] = value
    _resign(row)
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_provenance_uses_resolved_scenario_budget_and_timestep(tmp_path, boundary):
    """The producer's resolved settings remain valid without explicit campaign overrides."""
    from dataclasses import replace

    cfg, scenario, row = _runtime_case(tmp_path)
    cfg = replace(cfg, horizon=None, dt=None, horizon_policy=None, protocol_version="0.0.9")
    scenario["simulation_config"]["time_per_step_in_secs"] = 0.2
    scoped = expected_arm_scenarios(
        [scenario], cfg=cfg, planner=cfg.planners[0], kinematics="differential_drive"
    )[0]
    row["scenario_params"] = producer_params(
        scoped,
        1001,
        algo="social_force",
        horizon=None,
        dt=None,
        observation_mode="socnav_state",
    )
    _attach_producer_provenance_settings(row, horizon=600, dt=0.2)
    _resign(row)
    producer_record_identity(row)
    result = _configured_check(tmp_path, cfg, scenario, row, boundary)
    if boundary == "admission":
        assert result["benchmark_success_allowed"] is True
    else:
        assert result[0].verdict == "skip-complete"


@pytest.mark.parametrize(
    "field,value",
    [
        ("config_hash", "contradictory-config"),
        (
            "telemetry_profile",
            {
                "schema_version": "analysis-telemetry-profile.v1",
                "analysis_trace": "all",
                "planner_debug_trace": "none",
            },
        ),
    ],
)
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_separate_provenance_copy_cannot_contradict_campaign(tmp_path, field, value, boundary):
    """Neither provenance format can silently mislabel an admitted or reused row."""
    cfg, scenario, row = _runtime_case(tmp_path)
    row["provenance"].update({"config_hash": row["config_hash"], "telemetry_profile": None})
    clean = _configured_check(tmp_path, cfg, scenario, row, boundary)
    if boundary == "admission":
        assert clean["benchmark_success_allowed"] is True
    else:
        assert clean[0].verdict == "skip-complete"
    row["provenance"][field] = value
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize(
    "profile,trace_available",
    [
        ("off", False),
        ("debug-only", False),
        ("all", False),
        ("all", True),
    ],
)
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_separate_provenance_accepts_actual_producer_hash_and_profile(
    tmp_path, profile, trace_available, boundary
):
    """Both trace hashes and trace-unavailable fallback hashes bind trusted inputs."""
    from robot_sf.benchmark.map_runner.map_runner_episode import _build_episode_record_dict
    from robot_sf.benchmark.map_runner.map_runner_identity import (
        _scenario_with_episode_seed_defaults,
    )
    from robot_sf.benchmark.observation_noise import normalize_observation_noise_spec
    from robot_sf.benchmark.tracking_precision_contract import normalize_tracking_precision_spec

    cfg, scenario, row = _runtime_case(tmp_path)
    enabled = profile == "all"
    telemetry = {
        "analysis_trace": "all" if enabled else "off",
        "planner_debug_trace": "all" if profile != "off" else "none",
    }
    cfg = replace(cfg, telemetry=telemetry)
    scoped = expected_arm_scenarios(
        [scenario], cfg=cfg, planner=cfg.planners[0], kinematics="differential_drive"
    )[0]
    row["scenario_params"] = producer_params(
        scoped,
        1001,
        algo="social_force",
        horizon=600,
        dt=0.1,
        observation_mode="socnav_state",
        record_simulation_step_trace=enabled,
    )
    _resign(row)
    algo_meta = {}
    if enabled:
        algo_meta["telemetry"] = {"schema_version": "analysis-telemetry-profile.v1", **telemetry}
    if trace_available:
        prepared = _scenario_with_episode_seed_defaults(scoped, seed=1001)
        algo_meta["analysis_trace"] = {
            "git_hash": "commit-a",
            "config_hash": _config_hash(
                {key: value for key, value in prepared.items() if key not in {"seed", "repeats"}}
            ),
        }
    produced = _build_episode_record_dict(
        scenario_id=row["scenario_id"],
        seed=1001,
        scenario_params=row["scenario_params"],
        metrics={},
        safety_predicates={},
        public_requirement_events={},
        algo_meta=algo_meta,
        noise_spec=normalize_observation_noise_spec(None),
        noise_stats={},
        tracking_precision_spec=normalize_tracking_precision_spec(None),
        algo="social_force",
        active_observation_mode="socnav_state",
        active_observation_level="tracked_agents_no_noise",
        ts_start="",
        ts_end="",
        status="success",
        steps_taken=1,
        horizon_val=600,
        wall_time=1.0,
        termination_reason="success",
        outcome={},
        contradictions=[],
        view_integrity=None,
    )
    row["provenance"] = produced["provenance"]
    row["algorithm_metadata"] = produced["algorithm_metadata"]
    producer_record_identity(row, telemetry_profile=algo_meta.get("telemetry"))
    result = _configured_check(tmp_path, cfg, scenario, row, boundary)
    if boundary == "admission":
        assert result["benchmark_success_allowed"] is True, result["blockers"]
    else:
        assert result[0].verdict == "skip-complete"


@pytest.mark.parametrize("field", ["config_hash", "telemetry_profile"])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_separate_provenance_requires_runtime_identity_fields(tmp_path, field, boundary):
    """A present producer block cannot omit either of its runtime identity labels."""
    cfg, scenario, row = _runtime_case(tmp_path)
    row["provenance"].update({"config_hash": row["config_hash"], "telemetry_profile": None})
    del row["provenance"][field]
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("provenance", [None, [], "invalid"])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_separate_provenance_refuses_malformed_blocks(tmp_path, provenance, boundary):
    """Malformed separate provenance is a blocker, not a bypass or exception."""
    cfg, scenario, row = _runtime_case(tmp_path)
    row["provenance"] = provenance
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("field,value", [("run_horizon", 60), ("future_runtime_setting", True)])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_separate_provenance_refuses_extra_runtime_claims(tmp_path, field, value, boundary):
    """Known mirrors must match, and new provenance fields need an explicit identity decision."""
    cfg, scenario, row = _runtime_case(tmp_path)
    row["provenance"][field] = value
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_separate_provenance_git_commit_cannot_disagree(tmp_path, boundary):
    """A divergent separate provenance commit cannot authorize admission or reuse."""
    cfg, scenario, row = _runtime_case(tmp_path)
    row["provenance"].update(
        {"git_hash": "commit-a", "config_hash": row["config_hash"], "telemetry_profile": None}
    )
    clean = _configured_check(tmp_path, cfg, scenario, row, boundary)
    if boundary == "admission":
        assert clean["benchmark_success_allowed"] is True
    else:
        assert clean[0].verdict == "skip-complete"
    row["provenance"]["git_hash"] = "different-commit"
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


_IDENTITY_COPIES = [
    (field.name, path)
    for field in IDENTITY_FIELDS
    for path in field.copies + field.conditional_copies
]


@pytest.mark.parametrize(
    "field,path",
    _IDENTITY_COPIES,
    ids=[name + ":" + ".".join(path) for name, path in _IDENTITY_COPIES],
)
@pytest.mark.parametrize("mistake", ["mutate", "omit"])
@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_every_declared_identity_copy_is_required_and_bound(
    tmp_path, field, path, mistake, boundary
):
    """Each schema field at every producer copy refuses a planted disagreement or omission."""
    cfg, scenario, row = _runtime_case(tmp_path)
    if path[0] == "algorithm_metadata":
        from robot_sf.benchmark.map_runner.map_runner_identity import (
            _scenario_with_episode_seed_defaults,
        )

        telemetry = {"analysis_trace": "all", "planner_debug_trace": "none"}
        cfg = replace(cfg, telemetry=telemetry)
        scoped = expected_arm_scenarios(
            [scenario], cfg=cfg, planner=cfg.planners[0], kinematics="differential_drive"
        )[0]
        row["scenario_params"] = producer_params(
            scoped,
            1001,
            algo="social_force",
            horizon=600,
            dt=0.1,
            observation_mode="socnav_state",
            record_simulation_step_trace=True,
        )
        prepared = _scenario_with_episode_seed_defaults(scoped, seed=1001)
        row["algorithm_metadata"] = {
            "telemetry": {"schema_version": "analysis-telemetry-profile.v1", **telemetry},
            "analysis_trace": {
                "git_hash": "commit-a",
                "config_hash": _config_hash(
                    {
                        key: value
                        for key, value in prepared.items()
                        if key not in {"seed", "repeats"}
                    }
                ),
            },
        }
        producer_record_identity(row)
    clean = _configured_check(tmp_path, cfg, scenario, row, boundary)
    if boundary == "admission":
        assert clean["benchmark_success_allowed"] is True
    else:
        assert clean[0].verdict == "skip-complete"
    target = row
    for part in path[:-1]:
        target = target[part]
    value = target[path[-1]]
    if mistake == "omit":
        del target[path[-1]]
    elif field == "seed":
        target[path[-1]] = 1002
    elif isinstance(value, dict):
        target[path[-1]] = {**value, "contradictory_identity": True}
    elif isinstance(value, bool):
        target[path[-1]] = not value
    elif isinstance(value, (int, float)):
        target[path[-1]] = value + 1
    else:
        target[path[-1]] = "contradictory-identity"
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


@pytest.mark.parametrize("boundary", ["admission", "resume"])
@pytest.mark.parametrize("commit", [None, "different-campaign-commit"])
def test_campaign_commit_is_required_and_authoritative(tmp_path, boundary, commit):
    """Missing or divergent trusted campaign provenance cannot authorize saved rows."""
    cfg, scenario, row = _runtime_case(tmp_path)
    if boundary == "admission":
        report = _configured_check(tmp_path, cfg, scenario, row, boundary, campaign_commit=commit)
        assert report["benchmark_success_allowed"] is False
    else:
        with pytest.raises(ResumeMismatchError, match="resume.*identity"):
            _configured_check(tmp_path, cfg, scenario, row, boundary, campaign_commit=commit)


@pytest.mark.parametrize("boundary", ["admission", "resume"])
def test_agreeing_commit_copies_cannot_override_trusted_campaign(tmp_path, boundary):
    """Even agreement among every saved Git copy cannot replace the selected source version."""
    cfg, scenario, row = _runtime_case(tmp_path)
    field = next(field for field in IDENTITY_FIELDS if field.name == "git_hash")
    for path in field.copies:
        target = row
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = "different-commit"
    _assert_configured_rejection(tmp_path, cfg, scenario, row, boundary)


def test_forged_saved_manifest_and_commit_copies_do_not_select_resume_source(tmp_path):
    """Resume compares the saved campaign to the actual execution source, not saved claims."""
    from robot_sf.benchmark.utils import _git_hash_fallback

    cfg, scenario, row = _runtime_case(tmp_path)
    paths = next(field.copies for field in IDENTITY_FIELDS if field.name == "git_hash")
    source = _git_hash_fallback()
    for path in paths:
        target = row
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = source
    arm = tmp_path / "runs/social_force__differential_drive"
    arm.mkdir(parents=True)
    _write(arm, row)
    manifest = tmp_path / "campaign_manifest.json"
    manifest.write_text(json.dumps({"git": {"commit": source}}))
    kwargs = {
        "planners": [PLANNER],
        "kinematics_matrix": ["differential_drive"],
        "scenarios": [scenario],
        "cfg": cfg,
    }
    assert build_resume_plan(tmp_path / "runs", **kwargs)[0].verdict == "skip-complete"
    for path in paths:
        target = row
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = "forged-source"
    _write(arm, row)
    manifest.write_text(json.dumps({"git": {"commit": "forged-source"}}))
    with pytest.raises(ResumeMismatchError, match="commit differs from executing source"):
        build_resume_plan(tmp_path / "runs", **kwargs)
