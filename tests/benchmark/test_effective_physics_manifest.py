"""Native runtime -> episode -> final campaign manifest physics regression proofs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.camera_ready import campaign
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.benchmark.map_runner.map_runner import _build_policy
from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode

ROOT = Path(__file__).resolve().parents[2]


def _episode(
    *, radius=0.47, tier="typical", grid=False, robot_force=True, reverse_cap=None, profile=None
):
    cfg = load_campaign_config(ROOT / "configs/benchmarks/camera_ready_smoke_all_planners.yaml")
    scenario = copy.deepcopy(
        next(
            s
            for s in _load_campaign_scenarios(cfg, ROOT)
            if s["name"] == "francis2023_blind_corner"
        )
    )
    scenario["seeds"] = [1001]
    scenario.setdefault("simulation_config", {}).update(
        ped_radius=radius,
        prf_config={"is_active": robot_force},
    )
    from dataclasses import replace

    from robot_sf.benchmark.map_runner import map_runner_episode

    original_factory = map_runner_episode.make_robot_env

    def configured_factory(**kwargs):
        config = kwargs["config"]
        config.sim_config = replace(
            config.sim_config, ped_speed_tier=tier, desired_speed_mean=None, desired_speed_std=None
        )
        if profile is not None:
            config.sim_config = replace(config.sim_config, **profile)
        if grid:
            from robot_sf.nav.occupancy_grid import GridConfig

            config.use_occupancy_grid = True
            config.grid_config = GridConfig()
        if reverse_cap is not None:
            config.robot_config = replace(
                config.robot_config, limited_reverse=True, max_reverse_speed=reverse_cap
            )
        return original_factory(**kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(map_runner_episode, "make_robot_env", configured_factory)
        return run_map_episode(
            scenario,
            1001,
            horizon=2,
            dt=0.1,
            algo="goal",
            record_forces=True,
            snqi_weights=None,
            snqi_baseline=None,
            scenario_path=ROOT / "scoped_scenarios.json",
            policy_builder=_build_policy,
        )


def _manifest(tmp_path, row):
    run = tmp_path / "runs" / "goal"
    run.mkdir(parents=True, exist_ok=True)
    (run / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    payload = campaign._build_campaign_manifest_payload(
        SimpleNamespace(
            campaign_root=tmp_path,
            reports_dir=tmp_path / "reports",
            manifest_payload={"schema_version": campaign.CAMPAIGN_SCHEMA_VERSION},
        ),
        outcome=SimpleNamespace(runtime_sec=0.1, campaign_finished_at_utc="2026-10-08T00:00:00Z"),
        snqi=None,
        run_meta={},
        table_paths={
            key: tmp_path / (key + ".json")
            for key in (
                "seed_variability_json_path",
                "seed_variability_csv_path",
                "seed_episode_rows_csv_path",
                "statistical_sufficiency_json_path",
            )
        },
    )
    campaign._write_json(tmp_path / "campaign_manifest.json", payload)
    return json.loads((tmp_path / "campaign_manifest.json").read_text())


def test_changed_live_config_reaches_final_campaign_manifest(tmp_path):
    """The writer must preserve nondefault runtime radii and distribution parameters."""
    row = _episode()
    manifest = _manifest(tmp_path, row)
    assert manifest["schema_version"] == "benchmark-camera-ready-campaign.v2"
    params = manifest["release_design_parameters"]
    assert params["pedestrian_physical_radius_m"] == 0.47
    assert params["pedestrian_metric_radius_m"] == 0.47
    assert params["pedestrian_force_radius_m"] == 0.35  # live PedState remains separate on main
    assert params["pedestrian_speed_mean_m_s"] == 1.3
    assert params["pedestrian_speed_sd_m_s"] == 0.2
    assert params["pedestrian_speed_cap_m_s"] == 3.0
    assert params["pedestrian_robot_force_activation_range_m"] == 3.35
    assert params["pedestrian_robot_steering_enabled"] is False
    assert params["ttc_definition"]["geometry"] == "center_based"
    assert len(params) == 15
    assert manifest["effective_physics_samples"][0]["physics"] == row["effective_physics"]
    changed = _manifest(tmp_path, _episode(radius=0.53, tier="brisk"))
    assert changed["release_design_parameters"]["pedestrian_physical_radius_m"] == 0.53
    assert changed["release_design_parameters"]["pedestrian_speed_mean_m_s"] == 1.6
    assert changed["release_design_parameters"]["pedestrian_speed_sd_m_s"] == 0.2


def test_manifest_preserves_v1_without_runtime_witnesses(tmp_path):
    """A resumed config-only campaign remains writable without invented physics."""
    row = _episode()
    row.pop("effective_physics", None)
    manifest = _manifest(tmp_path, row)
    assert manifest["schema_version"] == "benchmark-camera-ready-campaign.v1"
    assert "release_design_parameters" not in manifest
    assert not any(key.startswith("effective_physics") for key in manifest)


def test_manifest_rejects_declared_live_mismatch(tmp_path):
    """Contradictory intake fields must block the real manifest writer."""
    row = _episode()
    row["release_design_parameters"] = {"pedestrian_physical_radius_m": 9.0}
    with pytest.raises(ValueError, match="contradict runtime witness"):
        _manifest(tmp_path, row)


def test_snapshot_validator_checks_live_value_and_required_fields(monkeypatch):
    """Validate against a still-live environment, rejecting stale or incomplete records."""
    assert "effective_physics" in _episode(), "native episode lacks runtime physics witness"
    from robot_sf.benchmark import effective_physics
    from robot_sf.benchmark.map_runner import map_runner_episode

    original = map_runner_episode.capture_effective_physics
    captured = []

    def inspect(env):
        snapshot = original(env)
        effective_physics.validate_effective_physics(snapshot, env=env)
        stale = copy.deepcopy(snapshot)
        stale["release_design_parameters"]["pedestrian_physical_radius_m"] = 8.0
        with pytest.raises(ValueError, match="differs from live simulator"):
            effective_physics.validate_effective_physics(stale, env=env)
        incomplete = copy.deepcopy(snapshot)
        incomplete["release_design_parameters"].pop("wall_force_law")
        from jsonschema import ValidationError

        with pytest.raises(ValidationError, match="wall_force_law"):
            effective_physics.validate_effective_physics(incomplete)
        captured.append(snapshot)
        return snapshot

    monkeypatch.setattr(map_runner_episode, "capture_effective_physics", inspect)
    _episode()
    assert len(captured) == 1


def test_legacy_speed_is_explicit_and_not_fabricated(tmp_path):
    """Spawn-coupled speeds have per-agent caps, not configured normal parameters."""
    row = _episode(tier=None)
    manifest = _manifest(tmp_path, row)
    speed = manifest["effective_physics_samples"][0]["physics"]["pedestrian_speed_model"]
    assert speed["identity"] == "spawn_coupled_v1"
    assert speed["mean_m_s"] is None
    assert speed["sd_m_s"] is None
    assert speed["cap_m_s"] is None
    assert "pedestrian_speed_mean_m_s" not in manifest["release_design_parameters"]


def test_manifest_rejects_invalid_numeric_witness(tmp_path):
    """Even agreeing episode copies cannot admit a negative physical radius."""
    row = _episode()
    assert "effective_physics" in row, "native episode lacks runtime physics witness"
    row["effective_physics"]["release_design_parameters"]["pedestrian_physical_radius_m"] = -1
    row["release_design_parameters"]["pedestrian_physical_radius_m"] = -1
    from jsonschema import ValidationError

    with pytest.raises(ValidationError, match="minimum"):
        _manifest(tmp_path, row)


def test_scenario_specific_physics_does_not_become_global(tmp_path):
    """Mixed radii remain in scoped witnesses, without a false campaign-wide radius."""
    first = _episode(radius=0.47)
    second = _episode(radius=0.53)
    assert "effective_physics" in first, "native episode lacks runtime physics witness"
    _manifest(tmp_path, first)
    other = tmp_path / "runs" / "other"
    other.mkdir()
    (other / "episodes.jsonl").write_text(json.dumps(second) + "\n")
    manifest = _manifest(tmp_path, first)
    assert "pedestrian_physical_radius_m" not in manifest["release_design_parameters"]
    assert manifest["effective_physics_episode_count"] == 2
    assert {
        s["physics"]["release_design_parameters"]["pedestrian_physical_radius_m"]
        for s in manifest["effective_physics_samples"]
    } == {0.47, 0.53}
    from robot_sf.benchmark.effective_physics import validate_campaign_physics

    manifest["release_design_parameters"]["pedestrian_physical_radius_m"] = 0.47
    with pytest.raises(ValueError, match="global design parameters contradict"):
        validate_campaign_physics(manifest)


def test_live_disabled_force_grid_radii_and_reverse_cap(tmp_path):
    """Record active components, rasterized geometry and the plant's effective reverse bound."""
    row = _episode(grid=True, robot_force=False, reverse_cap=0.31)
    assert "effective_physics" in row, "native episode lacks runtime physics witness"
    manifest = _manifest(tmp_path, row)
    physics = manifest["effective_physics_samples"][0]["physics"]
    params = physics["release_design_parameters"]
    assert params["pedestrian_robot_force_enabled"] is False
    assert params["pedestrian_robot_force_law"]["parameters"] == []
    assert params["robot_reverse_speed_cap_m_s"] == 0.31
    grid = physics["radius_roles"]["occupancy_grid"]
    assert grid["enabled"] is True
    assert grid["radii_m"] and set(grid["radii_m"]) == {0.35}
    assert physics["integration"] == {"dt_s": 0.1, "integrator": "semi_implicit_euler"}
    assert physics["robot_kinematics"] == "DifferentialDriveRobot"
    assert physics["pedestrian_contact_law"]["hard_nonpenetration"] is False


@pytest.fixture(scope="module")
def native_physics_row():
    """One native runtime witness reused by serialization-only review regressions."""
    return _episode()


@pytest.mark.parametrize(
    "field",
    [
        "robot_kinematics",
        "radius_roles",
        "pedestrian_model",
        "integration",
        "pedestrian_contact_law",
        "group_forces",
        "per_agent_caps_m_s",
    ],
)
def test_manifest_retains_non_intake_changes_for_same_config(tmp_path, native_physics_row, field):
    """Differing full snapshots under one scenario/config identity must survive serialization."""
    first = copy.deepcopy(native_physics_row)
    second = copy.deepcopy(first)
    second["episode_id"] += "--second"
    second["seed"] = 1002
    physics = second["effective_physics"]
    if field == "robot_kinematics":
        physics[field] = "BicycleDriveRobot"
    elif field == "radius_roles":
        physics[field]["placement_m"] = 0.51
    elif field == "pedestrian_model":
        physics[field] = "hsfm_total_force_v1"
    elif field == "integration":
        physics[field]["dt_s"] = 0.2
    elif field == "pedestrian_contact_law":
        physics[field]["parameters"]["factor"] = 6.0
    elif field == "group_forces":
        physics[field] = [{"identity": "GroupGazeForce", "parameters": {"factor": 6.0}}]
    else:
        physics["pedestrian_speed_model"][field][0] += 0.01
    _manifest(tmp_path, first)
    (tmp_path / "runs" / "goal" / "episodes.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n"
    )
    from robot_sf.benchmark.effective_physics import campaign_physics, validate_campaign_physics

    manifest = {"schema_version": campaign.CAMPAIGN_SCHEMA_VERSION, **campaign_physics(tmp_path)}
    campaign._write_json(tmp_path / "campaign_manifest.json", manifest)
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert len(manifest["effective_physics_samples"]) == 2
    assert [sample["physics"] for sample in manifest["effective_physics_samples"]] == [
        first["effective_physics"],
        second["effective_physics"],
    ]
    assert validate_campaign_physics(manifest)["status"] == "complete"


def test_manifest_reports_mixed_witnesses_as_incomplete(tmp_path, native_physics_row):
    """A missing resumed row is retained as an explicit marker, without global physics claims."""
    first = copy.deepcopy(native_physics_row)
    missing = copy.deepcopy(first)
    missing["episode_id"] += "--old"
    missing["seed"] = 1002
    missing.pop("effective_physics")
    missing.pop("release_design_parameters")
    (tmp_path / "runs" / "other").mkdir(parents=True)
    (tmp_path / "runs" / "other" / "episodes.jsonl").write_text(json.dumps(missing) + "\n")
    manifest = _manifest(tmp_path, first)
    assert manifest["schema_version"] == "benchmark-camera-ready-campaign.v2"
    assert manifest["release_design_parameters"] == {}
    assert manifest["effective_physics_episode_count"] == 2
    assert manifest["effective_physics_witnessed_episode_count"] == 1
    assert manifest["effective_physics_missing_episode_count"] == 1
    absent = next(
        s for s in manifest["effective_physics_samples"] if s["episode_id"] == missing["episode_id"]
    )
    assert absent["physics_witness"] == "missing"
    assert absent["scenario_id"] == first["scenario_id"]
    assert "physics" not in absent
    from robot_sf.benchmark.effective_physics import validate_campaign_physics

    report = validate_campaign_physics(manifest)
    assert report["status"] == "incomplete"
    assert report["missing"] == [absent]
    # A claimed global value remains invalid while any episode is unwitnessed.
    manifest["release_design_parameters"] = {"pedestrian_force_radius_m": 0.35}
    with pytest.raises(ValueError, match="global design parameters.*incomplete"):
        validate_campaign_physics(manifest)


def test_capture_before_grid_regeneration_records_empty_radii():
    """An episode closed before reset/step can still snapshot an enabled, ungenerated grid."""
    from robot_sf.benchmark.effective_physics import capture_effective_physics
    from robot_sf.gym_env.environment_factory import make_robot_env
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.nav.occupancy_grid import GridConfig

    env = make_robot_env(
        config=RobotSimulationConfig(use_occupancy_grid=True, grid_config=GridConfig()), seed=1001
    )
    try:
        snapshot = capture_effective_physics(env)
        assert snapshot["radius_roles"]["occupancy_grid"]["enabled"] is True
        assert snapshot["radius_roles"]["occupancy_grid"]["radii_m"] == []
    finally:
        env.close()


@pytest.mark.parametrize(
    "truncated,identity,normal_rule",
    [
        (False, "clipped_normal_v1", "clip_normal_to_[0,high]"),
        (True, "rejection_truncated_normal_v1", "reject_normal_outside_[0,high]"),
    ],
)
def test_profile_controls_reach_native_physics_manifest(tmp_path, truncated, identity, normal_rule):
    """Native profile witnesses must retain the sampler law and explicit force radius."""
    row = _episode(
        radius=0.25,
        tier=None,
        profile={
            "ped_force_radius": 0.25,
            "desired_speed_mean": 1.29,
            "desired_speed_std": 0.19,
            "desired_speed_seed": 1001,
            "desired_speed_truncated": truncated,
        },
    )
    manifest = _manifest(tmp_path, row)
    physics = manifest["effective_physics_samples"][0]["physics"]
    speed = physics["pedestrian_speed_model"]
    assert speed["identity"] == identity
    assert speed["cap_rule"] == normal_rule + "; velocity_norm<=per_agent_desired_speed"
    assert (speed["mean_m_s"], speed["sd_m_s"], speed["cap_m_s"]) == (1.29, 0.19, 3.0)
    assert manifest["release_design_parameters"]["pedestrian_force_radius_m"] == 0.25
    assert physics["radius_roles"]["physical_contact_m"] == 0.25
    assert physics["radius_roles"]["metric_m"] == 0.25
    assert all(0 <= value <= 3 for value in speed["per_agent_caps_m_s"])
