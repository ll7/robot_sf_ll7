"""Regression tests for map-runner resume identity scoping."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.benchmark import map_runner

SCHEMA_PATH = str(
    Path(__file__).resolve().parents[2] / "robot_sf/benchmark/schemas/episode.schema.v1.json"
)


def _minimal_map_scenario() -> dict[str, object]:
    """Return a minimal supported map scenario with a deterministic seed list."""
    return {
        "name": "resume-identity-smoke",
        "metadata": {"supported": True},
        "map_file": "maps/svg_maps/classic_crossing.svg",
        "simulation_config": {"max_episode_steps": 5},
        "seeds": [1],
    }


def test_resume_identity_is_algorithm_aware(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A second algorithm run must not be skipped by resume state from the first run."""
    written_ids: set[str] = set()

    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})

    def _fake_worker(job: tuple[dict, int, dict]) -> dict[str, str]:
        """Compute the same episode identity that the real worker would return."""
        scenario, seed, params = job
        identity_payload = map_runner._scenario_identity_payload(
            scenario,
            algo=str(params.get("algo", "goal")),
            algo_config=dict(params.get("algo_config", {})),
            horizon=params.get("horizon"),
            dt=params.get("dt"),
            record_forces=bool(params.get("record_forces", True)),
            observation_mode=params.get("observation_mode"),
            observation_noise=params.get("observation_noise"),
        )
        return {"episode_id": map_runner._compute_map_episode_id(identity_payload, int(seed))}

    def _fake_write(_handle, _schema: dict, record: dict[str, str]) -> None:
        """Record written episode IDs without touching JSONL output."""
        written_ids.add(record["episode_id"])

    monkeypatch.setattr(map_runner, "_run_map_job_worker", _fake_worker)
    monkeypatch.setattr(map_runner, "_write_validated_to_handle", _fake_write)

    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text("", encoding="utf-8")

    first = map_runner.run_map_batch(
        [_minimal_map_scenario()],
        out_path,
        schema_path=SCHEMA_PATH,
        algo="goal",
        resume=False,
    )
    assert first["written"] == 1

    monkeypatch.setattr(map_runner, "index_existing", lambda _path: set(written_ids))

    second = map_runner.run_map_batch(
        [_minimal_map_scenario()],
        out_path,
        schema_path=SCHEMA_PATH,
        algo="social_force",
        resume=True,
    )
    assert second["written"] == 1


def test_resume_identity_distinguishes_uncertainty_envelope_alpha() -> None:
    """Scenario-level uncertainty-envelope alpha changes the episode identity."""
    scenario_alpha_zero = _minimal_map_scenario()
    scenario_alpha_zero["simulation_config"] = {
        "pedestrian_uncertainty_envelope_enabled": True,
        "pedestrian_uncertainty_alpha_mps": 0.0,
    }
    scenario_alpha_point_one = _minimal_map_scenario()
    scenario_alpha_point_one["simulation_config"] = {
        "pedestrian_uncertainty_envelope_enabled": True,
        "pedestrian_uncertainty_alpha_mps": 0.1,
    }

    payload_zero = map_runner._scenario_identity_payload(
        scenario_alpha_zero,
        algo="prediction_mpc",
        algo_config=map_runner._apply_scenario_uncertainty_envelope_config(
            "prediction_mpc", {}, scenario_alpha_zero
        ),
        horizon=None,
        dt=None,
        record_forces=True,
    )
    payload_point_one = map_runner._scenario_identity_payload(
        scenario_alpha_point_one,
        algo="prediction_mpc",
        algo_config=map_runner._apply_scenario_uncertainty_envelope_config(
            "prediction_mpc", {}, scenario_alpha_point_one
        ),
        horizon=None,
        dt=None,
        record_forces=True,
    )

    assert map_runner._compute_map_episode_id(
        payload_zero, 1
    ) != map_runner._compute_map_episode_id(payload_point_one, 1)


def test_scenario_uncertainty_envelope_config_threads_only_supported_planners() -> None:
    """Scenario envelope fields become planner config for MPC-family adapters only."""
    scenario = {
        "name": "uncertainty-envelope-config",
        "simulation_config": {
            "pedestrian_uncertainty_envelope_enabled": True,
            "pedestrian_uncertainty_alpha_mps": 0.1,
        },
    }

    threaded = map_runner._apply_scenario_uncertainty_envelope_config(
        "prediction_mpc", {"horizon_steps": 3}, scenario
    )
    unrelated = map_runner._apply_scenario_uncertainty_envelope_config(
        "goal", {"horizon_steps": 3}, scenario
    )

    assert threaded["pedestrian_uncertainty_envelope_enabled"] is True
    assert threaded["pedestrian_uncertainty_alpha_mps"] == 0.1
    assert unrelated == {"horizon_steps": 3}


def test_resume_identity_uses_identity_algo_observation_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Resume skipping should resolve observation mode per resolved scenario algorithm."""
    scenarios = [
        {"name": "goal", "metadata": {"supported": True}, "simulation_config": {}},
        {"name": "ppo", "metadata": {"supported": True}, "simulation_config": {}},
    ]
    runs: list[dict[str, object]] = []

    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})
    monkeypatch.setattr(
        map_runner,
        "_select_seeds",
        lambda _scenario, suite_seeds=None, suite_key=None: [1],
    )
    monkeypatch.setattr(
        map_runner,
        "_compute_map_episode_id",
        lambda payload, seed: (
            f"{payload.get('id')}--{payload.get('algo')}--{payload.get('observation_mode')}"
        ),
    )

    def fake_resolve(
        default_algo: str, algo_config_path: str | None, algo_config: dict, scenario: dict
    ):
        if scenario.get("name") == "goal":
            return "goal", {}
        return "ppo", {}

    monkeypatch.setattr(
        map_runner,
        "_resolve_policy_search_candidate_runtime",
        fake_resolve,
    )
    monkeypatch.setattr(
        map_runner,
        "_run_map_job_worker",
        lambda job: runs.append(job) or {"algorithm_metadata": {}},
    )
    monkeypatch.setattr(map_runner, "_write_validated_to_handle", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        map_runner,
        "index_existing",
        lambda _path: {
            "goal--goal--goal_state",
            "ppo--ppo--sensor_fusion_state",
        },
    )

    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text("", encoding="utf-8")

    result = map_runner.run_map_batch(
        scenarios,
        out_path,
        schema_path=tmp_path / "schema.json",
        resume=True,
    )

    assert result["written"] == 0
    assert runs == []


def test_resume_identity_uses_identity_algo_observation_level(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Resume skipping should resolve observation level per resolved scenario algorithm."""
    scenarios = [
        {"name": "goal", "metadata": {"supported": True}, "simulation_config": {}},
        {"name": "ppo", "metadata": {"supported": True}, "simulation_config": {}},
    ]
    runs: list[dict[str, object]] = []

    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})
    monkeypatch.setattr(
        map_runner,
        "_select_seeds",
        lambda _scenario, suite_seeds=None, suite_key=None: [1],
    )
    monkeypatch.setattr(
        map_runner,
        "_compute_map_episode_id",
        lambda payload, seed: (
            f"{payload.get('id')}--{payload.get('algo')}--{payload.get('observation_level')}"
        ),
    )

    def fake_resolve(
        default_algo: str, algo_config_path: str | None, algo_config: dict, scenario: dict
    ):
        if scenario.get("name") == "goal":
            return "goal", {}
        return "ppo", {}

    monkeypatch.setattr(
        map_runner,
        "_resolve_policy_search_candidate_runtime",
        fake_resolve,
    )
    monkeypatch.setattr(
        map_runner,
        "_run_map_job_worker",
        lambda job: runs.append(job) or {"algorithm_metadata": {}},
    )
    monkeypatch.setattr(map_runner, "_write_validated_to_handle", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        map_runner,
        "index_existing",
        lambda _path: {
            "goal--goal--oracle_full_state",
            "ppo--ppo--lidar_2d",
        },
    )

    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text("", encoding="utf-8")

    result = map_runner.run_map_batch(
        scenarios,
        out_path,
        schema_path=tmp_path / "schema.json",
        resume=True,
    )

    assert result["written"] == 0
    assert runs == []


def test_resume_identity_includes_algo_config_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Different config payloads for the same algorithm should produce distinct run identities."""
    written_ids: set[str] = set()

    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})

    def _fake_worker(job: tuple[dict, int, dict]) -> dict[str, str]:
        """Compute episode identity including the algorithm configuration hash."""
        scenario, seed, params = job
        identity_payload = map_runner._scenario_identity_payload(
            scenario,
            algo=str(params.get("algo", "goal")),
            algo_config=dict(params.get("algo_config", {})),
            horizon=params.get("horizon"),
            dt=params.get("dt"),
            record_forces=bool(params.get("record_forces", True)),
            observation_mode=params.get("observation_mode"),
            observation_noise=params.get("observation_noise"),
        )
        return {"episode_id": map_runner._compute_map_episode_id(identity_payload, int(seed))}

    def _fake_write(_handle, _schema: dict, record: dict[str, str]) -> None:
        """Record written episode IDs for resume-index simulation."""
        written_ids.add(record["episode_id"])

    monkeypatch.setattr(map_runner, "_run_map_job_worker", _fake_worker)
    monkeypatch.setattr(map_runner, "_write_validated_to_handle", _fake_write)

    cfg_a = tmp_path / "algo_a.yaml"
    cfg_b = tmp_path / "algo_b.yaml"
    cfg_a.write_text(yaml.safe_dump({"max_speed": 0.8}), encoding="utf-8")
    cfg_b.write_text(yaml.safe_dump({"max_speed": 1.2}), encoding="utf-8")

    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text("", encoding="utf-8")

    first = map_runner.run_map_batch(
        [_minimal_map_scenario()],
        out_path,
        schema_path=SCHEMA_PATH,
        algo="goal",
        algo_config_path=str(cfg_a),
        resume=False,
    )
    assert first["written"] == 1

    monkeypatch.setattr(map_runner, "index_existing", lambda _path: set(written_ids))

    second = map_runner.run_map_batch(
        [_minimal_map_scenario()],
        out_path,
        schema_path=SCHEMA_PATH,
        algo="goal",
        algo_config_path=str(cfg_b),
        resume=True,
    )
    assert second["written"] == 1


def test_scenario_identity_ignores_seed_schedule_fields() -> None:
    """Identity hash should be stable even when scenario seed schedule changes."""
    scenario_a = _minimal_map_scenario()
    scenario_b = {**_minimal_map_scenario(), "seeds": [7, 9, 11]}

    payload_a = map_runner._scenario_identity_payload(
        scenario_a,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
    )
    payload_b = map_runner._scenario_identity_payload(
        scenario_b,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
    )

    episode_a = map_runner._compute_map_episode_id(payload_a, seed=1)
    episode_b = map_runner._compute_map_episode_id(payload_b, seed=1)
    assert episode_a == episode_b


def test_scenario_identity_ignores_disabled_safety_wrapper() -> None:
    """Disabled safety-wrapper config must not change wrapper-off episode identity."""

    scenario = _minimal_map_scenario()
    baseline = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
    )
    wrapper_off = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        safety_wrapper={"enabled": False, "arm_key": "wrapper_off"},
    )
    wrapper_on = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
    )

    assert wrapper_off == baseline
    assert wrapper_on != baseline


def test_scenario_identity_normalizes_enabled_safety_wrapper_defaults() -> None:
    """Resume identity hashes effective wrapper-on runtime config."""

    scenario = _minimal_map_scenario()

    minimal = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
    )
    explicit_defaults = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        safety_wrapper={
            "enabled": True,
            "arm_key": "wrapper_on",
            "pedestrian_caution_radius_m": 2.0,
            "capped_speed_m_s": 0.5,
            "ttc_veto_threshold_s": 1.0,
            "clearance_veto_m": 0.3,
            "fail_on_native_action": True,
            "fail_on_unsupported_command": True,
            "record_step_trace": False,
            "false_stop_lookahead_s": 2.0,
        },
    )

    assert minimal == explicit_defaults


def test_scenario_identity_includes_observation_noise_hash() -> None:
    """Resume identity should distinguish clean and observation-noisy benchmark runs."""
    scenario = _minimal_map_scenario()

    clean = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
    )
    noisy = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        observation_noise={"profile": "unit", "pose_noise_std_m": 0.1},
    )

    assert "observation_noise_profile" not in clean
    assert noisy["observation_noise_profile"] == "unit"
    assert map_runner._compute_map_episode_id(clean, seed=1) != map_runner._compute_map_episode_id(
        noisy,
        seed=1,
    )


def test_scenario_identity_includes_record_planner_decision_trace() -> None:
    """Trace-enabled runs must not resume-reuse non-trace episode rows (issue #4425)."""
    scenario = _minimal_map_scenario()

    non_trace = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
    )
    trace = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        record_planner_decision_trace=True,
    )

    assert non_trace["record_planner_decision_trace"] is False
    assert trace["record_planner_decision_trace"] is True
    assert map_runner._compute_map_episode_id(
        non_trace, seed=1
    ) != map_runner._compute_map_episode_id(trace, seed=1)


def test_scenario_identity_includes_latency_stress_profile() -> None:
    """Resume identity should distinguish latency-stress profile variants."""
    scenario = _minimal_map_scenario()

    one_step = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=0.1,
        record_forces=True,
        latency_stress_profile={
            "name": "learned-policy-latency-stress-v0",
            "observation_delay_steps": 1,
            "action_delay_steps": 1,
            "planner_update_mode": "hold-last",
            "planner_update_period_steps": 2,
        },
    )
    two_step = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=0.1,
        record_forces=True,
        latency_stress_profile={
            "name": "learned-policy-latency-stress-v0",
            "observation_delay_steps": 2,
            "action_delay_steps": 1,
            "planner_update_mode": "hold-last",
            "planner_update_period_steps": 2,
        },
    )

    assert one_step["latency_stress_profile"]["observation_delay_steps"] == 1
    assert map_runner._compute_map_episode_id(
        one_step,
        seed=1,
    ) != map_runner._compute_map_episode_id(two_step, seed=1)


def test_resume_identity_uses_effective_latency_profile_dt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Omitted run dt should not make latency-profile resume identities drift."""
    written_ids: set[str] = set()
    latency_profile = {
        "name": "learned-policy-latency-stress-v0",
        "observation_delay_steps": 1,
        "action_delay_steps": 1,
        "planner_update_mode": "hold-last",
        "planner_update_period_steps": 2,
    }

    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})
    monkeypatch.setattr(
        map_runner,
        "_compute_map_episode_id",
        lambda payload, seed: f"{payload['latency_stress_profile']['action_delay_ms']}--{seed}",
    )

    def _fake_worker(job: tuple[dict, int, dict]) -> dict[str, str]:
        """Compute the same latency-scoped episode identity that the real worker returns."""
        scenario, seed, params = job
        identity_payload = map_runner._scenario_identity_payload(
            scenario,
            algo=str(params.get("algo", "goal")),
            algo_config=dict(params.get("algo_config", {})),
            horizon=params.get("horizon"),
            dt=params.get("dt"),
            record_forces=bool(params.get("record_forces", True)),
            observation_mode=params.get("observation_mode"),
            observation_noise=params.get("observation_noise"),
            latency_stress_profile=params.get("latency_stress_profile"),
        )
        return {"episode_id": map_runner._compute_map_episode_id(identity_payload, int(seed))}

    def _fake_write(_handle, _schema: dict, record: dict[str, str]) -> None:
        """Record written episode IDs without touching JSONL output."""
        written_ids.add(record["episode_id"])

    monkeypatch.setattr(map_runner, "_run_map_job_worker", _fake_worker)
    monkeypatch.setattr(map_runner, "_write_validated_to_handle", _fake_write)

    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text("", encoding="utf-8")

    first = map_runner.run_map_batch(
        [_minimal_map_scenario()],
        out_path,
        schema_path=SCHEMA_PATH,
        latency_stress_profile=latency_profile,
        resume=False,
    )
    assert first["written"] == 1
    assert first["latency_stress_profile"]["action_delay_ms"] == 100.0

    monkeypatch.setattr(map_runner, "index_existing", lambda _path: set(written_ids))

    second = map_runner.run_map_batch(
        [_minimal_map_scenario()],
        out_path,
        schema_path=SCHEMA_PATH,
        latency_stress_profile=latency_profile,
        resume=True,
    )

    assert second["written"] == 0


def test_scenario_identity_includes_observation_mode() -> None:
    """Observation-mode parity runs should not collide in resume identity."""
    scenario = _minimal_map_scenario()
    goal_state_payload = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        observation_mode="goal_state",
    )
    socnav_state_payload = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        observation_mode="socnav_state",
    )

    assert map_runner._compute_map_episode_id(goal_state_payload, seed=1) != (
        map_runner._compute_map_episode_id(socnav_state_payload, seed=1)
    )


def test_scenario_identity_includes_benchmark_track() -> None:
    """Track-aware rows should not collide when only the observation track changes."""
    scenario = _minimal_map_scenario()
    grid_track_payload = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        observation_mode="socnav_state",
        observation_level="tracked_agents_no_noise",
        benchmark_track="grid_socnav_v1",
        track_schema_version="observation-track.v1",
    )
    lidar_track_payload = map_runner._scenario_identity_payload(
        scenario,
        algo="goal",
        algo_config={},
        horizon=None,
        dt=None,
        record_forces=True,
        observation_mode="socnav_state",
        observation_level="tracked_agents_no_noise",
        benchmark_track="lidar_2d_v1",
        track_schema_version="observation-track.v1",
    )

    assert map_runner._compute_map_episode_id(grid_track_payload, seed=1) != (
        map_runner._compute_map_episode_id(lidar_track_payload, seed=1)
    )


def test_resume_identity_uses_identity_benchmark_track(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Resume skipping should treat benchmark_track as part of row identity."""
    scenario = _minimal_map_scenario()
    written_ids: set[str] = set()

    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})

    def _fake_worker(job: tuple[dict, int, dict]) -> dict[str, str]:
        """Compute the same track-scoped episode identity that the real worker returns."""
        scenario_payload, seed, params = job
        identity_payload = map_runner._scenario_identity_payload(
            scenario_payload,
            algo=str(params.get("algo", "goal")),
            algo_config=dict(params.get("algo_config", {})),
            horizon=params.get("horizon"),
            dt=params.get("dt"),
            record_forces=bool(params.get("record_forces", True)),
            observation_mode=params.get("observation_mode"),
            observation_level=params.get("observation_level"),
            benchmark_track=params.get("benchmark_track"),
            track_schema_version=params.get("track_schema_version"),
            observation_noise=params.get("observation_noise"),
        )
        return {"episode_id": map_runner._compute_map_episode_id(identity_payload, int(seed))}

    def _fake_write(_handle, _schema: dict, record: dict[str, str]) -> None:
        """Record written episode IDs without touching JSONL output."""
        written_ids.add(record["episode_id"])

    monkeypatch.setattr(map_runner, "_run_map_job_worker", _fake_worker)
    monkeypatch.setattr(map_runner, "_write_validated_to_handle", _fake_write)

    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text("", encoding="utf-8")

    first = map_runner.run_map_batch(
        [scenario],
        out_path,
        schema_path=SCHEMA_PATH,
        observation_level="tracked_agents_no_noise",
        benchmark_track="grid_socnav_v1",
        track_schema_version="observation-track.v1",
        resume=False,
    )
    assert first["written"] == 1

    monkeypatch.setattr(map_runner, "index_existing", lambda _path: set(written_ids))

    second = map_runner.run_map_batch(
        [scenario],
        out_path,
        schema_path=SCHEMA_PATH,
        observation_level="tracked_agents_no_noise",
        benchmark_track="lidar_2d_v1",
        track_schema_version="observation-track.v1",
        resume=True,
    )
    assert second["written"] == 1
