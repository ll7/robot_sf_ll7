"""Regression tests for map-runner resume identity scoping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark import map_runner

if TYPE_CHECKING:
    from pathlib import Path


SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


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

    def _fake_write(_out: Path, _schema: dict, record: dict[str, str]) -> None:
        """Record written episode IDs without touching JSONL output."""
        written_ids.add(record["episode_id"])

    monkeypatch.setattr(map_runner, "_run_map_job_worker", _fake_worker)
    monkeypatch.setattr(map_runner, "_write_validated", _fake_write)

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
    monkeypatch.setattr(map_runner, "_write_validated", lambda *args, **kwargs: None)

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

    def _fake_write(_out: Path, _schema: dict, record: dict[str, str]) -> None:
        """Record written episode IDs for resume-index simulation."""
        written_ids.add(record["episode_id"])

    monkeypatch.setattr(map_runner, "_run_map_job_worker", _fake_worker)
    monkeypatch.setattr(map_runner, "_write_validated", _fake_write)

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
