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
        scenario, seed, params = job
        identity_payload = map_runner._scenario_identity_payload(
            scenario,
            algo=str(params.get("algo", "goal")),
            algo_config=dict(params.get("algo_config", {})),
            horizon=params.get("horizon"),
            dt=params.get("dt"),
            record_forces=bool(params.get("record_forces", True)),
        )
        return {"episode_id": map_runner._compute_map_episode_id(identity_payload, int(seed))}

    def _fake_write(_out: Path, _schema: dict, record: dict[str, str]) -> None:
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


def test_resume_identity_includes_algo_config_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Different config payloads for the same algorithm should produce distinct run identities."""
    written_ids: set[str] = set()

    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})

    def _fake_worker(job: tuple[dict, int, dict]) -> dict[str, str]:
        scenario, seed, params = job
        identity_payload = map_runner._scenario_identity_payload(
            scenario,
            algo=str(params.get("algo", "goal")),
            algo_config=dict(params.get("algo_config", {})),
            horizon=params.get("horizon"),
            dt=params.get("dt"),
            record_forces=bool(params.get("record_forces", True)),
        )
        return {"episode_id": map_runner._compute_map_episode_id(identity_payload, int(seed))}

    def _fake_write(_out: Path, _schema: dict, record: dict[str, str]) -> None:
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
