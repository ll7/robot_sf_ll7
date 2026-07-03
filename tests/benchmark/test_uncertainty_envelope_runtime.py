"""Runtime smoke coverage for pedestrian uncertainty-envelope scenario threading."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark import map_runner

SCHEMA_PATH = str(
    Path(__file__).resolve().parents[2] / "robot_sf/benchmark/schemas/episode.schema.v1.json"
)


def _scenario(alpha: float) -> dict[str, object]:
    """Return a tiny deterministic scenario arm for alpha-comparison smoke tests."""
    return {
        "name": f"uncertainty-envelope-alpha-{alpha}",
        "metadata": {"supported": True},
        "map_file": "maps/svg_maps/classic_crossing.svg",
        "simulation_config": {
            "max_episode_steps": 2,
            "pedestrian_uncertainty_envelope_enabled": True,
            "pedestrian_uncertainty_alpha_mps": alpha,
        },
        "seeds": [4141],
    }


def test_uncertainty_envelope_alpha_comparison_smoke_records_disposable_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Fixed-seed alpha smoke runs both arms and records temp-artifact disposition."""
    written_records: list[dict[str, object]] = []

    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})
    monkeypatch.setattr(
        map_runner,
        "_select_seeds",
        lambda _scenario, suite_seeds=None, suite_key=None: [4141],
    )
    monkeypatch.setattr(map_runner, "index_existing", lambda _path: set())

    def _fake_worker(job: tuple[dict, int, dict]) -> dict[str, object]:
        """Return deterministic smoke metrics keyed by effective scenario alpha."""
        scenario, seed, params = job
        effective_cfg = map_runner._apply_scenario_uncertainty_envelope_config(
            str(params["algo"]),
            dict(params.get("algo_config", {})),
            scenario,
        )
        alpha = float(effective_cfg["pedestrian_uncertainty_alpha_mps"])
        identity_payload = map_runner._scenario_identity_payload(
            scenario,
            algo=str(params["algo"]),
            algo_config=effective_cfg,
            horizon=params.get("horizon"),
            dt=params.get("dt"),
            record_forces=bool(params.get("record_forces", True)),
            observation_mode=params.get("observation_mode"),
        )
        return {
            "episode_id": map_runner._compute_map_episode_id(identity_payload, int(seed)),
            "scenario_id": scenario["name"],
            "seed": int(seed),
            "metrics": {"mean_clearance": 1.0 + alpha},
            "planner_runtime": {
                "pedestrian_uncertainty_envelope": {
                    "enabled": True,
                    "alpha_mps": alpha,
                    "artifact_disposition": "pytest_tmp_path_disposable_not_benchmark_evidence",
                }
            },
        }

    def _fake_write(_handle, _schema: dict, record: dict[str, object]) -> None:
        """Record would-be JSONL rows in memory for smoke assertions."""
        written_records.append(record)

    monkeypatch.setattr(map_runner, "_run_map_job_worker", _fake_worker)
    monkeypatch.setattr(map_runner, "_write_validated_to_handle", _fake_write)

    algo_config_path = tmp_path / "prediction_mpc_testing.yaml"
    algo_config_path.write_text("allow_testing_algorithms: true\n", encoding="utf-8")

    result = map_runner.run_map_batch(
        [_scenario(0.0), _scenario(0.1)],
        tmp_path / "episodes.jsonl",
        schema_path=SCHEMA_PATH,
        algo="prediction_mpc",
        algo_config_path=str(algo_config_path),
        benchmark_profile="experimental",
        horizon=2,
        dt=0.1,
        record_forces=False,
        resume=False,
    )
    disposition_path = tmp_path / "artifact_disposition.json"
    disposition_path.write_text(
        json.dumps(
            {
                "path": str(tmp_path),
                "disposition": "pytest_tmp_path_disposable_not_benchmark_evidence",
                "benchmark_claim": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert result["written"] == 2
    by_alpha = {
        record["planner_runtime"]["pedestrian_uncertainty_envelope"]["alpha_mps"]: record
        for record in written_records
    }
    assert by_alpha[0.1]["metrics"]["mean_clearance"] > by_alpha[0.0]["metrics"]["mean_clearance"]
    assert all(
        record["planner_runtime"]["pedestrian_uncertainty_envelope"]["artifact_disposition"]
        == "pytest_tmp_path_disposable_not_benchmark_evidence"
        for record in written_records
    )
    assert json.loads(disposition_path.read_text(encoding="utf-8"))["benchmark_claim"] is False
