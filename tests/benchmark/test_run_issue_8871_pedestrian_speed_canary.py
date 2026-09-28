"""Focused contracts for the issue #8871 native activation canary runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.sim.sim_config import SimulationSettings
from scripts.benchmark import run_issue_8871_pedestrian_speed_canary as canary
from scripts.validation.build_issue_8871_pedestrian_speed_canary import (
    build_manifest,
    load_canary_config,
)


def _row(*, regime_id: str = "slow_distributed", horizon_steps: int = 4) -> dict:
    configured_mean = {"slow_distributed": 0.65, "typical_distributed": 1.3}.get(regime_id)
    return {
        "identity_key": f"scenario__{regime_id}__orca__311",
        "scenario_id": "classic_head_on_corridor_medium",
        "regime_id": regime_id,
        "planner_id": "orca",
        "seed": 311,
        "horizon_steps": horizon_steps,
        "dt_seconds": 0.1,
        "runtime_controls": {
            "ped_speed_tier": None,
            "desired_speed_mean": configured_mean,
            "desired_speed_std": None if configured_mean is None else 0.2,
            "desired_speed_seed": None if configured_mean is None else "episode_seed",
        },
    }


def _record(
    *,
    preferred: tuple[float, ...] = (0.65, 0.7, 0.6, 0.8),
    actual_by_step: tuple[tuple[float, ...], ...] = (
        (0.65, 0.5, 0.5, 0.5),
        (0.65, 0.65, 0.5, 0.5),
        (0.65, 0.65, 0.65, 0.5),
        (0.65, 0.65, 0.65, 0.65),
    ),
) -> dict:
    reset = {
        "pedestrians": [
            {"id": f"simulator-slot-{index}", "velocity": [0.5, 0.0]}
            for index in range(len(preferred))
        ]
    }
    steps = []
    for step_index, speeds in enumerate(actual_by_step):
        transitions = [
            {
                "simulator_pedestrian_id": f"pysf-{ped_index}",
                "dynamics": {"preferred_speed_mps": preferred[ped_index]},
                "post_integration": {"velocity_xy": [speed, 0.0]},
            }
            for ped_index, speed in enumerate(speeds)
        ]
        steps.append(
            {
                "step": step_index,
                "oracle_transition_trace": {"transitions": transitions},
            }
        )
    return {
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "adapter"},
            "simulation_step_trace": {"dt": 0.1, "reset": reset, "steps": steps},
        }
    }


def test_extracts_native_runtime_speed_and_velocity_diagnostics() -> None:
    diagnostics = canary.extract_activation_diagnostics(
        _record(
            preferred=(1.25, 1.3, 1.2, 1.35),
            actual_by_step=(
                (1.3, 0.5, 0.5, 0.5),
                (1.3, 1.3, 0.5, 0.5),
                (1.3, 1.3, 1.3, 0.5),
                (1.3, 1.3, 1.3, 1.3),
            ),
        ),
        _row(regime_id="typical_distributed"),
    )

    assert diagnostics["initial_spawn_speed_mean_m_s"] == pytest.approx(0.5)
    assert diagnostics["initial_spawn_speed_peak_m_s"] == pytest.approx(0.5)
    assert diagnostics["desired_speed_activation_fraction"] == pytest.approx(1.0)
    assert diagnostics["acceleration_transient_steps"] == 4
    assert diagnostics["time_to_desired_speed_target_seconds"] == pytest.approx(0.4)
    assert diagnostics["runtime_max_speed_m_s_by_pedestrian"] == {
        "pysf-0": 1.25,
        "pysf-1": 1.3,
        "pysf-2": 1.2,
        "pysf-3": 1.35,
    }
    assert diagnostics["initial_spawn_velocity_xy_by_pedestrian"] == {
        "simulator-slot-0": [0.5, 0.0],
        "simulator-slot-1": [0.5, 0.0],
        "simulator-slot-2": [0.5, 0.0],
        "simulator-slot-3": [0.5, 0.0],
    }
    assert diagnostics["final_post_integration_velocity_xy_by_pedestrian"]["pysf-0"] == [
        1.3,
        0.0,
    ]


def test_inactive_intervention_is_finite_and_does_not_get_retried_by_classifier() -> None:
    row = _row(regime_id="typical_distributed")
    record = _record(
        preferred=(1.3, 1.3, 1.3, 1.3),
        actual_by_step=((0.5, 0.5, 0.5, 0.5),) * 4,
    )

    diagnostics = canary.extract_activation_diagnostics(record, row)

    assert diagnostics["desired_speed_activation_fraction"] == 0.0
    assert diagnostics["time_to_desired_speed_target_seconds"] == pytest.approx(0.4)
    assert diagnostics["acceleration_transient_steps"] == 4


def test_legacy_reference_keeps_target_fields_descriptive_only() -> None:
    diagnostics = canary.extract_activation_diagnostics(
        _record(preferred=(0.65, 0.65, 0.65, 0.65)),
        _row(regime_id="legacy_default"),
    )

    assert diagnostics["configured_desired_speed_mean_m_s"] is None
    assert diagnostics["time_to_desired_speed_target_seconds"] is None
    assert diagnostics["desired_speed_activation_fraction"] is None
    assert diagnostics["runtime_max_speed_m_s_by_pedestrian"]["pysf-0"] == pytest.approx(0.65)


def test_execution_disposition_rejects_foresight_fallback() -> None:
    record = _record()
    record["algorithm_metadata"]["foresight_prediction"] = {"fallback_used": True}

    assert canary._execution_disposition(record) == (
        "degraded",
        "foresight_prediction.fallback_used=true",
    )


def test_runtime_binding_sets_explicit_seeded_controls_and_restores_builder(monkeypatch) -> None:
    from robot_sf.benchmark.map_runner import map_runner_episode

    fake_config = SimpleNamespace(sim_config=SimulationSettings())

    def fake_builder(scenario: dict, *, scenario_path: Path) -> SimpleNamespace:
        return fake_config

    monkeypatch.setattr(map_runner_episode, "_build_env_config", fake_builder)
    controls = _row()["runtime_controls"]
    scenario = canary.build_execution_scenario({"name": "fixture"}, _row())

    with canary._runtime_binding_context(controls, seed=311):
        bound = map_runner_episode._build_env_config(scenario, scenario_path=Path("fixture.yaml"))

    assert bound.sim_config.desired_speed_mean == pytest.approx(0.65)
    assert bound.sim_config.desired_speed_std == pytest.approx(0.2)
    assert bound.sim_config.desired_speed_seed == 311
    assert map_runner_episode._build_env_config is fake_builder


def test_runtime_binding_rejects_invalid_native_control() -> None:
    config = SimpleNamespace(sim_config=SimulationSettings())
    with pytest.raises(ValueError, match="desired_speed_mean"):
        canary._apply_runtime_speed_controls(
            config,
            {
                "ped_speed_tier": None,
                "desired_speed_mean": -1.0,
                "desired_speed_std": 0.2,
                "desired_speed_seed": "episode_seed",
            },
            seed=311,
        )


def test_manifest_loader_rejects_source_drift(tmp_path: Path) -> None:
    manifest = build_manifest(load_canary_config(), source_commit="a" * 40)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(canary.CanaryError, match="source drift"):
        canary.load_execution_manifest(manifest_path, current_head="b" * 40)


def test_registry_checkpoint_accepts_release_asset_filename(monkeypatch, tmp_path: Path) -> None:
    model_bytes = b"frozen model bytes"
    expected_sha = hashlib.sha256(model_bytes).hexdigest()
    registry_path = tmp_path / "model" / "registry.yaml"
    registry_path.parent.mkdir()
    registry_path.write_text(
        "models:\n"
        "  - model_id: fixture-model\n"
        "    local_path: output/model_cache/fixture-model/model.zip\n"
        "    github_release:\n"
        "      asset_name: fixture-model-release.zip\n"
        f"      sha256: {expected_sha}\n",
        encoding="utf-8",
    )
    staged = tmp_path / "staged" / "fixture-model"
    staged.mkdir(parents=True)
    artifact = staged / "fixture-model-release.zip"
    artifact.write_bytes(model_bytes)
    monkeypatch.setattr(canary, "REPO_ROOT", tmp_path)

    result = canary._registry_checkpoint("fixture-model", tmp_path / "staged")

    assert result["path_label"] == "fixture-model/fixture-model-release.zip"
    assert result["sha256"] == expected_sha


def test_ppo_binding_preserves_registry_observation_contract_with_staged_path(
    tmp_path: Path,
) -> None:
    model_id = "ppo-fixture"
    staged = tmp_path / "ppo-model.zip"
    staged.write_bytes(b"frozen model bytes")
    promotion = {
        "benchmark_track": "grid_socnav_v1",
        "track_schema_version": "observation-track.v1",
        "observation_level": "tracked_agents_no_noise",
        "observation_mode": "dict",
    }

    bound = canary._bind_checkpoint_paths(
        "ppo",
        {"model_id": model_id, "obs_mode": "dict"},
        {
            model_id: {
                "path": staged,
                "benchmark_promotion": promotion,
            }
        },
    )

    assert bound["model_id"] is None
    assert bound["model_path"] == str(staged)
    assert bound["benchmark_promotion"] == promotion
    assert bound["benchmark_promotion"] is not promotion
    resolved = canary.resolve_learned_checkpoint_observation_contract("ppo", bound)
    assert resolved["metadata_source"] == "algo_config.benchmark_promotion"
    assert resolved["active_observation_mode"] == "socnav_state"
    assert resolved["observation_level"] == "tracked_agents_no_noise"


def test_ppo_binding_without_registry_observation_contract_fails_closed(tmp_path: Path) -> None:
    model_id = "ppo-fixture"
    staged = tmp_path / "ppo-model.zip"
    staged.write_bytes(b"frozen model bytes")

    with pytest.raises(canary.CanaryError, match="authoritative benchmark_promotion"):
        canary._bind_checkpoint_paths(
            "ppo",
            {"model_id": model_id, "obs_mode": "dict"},
            {model_id: {"path": staged, "benchmark_promotion": None}},
        )


def test_journal_event_is_json_lines_and_flushes(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    with journal_path.open("w", encoding="utf-8") as stream:
        canary._append_journal_event(stream, "row_finished", identity_key="x", row_status="native")

    assert json.loads(journal_path.read_text(encoding="utf-8")) == {
        "event": "row_finished",
        "identity_key": "x",
        "row_status": "native",
    }
