"""Tests for the adversarial manifest planner-smoke runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.scenario_manifest import (
    AdversarialScenarioManifest,
    ManifestCategory,
    ValidationRecord,
    build_manifest,
)
from scripts.tools import run_adversarial_manifest_smoke as smoke

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = _REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _write_template(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "template",
                        "map_id": "classic_cross_trap",
                        "simulation_config": {"max_episode_steps": 30, "ped_density": 0.1},
                        "robot_config": {},
                        "metadata": {"archetype": "test"},
                        "seeds": [1],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_space(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 1.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": 0.5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_run_smoke_materializes_valid_manifest_and_summarizes_planner_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The smoke runner should generate, materialize, run planners, and write summary JSON."""
    template = tmp_path / "template.yaml"
    space = tmp_path / "space.yaml"
    output_dir = tmp_path / "output"
    summary_json = tmp_path / "summary.json"
    _write_template(template)
    _write_space(space)
    calls: list[dict[str, Any]] = []

    def fake_run_batch(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        out_path = Path(kwargs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "scenario_id": "template_manifest_0000",
            "seed": 7,
            "status": "success",
            "termination_reason": "goal_reached",
            "metrics": {"success": 1.0, "collisions": 0.0, "time_to_goal_norm": 0.5},
        }
        out_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {
            "status": "success",
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "out_path": str(out_path),
        }

    monkeypatch.setattr(smoke, "run_batch", fake_run_batch)
    args = argparse.Namespace(
        search_space=space,
        scenario_template=template,
        count=1,
        seed=42,
        max_valid=1,
        output_dir=output_dir,
        summary_json=summary_json,
        schema=_SCHEMA,
        generator_family="random",
        planner=["goal", "social_force"],
        horizon=12,
        dt=0.1,
        workers=1,
    )

    payload = smoke.run_smoke(args)

    assert payload["schema_version"] == "adversarial_manifest_smoke_summary.v1"
    assert payload["evidence_classification"] == "adversarial_smoke"
    assert payload["result_classification"] == "smoke_passed"
    assert payload["validator_metadata"]["manifest_schema_version"] == (
        "adversarial_scenario_manifest.v1"
    )
    assert (
        payload["validator_metadata"]["validator_ref"]
        == "robot_sf.adversarial.scenario_manifest.validate_manifest_payload"
    )
    assert payload["generation"]["valid"] == 1
    assert payload["materialized"]["matrix_path"].endswith("materialized_matrix.yaml")
    assert len(payload["planner_runs"]) == 2
    assert payload["planner_runs"][0]["metrics"]["success"]["mean"] == 1.0
    assert len(payload["manifests"]) == 1
    assert payload["manifests"][0]["manifest_yaml_sha256"] is not None
    assert len(payload["manifests"][0]["manifest_yaml_sha256"]) == 64
    assert len(calls) == 2
    assert calls[0]["algo"] == "goal"
    assert calls[0]["record_forces"] is False
    loaded_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    contract = loaded_summary["execution_contract"]
    assert contract["planner_pair"] == ["goal", "social_force"]
    assert contract["planner_count"] == 2
    assert contract["seeds"] == [7]
    assert contract["horizon"] == 12
    assert contract["dt"] == 0.1
    assert contract["max_valid"] == 1
    assert contract["generated_candidates"] == 1
    assert contract["valid_candidates"] == 1
    assert contract["invalid_candidates"] == 0
    assert contract["degenerate_candidates"] == 0
    assert contract["outcome"] == "smoke_passed"
    assert loaded_summary["planner_runs"][1]["planner"] == "social_force"
    materialized = yaml.safe_load(
        Path(payload["materialized"]["matrix_path"]).read_text(encoding="utf-8")
    )
    scenario = materialized["scenarios"][0]
    assert scenario["simulation_config"]["ped_density"] == 0.1
    assert scenario["route_overrides_file"] == "routes/candidate_0000_route_overrides.yaml"
    assert "single_pedestrians" not in scenario
    assert scenario["metadata"]["adversarial_manifest_runtime"]["benchmark_frozen"] is False
    route_path = output_dir / scenario["route_overrides_file"]
    assert yaml.safe_load(route_path.read_text(encoding="utf-8"))["robot_routes"][0][
        "waypoints"
    ] == [[1.0, 2.0], [5.0, 2.0]]


def test_run_smoke_reports_no_valid_manifests_without_planner_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """If generated manifests are invalid, the runner should not invoke planners."""
    template = tmp_path / "template.yaml"
    space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(space)
    payload = yaml.safe_load(space.read_text(encoding="utf-8"))
    payload["constraints"]["min_start_goal_distance_m"] = 99.0
    space.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def fail_run_batch(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("planner should not run without valid manifests")

    def fail_materialize_matrix(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("materialization should not run without valid manifests")

    monkeypatch.setattr(smoke, "run_batch", fail_run_batch)
    monkeypatch.setattr(smoke, "_materialize_matrix", fail_materialize_matrix)
    args = argparse.Namespace(
        search_space=space,
        scenario_template=template,
        count=1,
        seed=42,
        max_valid=1,
        output_dir=tmp_path / "output",
        summary_json=tmp_path / "summary.json",
        schema=_SCHEMA,
        generator_family="random",
        planner=["goal"],
        horizon=12,
        dt=0.1,
        workers=1,
    )

    payload = smoke.run_smoke(args)

    assert payload["result_classification"] == "no_valid_manifests"
    assert payload["planner_runs"] == []
    assert payload["materialized"]["matrix_path"] is None


def _smoke_candidate() -> CandidateSpec:
    return CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )


def test_run_smoke_only_materializes_valid_manifests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Invalid and degenerate manifests should be filtered before materialization."""
    template = tmp_path / "template.yaml"
    space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(space)
    output_dir = tmp_path / "output"

    valid_manifest = build_manifest(_smoke_candidate())
    manifests = [
        AdversarialScenarioManifest(
            validation=ValidationRecord(status=ManifestCategory.INVALID),
        ),
        valid_manifest,
        AdversarialScenarioManifest(
            validation=ValidationRecord(status=ManifestCategory.DEGENERATE),
        ),
    ]

    materialized_inputs: list[list[AdversarialScenarioManifest]] = []
    run_batch_calls: list[str] = []

    def fake_generate_manifests(
        *_args: Any, **_kwargs: Any
    ) -> tuple[
        list[AdversarialScenarioManifest],
        dict[str, Any],
    ]:
        return manifests, {
            "total_candidates": 3,
            "valid": 1,
            "invalid": 1,
            "degenerate": 1,
            "rejection_reasons": {},
        }

    def fake_materialize_matrix(
        materialized: list[AdversarialScenarioManifest],
        **_kwargs: Any,
    ) -> tuple[Path, list[dict[str, Any]]]:
        materialized_inputs.append(materialized)
        matrix_path = output_dir / "materialized_matrix.yaml"
        matrix_path.write_text(
            yaml.safe_dump({"scenarios": []}, sort_keys=False),
            encoding="utf-8",
        )
        return matrix_path, [{"scenario_seed": 7}]

    def fake_run_batch(**kwargs: Any) -> dict[str, Any]:
        run_batch_calls.append(str(kwargs.get("scenarios_or_path")))
        out_path = Path(kwargs["out_path"])
        out_path.write_text("", encoding="utf-8")
        return {
            "status": "success",
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
        }

    monkeypatch.setattr(smoke, "generate_manifests", fake_generate_manifests)
    monkeypatch.setattr(smoke, "_materialize_matrix", fake_materialize_matrix)
    monkeypatch.setattr(smoke, "run_batch", fake_run_batch)
    args = argparse.Namespace(
        search_space=space,
        scenario_template=template,
        count=3,
        seed=42,
        max_valid=1,
        output_dir=output_dir,
        summary_json=tmp_path / "summary.json",
        schema=_SCHEMA,
        generator_family="random",
        planner=["goal"],
        horizon=12,
        dt=0.1,
        workers=1,
    )

    payload = smoke.run_smoke(args)

    assert materialized_inputs == [[valid_manifest]]
    assert payload["result_classification"] == "smoke_passed"
    assert payload["materialized"]["selected_valid_candidates"] == [{"scenario_seed": 7}]
    assert run_batch_calls == [str(output_dir / "materialized_matrix.yaml")]
