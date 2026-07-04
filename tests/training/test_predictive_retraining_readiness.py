"""Tests predictive retraining readiness decision packets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.training.predictive_retraining_readiness import (
    DECISION_BLOCKED,
    PredictiveRetrainingReadinessError,
    evaluate_retraining_readiness,
    load_readiness_packet,
)
from scripts.training import check_predictive_retraining_readiness as readiness_cli


def _write_packet(tmp_path: Path, payload: dict[str, object]) -> Path:
    packet = tmp_path / "packet.yaml"
    packet.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return packet


def _write_pipeline_config(tmp_path: Path) -> Path:
    scenario_dir = tmp_path / "configs"
    scenario_dir.mkdir()
    scenario_matrix = scenario_dir / "scenarios.yaml"
    hard_seed_manifest = scenario_dir / "hard_seeds.yaml"
    planner_grid = scenario_dir / "planner_grid.yaml"
    for path in (scenario_matrix, hard_seed_manifest, planner_grid):
        path.write_text("{}\n", encoding="utf-8")

    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "test"},
                "output": {
                    "root": "output/tmp/predictive_planner/pipeline",
                    "provenance": {
                        "status": "expected_missing_until_training",
                        "checkpoint_path": (
                            "output/tmp/predictive_planner/pipeline/training/test/"
                            "predictive_model.pt"
                        ),
                        "checkpoint_provenance_path": (
                            "output/tmp/predictive_planner/pipeline/training/test/"
                            "checkpoint_provenance.json"
                        ),
                        "hard_seed_evaluation_summary": (
                            "output/tmp/predictive_planner/pipeline/evaluation/test/summary.json"
                        ),
                    },
                },
                "scenarios": {
                    "scenario_matrix": str(scenario_matrix),
                    "hard_seed_manifest": str(hard_seed_manifest),
                    "planner_grid": str(planner_grid),
                },
                "base_collection": {"max_steps": 10},
                "hardcase_collection": {"max_steps": 10},
                "training": {"model_id": "test_predictive_model"},
                "evaluation": {
                    "horizon": 120,
                    "dt": 0.1,
                    "workers": 1,
                    "campaign_workers": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return pipeline_config


def _complete_payload(tmp_path: Path) -> dict[str, object]:
    weighting_spec = tmp_path / "weighting.yaml"
    pipeline_config = _write_pipeline_config(tmp_path)
    weighting_spec.write_text("profile_id: test\n", encoding="utf-8")
    return {
        "schema_version": "predictive-retraining-readiness.v1",
        "issue": "#3214",
        "candidate_id": "crossing_conflict_weighted_predictive_retraining",
        "claim_boundary": (
            "No Slurm submission, no full benchmark campaign run, no paper claim edit."
        ),
        "inputs": {
            "weighting_spec": str(weighting_spec),
            "pipeline_config": str(pipeline_config),
        },
        "data_prerequisites": {
            "base_collection": "pipeline_config.base_collection",
            "hardcase_collection": "pipeline_config.hardcase_collection",
            "weighting_spec": "inputs.weighting_spec",
        },
        "expected_checkpoint_lineage": {
            "candidate_model_id": "test_predictive_model",
            "checkpoint_path": "output/tmp/predictive_planner/pipeline/training/test/predictive_model.pt",
            "checkpoint_provenance_path": (
                "output/tmp/predictive_planner/pipeline/training/test/checkpoint_provenance.json"
            ),
        },
        "evaluation_config": {
            "hard_seed_benchmark": "configs/hard_seeds.yaml",
            "planner_grid": "configs/planner_grid.yaml",
            "summary_path": "output/tmp/predictive_planner/pipeline/evaluation/test/summary.json",
        },
        "output_roots": {
            "pipeline_root": "output/tmp/predictive_planner/pipeline",
            "training_root": "output/tmp/predictive_planner/pipeline/training",
            "evaluation_root": "output/tmp/predictive_planner/pipeline/evaluation",
        },
        "prior_result": {
            "status": "verified_negative",
            "evidence_status": "diagnostic negative result only; not paper claim",
            "source_issue": "#3254",
        },
        "launch_decision": {
            "state": "blocked_until_control_law_change",
            "compute_submit_allowed": False,
            "fail_closed_on_missing": True,
            "required_before_rerun": [
                "control_law_change_config",
                "checkpoint_provenance_plan",
                "hard_seed_evaluation_plan",
            ],
        },
    }


def test_default_issue_3214_packet_is_complete_but_launch_blocked() -> None:
    """Checked-in packet complete but must not authorize blind retraining."""

    repo = Path(__file__).resolve().parents[2]
    report = evaluate_retraining_readiness(repo_root=repo)

    assert report["issue"] == "#3214"
    assert report["packet_complete"] is True
    assert report["launch_ready"] is False
    assert report["decision"] == DECISION_BLOCKED
    assert report["prior_result_status"] == "verified_negative"
    assert report["compute_submit_allowed"] is False
    assert report["blockers"] == []


def test_complete_negative_result_packet_blocks_without_missing_prerequisites(
    tmp_path: Path,
) -> None:
    """A complete packet reports negative-result decision instead missing inputs."""

    packet = _write_packet(tmp_path, _complete_payload(tmp_path))

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is True
    assert report["launch_ready"] is False
    assert report["decision"] == DECISION_BLOCKED
    assert report["blockers"] == []


def test_missing_launch_packet_sections_fail_closed(tmp_path: Path) -> None:
    """Missing launch-packet sections surface as explicit blockers."""

    payload = _complete_payload(tmp_path)
    payload.pop("expected_checkpoint_lineage")
    payload.pop("evaluation_config")
    packet = _write_packet(tmp_path, payload)

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is False
    assert report["launch_ready"] is False
    assert any("expected_checkpoint_lineage" in blocker for blocker in report["blockers"])
    assert any("evaluation_config" in blocker for blocker in report["blockers"])


def test_missing_pipeline_retraining_prerequisites_fail_closed(tmp_path: Path) -> None:
    """Missing data, evaluation, provenance, or output-root config blocks launch readiness."""

    payload = _complete_payload(tmp_path)
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline = yaml.safe_load(pipeline_config.read_text(encoding="utf-8"))
    pipeline.pop("hardcase_collection")
    pipeline["scenarios"]["hard_seed_manifest"] = "missing_hard_seeds.yaml"
    pipeline["output"]["root"] = "/tmp/not-a-repo-output"
    pipeline["output"]["provenance"].pop("checkpoint_provenance_path")
    pipeline["output"]["provenance"]["status"] = "ready"
    pipeline["evaluation"].pop("horizon")
    pipeline_config.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
    packet = _write_packet(tmp_path, payload)

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is False
    assert report["launch_ready"] is False
    assert any(
        "pipeline_config.hardcase_collection must be mapping" in blocker
        for blocker in report["blockers"]
    )
    assert any(
        "pipeline_config.scenarios.hard_seed_manifest does not exist" in blocker
        for blocker in report["blockers"]
    )
    assert any(
        "pipeline_config.output.root must stay under output/" in blocker
        for blocker in report["blockers"]
    )
    assert any(
        "pipeline_config.output.provenance.checkpoint_provenance_path" in blocker
        for blocker in report["blockers"]
    )
    assert any(
        "pipeline_config.output.provenance.status" in blocker for blocker in report["blockers"]
    )
    assert any(
        "pipeline_config.evaluation.horizon must be set" in blocker
        for blocker in report["blockers"]
    )


def test_packet_pipeline_summary_mismatch_fails_closed(tmp_path: Path) -> None:
    """Divergent packet and pipeline hard-seed summary targets block launch readiness."""

    payload = _complete_payload(tmp_path)
    payload["evaluation_config"] = {
        **payload["evaluation_config"],
        "summary_path": "output/tmp/predictive_planner/pipeline/evaluation/stale/summary.json",
    }
    packet = _write_packet(tmp_path, payload)

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is False
    assert report["launch_ready"] is False
    assert any(
        "evaluation_config.summary_path must match "
        "pipeline_config.output.provenance.hard_seed_evaluation_summary" in blocker
        for blocker in report["blockers"]
    )


def test_missing_retraining_prerequisites_fail_closed(tmp_path: Path) -> None:
    """Missing control-law/provenance prerequisites keep packet incomplete."""

    payload = _complete_payload(tmp_path)
    payload["launch_decision"] = {
        "state": "blocked_until_control_law_change",
        "compute_submit_allowed": False,
        "fail_closed_on_missing": True,
        "required_before_rerun": ["control_law_change_config"],
    }
    packet = _write_packet(tmp_path, payload)

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is False
    assert report["launch_ready"] is False
    assert any("checkpoint_provenance_plan" in blocker for blocker in report["blockers"])
    assert any("hard_seed_evaluation_plan" in blocker for blocker in report["blockers"])


def test_missing_input_path_fails_closed(tmp_path: Path) -> None:
    """Missing public config inputs surfaced blockers, not launch-ready state."""

    payload = _complete_payload(tmp_path)
    payload["inputs"] = {
        "weighting_spec": str(tmp_path / "missing_weighting.yaml"),
        "pipeline_config": str(tmp_path / "missing_pipeline.yaml"),
    }
    packet = _write_packet(tmp_path, payload)

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is False
    assert report["launch_ready"] is False
    assert any("inputs.weighting_spec does not exist" in blocker for blocker in report["blockers"])
    assert any("inputs.pipeline_config does not exist" in blocker for blocker in report["blockers"])


def test_cli_require_launch_ready_exits_blocked(monkeypatch, tmp_path: Path, capsys) -> None:
    """CLI can be used as fail-closed gate if compute authorization exists later."""

    packet = _write_packet(tmp_path, _complete_payload(tmp_path))
    monkeypatch.setattr(
        readiness_cli,
        "_parse_args",
        lambda: readiness_cli.argparse.Namespace(
            packet=packet,
            repo_root=tmp_path,
            require_launch_ready=True,
        ),
    )

    assert readiness_cli.main() == 3
    report = json.loads(capsys.readouterr().out)
    assert report["decision"] == DECISION_BLOCKED
    assert report["launch_ready"] is False


def test_load_readiness_packet_rejects_non_mapping(tmp_path: Path) -> None:
    """Malformed packet YAML remains operator error."""

    packet = tmp_path / "bad.yaml"
    packet.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(PredictiveRetrainingReadinessError) as exc_info:
        load_readiness_packet(packet)
    assert "must be a mapping" in str(exc_info.value)
