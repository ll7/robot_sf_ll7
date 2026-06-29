"""Tests for the crossing-conflict predictive retraining launch preflight."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.training.predictive_retrain_preflight import (
    PredictiveRetrainPreflightError,
    build_retrain_decision_packet,
    validate_retrain_preflight,
)
from scripts.validation.validate_predictive_retrain_preflight import main as validate_cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_CONFIG = (
    _REPO_ROOT
    / "configs"
    / "training"
    / "predictive"
    / "predictive_crossing_conflict_weighted_issue_3254.yaml"
)


def _base_config(tmp_path: Path) -> dict[str, object]:
    """Build a minimal, internally consistent pipeline config in ``tmp_path``.

    Referenced data prerequisites are created as files so the preflight's
    existence checks pass; the config dir is ``tmp_path``.
    """
    (tmp_path / "scenarios.yaml").write_text("- name: a\n", encoding="utf-8")
    (tmp_path / "hard_seeds.yaml").write_text("a: [1, 2]\n", encoding="utf-8")
    (tmp_path / "planner_grid.yaml").write_text("variants: []\n", encoding="utf-8")
    (tmp_path / "algo.yaml").write_text(
        "predictive_model_id: predictive_proxy_selected_v1\n",
        encoding="utf-8",
    )
    registry_dir = tmp_path / "model"
    registry_dir.mkdir()
    (registry_dir / "registry.yaml").write_text(
        yaml.safe_dump({"models": [{"model_id": "predictive_proxy_selected_v1"}]}),
        encoding="utf-8",
    )
    (tmp_path / "weighting.yaml").write_text(
        yaml.safe_dump(
            {
                "weighting": {
                    "rule": "repeat_hardcase_rows",
                    "hardcase_family": "crossing_conflict",
                    "hardcase_repeat": 8,
                    "shuffle_seed": 3214,
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "model_family": "predictive_ego_v1",
        "experiment": {"run_id": "demo_crossing_conflict"},
        "output": {"root": "output/tmp/predictive_planner/pipeline"},
        "provenance": {
            "status": "expected_missing_until_training",
            "checkpoint_path": "output/tmp/predictive_planner/pipeline/training/demo/predictive_model.pt",
            "checkpoint_provenance_path": "output/tmp/predictive_planner/pipeline/training/demo/checkpoint_provenance.json",
            "hard_seed_evaluation_summary": "output/tmp/predictive_planner/pipeline/evaluation/demo_hard_seeds/summary.json",
        },
        "scenarios": {
            "scenario_matrix": "scenarios.yaml",
            "hard_seed_manifest": "hard_seeds.yaml",
            "planner_grid": "planner_grid.yaml",
        },
        "base_collection": {"ego_conditioning": True},
        "hardcase_collection": {"ego_conditioning": True},
        "mixing": {"weighting_spec": "weighting.yaml"},
        "training": {
            "model_id": "demo_crossing_conflict_xl_ego",
            "epochs": 400,
            "max_val_ade": 1.0,
            "max_val_fde": 1.8,
        },
        "evaluation": {
            "baseline_model_id": "predictive_proxy_selected_v1",
            "baseline_algo_config": "algo.yaml",
            "workers": 1,
            "horizon": 120,
            "dt": 0.1,
        },
    }


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_real_crossing_conflict_config_passes() -> None:
    """The committed #3254 launch config validates against the repo tree."""
    report = validate_retrain_preflight(_REAL_CONFIG, repo_root=_REPO_ROOT)
    assert report["status"] == "valid"
    assert report["run_id"] == "predictive_crossing_conflict_weighted_issue_3254"
    assert report["feature_compatibility"]["ego_conditioning"] is True
    assert report["mixing"]["weighting_rule"] == "repeat_hardcase_rows"
    assert report["mixing"]["hardcase_family"] == "crossing_conflict"
    assert report["evaluation"]["baseline_model_id"] == "predictive_proxy_selected_v1"
    assert report["provenance"]["status"] == "expected_missing_until_training"
    assert len(report["provenance"]["missing_artifacts"]) == 3


def test_minimal_config_passes(tmp_path: Path) -> None:
    """A complete synthetic config with present prerequisites validates."""
    path = _write_config(tmp_path, _base_config(tmp_path))
    report = validate_retrain_preflight(path, repo_root=tmp_path)
    assert report["status"] == "valid"
    assert report["output_root"] == "output/tmp/predictive_planner/pipeline"


def test_missing_scenario_reference_fails(tmp_path: Path) -> None:
    """A non-existent data prerequisite is rejected."""
    config = _base_config(tmp_path)
    config["scenarios"]["hard_seed_manifest"] = "does_not_exist.yaml"  # type: ignore[index]
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="hard_seed_manifest does not exist"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_ego_conditioning_mismatch_fails(tmp_path: Path) -> None:
    """Mismatched base/hard-case ego conditioning breaks the feature-width contract."""
    config = _base_config(tmp_path)
    config["hardcase_collection"] = {"ego_conditioning": False}
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="ego_conditioning"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_model_family_mismatch_fails(tmp_path: Path) -> None:
    """Mismatched collection model families are rejected as a lineage hazard."""
    config = _base_config(tmp_path)
    config["hardcase_collection"] = {"ego_conditioning": True, "model_family": "other_family"}
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="model families must match"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_unsupported_weighting_rule_fails(tmp_path: Path) -> None:
    """A weighting rule the dataset builder would reject fails closed here too."""
    config = _base_config(tmp_path)
    (tmp_path / "weighting.yaml").write_text(
        yaml.safe_dump(
            {"weighting": {"rule": "made_up_rule", "hardcase_family": "x", "hardcase_repeat": 2}}
        ),
        encoding="utf-8",
    )
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="unsupported weighting rule"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_missing_trajectory_gate_fails(tmp_path: Path) -> None:
    """The trajectory gate (val_ade/val_fde) must stay present and positive."""
    config = _base_config(tmp_path)
    del config["training"]["max_val_fde"]  # type: ignore[attr-defined]
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="max_val_fde"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_missing_evaluation_block_fails(tmp_path: Path) -> None:
    """A config without an evaluation block is rejected."""
    config = _base_config(tmp_path)
    del config["evaluation"]
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="evaluation must be a mapping"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_missing_baseline_algo_config_fails(tmp_path: Path) -> None:
    """Evaluation must name an explicit algo config; no implicit fallback."""
    config = _base_config(tmp_path)
    config["evaluation"]["baseline_algo_config"] = "missing.yaml"  # type: ignore[index]
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="baseline_algo_config"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_baseline_algo_config_must_match_lineage(tmp_path: Path) -> None:
    """Baseline algo config must point at the expected predictive v1 model."""
    config = _base_config(tmp_path)
    (tmp_path / "algo.yaml").write_text(
        "predictive_model_id: predictive_proxy_selected_v2_full\n",
        encoding="utf-8",
    )
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="predictive_model_id"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_missing_provenance_block_fails(tmp_path: Path) -> None:
    """Expected checkpoint/provenance paths are required even before training."""
    config = _base_config(tmp_path)
    del config["provenance"]
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="provenance must be a mapping"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_missing_required_evaluation_key_fails(tmp_path: Path) -> None:
    """Evaluation block must include runner settings pipeline expects."""
    config = _base_config(tmp_path)
    del config["evaluation"]["horizon"]  # type: ignore[index]
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="evaluation.horizon required"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_baseline_model_id_must_be_issue_3254_lineage(tmp_path: Path) -> None:
    """Baseline lineage is pinned to predictive_proxy_selected_v1."""
    config = _base_config(tmp_path)
    config["evaluation"]["baseline_model_id"] = "predictive_proxy_selected_v2_full"  # type: ignore[index]
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="baseline_model_id"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_baseline_algo_config_must_be_mapping(tmp_path: Path) -> None:
    """Algo config path must load as a YAML mapping."""
    config = _base_config(tmp_path)
    (tmp_path / "algo.yaml").write_text("- not\n- mapping\n", encoding="utf-8")
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="YAML mapping"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_invalid_baseline_algo_config_yaml_is_collected(tmp_path: Path) -> None:
    """Invalid baseline algo YAML reported as preflight error."""
    config = _base_config(tmp_path)
    (tmp_path / "algo.yaml").write_text("predictive_model_id: [\n", encoding="utf-8")
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError) as exc_info:
        validate_retrain_preflight(path, repo_root=tmp_path)
    message = str(exc_info.value)
    assert "evaluation.baseline_algo_config" in message
    assert "not readable as valid YAML" in message


def test_baseline_registry_must_have_models_list(tmp_path: Path) -> None:
    """Registry has to expose a models list for baseline lineage lookup."""
    config = _base_config(tmp_path)
    (tmp_path / "model" / "registry.yaml").write_text("version: 1\n", encoding="utf-8")
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="models list"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_invalid_baseline_registry_yaml_is_collected(tmp_path: Path) -> None:
    """Invalid model registry YAML reported as preflight error."""
    config = _base_config(tmp_path)
    (tmp_path / "model" / "registry.yaml").write_text("models: [\n", encoding="utf-8")
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError) as exc_info:
        validate_retrain_preflight(path, repo_root=tmp_path)
    message = str(exc_info.value)
    assert "model registry" in message
    assert "not readable as valid YAML" in message


def test_baseline_registry_must_contain_model_id(tmp_path: Path) -> None:
    """Baseline model id must be present in the registry."""
    config = _base_config(tmp_path)
    (tmp_path / "model" / "registry.yaml").write_text(
        yaml.safe_dump({"models": [{"model_id": "other"}]}),
        encoding="utf-8",
    )
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="not found in model registry"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_provenance_fields_and_status_are_required(tmp_path: Path) -> None:
    """Expected output artifacts are declared and labelled prepare-only."""
    config = _base_config(tmp_path)
    config["provenance"] = {
        "status": "available",
        "checkpoint_path": "",
        "checkpoint_provenance_path": "prov.json",
    }
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError) as exc_info:
        validate_retrain_preflight(path, repo_root=tmp_path)
    message = str(exc_info.value)
    assert "provenance.checkpoint_path" in message
    assert "provenance.hard_seed_evaluation_summary" in message
    assert "expected_missing_until_training" in message


def test_missing_output_root_fails(tmp_path: Path) -> None:
    """An empty output root is rejected (output discipline)."""
    config = _base_config(tmp_path)
    config["output"] = {"root": ""}
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError, match="root must be a non-empty string"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_aggregates_multiple_errors(tmp_path: Path) -> None:
    """All independent failures surface in one error message."""
    config = _base_config(tmp_path)
    config["experiment"] = {}
    config["output"] = {"root": ""}
    path = _write_config(tmp_path, config)
    with pytest.raises(PredictiveRetrainPreflightError) as exc_info:
        validate_retrain_preflight(path, repo_root=tmp_path)
    message = str(exc_info.value)
    assert "run_id must be a non-empty string" in message
    assert "root must be a non-empty string" in message


def test_cli_returns_zero_on_valid_real_config(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI exits 0 and prints a report for the committed config."""
    code = validate_cli_main(
        ["--config", str(_REAL_CONFIG), "--repo-root", str(_REPO_ROOT), "--json"]
    )
    assert code == 0
    payload = capsys.readouterr().out
    assert '"status": "valid"' in payload


def test_cli_returns_two_on_invalid_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI exits 2 and emits an invalid report for a broken config."""
    config = _base_config(tmp_path)
    config["scenarios"]["scenario_matrix"] = "missing.yaml"  # type: ignore[index]
    path = _write_config(tmp_path, config)
    code = validate_cli_main(["--config", str(path), "--repo-root", str(tmp_path), "--json"])
    assert code == 2
    assert '"status": "invalid"' in capsys.readouterr().out


def test_config_must_be_mapping(tmp_path: Path) -> None:
    """A non-mapping YAML document is rejected."""
    path = tmp_path / "pipeline.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(PredictiveRetrainPreflightError, match="config must be a YAML mapping"):
        validate_retrain_preflight(path, repo_root=tmp_path)


def test_decision_packet_ready_from_valid_preflight(tmp_path: Path) -> None:
    """A valid prepare-only config becomes a no-submit ready packet."""
    path = _write_config(tmp_path, _base_config(tmp_path))

    packet = build_retrain_decision_packet(path, repo_root=tmp_path)

    assert packet["schema_version"] == "predictive_retrain_decision_packet.v1"
    assert packet["decision"] == "ready"
    assert packet["preflight_status"] == "valid"
    assert packet["submitted"] is False
    assert packet["blockers"] == []
    assert len(packet["expected_missing_outputs"]) == 3
    assert packet["expected_costs"]["training_epochs"] == 400
    assert packet["expected_costs"]["output_root"] == "output/tmp/predictive_planner/pipeline"
    assert "no Slurm/GPU submission" in packet["out_of_scope"]
    assert packet["preflight_report"]["status"] == "valid"


def test_decision_packet_blocks_invalid_preflight(tmp_path: Path) -> None:
    """Invalid static preflight is reported as blocked, not submitted."""
    config = _base_config(tmp_path)
    config["scenarios"]["planner_grid"] = "missing.yaml"  # type: ignore[index]
    path = _write_config(tmp_path, config)

    packet = build_retrain_decision_packet(path, repo_root=tmp_path)

    assert packet["decision"] == "blocked"
    assert packet["preflight_status"] == "invalid"
    assert packet["submitted"] is False
    assert "planner_grid" in packet["blockers"][0]


def test_decision_packet_reports_missing_preflight_metadata(tmp_path: Path) -> None:
    """Missing provenance metadata has its own status before full validation."""
    config = _base_config(tmp_path)
    del config["provenance"]
    path = _write_config(tmp_path, config)

    packet = build_retrain_decision_packet(path, repo_root=tmp_path)

    assert packet["decision"] == "missing_preflight_metadata"
    assert packet["preflight_status"] == "missing"
    assert packet["submitted"] is False
    assert packet["blockers"] == ["provenance metadata block is missing"]


def test_decision_packet_cli_is_no_submit_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI decision-packet mode emits JSON and keeps submission false."""
    path = _write_config(tmp_path, _base_config(tmp_path))

    code = validate_cli_main(
        ["--config", str(path), "--repo-root", str(tmp_path), "--decision-packet"]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert '"decision": "ready"' in output
    assert '"submitted": false' in output
