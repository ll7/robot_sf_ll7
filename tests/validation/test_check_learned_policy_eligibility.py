"""Tests for the learned-policy eligibility checklist helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.validation.check_learned_policy_eligibility import (
    load_candidate_spec,
    main,
    validate_learned_policy_eligibility,
)

if TYPE_CHECKING:
    from pathlib import Path


def _eligible_spec() -> dict[str, object]:
    """Return a complete learned-policy metadata spec."""
    return {
        "verdict": "eligible_for_adapter",
        "observation_t": "decision step t before action selection",
        "observation_fields": {
            "deployment_observable": [
                "robot pose",
                "current goal",
                "visible pedestrian positions and velocities",
            ],
            "training_only": ["expert demonstrations used for imitation loss"],
            "forbidden_evaluation_time": [],
        },
        "split_provenance": {
            "training_data_source": "synthetic training split",
            "validation_split": "held-out maps validation_v1",
            "test_split": "Robot SF benchmark test scenarios excluded from training",
            "checkpoint_or_model_provenance": "wandb artifact candidate:v1",
            "privileged_training_inputs": "critic-only future return labels",
            "privileged_training_inputs_enter_evaluation": False,
            "normalization_statistics": "fit on training episodes only",
            "normalization_statistics_fit_on_training_only": True,
            "evidence_source": "Robot SF local smoke command",
        },
        "action_contract": {
            "output_family": "velocity_command",
            "frame": "robot local frame",
            "units": "m/s and rad/s",
            "bounds": "linear <= 1.0 m/s, angular <= 1.5 rad/s",
            "kinematics_compatibility": "differential-drive compatible",
            "projection_policy": "clip to Robot SF planner command bounds",
            "raw_to_robot_sf_action": "map raw velocity vector to action dictionary",
            "guard_or_projection_policy": "log clipping and fallback separately",
        },
        "per_step_logging": {
            "raw_model_action": "logged as candidate.raw_model_action",
            "adapted_action": "logged as candidate.adapted_action",
            "post_guard_action": "logged as candidate.post_guard_action",
            "guard_applied": "boolean flag",
            "guard_or_fallback_reason": "stable string or none",
            "observation_level": "global observation level",
            "planner_observation_mode": "candidate preprocessing mode",
            "action_bounds": "bounds active for the step",
            "action_projection_metadata": "clip/projection details",
        },
        "candidate_registry": {
            "entry_planned": True,
            "candidate_config_path": "configs/policy_search/candidates/example.yaml",
            "adapter_path": "robot_sf/planner/example.py",
            "smoke_or_validation_command": (
                "uv run python scripts/validation/run_policy_search_candidate.py "
                "--candidate example --stage smoke"
            ),
            "missing_checkpoint_policy": "fail closed with not_available",
            "unsupported_observation_policy": "fail closed before benchmark run",
            "guard_activation_policy": "report as caveat, not success evidence",
        },
        "benchmark_promotion": {
            "claim_boundary": "benchmark_promoted",
            "benchmark_track": "grid_socnav_v1",
            "track_schema_version": "observation-track.v1",
            "observation_level": "tracked_agents_no_noise",
            "observation_mode": "socnav_state",
            "allowed_observation_keys": [
                "robot_state",
                "goal",
                "tracked_agents",
                "occupancy_grid",
            ],
            "goal_encoding": "current route goal included in planner observation",
            "sensor_geometry": "local occupancy-grid/SocNav state, no LiDAR ray geometry",
            "privileged_input_status": "no evaluation-time privileged inputs",
            "reference": "docs/context/issue_1612_observation_track_architecture.md",
        },
    }


def test_complete_eligible_spec_passes() -> None:
    """A complete metadata spec should satisfy the checklist preflight."""
    assert validate_learned_policy_eligibility(_eligible_spec()) == []


def test_missing_fields_are_reported() -> None:
    """The helper should flag absent checklist inputs with stable paths."""
    spec = _eligible_spec()
    spec["observation_t"] = ""
    spec["observation_fields"] = {"deployment_observable": []}
    spec["per_step_logging"] = {"raw_model_action": "yes"}

    issues = validate_learned_policy_eligibility(spec)
    paths = {issue.path for issue in issues}

    assert "observation_t" in paths
    assert "observation_fields.deployment_observable" in paths
    assert "observation_fields.training_only" in paths
    assert "observation_fields.forbidden_evaluation_time" in paths
    assert "per_step_logging.post_guard_action" in paths
    assert "per_step_logging.guard_or_fallback_reason" in paths


def test_forbidden_evaluation_fields_block_eligible_verdict() -> None:
    """Eligible verdicts must not depend on forbidden evaluation-time fields."""
    spec = _eligible_spec()
    spec["observation_fields"]["forbidden_evaluation_time"] = [
        "future pedestrian trajectory",
    ]

    issues = validate_learned_policy_eligibility(spec)

    assert any(
        issue.path == "observation_fields.forbidden_evaluation_time"
        and "eligible verdicts" in issue.message
        for issue in issues
    )


def test_training_only_or_oracle_can_record_forbidden_fields() -> None:
    """Oracle classifications should still record the forbidden inputs they need."""
    spec = _eligible_spec()
    spec["verdict"] = "training_only_or_oracle"
    spec["observation_fields"]["forbidden_evaluation_time"] = [
        "future collision label",
    ]
    spec["benchmark_promotion"] = {
        "claim_boundary": "research_only",
        "non_benchmark_reason": "Oracle-training spec is not a benchmark-promoted checkpoint.",
    }

    assert validate_learned_policy_eligibility(spec) == []


def test_benchmark_promotion_requires_observation_track_metadata() -> None:
    """Learned checkpoints cannot be promoted without explicit observation-track metadata."""
    spec = _eligible_spec()
    spec["benchmark_promotion"] = {"claim_boundary": "benchmark_promoted"}

    issues = validate_learned_policy_eligibility(spec)
    paths = {issue.path for issue in issues}

    assert "benchmark_promotion.benchmark_track" in paths
    assert "benchmark_promotion.observation_level" in paths
    assert "benchmark_promotion.allowed_observation_keys" in paths


def test_research_only_promotion_boundary_can_pass_without_track_fields() -> None:
    """Research-only candidates should declare the boundary without pretending to be benchmark rows."""
    spec = _eligible_spec()
    spec["verdict"] = "eligible_for_research_only"
    spec["benchmark_promotion"] = {
        "claim_boundary": "research_only",
        "non_benchmark_reason": "Adapter smoke only; not benchmark promotion evidence.",
    }

    assert validate_learned_policy_eligibility(spec) == []


def test_missing_provenance_booleans_are_reported() -> None:
    """Whether-style provenance checklist fields should be explicit."""
    spec = _eligible_spec()
    del spec["split_provenance"]["privileged_training_inputs_enter_evaluation"]
    del spec["split_provenance"]["normalization_statistics_fit_on_training_only"]

    issues = validate_learned_policy_eligibility(spec)
    paths = {issue.path for issue in issues}

    assert "split_provenance.privileged_training_inputs_enter_evaluation" in paths
    assert "split_provenance.normalization_statistics_fit_on_training_only" in paths


def test_cli_reports_json_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command-line helper should return nonzero when any spec fails."""
    valid = tmp_path / "valid.yaml"
    invalid = tmp_path / "invalid.yaml"
    valid.write_text(yaml.safe_dump(_eligible_spec()), encoding="utf-8")
    invalid.write_text("verdict: eligible_for_adapter\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_learned_policy_eligibility.py",
            "--json",
            str(valid),
            str(invalid),
        ],
    )

    assert main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output[str(valid)] == []
    assert any(issue["path"] == "observation_t" for issue in output[str(invalid)])
    loaded = load_candidate_spec(valid)

    assert loaded["verdict"] == "eligible_for_adapter"


def test_json_spec_loading(tmp_path: Path) -> None:
    """YAML and JSON specs should both be accepted by the loader."""
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_eligible_spec()), encoding="utf-8")

    assert load_candidate_spec(path)["action_contract"]["output_family"] == "velocity_command"


def test_loader_errors_include_path_and_line(tmp_path: Path) -> None:
    """Malformed specs should fail closed with actionable file context."""
    path = tmp_path / "broken.json"
    path.write_text('{\n  "verdict": "eligible_for_adapter",\n', encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{path}.*line 3"):
        load_candidate_spec(path)


def test_loader_rejects_non_mapping_specs(tmp_path: Path) -> None:
    """Top-level sequences should not be accepted as candidate metadata."""
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"top-level mapping.*{path}"):
        load_candidate_spec(path)
