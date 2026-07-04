"""Tests for the predictive planner v2 same-seed comparison readiness preflight (#1490).

These are focused synthetic preflight tests: they assert behavior on missing
seed/conditioning/provenance metadata and the fail-closed blocked Slurm gate. No
benchmark, training, or Slurm execution is performed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark.predictive_v2_comparison_readiness import (
    DEFAULT_CONTRACT_PATH,
    PredictiveV2ComparisonReadinessError,
    validate_predictive_v2_comparison_readiness,
)
from robot_sf.common.artifact_paths import get_repository_root
from scripts.validation.validate_predictive_v2_comparison_readiness import main as cli_main

if TYPE_CHECKING:
    from pathlib import Path

REPO_ROOT = get_repository_root()


def _write_contract(tmp_path: Path, contract: dict) -> Path:
    """Write a contract dict to a YAML file and return its path."""
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    return path


def _minimal_contract(tmp_path: Path) -> dict:
    """Build a self-consistent contract whose referenced paths all exist under tmp_path."""
    # Create stand-in config and manifest files so the provenance stage passes.
    for name in (
        "baseline.yaml",
        "obstacle_only.yaml",
        "ego_only.yaml",
        "ego_obstacle_combined.yaml",
        "seed_manifest.yaml",
        "hard_seeds.yaml",
        "scenario_matrix.yaml",
        "planner_grid.yaml",
    ):
        (tmp_path / name).write_text("placeholder: true\n", encoding="utf-8")

    return {
        "ego_motion_channel_producers": {
            "same_seed_hardcase_runtime_robot_speed_v1": {
                "producer_key": "same_seed_hardcase_runtime_robot_speed_v1",
            },
        },
        "row_identifiers": {
            "baseline": {
                "schema_name": "predictive_legacy_v1",
                "input_dim": 4,
                "config": "baseline.yaml",
            },
            "obstacle_only": {
                "schema_name": "predictive_obstacle_features_v1",
                "input_dim": 10,
                "config": "obstacle_only.yaml",
            },
            "ego_only": {
                "schema_name": "predictive_ego_v1",
                "input_dim": 9,
                "config": "ego_only.yaml",
                "ego_motion_channel_producer": "same_seed_hardcase_runtime_robot_speed_v1",
            },
            "ego_obstacle_combined": {
                "schema_name": "predictive_obstacle_features_v1",
                "input_dim": 15,
                "config": "ego_obstacle_combined.yaml",
                "ego_motion_channel_producer": "same_seed_hardcase_runtime_robot_speed_v1",
            },
        },
        "same_seed_comparability": {
            "seed_manifest": "seed_manifest.yaml",
            "hard_seed_manifest": "hard_seeds.yaml",
            "scenario_matrix": "scenario_matrix.yaml",
            "planner_grid": "planner_grid.yaml",
            "seed": 42,
        },
        "metric_separation": {
            "forecast_metrics": ["ADE", "FDE"],
            "navigation_metrics": ["success_rate", "collision_rate"],
        },
    }


def _run(contract_path: Path, repo_root: Path, **kwargs) -> dict:
    """Convenience wrapper for the validator."""
    return validate_predictive_v2_comparison_readiness(
        contract_path=contract_path, repo_root=repo_root, **kwargs
    )


def test_complete_metadata_is_blocked_by_default(tmp_path: Path) -> None:
    """A metadata-complete contract is still blocked when no gate clearance is supplied."""
    contract_path = _write_contract(tmp_path, _minimal_contract(tmp_path))
    report = _run(contract_path, tmp_path)

    assert report["status"] == "blocked"
    for stage in (
        "variant_completeness",
        "provenance",
        "ego_obstacle_conditioning",
        "same_seed_schedule",
    ):
        assert report["stages"][stage]["status"] == "passed"
    assert report["stages"]["blocked_slurm_gate"]["status"] == "blocked"
    packet = report["decision_packet"]
    assert packet["decision"] == "no_go"
    assert packet["candidate_queue_entry"]["current_state"] == "deferred_blocked"
    assert packet["candidate_queue_entry"]["submission_authorized_by_this_packet"] is False
    assert packet["exact_command_if_go"] is None
    assert packet["exact_readiness_command_to_clear_gate"].startswith(
        "uv run python scripts/validation/validate_predictive_v2_comparison_readiness.py"
    )
    assert {item["issue"] for item in packet["completed_or_running_jobs"]} >= {
        "#1543",
        "#1897",
        "#2275",
        "#2916",
    }
    assert packet["evidence_gap"]["status"] == "open"


def test_gate_clears_only_with_continue_and_hypothesis(tmp_path: Path) -> None:
    """The blocked gate clears only with a 'continue' coupling gate, provenance, and ack."""
    contract_path = _write_contract(tmp_path, _minimal_contract(tmp_path))
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "recommendation": {
                    "decision": "continue",
                    "evidence_path": "docs/context/evidence/issue_1490/gate.json",
                }
            }
        ),
        encoding="utf-8",
    )

    ready = _run(contract_path, tmp_path, coupling_gate_path=gate, revised_hypothesis_recorded=True)
    assert ready["status"] == "ready"
    assert ready["stages"]["blocked_slurm_gate"]["status"] == "passed"
    packet = ready["decision_packet"]
    assert packet["decision"] == "go"
    assert packet["candidate_queue_entry"]["current_state"] == "ready_for_bounded_launch"
    assert packet["exact_command_if_go"] is not None
    assert "--revised-hypothesis-recorded" in packet["exact_command_if_go"]

    # Missing maintainer acknowledgement keeps it blocked.
    still_blocked = _run(contract_path, tmp_path, coupling_gate_path=gate)
    assert still_blocked["status"] == "blocked"


def test_continue_gate_without_provenance_stays_blocked(tmp_path: Path) -> None:
    """A go-like gate must name durable evidence before compute can be unblocked."""
    contract_path = _write_contract(tmp_path, _minimal_contract(tmp_path))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"recommendation": {"decision": "continue"}}), encoding="utf-8")

    report = _run(
        contract_path,
        tmp_path,
        coupling_gate_path=gate,
        revised_hypothesis_recorded=True,
    )

    assert report["status"] == "blocked"
    messages = report["stages"]["blocked_slurm_gate"]["messages"]
    assert any("evidence/provenance pointer" in message for message in messages)


def test_continue_gate_accepts_nested_root_provenance(tmp_path: Path) -> None:
    """Root-level nested provenance pointers are valid for future gate artifacts."""
    contract_path = _write_contract(tmp_path, _minimal_contract(tmp_path))
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "recommendation": "continue",
                "provenance": {
                    "artifacts": ["docs/context/evidence/issue_1490/gate.json"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = _run(
        contract_path,
        tmp_path,
        coupling_gate_path=gate,
        revised_hypothesis_recorded=True,
    )

    assert report["status"] == "ready"


def test_malformed_gate_artifacts_stay_blocked(tmp_path: Path) -> None:
    """Malformed/non-object gate artifacts fail closed with actionable messages."""
    contract_path = _write_contract(tmp_path, _minimal_contract(tmp_path))

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    malformed_report = _run(
        contract_path,
        tmp_path,
        coupling_gate_path=malformed,
        revised_hypothesis_recorded=True,
    )
    assert malformed_report["status"] == "blocked"
    assert any(
        "malformed JSON" in message
        for message in malformed_report["stages"]["blocked_slurm_gate"]["messages"]
    )

    non_object = tmp_path / "non_object.json"
    non_object.write_text(json.dumps(["continue"]), encoding="utf-8")
    non_object_report = _run(
        contract_path,
        tmp_path,
        coupling_gate_path=non_object,
        revised_hypothesis_recorded=True,
    )
    assert non_object_report["status"] == "blocked"
    assert any(
        "JSON object" in message
        for message in non_object_report["stages"]["blocked_slurm_gate"]["messages"]
    )


def test_failing_coupling_gate_stays_blocked(tmp_path: Path) -> None:
    """A coupling gate recommending stop/revise must not clear the gate (issue #1897 case)."""
    contract_path = _write_contract(tmp_path, _minimal_contract(tmp_path))
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps({"recommendation": "stop_predictive_v2_expansion"}), encoding="utf-8"
    )

    report = _run(
        contract_path, tmp_path, coupling_gate_path=gate, revised_hypothesis_recorded=True
    )
    assert report["status"] == "blocked"
    messages = report["stages"]["blocked_slurm_gate"]["messages"]
    assert any("expected 'continue'" in m for m in messages)


def test_missing_variant_is_incomplete(tmp_path: Path) -> None:
    """Dropping a required variant reports an incomplete metadata stage."""
    contract = _minimal_contract(tmp_path)
    del contract["row_identifiers"]["ego_obstacle_combined"]
    contract_path = _write_contract(tmp_path, contract)

    report = _run(contract_path, tmp_path)
    assert report["status"] == "incomplete"
    assert report["stages"]["variant_completeness"]["status"] == "failed"
    assert any(
        "ego_obstacle_combined" in m for m in report["stages"]["variant_completeness"]["messages"]
    )


def test_missing_ego_producer_is_incomplete(tmp_path: Path) -> None:
    """An ego variant without conditioning metadata fails the conditioning stage."""
    contract = _minimal_contract(tmp_path)
    del contract["row_identifiers"]["ego_only"]["ego_motion_channel_producer"]
    contract_path = _write_contract(tmp_path, contract)

    report = _run(contract_path, tmp_path)
    assert report["status"] == "incomplete"
    assert report["stages"]["ego_obstacle_conditioning"]["status"] == "failed"


def test_mixed_ego_producers_not_comparable(tmp_path: Path) -> None:
    """Different ego producers across ego variants break same-seed comparability."""
    contract = _minimal_contract(tmp_path)
    contract["ego_motion_channel_producers"]["standalone_rollout_velocity_xy_preferred_v1"] = {
        "producer_key": "standalone_rollout_velocity_xy_preferred_v1",
    }
    contract["row_identifiers"]["ego_obstacle_combined"]["ego_motion_channel_producer"] = (
        "standalone_rollout_velocity_xy_preferred_v1"
    )
    contract_path = _write_contract(tmp_path, contract)

    report = _run(contract_path, tmp_path)
    assert report["stages"]["ego_obstacle_conditioning"]["status"] == "failed"
    assert any("mixed" in m for m in report["stages"]["ego_obstacle_conditioning"]["messages"])


def test_missing_provenance_path_is_incomplete(tmp_path: Path) -> None:
    """A config path that does not exist on disk fails the provenance stage."""
    contract = _minimal_contract(tmp_path)
    contract["row_identifiers"]["baseline"]["config"] = "does_not_exist.yaml"
    contract_path = _write_contract(tmp_path, contract)

    report = _run(contract_path, tmp_path)
    assert report["status"] == "incomplete"
    assert report["stages"]["provenance"]["status"] == "failed"


def test_missing_seed_value_is_incomplete(tmp_path: Path) -> None:
    """Dropping the fixed seed fails the same-seed schedule stage."""
    contract = _minimal_contract(tmp_path)
    del contract["same_seed_comparability"]["seed"]
    contract_path = _write_contract(tmp_path, contract)

    report = _run(contract_path, tmp_path)
    assert report["status"] == "incomplete"
    assert report["stages"]["same_seed_schedule"]["status"] == "failed"


def test_missing_contract_raises(tmp_path: Path) -> None:
    """A non-existent contract path raises a readiness error."""
    try:
        _run(tmp_path / "nope.yaml", tmp_path)
    except PredictiveV2ComparisonReadinessError as exc:
        assert "not found" in str(exc)
    else:  # pragma: no cover - explicit failure
        raise AssertionError("expected PredictiveV2ComparisonReadinessError")


def test_committed_contract_is_metadata_complete_but_blocked() -> None:
    """The real committed contract should be metadata-complete yet blocked by default.

    This is the integration assertion: the same-seed predictive-v2 comparison has all its
    prerequisite metadata declared, but the lane stays fail-closed behind the maintainer
    gate (#2916) until a revised hypothesis and passing coupling gate are recorded.
    """
    contract_path = REPO_ROOT / DEFAULT_CONTRACT_PATH
    report = _run(contract_path, REPO_ROOT)

    assert report["status"] == "blocked"
    assert report["stages"]["variant_completeness"]["status"] == "passed"
    assert report["stages"]["provenance"]["status"] == "passed"
    assert report["stages"]["ego_obstacle_conditioning"]["status"] == "passed"
    assert report["stages"]["same_seed_schedule"]["status"] == "passed"
    assert report["stages"]["blocked_slurm_gate"]["status"] == "blocked"


def test_cli_default_exit_code_is_blocked(capsys) -> None:
    """The CLI exits 2 (not ready) by default against the committed contract."""
    exit_code = cli_main(["--json"])
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"


def test_cli_markdown_gate_clears(tmp_path: Path, capsys) -> None:
    """A Markdown coupling gate recommending continue clears the CLI gate to ready."""
    contract = _minimal_contract(tmp_path)
    contract_path = _write_contract(tmp_path, contract)
    gate = tmp_path / "gate.md"
    gate.write_text(
        "# Coupling gate\n\n"
        "- recommendation: continue\n"
        "- evidence_path: docs/context/evidence/issue_1490/gate.md\n",
        encoding="utf-8",
    )

    exit_code = cli_main(
        [
            "--contract",
            str(contract_path),
            "--repo-root",
            str(tmp_path),
            "--coupling-gate",
            str(gate),
            "--revised-hypothesis-recorded",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"


# Guard against accidental mutation of the shared REQUIRED_VARIANTS template in tests.
def test_minimal_contract_is_independent(tmp_path: Path) -> None:
    """Each helper call returns a fresh contract; mutating one must not affect the next."""
    first = _minimal_contract(tmp_path)
    first["row_identifiers"]["baseline"]["input_dim"] = 999
    del first["same_seed_comparability"]["seed"]

    second = _minimal_contract(tmp_path)
    assert second["row_identifiers"]["baseline"]["input_dim"] == 4
    assert second["same_seed_comparability"]["seed"] == 42
