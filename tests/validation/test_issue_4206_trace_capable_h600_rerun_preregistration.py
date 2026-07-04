"""Tests for the issue #4206 trace-capable h600 re-run pre-registration contract.

The checker is fail-closed: it must accept the checked-in contract and reject
every way the eventual re-run could silently drop a required output, weaken trace
capture, shrink the roster/seeds, substitute geometry buckets, or submit from a
pre-registration PR.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.failure_mechanism_taxonomy import REQUIRED_MECHANISM_FIELDS
from scripts.validation.check_issue_4206_trace_capable_h600_rerun_preregistration import (
    CONFIG_SCHEMA_VERSION,
    RerunPreregistrationError,
    build_dry_run_manifest,
    load_preregistration,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml"
)
RUN_CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v1_h600_trace_capable_rerun.yaml"
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "prereg.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def base_payload() -> dict:
    """Return a mutable copy of the checked-in valid pre-registration payload."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_checked_in_config_is_valid() -> None:
    """The checked-in contract passes and declares the expected shape."""
    payload = load_preregistration(CONFIG_PATH)
    assert payload["schema_version"] == CONFIG_SCHEMA_VERSION
    assert payload["issue"] == 4206
    # Required-field list is exactly the canonical taxonomy contract.
    assert (
        tuple(payload["required_outputs"]["failure_mechanism"]["required_fields"])
        == REQUIRED_MECHANISM_FIELDS
    )


def test_dry_run_manifest_declares_no_submission(base_payload: dict) -> None:
    """The manifest is a pre-registration summary that submits nothing."""
    manifest = build_dry_run_manifest(base_payload)
    assert manifest["submits_campaign"] is False
    assert manifest["derives_mechanism_labels"] is False
    assert manifest["predecessor_jobs"] == [13268, 13273]
    assert manifest["planner_arm_count"] == 12
    assert manifest["seeds"] == [20, 21, 22, 23, 24]


def test_main_passes_on_checked_in_config(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI returns 0 and prints PASS on the checked-in contract."""
    exit_code = main(["--config", str(CONFIG_PATH)])
    assert exit_code == 0
    assert "PASS" in capsys.readouterr().out


def test_main_writes_manifest_without_submitting(tmp_path: Path) -> None:
    """--manifest-out writes a dry-run manifest and never submits a campaign."""
    manifest_path = tmp_path / "manifest.json"
    exit_code = main(["--config", str(CONFIG_PATH), "--manifest-out", str(manifest_path)])
    assert exit_code == 0
    assert manifest_path.is_file()


def test_required_mechanism_field_drift_fails_closed(tmp_path: Path, base_payload: dict) -> None:
    """Dropping a canonical mechanism field from the contract fails closed."""
    base_payload["required_outputs"]["failure_mechanism"]["required_fields"].pop()
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="REQUIRED_MECHANISM_FIELDS"):
        load_preregistration(path)


def test_missing_interaction_exposure_field_fails_closed(
    tmp_path: Path, base_payload: dict
) -> None:
    """Dropping a canonical interaction-exposure field fails closed."""
    base_payload["required_outputs"]["interaction_exposure"]["required_fields"].pop()
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="INTERACTION_EXPOSURE_REQUIRED_FIELDS"):
        load_preregistration(path)


def test_trace_capture_not_required_fails_closed(tmp_path: Path, base_payload: dict) -> None:
    """A re-run that does not require trace capture is rejected."""
    base_payload["trace_capture"]["required"] = False
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="trace_capture.required"):
        load_preregistration(path)


def test_missing_trace_record_flag_fails_closed(tmp_path: Path, base_payload: dict) -> None:
    """Turning off a trace record flag fails closed."""
    base_payload["trace_capture"]["record_simulation_step_trace"] = False
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="record_simulation_step_trace"):
        load_preregistration(path)


def test_all_not_derivable_success_rejected(tmp_path: Path, base_payload: dict) -> None:
    """A contract allowing an all-unknown re-run to count as success is rejected."""
    base_payload["fail_closed_exclusions"]["all_not_derivable_output_is_success"] = True
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="all_not_derivable"):
        load_preregistration(path)


def test_geometry_bucket_substitution_rejected(tmp_path: Path, base_payload: dict) -> None:
    """Allowing geometry buckets to substitute mechanism labels is rejected."""
    base_payload["fail_closed_exclusions"]["geometry_buckets_may_substitute_mechanism_labels"] = (
        True
    )
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="geometry_buckets"):
        load_preregistration(path)


def test_min_labeled_fraction_out_of_range_fails_closed(tmp_path: Path, base_payload: dict) -> None:
    """A zero minimum labeled fraction (any output passes) is rejected."""
    base_payload["required_outputs"]["failure_mechanism"]["min_trace_verified_labeled_fraction"] = (
        0.0
    )
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="min_trace_verified_labeled_fraction"):
        load_preregistration(path)


def test_empty_roster_fails_closed(tmp_path: Path, base_payload: dict) -> None:
    """An empty planner roster is rejected."""
    base_payload["planner_roster"]["structural_classes"] = {}
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="structural_classes"):
        load_preregistration(path)


def test_duplicate_planner_key_fails_closed(tmp_path: Path, base_payload: dict) -> None:
    """A planner key declared in two structural classes is rejected."""
    base_payload["planner_roster"]["structural_classes"]["predictive"]["planner_keys"].append("ppo")
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="more than one class"):
        load_preregistration(path)


def test_empty_seeds_fail_closed(tmp_path: Path, base_payload: dict) -> None:
    """An empty seed schedule is rejected."""
    base_payload["seeds"]["schedule"] = []
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="seeds.schedule"):
        load_preregistration(path)


def test_submit_in_this_pr_fails_closed(tmp_path: Path, base_payload: dict) -> None:
    """A contract that submits in this PR is rejected (pre-registration only)."""
    base_payload["queue_plan"]["submit_in_this_pr"] = True
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="submit_in_this_pr"):
        load_preregistration(path)


def test_missing_provenance_predecessors_fails_closed(tmp_path: Path, base_payload: dict) -> None:
    """A contract without predecessor-run provenance is rejected."""
    base_payload["provenance"]["predecessor_runs"] = []
    path = _write_config(tmp_path, base_payload)
    with pytest.raises(RerunPreregistrationError, match="predecessor_runs"):
        load_preregistration(path)


def test_runnable_h600_config_matches_preregistration_contract() -> None:
    """Issue #4404 runnable config preserves the issue #4350 pre-registration identity."""
    from scripts.validation.check_issue_4206_trace_capable_h600_rerun_preregistration import (
        validate_runnable_config_pair,
    )

    payload = load_preregistration(CONFIG_PATH)
    manifest = validate_runnable_config_pair(payload, RUN_CONFIG_PATH)
    expected_keys = [
        key
        for spec in payload["planner_roster"]["structural_classes"].values()
        for key in spec["planner_keys"]
    ]
    assert manifest["planner_keys"] == expected_keys
    assert manifest["planner_arm_count"] == len(expected_keys) == 12
    assert manifest["seeds"] == [20, 21, 22, 23, 24]
    assert manifest["horizon"] == 600
    assert manifest["expected_scenario_matrix_hash"] == "c10df617a87c"
    assert manifest["trace_capture"] == {
        "record_planner_decision_trace": True,
        "record_simulation_step_trace": True,
    }


def test_runnable_h600_config_loads_trace_capture_flags() -> None:
    """Camera-ready config loader preserves trace switches for the runner."""
    from robot_sf.benchmark.camera_ready._config import load_campaign_config

    cfg = load_campaign_config(RUN_CONFIG_PATH)
    assert cfg.name == "paper_experiment_matrix_v1_h600_trace_capable_rerun"
    assert cfg.horizon == 600
    assert tuple(cfg.seed_policy.seeds) == (20, 21, 22, 23, 24)
    assert cfg.record_planner_decision_trace is True
    assert cfg.record_simulation_step_trace is True
    assert [planner.key for planner in cfg.planners] == [
        "scenario_adaptive_hybrid_orca_v1",
        "hybrid_rule_v3_fast_progress_static_escape",
        "ppo",
        "guarded_ppo",
        "prediction_planner",
        "prediction_mpc",
        "prediction_mpc_cbf",
        "goal",
        "social_force",
        "orca",
        "socnav_sampling",
        "sacadrl",
    ]
