"""Tests for the Issue #2444 AMMV divergence classification runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis.run_ammv_divergence_classification_issue_2444 import (
    DEFAULT_PROBES,
    RESULT_CLASSIFICATIONS,
    build_selection_block,
    classify_divergence,
    main,
    run_classification,
)

AMMV_CONFIG = Path("configs/baselines/social_force_ammv_aware.yaml")

# Required fields for the compact ``ammv_divergence_selection`` block (acceptance criterion).
_SELECTION_FIELDS = {
    "scenario_id",
    "seed",
    "default_candidate",
    "ammv_candidate",
    "frame_count",
    "max_robot_state_delta",
    "max_pedestrian_state_delta",
    "max_selected_action_delta",
    "max_ammv_force_delta",
    "outcome_changed",
    "mechanism_activation_observed",
    "result_classification",
}


def _identical_probe_result(name: str = "synthetic_zero_pair") -> dict[str, object]:
    """A default/AMMV pair with zero AMMV force and zero paired delta (a rendering fixture)."""
    zero_trace = {
        "mean_robot_speed_mps": 1.0,
        "max_abs_lateral_velocity_mps": 0.0,
        "final_robot_lateral_offset_m": 0.0,
        "min_robot_ped_clearance_m": 0.5,
        "max_ammv_force_magnitude": 0.0,
        "max_intrusion_count": 0,
    }
    return {
        "name": name,
        "steps": 20,
        "paired_delta": {
            "mean_robot_speed_mps": 0.0,
            "max_abs_lateral_velocity_mps": 0.0,
            "final_robot_lateral_offset_m": 0.0,
            "min_robot_ped_clearance_m": 0.0,
        },
        "traces": {
            "default_social_force": dict(zero_trace),
            "ammv_social_force": dict(zero_trace),
        },
    }


def _divergent_probe_result(name: str = "synthetic_nonzero_pair") -> dict[str, object]:
    """A pair with active AMMV force and a nonzero paired delta (genuine behavioral difference)."""
    return {
        "name": name,
        "steps": 24,
        "paired_delta": {
            "mean_robot_speed_mps": 0.07,
            "max_abs_lateral_velocity_mps": 0.78,
            "final_robot_lateral_offset_m": 0.2,
            "min_robot_ped_clearance_m": 0.14,
        },
        "traces": {
            "default_social_force": {"min_robot_ped_clearance_m": 0.1},
            "ammv_social_force": {
                "min_robot_ped_clearance_m": 0.24,
                "max_ammv_force_magnitude": 2.64,
                "max_intrusion_count": 1,
            },
        },
    }


def test_zero_divergence_guard_rejects_identical_pair_as_behavioral_evidence() -> None:
    """An identical (zero-force, zero-delta) pair must NOT be treated as behavioral evidence."""
    block = build_selection_block(_identical_probe_result(), seed=7)

    assert block["mechanism_activation_observed"] is False
    assert block["max_ammv_force_delta"] == 0.0
    assert block["max_robot_state_delta"] == 0.0
    assert block["result_classification"] == "ammv_inactive_under_tested_settings"

    # The aggregate guard refuses to call this slice a nonzero divergence.
    assert classify_divergence([block]) == "ammv_inactive_under_tested_settings"


def test_nonzero_divergence_branch() -> None:
    """An active AMMV pair with nonzero deltas classifies as a found divergence."""
    block = build_selection_block(_divergent_probe_result(), seed=3202)

    assert block["mechanism_activation_observed"] is True
    assert block["max_ammv_force_delta"] > 0.0
    assert block["max_robot_state_delta"] > 0.0
    assert block["result_classification"] == "nonzero_divergence_found"
    assert classify_divergence([block]) == "nonzero_divergence_found"


def test_non_finite_deltas_are_ignored_when_computing_maxima() -> None:
    """NaN/Inf values must not dominate the diagnostic max-delta fields."""
    probe = _divergent_probe_result()
    probe["paired_delta"] = {
        "mean_robot_speed_mps": float("nan"),
        "max_abs_lateral_velocity_mps": float("inf"),
        "final_robot_lateral_offset_m": 0.25,
        "min_robot_ped_clearance_m": -0.5,
    }

    block = build_selection_block(probe, seed=3202)

    assert block["max_robot_state_delta"] == 0.5
    assert block["max_selected_action_delta"] == 0.0
    assert block["result_classification"] == "nonzero_divergence_found"


def test_blocked_branch_when_no_probes_run() -> None:
    """An empty selection list maps to the missing-instrumentation classification."""
    assert classify_divergence([]) == "blocked_missing_instrumentation"


def test_blocked_branch_when_ammv_config_missing(tmp_path: Path) -> None:
    """A missing AMMV config yields a named blocker, not a fabricated result."""
    summary = run_classification(ammv_config=tmp_path / "does_not_exist.yaml")

    assert summary["result_classification"] == "blocked_missing_instrumentation"
    assert "does_not_exist.yaml" in summary["blocker"]


def test_selection_block_has_all_required_fields() -> None:
    """The selection block must carry every acceptance-required field."""
    block = build_selection_block(_divergent_probe_result(), seed=1)
    assert set(block) == _SELECTION_FIELDS
    # Pedestrian state is simulator-owned in the direct probe and reported as None, not a hidden 0.
    assert block["max_pedestrian_state_delta"] is None


def test_real_slice_finds_nonzero_divergence() -> None:
    """The real direct-probe slice exposes a genuine AMMV behavioral difference."""
    summary = run_classification(ammv_config=AMMV_CONFIG)

    assert summary["result_classification"] in RESULT_CLASSIFICATIONS
    assert summary["result_classification"] == "nonzero_divergence_found"
    assert summary["claim_boundary"] == "diagnostic_only"
    assert "issue_2434" in json.dumps(summary["issue_2434_baseline"])
    assert len(summary["selections"]) == len(DEFAULT_PROBES)
    assert any(b["mechanism_activation_observed"] for b in summary["selections"])


def test_determinism_same_classification_and_deltas() -> None:
    """Two runs of the real slice produce identical classification and deltas."""
    first = run_classification(ammv_config=AMMV_CONFIG)
    second = run_classification(ammv_config=AMMV_CONFIG)

    assert first["result_classification"] == second["result_classification"]
    assert first["selections"] == second["selections"]


def test_main_writes_evidence_pack(tmp_path: Path) -> None:
    """The CLI writes a JSON+README pack and prints the classification."""
    out_dir = tmp_path / "pack"
    exit_code = main(["--ammv-config", str(AMMV_CONFIG), "--output-dir", str(out_dir)])

    assert exit_code == 0
    summary_path = out_dir / "ammv_divergence_classification.json"
    assert summary_path.exists()
    assert (out_dir / "README.md").exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["result_classification"] in RESULT_CLASSIFICATIONS
    assert payload["schema_version"].startswith("issue_2444")


@pytest.mark.parametrize("classification", RESULT_CLASSIFICATIONS)
def test_classification_vocabulary_is_stable(classification: str) -> None:
    """Guard the closed classification vocabulary against silent drift."""
    assert classification in {
        "nonzero_divergence_found",
        "ammv_inactive_under_tested_settings",
        "blocked_missing_instrumentation",
    }
