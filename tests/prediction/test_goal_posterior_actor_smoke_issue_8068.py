"""Tests for the issue #8068 observation-only actor smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "benchmark" / "run_goal_posterior_actor_smoke_issue_8068.py"
_CONFIG_PATH = _REPO_ROOT / "configs" / "benchmarks" / "issue_8068_goal_posterior_actor_smoke.yaml"
_SPEC = importlib.util.spec_from_file_location("issue_8068_goal_posterior_smoke", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_report = _MODULE.build_report


def test_actor_smoke_report_is_observation_only_and_rotated() -> None:
    """Smoke report exposes uncertainty and no simulator identity input."""

    report = build_report(_CONFIG_PATH)

    assert report["schema_version"] == "issue_8068_goal_posterior_actor_smoke.v1"
    assert report["source_contract"] == "observation_only"
    assert report["oracle_identity_input_present"] is False
    assert len(report["scenarios"]) == 10
    reports = {row["case_id"]: row for row in report["scenarios"]}
    assert (
        reports["aligned_axes"]["probabilities"]["east"]
        > reports["aligned_axes"]["probabilities"]["west"]
    )
    assert (
        reports["same_ray_near_far"]["probabilities"]["near"]
        == (reports["same_ray_near_far"]["probabilities"]["far"])
    )
    assert reports["stationary_prior"]["unknown_candidate_probability"] == 0.1
    assert "stationary_below_velocity_min_mps" in reports["stationary_prior"]["blockers"]
    assert reports["candidate_misspecification"]["unknown_candidate_probability"] > 0.5
    assert "unknown_hypothesis_dominant" in reports["candidate_misspecification"]["blockers"]
    assert reports["no_public_candidates"]["mode"] == "unavailable"
    assert reports["no_public_candidates"]["unknown_candidate_probability"] == 1.0
    assert (
        reports["aligned_axes"]["probabilities"] == reports["aligned_axes_rotated"]["probabilities"]
    )
