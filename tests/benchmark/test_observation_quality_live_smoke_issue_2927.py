"""Tests for issue #2927 observation-quality live-smoke evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.observation_quality import ObservationQuality

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/build_observation_quality_live_smoke_issue_2927.py"
SOURCE_SUMMARY_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_3233_near_field_observation_noise/summary.json"
)
_LOADED_MOD = None


def _load_script():
    """Load the issue #2927 evidence builder as a module."""

    global _LOADED_MOD
    if _LOADED_MOD is not None:
        return _LOADED_MOD
    spec = importlib.util.spec_from_file_location(
        "build_observation_quality_live_smoke_issue_2927", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_observation_quality_live_smoke_issue_2927"] = mod
    spec.loader.exec_module(mod)
    _LOADED_MOD = mod
    return _LOADED_MOD


def _source_summary() -> dict:
    """Load the committed near-field live-smoke summary used as source evidence."""

    return json.loads(SOURCE_SUMMARY_PATH.read_text(encoding="utf-8"))


def test_report_attaches_valid_observation_quality_metadata() -> None:
    """The issue #2927 report should expose validated observation_quality.v1 groups."""

    report = _load_script().build_report(_source_summary())

    clean = report["observation_quality"]["clean"]
    perturbed = report["observation_quality"]["perturbed"]
    clean_quality = ObservationQuality.from_dict(clean["fields"])
    perturbed_quality = ObservationQuality.from_dict(perturbed["fields"])

    assert report["schema_version"] == "issue_2927_observation_quality_live_smoke.v1"
    assert clean["schema_version"] == "observation_quality.v1"
    assert perturbed["schema_version"] == "observation_quality.v1"
    assert clean_quality.false_negative_rate == 0.0
    assert perturbed_quality.false_negative_rate == 0.5
    assert perturbed_quality.false_positive_rate == 0.0
    assert perturbed_quality.range_limit_m is None
    assert "hardware-calibrated" in perturbed_quality.notes


def test_report_keeps_live_smoke_claim_boundary_and_exclusions() -> None:
    """Live-smoke evidence should stay diagnostic and fail closed on unavailable rows."""

    report = _load_script().build_report(_source_summary())

    assert report["issue"] == 2927
    assert report["execution_boundary"]["evidence_status"] == "smoke evidence"
    assert report["execution_boundary"]["near_field_satisfied"] is True
    assert report["execution_boundary"]["fallback_rows"] == []
    assert report["execution_boundary"]["degraded_rows"] == []
    assert report["execution_boundary"]["not_available_rows"] == [
        {
            "row": "false_positive_actor_injection",
            "reason": "not modeled by the source live-smoke perturbation",
            "classification": "explicitly_excluded",
        }
    ]
    assert "planner superiority" in report["claim_boundary"]
    assert "hardware-calibrated" in report["claim_boundary"]


def test_report_rejects_non_boolean_near_field_satisfied() -> None:
    """Near-field evidence flags should not be accepted through truthiness coercion."""

    source = _source_summary()
    source["near_field_target"]["satisfied"] = "false"

    with pytest.raises(ValueError, match="near_field_target.satisfied must be a boolean"):
        _load_script().build_report(source)


def test_report_summarizes_false_negative_and_false_positive_safety_effects() -> None:
    """The safety summary should name both false-negative and false-positive outcomes."""

    report = _load_script().build_report(_source_summary())
    effects = report["safety_effects"]

    assert effects["false_negative"]["effect"] == (
        "non_null_behavior_delta_with_false_negative_perturbation"
    )
    assert effects["false_negative"]["missed_actor_observations_delta"] == 3
    assert effects["false_negative"]["collision_delta"]["pedestrian"]["delta"] == 1
    assert effects["false_positive"]["effect"] == "not_available_excluded"
    assert effects["false_positive"]["false_positive_actor_rows"] == 0
    assert "explicitly excluded" in effects["false_positive"]["rationale"]


def test_cli_writes_compact_evidence_bundle(tmp_path: Path) -> None:
    """The CLI should write JSON and Markdown evidence artifacts."""

    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "README.md"

    assert (
        _load_script().main(
            [
                "--source-summary-json",
                str(SOURCE_SUMMARY_PATH),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert report["issue"] == 2927
    assert "Issue #2927 Observation-Quality Live Smoke" in markdown
    assert "False-positive effect" in markdown
    assert "planner superiority" in markdown
