"""Contract, determinism, and fail-closed tests for issue #6768 ranker comparison.

These tests prove:
- The committed config is a matched, disjoint held-out comparison with full
  provenance and a pinned generator/config hash.
- The report carries decomposed validity, hard-gate, selection, reliability,
  timing, and unavailable-denominator sections.
- The report is deterministic for a pinned generation timestamp (the metric
  content is byte-identical across runs; with a pinned clock the full report is
  byte-identical).
- The script fails closed on split overlap, missing fixture provenance,
  non-finite values, unequal candidate budgets, and generator/config hash
  mismatch.
- The Markdown report leads with the diagnostic-only claim boundary.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tests.support.script_loader import load_script_module

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "analysis" / "compare_risk_ranker_generators_issue_6768.py"
_DEFAULT_CONFIG = (
    _REPO_ROOT / "configs" / "analysis" / "issue_6768_risk_ranker_generator_comparison.yaml"
)
_HELD_OUT_FIXTURE = (
    _REPO_ROOT / "tests" / "fixtures" / "risk_ranker" / "issue_6768_risk_ranker_held_out_v1.yaml"
)
_CALIBRATION_FIXTURE = (
    _REPO_ROOT / "tests" / "fixtures" / "risk_ranker" / "issue_6768_risk_ranker_calibration_v1.yaml"
)

PINNED_GENERATED_AT = "2026-08-06T00:00:00+00:00"


def _deterministic_view(report: dict) -> dict:
    """Return the report content that must be deterministic for a pinned timestamp.

    Wall-clock timing (top-level and per-arm) and the runtime git-status stamp
    are inherently measured values; every other field is a pure function of the
    pinned inputs.
    """
    view = copy.deepcopy(report)
    view.pop("timing")
    for case in view["cases"]:
        for generator_name in ("deterministic_primitive", "rbf"):
            case["arms"][generator_name].pop("timing")
    view["provenance"].pop("generated_at_utc")
    view["provenance"].pop("git_status_short")
    return view


def _fixture_yaml(*, split: str, case_ids: list[str], provenance_source: str) -> str:
    """Render a minimal fixture file with per-case provenance."""
    cases = []
    for case_id in case_ids:
        cases.append(
            {
                "case_id": case_id,
                "split": split,
                "status": "valid",
                "provenance": {"source": provenance_source},
                "start_position": [0.0, 0.0],
                "local_goal": [2.0, 0.0],
                "pedestrians": [],
            }
        )
    return yaml.safe_dump(
        {
            "schema_version": "issue_6768_risk_ranker_generator_comparison_fixture.v1",
            "split": split,
            "claim_boundary": "diagnostic_only; test fixture",
            "provenance": {
                "fixture_source": provenance_source,
                "generated_by": "issue #6768 test",
                "disjoint_from": "issue_6768_risk_ranker_held_out_v1.yaml",
            },
            "cases": cases,
        },
        sort_keys=False,
    )


def _write_fixture(path: Path, *, split: str, case_ids: list[str]) -> Path:
    """Write a custom fixture file for fail-closed tests."""
    path.write_text(
        _fixture_yaml(
            split=split,
            case_ids=case_ids,
            provenance_source=f"issue #6768 test fixture {path.name}",
        ),
        encoding="utf-8",
    )
    return path


def _config_overrides(tmp_path: Path, **overrides: object) -> Path:
    """Write a config copy with the given top-level overrides applied."""
    payload = yaml.safe_load(_DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        payload[key] = value
    target = tmp_path / "overridden_config.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def _nested_config_overrides(
    tmp_path: Path, *, fixture_path: Path, calibration_path: Path, **nested: object
) -> Path:
    """Write a config copy with nested-block overrides and tmp fixture paths."""
    payload = yaml.safe_load(_DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["fixture_path"] = str(fixture_path)
    payload["calibration_fixture_path"] = str(calibration_path)
    for block, fields in nested.items():
        payload[block].update(fields)
    target = tmp_path / "nested_override_config.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


@pytest.fixture(scope="module")
def cmp_module():
    """Load the issue #6768 comparison script once per test module."""
    return load_script_module(_SCRIPT_PATH, name="compare_risk_ranker_generators_issue_6768")


@pytest.fixture(scope="module")
def report(cmp_module):
    """Build the committed report once with a pinned generation timestamp."""
    return cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)


# ---------------------------------------------------------------------------
# Report schema and claim boundary
# ---------------------------------------------------------------------------


class TestReportSchema:
    """Prove the report carries the required schema and claim-boundary fields."""

    def test_schema_version(self, report):
        assert report["schema_version"] == "issue_6768_risk_ranker_generator_comparison.v1"

    def test_evidence_status_diagnostic_only(self, report):
        assert report["evidence_status"] == "diagnostic_only"
        assert report["diagnostic_only"] is True

    def test_required_sections_present(self, report):
        for section in (
            "candidate_validity",
            "hard_gate_rejection",
            "selection_differences",
            "model_risk_reliability",
            "timing",
            "unavailable_denominators",
            "split_integrity",
            "matched_comparison",
            "fallback_degraded_exclusions",
        ):
            assert section in report, f"missing report section: {section}"

    def test_provenance_fields(self, report):
        provenance = report["provenance"]
        assert len(provenance["git_commit_sha"]) == 40
        assert provenance["seed"] == 6768
        assert provenance["config_sha256"]
        assert provenance["fixture_sha256"]
        assert provenance["calibration_fixture_sha256"]
        assert provenance["generator_config_hash"] == provenance["expected_generator_config_hash"]
        assert provenance["pinned_generated_at_utc"] is True

    def test_split_integrity_disjoint(self, report):
        split = report["split_integrity"]
        assert split["disjoint"] is True
        assert split["case_ids_overlap"] is False
        assert split["evaluation_split"] == "held_out"
        assert split["calibration_split"] == "calibration"
        assert set(split["evaluation_case_ids"]) & set(split["calibration_case_ids"]) == set()

    def test_matched_comparison_equal_conditions(self, report):
        matched = report["matched_comparison"]
        assert matched["candidate_budget"] == 4
        assert matched["evaluation_case_count"] == 4
        assert matched["same_start_states_local_goals_actor_predictions"] is True
        assert matched["same_risk_estimator_config"] is True
        assert matched["same_ranking_weights"] is True
        assert matched["same_hard_gate_configs"] is True
        assert matched["same_horizon_and_timestep"] is True
        assert matched["planner_loop_wiring"] == "not_run; intentionally out of scope"

    def test_equal_budgets_across_arms(self, report):
        validity = report["candidate_validity"]["by_generator"]
        primitive = validity["deterministic_primitive"]
        rbf = validity["rbf"]
        assert primitive["candidate_count"] == 16
        assert primitive["valid_count"] == 16
        assert primitive["invalid_count"] == 0
        assert rbf["candidate_count"] == primitive["candidate_count"]

    def test_decomposed_components_reported_per_candidate(self, report):
        for case in report["cases"]:
            for generator_name in ("deterministic_primitive", "rbf"):
                candidates = case["arms"][generator_name]["decomposed_candidates"]
                assert len(candidates) == 4
                for candidate in candidates:
                    components = candidate["components"]
                    for component in (
                        "calibrated_collision_risk",
                        "travel_time_s",
                        "integrated_jerk",
                        "path_length_m",
                        "clearance_penalty",
                    ):
                        assert component in components, f"missing component {component}"
                        assert json.loads(json.dumps(components[component])) is not None

    def test_reliability_diagnostic_passes_on_committed_fixture(self, report):
        for generator_name in ("deterministic_primitive", "rbf"):
            reliability = report["model_risk_reliability"]["by_generator"][generator_name]
            assert reliability["declared_outcome_cases"] == 3
            assert reliability["declared_outcome_agreeing_candidates"] == 3
            assert reliability["declared_outcome_disagreeing_candidates"] == []
            assert reliability["declared_outcome_status"] == "pass"
            assert reliability["status"] == "pass"

    def test_unavailable_denominators_have_reasons(self, report):
        unavailable = report["unavailable_denominators"]
        assert unavailable, "expected at least one unavailable denominator on this fixture set"
        for entry in unavailable:
            assert entry["denominator"] is None
            assert entry["reason"]

    def test_fallback_degraded_exclusions(self, report):
        exclusions = report["fallback_degraded_exclusions"]
        assert exclusions["fallback_rows_excluded"] == 0
        assert exclusions["degraded_rows_excluded"] == 0
        assert exclusions["provenance_incomplete_rows_excluded"] == 0

    def test_report_is_json_serializable(self, report):
        encoded = json.dumps(report, sort_keys=True)
        assert encoded.startswith("{")

    def test_selected_identity_reported(self, report):
        for case in report["cases"]:
            for generator_name in ("deterministic_primitive", "rbf"):
                selection = case["arms"][generator_name]["selection"]
                assert "selected" in selection
                assert "selected_action_id" in selection
                assert "selected_role" in selection


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Prove the report is deterministic for a pinned generation timestamp."""

    def test_metric_content_deterministic(self, cmp_module):
        """Two real runs with a pinned timestamp share identical metric content."""
        first = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)
        second = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)
        assert _deterministic_view(first) == _deterministic_view(second)

    def test_full_report_byte_deterministic_with_pinned_clock(self, cmp_module):
        """With a pinned wall clock, the full report is byte-identical."""
        with patch.object(cmp_module, "_perf_counter_ns", return_value=1_000_000_000):
            first = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)
            second = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# ---------------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------------


class TestFailClosed:
    """Prove the analysis fails closed on every declared stop condition."""

    def test_split_overlap_fails_closed(self, cmp_module, tmp_path):
        evaluation = _write_fixture(
            tmp_path / "held_out.yaml", split="held_out", case_ids=["shared_case"]
        )
        calibration = _write_fixture(
            tmp_path / "calibration.yaml", split="calibration", case_ids=["shared_case"]
        )
        config_path = _nested_config_overrides(
            tmp_path, fixture_path=evaluation, calibration_path=calibration
        )
        with pytest.raises(cmp_module.ComparisonError, match="overlap"):
            cmp_module.build_report(config_path)

    def test_missing_fixture_provenance_fails_closed(self, cmp_module, tmp_path):
        payload = yaml.safe_load(_HELD_OUT_FIXTURE.read_text(encoding="utf-8"))
        del payload["provenance"]["fixture_source"]
        missing = tmp_path / "no_provenance.yaml"
        missing.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(cmp_module.ComparisonError, match="provenance is incomplete"):
            cmp_module._load_fixture(missing)

    def test_non_finite_risk_config_fails_closed(self, cmp_module, tmp_path):
        config_path = _nested_config_overrides(
            tmp_path,
            fixture_path=_HELD_OUT_FIXTURE,
            calibration_path=_CALIBRATION_FIXTURE,
            risk_estimator={"velocity_std_m_s": float("nan")},
        )
        with pytest.raises(cmp_module.ComparisonError, match="must be finite"):
            cmp_module.build_report(config_path)

    def test_unequal_budget_fails_closed(self, cmp_module, tmp_path, monkeypatch):
        monkeypatch.setattr(cmp_module, "generate_rbf_candidates", lambda *args, **kwargs: [])
        config_path = _config_overrides(tmp_path, candidate_budget=4)
        with pytest.raises(cmp_module.ComparisonError, match="equal candidate budgets"):
            cmp_module.build_report(config_path)

    def test_generator_config_hash_mismatch_fails_closed(self, cmp_module, tmp_path):
        config_path = _config_overrides(tmp_path, expected_generator_config_hash="deadbeef")
        with pytest.raises(cmp_module.ComparisonError, match="hash mismatch"):
            cmp_module.build_report(config_path)

    def test_missing_expected_hash_fails_closed(self, cmp_module, tmp_path):
        config_path = _config_overrides(tmp_path, expected_generator_config_hash=None)
        with pytest.raises(cmp_module.ComparisonError, match="expected_generator_config_hash"):
            cmp_module.build_report(config_path)

    def test_check_config_passes_on_committed_config(self, cmp_module):
        exit_code = cmp_module.main(["--check-config", str(_DEFAULT_CONFIG)])
        assert exit_code == 0


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestMarkdown:
    """Prove the human-readable report leads with the diagnostic boundary."""

    def test_markdown_leads_with_claim_boundary(self, report, cmp_module):
        markdown = cmp_module.render_markdown(report)
        assert markdown.startswith("# Risk-ranker generator comparison")
        assert markdown.index("Claim boundary:") < markdown.index("## Matched comparison")
        assert "diagnostic_only" in markdown
        assert "not an online performance claim" in markdown

    def test_markdown_reports_all_sections(self, report, cmp_module):
        markdown = cmp_module.render_markdown(report)
        for heading in (
            "## Candidate validity",
            "## Hard-gate rejection",
            "## Selection differences",
            "## Model-risk reliability",
            "## Timing",
            "## Unavailable denominators",
            "## Provenance",
        ):
            assert heading in markdown
