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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.research.collision_risk import CandidateAction
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
        assert primitive["unique_waypoint_sequences"] == 16
        assert rbf["candidate_count"] == primitive["candidate_count"]
        assert primitive["finite_validity_rate"] == 1.0
        assert primitive["unique_validity_rate"] == 1.0
        assert rbf["unique_waypoint_sequences"] == 16
        assert rbf["finite_validity_rate"] == 1.0
        assert rbf["unique_validity_rate"] == 1.0

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
                assert "selected_trajectory_sha256" in selection
                if selection["selected"]:
                    assert len(selection["selected_trajectory_sha256"]) == 64
                else:
                    assert selection["selected_trajectory_sha256"] is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Prove the report is deterministic for a pinned generation timestamp."""

    def test_metric_content_deterministic(self, cmp_module, monkeypatch):
        """Two real runs with a pinned timestamp share identical metric content."""
        first = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)
        second = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)
        assert _deterministic_view(first) == _deterministic_view(second)

        original_rank_trajectories = cmp_module.rank_trajectories
        call_count = 0

        def return_non_repeatable_scores(*args, **kwargs):
            nonlocal call_count
            result = original_rank_trajectories(*args, **kwargs)
            call_count += 1
            if call_count == 2:
                result[0] = replace(
                    result[0], joint_contact_probability=result[0].joint_contact_probability + 0.01
                )
            return result

        monkeypatch.setattr(cmp_module, "rank_trajectories", return_non_repeatable_scores)
        non_repeatable = cmp_module.build_report(
            _DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT
        )
        primitive_reliability = non_repeatable["model_risk_reliability"]["by_generator"][
            "deterministic_primitive"
        ]
        assert (
            primitive_reliability["repeatable_risk_scores"]
            < primitive_reliability["candidate_count"]
        )
        assert primitive_reliability["model_score_status"] == "inconclusive"

    def test_full_report_byte_deterministic_with_pinned_timestamp(self, cmp_module, monkeypatch):
        """A pinned timestamp suppresses wall timing so the full report is byte-identical."""
        original_argv = list(cmp_module.sys.argv)
        monkeypatch.setattr(cmp_module.sys, "argv", [*original_argv, "--output", "/tmp/first.json"])
        first = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)
        monkeypatch.setattr(
            cmp_module.sys, "argv", [*original_argv, "--output", "/tmp/second.json"]
        )
        second = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert first["timing"]["measurement_status"] == "not_measured_for_pinned_determinism"
        assert first["timing"]["total"] == {
            "deterministic_primitive": None,
            "rbf": None,
        }
        assert first["provenance"]["git_status_short"] == ["omitted_for_pinned_determinism"]


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

    def test_missing_fixture_claim_boundary_fails_closed(self, cmp_module, tmp_path):
        payload = yaml.safe_load(_HELD_OUT_FIXTURE.read_text(encoding="utf-8"))
        del payload["claim_boundary"]
        missing = tmp_path / "no_claim_boundary.yaml"
        missing.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(cmp_module.ComparisonError, match="claim_boundary"):
            cmp_module._load_fixture(missing)

    def test_malformed_fixture_claim_boundary_fails_closed(self, cmp_module, tmp_path):
        payload = yaml.safe_load(_HELD_OUT_FIXTURE.read_text(encoding="utf-8"))
        payload["claim_boundary"] = "diagnostic_onlyness"
        malformed = tmp_path / "malformed_claim_boundary.yaml"
        malformed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(cmp_module.ComparisonError, match="claim_boundary"):
            cmp_module._load_fixture(malformed)

    def test_missing_case_status_fails_closed(self, cmp_module, tmp_path):
        payload = yaml.safe_load(_HELD_OUT_FIXTURE.read_text(encoding="utf-8"))
        del payload["cases"][0]["status"]
        missing = tmp_path / "no_case_status.yaml"
        missing.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(cmp_module.ComparisonError, match="not a valid evidence row"):
            cmp_module._load_fixture(missing)

    def test_fractional_pedestrian_id_fails_closed(self, cmp_module, tmp_path):
        payload = yaml.safe_load(_HELD_OUT_FIXTURE.read_text(encoding="utf-8"))
        payload["cases"][1]["pedestrians"][0]["id"] = 101.5
        malformed = tmp_path / "fractional_pedestrian_id.yaml"
        malformed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(cmp_module.ComparisonError, match="id.*must be an integer"):
            cmp_module._load_fixture(malformed)

    def test_missing_known_outcome_role_fails_closed(self, cmp_module, tmp_path):
        payload = yaml.safe_load(_HELD_OUT_FIXTURE.read_text(encoding="utf-8"))
        target_case = next(case for case in payload["cases"] if "known_contact_outcome" in case)
        del target_case["known_contact_outcome"]["candidate_role"]
        missing = tmp_path / "no_known_outcome_role.yaml"
        missing.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(cmp_module.ComparisonError, match="candidate_role"):
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

    def test_non_finite_unused_risk_config_fails_closed(self, cmp_module, tmp_path):
        config_path = _nested_config_overrides(
            tmp_path,
            fixture_path=_HELD_OUT_FIXTURE,
            calibration_path=_CALIBRATION_FIXTURE,
            risk_estimator={"deadline_ms": float("nan")},
        )
        with pytest.raises(cmp_module.ComparisonError, match="deadline_ms must be finite"):
            cmp_module.build_report(config_path)

    def test_fractional_candidate_budget_fails_closed(self, cmp_module, tmp_path):
        config_path = _config_overrides(tmp_path, candidate_budget=4.5)
        with pytest.raises(cmp_module.ComparisonError, match="candidate_budget must be an integer"):
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

    def test_candidate_validation_requires_exact_start_state(self, cmp_module):
        candidate = CandidateAction(
            action_id="primitive_test_0",
            waypoints=np.asarray(
                [[1_000_000.01, 0.0], [1_000_000.01, 0.0], [1_000_000.01, 0.0]],
                dtype=float,
            ),
            representation="test",
        )
        with pytest.raises(cmp_module.ComparisonError, match="invalid candidate contract"):
            cmp_module._validate_candidates(
                [candidate],
                case={"case_id": "exact_start", "start_position": [1_000_000.0, 0.0]},
                horizon_steps=2,
                budget=1,
            )

    def test_candidate_validation_rejects_duplicate_waypoint_sequences(self, cmp_module):
        """Distinct IDs must not hide duplicate candidate trajectories."""
        waypoints = np.zeros((3, 2), dtype=float)
        duplicate_waypoints = waypoints.copy()
        duplicate_waypoints[1, 0] = -0.0
        candidates = [
            CandidateAction(action_id="candidate_a", waypoints=waypoints),
            CandidateAction(action_id="candidate_b", waypoints=duplicate_waypoints),
        ]
        with pytest.raises(cmp_module.ComparisonError, match="duplicate_waypoint_sequences"):
            cmp_module._validate_candidates(
                candidates,
                case={"case_id": "duplicate_trajectory", "start_position": [0.0, 0.0]},
                horizon_steps=2,
                budget=2,
            )

    def test_aggregate_unique_rate_uses_waypoint_sequences(self, cmp_module):
        rows = [
            {
                "expected_budget": 2,
                "candidate_count": 2,
                "valid_count": 2,
                "invalid_count": 0,
                "unique_action_ids": 2,
                "unique_waypoint_sequences": 1,
                "finite_waypoint_sequences": 2,
                "shape_valid_sequences": 2,
                "start_state_valid_sequences": 2,
            }
        ]
        aggregate = cmp_module._aggregate_validity(rows)
        assert aggregate["unique_validity_rate"] == 0.5

    def test_invalid_pinned_timestamp_fails_closed(self, cmp_module):
        """A provenance timestamp must be a valid UTC ISO-8601 value."""
        with pytest.raises(cmp_module.ComparisonError, match="UTC ISO-8601"):
            cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc="not-a-timestamp")

    def test_declared_outcome_without_matching_role_is_inconclusive(self, cmp_module, tmp_path):
        payload = yaml.safe_load(_HELD_OUT_FIXTURE.read_text(encoding="utf-8"))
        target_case = next(case for case in payload["cases"] if "known_contact_outcome" in case)
        target_case["known_contact_outcome"]["candidate_role"] = "not_generated"
        evaluation = tmp_path / "held_out_missing_role.yaml"
        evaluation.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config_path = _nested_config_overrides(
            tmp_path,
            fixture_path=evaluation,
            calibration_path=_CALIBRATION_FIXTURE,
        )

        report = cmp_module.build_report(config_path, generated_at_utc=PINNED_GENERATED_AT)
        for generator_name in ("deterministic_primitive", "rbf"):
            aggregate = report["model_risk_reliability"]["by_generator"][generator_name]
            assert aggregate["declared_outcome_status"] == "inconclusive"
            assert aggregate["status"] == "inconclusive"
            per_case = report["model_risk_reliability"]["per_case"][target_case["case_id"]][
                generator_name
            ]
            assert per_case["declared_outcome_check"]["status"] == "inconclusive"
        unavailable_metrics = {entry["metric"] for entry in report["unavailable_denominators"]}
        assert "model_risk_deterministic_primitive_declared_outcome_agreement" in (
            unavailable_metrics
        )
        assert "model_risk_rbf_declared_outcome_agreement" in unavailable_metrics

    def test_degraded_risk_rows_do_not_pass_reliability(self, cmp_module, tmp_path):
        config_path = _nested_config_overrides(
            tmp_path,
            fixture_path=_HELD_OUT_FIXTURE,
            calibration_path=_CALIBRATION_FIXTURE,
            risk_estimator={"min_samples_for_estimate": 999},
        )
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        (
            risk_config,
            weights,
            verifier_config,
            actuator_config,
            primitive_config,
            rbf_config,
            candidate_budget,
        ) = cmp_module._build_configs(payload)
        payload["expected_generator_config_hash"] = cmp_module._generator_config_hash(
            candidate_budget=candidate_budget,
            risk_config=risk_config,
            weights=weights,
            verifier_config=verifier_config,
            actuator_config=actuator_config,
            primitive_config=primitive_config,
            rbf_config=rbf_config,
        )
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        report = cmp_module.build_report(config_path, generated_at_utc=PINNED_GENERATED_AT)
        assert report["fallback_degraded_exclusions"]["degraded_rows_excluded"] == 32
        for generator_name in ("deterministic_primitive", "rbf"):
            reliability = report["model_risk_reliability"]["by_generator"][generator_name]
            assert reliability["degraded_rows"] == 16
            assert reliability["status"] == "inconclusive"

    def test_incomplete_provenance_does_not_enter_outcome_denominator(
        self, cmp_module, monkeypatch
    ):
        original_rank_trajectories = cmp_module.rank_trajectories

        def return_incomplete_provenance(*args, **kwargs):
            rankings = original_rank_trajectories(*args, **kwargs)
            return [
                replace(
                    record,
                    provenance=replace(record.provenance, config_hash=""),
                )
                for record in rankings
            ]

        monkeypatch.setattr(cmp_module, "rank_trajectories", return_incomplete_provenance)
        report = cmp_module.build_report(_DEFAULT_CONFIG, generated_at_utc=PINNED_GENERATED_AT)

        for generator_name in ("deterministic_primitive", "rbf"):
            reliability = report["model_risk_reliability"]["by_generator"][generator_name]
            assert reliability["complete_provenance_rows"] == 0
            assert reliability["declared_outcome_agreeing_candidates"] == 0
            assert reliability["declared_outcome_status"] == "inconclusive"
            assert reliability["status"] == "inconclusive"

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
