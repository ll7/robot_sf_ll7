"""Tests for the rule-based collision-cause analyser on injected faults (#5443).

Extends the frozen attribution contract tests
(``test_collision_causal_attribution_issue_5443.py``) with the analyser-under-test
and the 14 deterministic fault-injection fixtures. These tests prove acceptance
criteria 3-5 on the injected fixtures:

* criterion 3 - simple injected causes reach >= 0.90 top-explanation accuracy and
  median temporal-localization error <= 1 control step;
* criterion 4 - deliberately ambiguous fixtures never receive a high-confidence
  single-cause verdict;
* criterion 5 - negative controls are not promoted to a concrete cause.

Selectable with ``pytest -k 'causal and attribution'`` (the issue command).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError

from robot_sf.benchmark.collision_cause_analyser import (
    ABSTAIN_CONFIDENCE,
    HIGH_CONFIDENCE,
    analyse_cause,
    analyse_suite,
    analyser_config,
    scenario_collides,
)
from robot_sf.benchmark.collision_cause_attribution import (
    CAUSE_INTERACTING_AMBIGUOUS,
    CAUSE_NONE,
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    MAX_MEDIAN_TEMPORAL_ERROR_STEPS,
    SIMPLE_ACCURACY_FLOOR,
    VERDICT_PASS,
    build_validation_report,
    validate_fixture_manifest,
)
from robot_sf.benchmark.last_avoidable_fixtures import (
    COLLISION_CAUSE_FIXTURE_BUILDERS,
    CollisionCauseFixture,
    KinematicScenario,
    ObservableTraceEvent,
    build_collision_cause_fixtures,
)
from robot_sf.benchmark.last_avoidable_replay import VERDICT_ALREADY_UNAVOIDABLE

_MANIFEST_PATH = (
    Path(__file__).parent / "fixtures" / "collision_cause_attribution_manifest_5443.json"
)


def _load_manifest() -> list[dict]:
    data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return data["fixtures"]


def _manifest_by_id() -> dict[str, dict]:
    return {f["fixture_id"]: f for f in _load_manifest()}


# --- Registry / manifest alignment -----------------------------------------


def test_causal_attribution_analyser_has_fourteen_fixture_builders() -> None:
    """The registry exposes exactly the 14 frozen-manifest fixtures."""
    builders = COLLISION_CAUSE_FIXTURE_BUILDERS
    assert len(builders) == 14
    manifest_ids = {f["fixture_id"] for f in _load_manifest()}
    assert set(builders) == manifest_ids


def test_causal_attribution_fixtures_carry_no_ground_truth_label() -> None:
    """Fixtures must not embed the manifest answer key (honesty contract)."""
    for fixture in build_collision_cause_fixtures():
        assert isinstance(fixture, CollisionCauseFixture)
        # The fixture dataclass has no cause_class field at all.
        assert not hasattr(fixture, "cause_class")


# --- Determinism and decisive-pattern verification --------------------------


@pytest.mark.parametrize("fixture", build_collision_cause_fixtures(), ids=lambda f: f.fixture_id)
def test_causal_attribution_fixture_replay_is_deterministic(fixture: CollisionCauseFixture) -> None:
    """Every fixture's last-avoidable replay is deterministic (criterion: replay determinism)."""
    from robot_sf.benchmark.collision_cause_analyser import run_replay

    report, _ = run_replay(fixture)
    assert report.determinism.deterministic is True


def test_causal_attribution_single_decisive_repairs_avoid_contact() -> None:
    """Each avoidable single-cause fault is counterfactually decisive (repair avoids)."""
    single_ids = {
        "obs_omission_01",
        "obs_delay_01",
        "prediction_miss_01",
        "candidate_omission_01",
        "bad_selection_01",
        "guard_omission_01",
        "infeasible_command_01",
        "route_trap_01",
    }
    for fixture in build_collision_cause_fixtures():
        if fixture.fixture_id not in single_ids:
            continue
        assert len(fixture.faults) == 1
        fault = fixture.faults[0]
        assert fault.repair_scenario is not None
        repaired = fault.repair_scenario(fixture.scenario)
        assert scenario_collides(fixture.scenario), "faulted baseline must collide"
        assert not scenario_collides(repaired), "decisive repair must avoid contact"


def test_causal_attribution_ambiguous_partial_repairs_each_still_collide() -> None:
    """Ambiguous fixtures: no single candidate repair alone avoids contact."""
    ambiguous_ids = {"ambiguous_pred_guard_01", "ambiguous_route_selection_01"}
    for fixture in build_collision_cause_fixtures():
        if fixture.fixture_id not in ambiguous_ids:
            continue
        assert len(fixture.faults) >= 2
        for fault in fixture.faults:
            assert fault.repair_scenario is not None
            repaired = fault.repair_scenario(fixture.scenario)
            assert scenario_collides(repaired), (
                f"partial repair for {fault.events[0].channel} must still collide"
            )


def test_causal_attribution_negative_control_repairs_are_absent_or_ineffective() -> None:
    """Negative controls carry no decisive repair (suspicious signal is non-causal)."""
    nc_ids = {"negative_control_jitter_01", "negative_control_guard_flap_01"}
    for fixture in build_collision_cause_fixtures():
        if fixture.fixture_id not in nc_ids:
            continue
        for fault in fixture.faults:
            assert fault.gates_applied_command is False, "negative control never gates the command"
            if fault.repair_scenario is None:
                continue
            repaired = fault.repair_scenario(fixture.scenario)
            assert scenario_collides(repaired), "negative-control repair must not avoid contact"


# --- End-to-end attribution: criteria 3, 4, 5 ------------------------------


def test_causal_attribution_suite_scores_pass_against_manifest() -> None:
    """End-to-end: analyser verdicts pass the frozen scorer (criteria 3-5)."""
    fixtures = build_collision_cause_fixtures()
    result = analyse_suite(fixtures)
    report = build_validation_report(_load_manifest(), result.verdict_mappings())
    assert report.status == "scored"
    assert report.report is not None
    assert report.report.verdict == VERDICT_PASS
    # Criterion 3.
    assert report.report.top_explanation_accuracy >= SIMPLE_ACCURACY_FLOOR
    assert report.report.median_temporal_localization_error is not None
    assert report.report.median_temporal_localization_error <= MAX_MEDIAN_TEMPORAL_ERROR_STEPS
    # Criterion 4 & 5 (honesty guards clean).
    assert report.report.ambiguous_high_confidence_violations == ()
    assert report.report.negative_control_promotions == ()


def test_causal_attribution_simple_fixtures_perfect_top_explanation() -> None:
    """Criterion 3: the ten unambiguous injected causes are all recovered (accuracy 1.0)."""
    fixtures = build_collision_cause_fixtures()
    manifest = validate_fixture_manifest(_load_manifest())
    manifest_by_id = {f.fixture_id: f for f in manifest}
    unambiguous = [
        f for f in fixtures if manifest_by_id[f.fixture_id].ambiguity_status == "unambiguous"
    ]
    verdicts = {analyse_cause(f).fixture_id: analyse_cause(f) for f in unambiguous}
    correct = sum(
        1
        for fx in unambiguous
        if verdicts[fx.fixture_id].predicted_cause == manifest_by_id[fx.fixture_id].cause_class
    )
    assert correct == len(unambiguous)
    assert correct / len(unambiguous) == pytest.approx(1.0)


def test_causal_attribution_ambiguous_fixtures_abstain_low_confidence() -> None:
    """Criterion 4: ambiguous fixtures abstain and stay below the high-confidence threshold."""
    ambiguous = [
        f
        for f in build_collision_cause_fixtures()
        if _manifest_by_id()[f.fixture_id]["ambiguity_status"] == "ambiguous"
    ]
    assert len(ambiguous) == 2
    for fixture in ambiguous:
        verdict = analyse_cause(fixture)
        assert verdict.abstained is True
        assert verdict.confidence < DEFAULT_HIGH_CONFIDENCE_THRESHOLD
        # A high-confidence concrete single-cause verdict would be a violation.
        assert not (
            verdict.predicted_cause not in {CAUSE_INTERACTING_AMBIGUOUS, CAUSE_NONE}
            and verdict.confidence >= DEFAULT_HIGH_CONFIDENCE_THRESHOLD
        )


def test_causal_attribution_negative_controls_not_promoted() -> None:
    """Criterion 5: negative controls abstain to 'none' and are never a concrete cause."""
    negatives = [
        f
        for f in build_collision_cause_fixtures()
        if _manifest_by_id()[f.fixture_id]["ambiguity_status"] == "negative_control"
    ]
    assert len(negatives) == 2
    for fixture in negatives:
        verdict = analyse_cause(fixture)
        assert verdict.abstained is True
        assert verdict.predicted_cause == CAUSE_NONE
        assert verdict.confidence < DEFAULT_HIGH_CONFIDENCE_THRESHOLD


def test_causal_attribution_metric_artifact_detected_from_reported_physical_mismatch() -> None:
    """The metric-artifact fixture is attributed from the reported/physical mismatch."""
    fixture = COLLISION_CAUSE_FIXTURE_BUILDERS["metric_artifact_01"]()
    verdict = analyse_cause(fixture)
    assert verdict.predicted_cause == "metric_artifact"
    assert verdict.confidence == HIGH_CONFIDENCE


def test_causal_attribution_metric_artifact_rejected_when_physical_contact_follows() -> None:
    """An early inflated-radius report is not an artifact if bodies later touch."""
    scenario = KinematicScenario(
        robot_x0=0.0,
        robot_speed0=5.0,
        ped_pos0=(5.0, 0.0),
        ped_vel0=(0.0, 0.0),
        collision_radius=1.2,
        physical_collision_radius=0.4,
    )
    fixture = CollisionCauseFixture(
        fixture_id="physical_contact_after_reported",
        scenario=scenario,
    )
    verdict = analyse_cause(fixture)
    assert verdict.predicted_cause != "metric_artifact"


def test_causal_attribution_already_unavoidable_from_pure_replay() -> None:
    """The already-unavoidable fixture is attributed from the replay verdict alone."""
    fixture = COLLISION_CAUSE_FIXTURE_BUILDERS["already_unavoidable_01"]()
    assert fixture.faults == ()
    verdict = analyse_cause(fixture)
    assert verdict.predicted_cause == "already_unavoidable_contact"
    assert verdict.predicted_activation_step == 18
    assert verdict.avoidable_pred is False


# --- Honesty: analyser reasons from evidence, not the answer key ------------


def test_causal_attribution_fixture_input_has_no_answer_key_fields() -> None:
    """Fixture faults expose trace values, never typed labels or target windows."""
    fixture = COLLISION_CAUSE_FIXTURE_BUILDERS["obs_omission_01"]()
    fault = fixture.faults[0]
    assert not hasattr(fault, "fault_type")
    assert not hasattr(fault, "activation_window")
    assert fault.events == (
        ObservableTraceEvent(
            step=12,
            channel="observation",
            field="detection_present",
            expected=True,
            observed=False,
        ),
    )


def test_causal_attribution_fixture_id_cannot_select_the_cause() -> None:
    """Permuting the scorer key leaves the observable-derived cause unchanged."""
    fixture = COLLISION_CAUSE_FIXTURE_BUILDERS["obs_omission_01"]()
    rekeyed = replace(fixture, fixture_id="route_trap_01")
    assert analyse_cause(fixture).predicted_cause == "observation_omission"
    assert analyse_cause(rekeyed).predicted_cause == "observation_omission"


def test_causal_attribution_answer_key_permutation_does_not_change_predictions() -> None:
    """Manifest labels are scorer-only and cannot feed back into the analyser."""
    fixtures = build_collision_cause_fixtures()
    before = [analyse_cause(fixture).predicted_cause for fixture in fixtures]
    manifest = _load_manifest()
    first = manifest[0]["cause_class"]
    manifest[0]["cause_class"] = manifest[7]["cause_class"]
    manifest[7]["cause_class"] = first
    after = [analyse_cause(fixture).predicted_cause for fixture in fixtures]
    assert after == before
    permuted_report = build_validation_report(
        manifest,
        analyse_suite(fixtures).verdict_mappings(),
    )
    assert permuted_report.report is not None
    assert permuted_report.report.verdict != VERDICT_PASS


def test_causal_attribution_already_unavoidable_replay_verdict() -> None:
    """The already-unavoidable and negative-control fixtures replay as unavoidable."""
    unavoidable_replay_ids = {
        "already_unavoidable_01",
        "negative_control_jitter_01",
        "negative_control_guard_flap_01",
    }
    from robot_sf.benchmark.collision_cause_analyser import run_replay

    for fixture in build_collision_cause_fixtures():
        if fixture.fixture_id not in unavoidable_replay_ids:
            continue
        report, _ = run_replay(fixture)
        assert report.verdict == VERDICT_ALREADY_UNAVOIDABLE


# --- Runner integration -----------------------------------------------------


def test_causal_attribution_runner_payload_passes_frozen_scorer(tmp_path: Path) -> None:
    """The analysis runner's verdict payload passes the frozen validation scorer."""
    from scripts.analysis.run_collision_cause_attribution_issue_5443 import (
        build_verdicts_payload,
    )

    payload = build_verdicts_payload()
    report = build_validation_report(_load_manifest(), payload["verdicts"])
    assert report.status == "scored"
    assert report.report is not None
    assert report.report.verdict == VERDICT_PASS
    # The payload exposes computed evidence for transparency.
    assert len(payload["verdicts"]) == 14
    assert len(payload["evidence"]) == 14
    assert all("replay_verdict" in e for e in payload["evidence"])
    provenance = payload["provenance"]
    assert len(provenance["git_commit"]) == 40
    assert provenance["manifest_sha256"] == hashlib.sha256(_MANIFEST_PATH.read_bytes()).hexdigest()
    encoded_config = json.dumps(
        analyser_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    assert provenance["analyser_config_sha256"] == hashlib.sha256(encoded_config).hexdigest()


def test_causal_attribution_runner_payload_schema_fails_closed_on_bad_provenance() -> None:
    """The committed run schema rejects malformed commit provenance."""
    from scripts.analysis.run_collision_cause_attribution_issue_5443 import (
        build_verdicts_payload,
        validate_verdicts_payload,
    )

    payload = build_verdicts_payload()
    payload["provenance"]["git_commit"] = "not-a-commit"
    with pytest.raises(JsonSchemaValidationError):
        validate_verdicts_payload(payload)


def test_causal_attribution_abstention_confidence_below_threshold() -> None:
    """Sanity: the analyser's abstention confidence is below the high-confidence gate."""
    assert ABSTAIN_CONFIDENCE < DEFAULT_HIGH_CONFIDENCE_THRESHOLD
    assert HIGH_CONFIDENCE >= DEFAULT_HIGH_CONFIDENCE_THRESHOLD
