"""Tests for the collision-cause attribution validation contract (issue #5443).

Selectable with ``pytest -k 'causal and attribution'`` (the issue's validation
command). Covers the frozen fixture contract, matrix coverage, pure scoring,
honesty guards, the JSON schema/enum sync, and fail-closed report building.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.collision_cause_attribution import (
    AMBIGUITY_AMBIGUOUS,
    AMBIGUITY_NEGATIVE_CONTROL,
    AMBIGUITY_UNAMBIGUOUS,
    CAUSE_CLASSES,
    CAUSE_INTERACTING_AMBIGUOUS,
    CAUSE_NONE,
    COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA,
    REPORT_STATUS_ANALYSER_UNAVAILABLE,
    REPORT_STATUS_SCORED,
    VERDICT_PASS,
    VERDICT_REVISE,
    AttributionVerdict,
    CollisionCauseAttributionError,
    GroundTruthFixture,
    build_validation_report,
    score_attribution,
    validate_fixture_manifest,
)

_MANIFEST_PATH = (
    Path(__file__).parent / "fixtures" / "collision_cause_attribution_manifest_5443.json"
)
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "collision_cause_attribution_fixture.v1.json"
)


def _load_manifest_fixtures() -> list[dict]:
    data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return data["fixtures"]


def _perfect_verdict(fixture: GroundTruthFixture) -> AttributionVerdict:
    """Build a maximally-correct, honestly-abstaining verdict for a fixture."""
    if fixture.ambiguity_status == AMBIGUITY_UNAMBIGUOUS:
        return AttributionVerdict(
            fixture_id=fixture.fixture_id,
            predicted_cause=fixture.cause_class,
            predicted_activation_step=fixture.activation_window[0],
            confidence=0.95,
            avoidable_pred=fixture.avoidable,
            abstained=False,
        )
    if fixture.ambiguity_status == AMBIGUITY_AMBIGUOUS:
        return AttributionVerdict(
            fixture_id=fixture.fixture_id,
            predicted_cause=CAUSE_INTERACTING_AMBIGUOUS,
            predicted_activation_step=None,
            confidence=0.4,
            avoidable_pred=fixture.avoidable,
            abstained=True,
        )
    return AttributionVerdict(
        fixture_id=fixture.fixture_id,
        predicted_cause=CAUSE_NONE,
        predicted_activation_step=None,
        confidence=0.2,
        avoidable_pred=fixture.avoidable,
        abstained=True,
    )


# --- Frozen fixture contract ------------------------------------------------


def test_causal_attribution_fixture_rejects_unknown_cause_class() -> None:
    """A fixture with an unknown cause_class fails closed."""
    with pytest.raises(CollisionCauseAttributionError, match="cause_class"):
        GroundTruthFixture(
            fixture_id="x",
            cause_class="mystery",
            activation_window=(1, 1),
            allowed_intervention="none",
            ambiguity_status=AMBIGUITY_UNAMBIGUOUS,
            avoidable=True,
        )


def test_causal_attribution_negative_control_must_be_causeless() -> None:
    """A negative control cannot declare a concrete cause or be avoidable."""
    with pytest.raises(CollisionCauseAttributionError, match="negative_control"):
        GroundTruthFixture(
            fixture_id="nc",
            cause_class="prediction_miss",
            activation_window=(-1, -1),
            allowed_intervention="none",
            ambiguity_status=AMBIGUITY_NEGATIVE_CONTROL,
            avoidable=False,
        )


def test_causal_attribution_ambiguous_requires_two_candidates() -> None:
    """An ambiguous fixture must list at least two candidate causes."""
    with pytest.raises(CollisionCauseAttributionError, match="candidate_causes"):
        GroundTruthFixture(
            fixture_id="amb",
            cause_class=CAUSE_INTERACTING_AMBIGUOUS,
            activation_window=(1, 3),
            allowed_intervention="none",
            ambiguity_status=AMBIGUITY_AMBIGUOUS,
            avoidable=True,
            candidate_causes=("prediction_miss",),
        )


def test_causal_attribution_already_unavoidable_cannot_be_avoidable() -> None:
    """An already-unavoidable-contact fixture must be non-avoidable."""
    with pytest.raises(CollisionCauseAttributionError, match="already_unavoidable"):
        GroundTruthFixture(
            fixture_id="ua",
            cause_class="already_unavoidable_contact",
            activation_window=(5, 5),
            allowed_intervention="none",
            ambiguity_status=AMBIGUITY_UNAMBIGUOUS,
            avoidable=True,
        )


def test_causal_attribution_window_must_be_ordered() -> None:
    """A non-negative-control fixture needs 0 <= start <= end."""
    with pytest.raises(CollisionCauseAttributionError, match="activation_window"):
        GroundTruthFixture(
            fixture_id="w",
            cause_class="route_trap",
            activation_window=(5, 2),
            allowed_intervention="reroute",
            ambiguity_status=AMBIGUITY_UNAMBIGUOUS,
            avoidable=True,
        )


def test_causal_attribution_fixture_roundtrips_through_mapping() -> None:
    """to_dict / from_mapping preserve the frozen fixture."""
    fixture = GroundTruthFixture(
        fixture_id="obs",
        cause_class="observation_delay",
        activation_window=(8, 11),
        allowed_intervention="remove_lag",
        ambiguity_status=AMBIGUITY_UNAMBIGUOUS,
        avoidable=True,
        notes="delay",
    )
    assert GroundTruthFixture.from_mapping(fixture.to_dict()) == fixture


# --- Manifest coverage ------------------------------------------------------


def test_causal_attribution_packaged_manifest_covers_matrix() -> None:
    """The tracked fixture manifest satisfies the predeclared validation matrix."""
    fixtures = validate_fixture_manifest(_load_manifest_fixtures())
    statuses = {f.ambiguity_status for f in fixtures}
    assert AMBIGUITY_AMBIGUOUS in statuses
    assert AMBIGUITY_NEGATIVE_CONTROL in statuses
    assert len(fixtures) == len({f.fixture_id for f in fixtures})


def test_causal_attribution_manifest_missing_class_fails_closed() -> None:
    """Dropping a required cause class makes coverage validation fail closed."""
    fixtures = [f for f in _load_manifest_fixtures() if f["cause_class"] != "route_trap"]
    with pytest.raises(CollisionCauseAttributionError, match="route_trap"):
        validate_fixture_manifest(fixtures)


def test_causal_attribution_manifest_rejects_duplicate_ids() -> None:
    """Duplicate fixture ids fail closed."""
    fixtures = _load_manifest_fixtures()
    fixtures.append(dict(fixtures[0]))
    with pytest.raises(CollisionCauseAttributionError, match="duplicate"):
        validate_fixture_manifest(fixtures)


# --- Scoring ----------------------------------------------------------------


def test_causal_attribution_perfect_verdicts_pass() -> None:
    """Correct, honestly-abstaining verdicts yield a passing report."""
    fixtures = validate_fixture_manifest(_load_manifest_fixtures())
    verdicts = [_perfect_verdict(f) for f in fixtures]
    report = score_attribution(fixtures, verdicts)
    assert report.verdict == VERDICT_PASS
    assert report.top_explanation_accuracy == pytest.approx(1.0)
    assert report.median_temporal_localization_error == 0.0
    assert report.avoidability_accuracy == pytest.approx(1.0)
    assert report.abstention_coverage == pytest.approx(1.0)
    assert report.ambiguous_high_confidence_violations == ()
    assert report.negative_control_promotions == ()


def test_causal_attribution_ambiguous_high_confidence_is_violation() -> None:
    """A confident single-cause verdict on an ambiguous fixture forces revise."""
    fixtures = validate_fixture_manifest(_load_manifest_fixtures())
    verdicts = [_perfect_verdict(f) for f in fixtures]
    amb = next(f for f in fixtures if f.ambiguity_status == AMBIGUITY_AMBIGUOUS)
    verdicts = [v for v in verdicts if v.fixture_id != amb.fixture_id]
    verdicts.append(
        AttributionVerdict(
            fixture_id=amb.fixture_id,
            predicted_cause="prediction_miss",
            predicted_activation_step=amb.activation_window[0],
            confidence=0.97,
            avoidable_pred=amb.avoidable,
            abstained=False,
        )
    )
    report = score_attribution(fixtures, verdicts)
    assert amb.fixture_id in report.ambiguous_high_confidence_violations
    assert report.verdict == VERDICT_REVISE


def test_causal_attribution_negative_control_promotion_is_violation() -> None:
    """Promoting a negative control to a concrete cause forces revise."""
    fixtures = validate_fixture_manifest(_load_manifest_fixtures())
    verdicts = [_perfect_verdict(f) for f in fixtures]
    nc = next(f for f in fixtures if f.ambiguity_status == AMBIGUITY_NEGATIVE_CONTROL)
    verdicts = [v for v in verdicts if v.fixture_id != nc.fixture_id]
    verdicts.append(
        AttributionVerdict(
            fixture_id=nc.fixture_id,
            predicted_cause="observation_omission",
            predicted_activation_step=1,
            confidence=0.9,
            avoidable_pred=False,
            abstained=False,
        )
    )
    report = score_attribution(fixtures, verdicts)
    assert nc.fixture_id in report.negative_control_promotions
    assert report.verdict == VERDICT_REVISE


def test_causal_attribution_below_accuracy_floor_revises() -> None:
    """Misclassifying many simple fixtures drops below the 0.90 floor -> revise."""
    fixtures = validate_fixture_manifest(_load_manifest_fixtures())
    verdicts = []
    flipped = 0
    for fixture in fixtures:
        verdict = _perfect_verdict(fixture)
        if fixture.ambiguity_status == AMBIGUITY_UNAMBIGUOUS and flipped < 3:
            wrong = "metric_artifact" if fixture.cause_class != "metric_artifact" else "route_trap"
            verdict = AttributionVerdict(
                fixture_id=fixture.fixture_id,
                predicted_cause=wrong,
                predicted_activation_step=fixture.activation_window[0],
                confidence=0.6,
                avoidable_pred=fixture.avoidable,
                abstained=False,
            )
            flipped += 1
        verdicts.append(verdict)
    report = score_attribution(fixtures, verdicts)
    assert report.top_explanation_accuracy < 0.90
    assert report.verdict == VERDICT_REVISE


def test_causal_attribution_large_temporal_error_revises() -> None:
    """A median temporal error above one control step forces revise."""
    fixtures = validate_fixture_manifest(_load_manifest_fixtures())
    verdicts = []
    for fixture in fixtures:
        verdict = _perfect_verdict(fixture)
        if fixture.ambiguity_status == AMBIGUITY_UNAMBIGUOUS:
            verdict = AttributionVerdict(
                fixture_id=fixture.fixture_id,
                predicted_cause=fixture.cause_class,
                predicted_activation_step=fixture.activation_window[1] + 5,
                confidence=0.95,
                avoidable_pred=fixture.avoidable,
                abstained=False,
            )
        verdicts.append(verdict)
    report = score_attribution(fixtures, verdicts)
    assert report.median_temporal_localization_error == 5.0
    assert report.verdict == VERDICT_REVISE


def test_causal_attribution_missing_verdict_fails_closed() -> None:
    """A fixture with no verdict fails closed rather than scoring partial."""
    fixtures = validate_fixture_manifest(_load_manifest_fixtures())
    verdicts = [_perfect_verdict(f) for f in fixtures[:-1]]
    with pytest.raises(CollisionCauseAttributionError, match="missing analyser verdict"):
        score_attribution(fixtures, verdicts)


# --- Report builder ---------------------------------------------------------


def test_causal_attribution_report_fails_closed_without_verdicts() -> None:
    """No analyser verdicts -> analyser_unavailable, not a fabricated pass."""
    report = build_validation_report(_load_manifest_fixtures())
    assert report.status == REPORT_STATUS_ANALYSER_UNAVAILABLE
    assert report.covered_matrix is True
    assert report.report is None


def test_causal_attribution_report_scores_when_verdicts_present() -> None:
    """Supplying verdicts moves the report to the scored status."""
    fixtures = validate_fixture_manifest(_load_manifest_fixtures())
    verdicts = [_perfect_verdict(f).__dict__ for f in fixtures]
    report = build_validation_report(_load_manifest_fixtures(), verdicts)
    assert report.status == REPORT_STATUS_SCORED
    assert report.report is not None
    assert report.report.verdict == VERDICT_PASS


# --- Schema / enum sync -----------------------------------------------------


def test_causal_attribution_schema_enum_matches_module() -> None:
    """The JSON schema cause_class enum stays in sync with the module's classes."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    enum = set(schema["properties"]["cause_class"]["enum"])
    assert enum == set(CAUSE_CLASSES)
    assert schema["properties"]["schema_version"]["const"] == (
        COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA
    )
