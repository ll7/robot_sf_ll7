"""Tests for robot_sf.benchmark.collision.collision_cause_attribution — validation scoring."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.collision.collision_cause_attribution import (
    AMBIGUITY_AMBIGUOUS,
    AMBIGUITY_NEGATIVE_CONTROL,
    AMBIGUITY_UNAMBIGUOUS,
    CAUSE_ALREADY_UNAVOIDABLE_CONTACT,
    CAUSE_BAD_SELECTION,
    CAUSE_CANDIDATE_OMISSION,
    CAUSE_GUARD_OMISSION,
    CAUSE_INFEASIBLE_APPLIED_COMMAND,
    CAUSE_INTERACTING_AMBIGUOUS,
    CAUSE_METRIC_ARTIFACT,
    CAUSE_NONE,
    CAUSE_OBSERVATION_DELAY,
    CAUSE_OBSERVATION_OMISSION,
    CAUSE_PREDICTION_MISS,
    CAUSE_ROUTE_TRAP,
    COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA,
    COLLISION_CAUSE_ATTRIBUTION_REPORT_SCHEMA,
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    MAX_MEDIAN_TEMPORAL_ERROR_STEPS,
    REPORT_STATUS_ANALYSER_UNAVAILABLE,
    REPORT_STATUS_SCORED,
    SIMPLE_ACCURACY_FLOOR,
    SIMPLE_CAUSE_CLASSES,
    VERDICT_PASS,
    VERDICT_REVISE,
    AttributionVerdict,
    CollisionCauseAttributionError,
    GroundTruthFixture,
    build_validation_report,
    score_attribution,
    validate_fixture_manifest,
)


def _fixture(
    fixture_id: str = "fix-1",
    cause_class: str = CAUSE_OBSERVATION_OMISSION,
    window: tuple[int, int] = (5, 10),
    intervention: str = "earlier_brake",
    ambiguity: str = AMBIGUITY_UNAMBIGUOUS,
    avoidable: bool = True,
    candidates: tuple[str, ...] = (),
    notes: str = "",
) -> GroundTruthFixture:
    """Build a valid unambiguous ground-truth fixture."""
    return GroundTruthFixture(
        fixture_id=fixture_id,
        cause_class=cause_class,
        activation_window=window,
        allowed_intervention=intervention,
        ambiguity_status=ambiguity,
        avoidable=avoidable,
        candidate_causes=candidates,
        notes=notes,
    )


def _ambiguous_fixture(fixture_id: str = "amb-1") -> GroundTruthFixture:
    """Build a valid ambiguous fixture."""
    return GroundTruthFixture(
        fixture_id=fixture_id,
        cause_class=CAUSE_INTERACTING_AMBIGUOUS,
        activation_window=(3, 8),
        allowed_intervention="none",
        ambiguity_status=AMBIGUITY_AMBIGUOUS,
        avoidable=False,
        candidate_causes=(CAUSE_OBSERVATION_OMISSION, CAUSE_BAD_SELECTION),
    )


def _negative_control(fixture_id: str = "neg-1") -> GroundTruthFixture:
    """Build a valid negative-control fixture."""
    return GroundTruthFixture(
        fixture_id=fixture_id,
        cause_class=CAUSE_NONE,
        activation_window=(-1, -1),
        allowed_intervention="none",
        ambiguity_status=AMBIGUITY_NEGATIVE_CONTROL,
        avoidable=False,
    )


def _verdict(
    fixture_id: str = "fix-1",
    predicted_cause: str = CAUSE_OBSERVATION_OMISSION,
    step: int | None = 7,
    confidence: float = 0.9,
    avoidable_pred: bool = True,
    abstained: bool = False,
) -> AttributionVerdict:
    """Build a valid attribution verdict."""
    return AttributionVerdict(
        fixture_id=fixture_id,
        predicted_cause=predicted_cause,
        predicted_activation_step=step,
        confidence=confidence,
        avoidable_pred=avoidable_pred,
        abstained=abstained,
    )


def _full_manifest() -> list[GroundTruthFixture]:
    """Build a manifest covering the full predeclared validation matrix."""
    fixtures = [
        _fixture("obs-omission", CAUSE_OBSERVATION_OMISSION),
        _fixture("pred-miss", CAUSE_PREDICTION_MISS),
        _fixture("cand-omission", CAUSE_CANDIDATE_OMISSION),
        _fixture("bad-sel", CAUSE_BAD_SELECTION),
        _fixture("guard-om", CAUSE_GUARD_OMISSION),
        _fixture("infeasible", CAUSE_INFEASIBLE_APPLIED_COMMAND),
        _fixture("route-trap", CAUSE_ROUTE_TRAP),
        _fixture("unavoidable", CAUSE_ALREADY_UNAVOIDABLE_CONTACT, avoidable=False),
        _fixture("metric-art", CAUSE_METRIC_ARTIFACT),
        _ambiguous_fixture(),
        _negative_control(),
    ]
    return fixtures


class TestConstants:
    """Verify module-level constants and class sets."""

    def test_simple_cause_classes_count(self) -> None:
        """SIMPLE_CAUSE_CLASSES must contain exactly 10 single-cause labels."""
        assert len(SIMPLE_CAUSE_CLASSES) == 10

    def test_cause_none_not_in_simple(self) -> None:
        """CAUSE_NONE and CAUSE_INTERACTING_AMBIGUOUS must not be in SIMPLE_CAUSE_CLASSES."""
        assert CAUSE_NONE not in SIMPLE_CAUSE_CLASSES
        assert CAUSE_INTERACTING_AMBIGUOUS not in SIMPLE_CAUSE_CLASSES

    def test_schema_version_strings(self) -> None:
        """Schema version constants must be non-empty versioned strings."""
        assert "v1" in COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA
        assert "v1" in COLLISION_CAUSE_ATTRIBUTION_REPORT_SCHEMA

    def test_stop_rule_thresholds(self) -> None:
        """Stop-rule thresholds must have sane values."""
        assert 0.0 < SIMPLE_ACCURACY_FLOOR <= 1.0
        assert MAX_MEDIAN_TEMPORAL_ERROR_STEPS >= 0.0
        assert 0.0 < DEFAULT_HIGH_CONFIDENCE_THRESHOLD <= 1.0


class TestGroundTruthFixture:
    """Tests for GroundTruthFixture construction and validation."""

    def test_valid_unambiguous_fixture(self) -> None:
        """A well-formed unambiguous fixture must construct without error."""
        fix = _fixture()
        assert fix.fixture_id == "fix-1"
        assert fix.cause_class == CAUSE_OBSERVATION_OMISSION
        assert fix.activation_window == (5, 10)
        assert fix.avoidable is True

    def test_empty_fixture_id_rejected(self) -> None:
        """An empty fixture_id must raise CollisionCauseAttributionError."""
        with pytest.raises(CollisionCauseAttributionError, match="fixture_id"):
            _fixture(fixture_id="   ")

    def test_invalid_cause_class_rejected(self) -> None:
        """An unsupported cause_class must raise CollisionCauseAttributionError."""
        with pytest.raises(CollisionCauseAttributionError, match="cause_class"):
            GroundTruthFixture(
                fixture_id="bad",
                cause_class="nonexistent_cause",
                activation_window=(0, 1),
                allowed_intervention="none",
                ambiguity_status=AMBIGUITY_UNAMBIGUOUS,
                avoidable=True,
            )

    def test_invalid_ambiguity_status_rejected(self) -> None:
        """An unsupported ambiguity_status must raise CollisionCauseAttributionError."""
        with pytest.raises(CollisionCauseAttributionError, match="ambiguity_status"):
            GroundTruthFixture(
                fixture_id="bad",
                cause_class=CAUSE_OBSERVATION_OMISSION,
                activation_window=(0, 1),
                allowed_intervention="none",
                ambiguity_status="unknown_status",
                avoidable=True,
            )

    def test_negative_control_requires_cause_none(self) -> None:
        """negative_control with a real cause_class must be rejected."""
        with pytest.raises(CollisionCauseAttributionError, match="negative_control"):
            GroundTruthFixture(
                fixture_id="bad-neg",
                cause_class=CAUSE_OBSERVATION_OMISSION,
                activation_window=(-1, -1),
                allowed_intervention="none",
                ambiguity_status=AMBIGUITY_NEGATIVE_CONTROL,
                avoidable=False,
            )

    def test_negative_control_must_be_non_avoidable(self) -> None:
        """negative_control fixtures must be non-avoidable."""
        with pytest.raises(CollisionCauseAttributionError, match="non-avoidable"):
            GroundTruthFixture(
                fixture_id="bad-neg",
                cause_class=CAUSE_NONE,
                activation_window=(-1, -1),
                allowed_intervention="none",
                ambiguity_status=AMBIGUITY_NEGATIVE_CONTROL,
                avoidable=True,
            )

    def test_negative_control_window_must_be_sentinel(self) -> None:
        """negative_control fixtures must use (-1, -1) window."""
        with pytest.raises(CollisionCauseAttributionError, match="activation_window"):
            GroundTruthFixture(
                fixture_id="bad-neg",
                cause_class=CAUSE_NONE,
                activation_window=(0, 5),
                allowed_intervention="none",
                ambiguity_status=AMBIGUITY_NEGATIVE_CONTROL,
                avoidable=False,
            )

    def test_unambiguous_rejects_interacting_ambiguous(self) -> None:
        """Unambiguous fixtures cannot declare interacting_ambiguous cause."""
        with pytest.raises(CollisionCauseAttributionError, match="unambiguous"):
            GroundTruthFixture(
                fixture_id="bad",
                cause_class=CAUSE_INTERACTING_AMBIGUOUS,
                activation_window=(0, 5),
                allowed_intervention="none",
                ambiguity_status=AMBIGUITY_UNAMBIGUOUS,
                avoidable=True,
            )

    def test_ambiguous_requires_interacting_ambiguous(self) -> None:
        """Ambiguous fixtures must declare interacting_ambiguous cause."""
        with pytest.raises(CollisionCauseAttributionError, match="ambiguous"):
            GroundTruthFixture(
                fixture_id="bad",
                cause_class=CAUSE_OBSERVATION_OMISSION,
                activation_window=(0, 5),
                allowed_intervention="none",
                ambiguity_status=AMBIGUITY_AMBIGUOUS,
                avoidable=True,
            )

    def test_ambiguous_requires_two_candidates(self) -> None:
        """Ambiguous fixtures must list at least two candidate causes."""
        with pytest.raises(CollisionCauseAttributionError, match="candidate_causes"):
            GroundTruthFixture(
                fixture_id="bad",
                cause_class=CAUSE_INTERACTING_AMBIGUOUS,
                activation_window=(0, 5),
                allowed_intervention="none",
                ambiguity_status=AMBIGUITY_AMBIGUOUS,
                avoidable=True,
                candidate_causes=(CAUSE_OBSERVATION_OMISSION,),
            )

    def test_already_unavoidable_must_be_non_avoidable(self) -> None:
        """already_unavoidable_contact fixtures must be non-avoidable."""
        with pytest.raises(CollisionCauseAttributionError, match="non-avoidable"):
            _fixture(cause_class=CAUSE_ALREADY_UNAVOIDABLE_CONTACT, avoidable=True)

    def test_invalid_window_rejected(self) -> None:
        """A window with start > end must be rejected for unambiguous fixtures."""
        with pytest.raises(CollisionCauseAttributionError, match="activation_window"):
            _fixture(window=(10, 5))

    def test_to_dict_round_trip(self) -> None:
        """to_dict must produce a JSON-safe payload that from_mapping can rebuild."""
        fix = _fixture(notes="test note")
        payload = fix.to_dict()
        assert payload["schema_version"] == COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA
        rebuilt = GroundTruthFixture.from_mapping(payload)
        assert rebuilt.fixture_id == fix.fixture_id
        assert rebuilt.cause_class == fix.cause_class
        assert rebuilt.activation_window == fix.activation_window
        assert rebuilt.notes == "test note"

    def test_from_mapping_rejects_wrong_schema(self) -> None:
        """from_mapping must reject a wrong schema_version."""
        with pytest.raises(CollisionCauseAttributionError, match="schema_version"):
            GroundTruthFixture.from_mapping({"schema_version": "wrong"})

    def test_from_mapping_rejects_bad_window(self) -> None:
        """from_mapping must reject a malformed activation_window."""
        with pytest.raises(CollisionCauseAttributionError, match="activation_window"):
            GroundTruthFixture.from_mapping(
                {
                    "schema_version": COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA,
                    "fixture_id": "x",
                    "cause_class": CAUSE_OBSERVATION_OMISSION,
                    "activation_window": "bad",
                    "ambiguity_status": AMBIGUITY_UNAMBIGUOUS,
                }
            )


class TestAttributionVerdict:
    """Tests for AttributionVerdict construction and validation."""

    def test_valid_verdict(self) -> None:
        """A well-formed verdict must construct without error."""
        verdict = _verdict()
        assert verdict.fixture_id == "fix-1"
        assert verdict.confidence == 0.9

    def test_invalid_predicted_cause_rejected(self) -> None:
        """An unsupported predicted_cause must raise."""
        with pytest.raises(CollisionCauseAttributionError, match="predicted_cause"):
            AttributionVerdict(
                fixture_id="x",
                predicted_cause="nonexistent",
                predicted_activation_step=0,
                confidence=0.5,
                avoidable_pred=True,
            )

    def test_confidence_out_of_range_rejected(self) -> None:
        """Confidence outside [0, 1] must raise."""
        with pytest.raises(CollisionCauseAttributionError, match="confidence"):
            AttributionVerdict(
                fixture_id="x",
                predicted_cause=CAUSE_OBSERVATION_OMISSION,
                predicted_activation_step=0,
                confidence=1.5,
                avoidable_pred=True,
            )

    def test_from_mapping_round_trip(self) -> None:
        """from_mapping must rebuild a verdict from its dict representation."""
        verdict = _verdict(step=None, abstained=True)
        mapping = {
            "fixture_id": verdict.fixture_id,
            "predicted_cause": verdict.predicted_cause,
            "predicted_activation_step": None,
            "confidence": verdict.confidence,
            "avoidable_pred": verdict.avoidable_pred,
            "abstained": True,
        }
        rebuilt = AttributionVerdict.from_mapping(mapping)
        assert rebuilt.fixture_id == verdict.fixture_id
        assert rebuilt.predicted_activation_step is None
        assert rebuilt.abstained is True


class TestValidateFixtureManifest:
    """Tests for validate_fixture_manifest coverage checks."""

    def test_full_manifest_passes(self) -> None:
        """A manifest covering the full matrix must validate successfully."""
        fixtures = _full_manifest()
        resolved = validate_fixture_manifest(fixtures)
        assert len(resolved) == len(fixtures)

    def test_empty_manifest_rejected(self) -> None:
        """An empty manifest must raise."""
        with pytest.raises(CollisionCauseAttributionError, match="empty"):
            validate_fixture_manifest([])

    def test_duplicate_ids_rejected(self) -> None:
        """Duplicate fixture_ids must raise."""
        fixtures = _full_manifest()
        fixtures.append(_fixture("obs-omission", CAUSE_OBSERVATION_DELAY))
        with pytest.raises(CollisionCauseAttributionError, match="duplicate"):
            validate_fixture_manifest(fixtures)

    def test_missing_observation_family_rejected(self) -> None:
        """A manifest without any observation-family fixture must fail."""
        fixtures = [
            _fixture("pred-miss", CAUSE_PREDICTION_MISS),
            _fixture("cand-om", CAUSE_CANDIDATE_OMISSION),
            _fixture("bad-sel", CAUSE_BAD_SELECTION),
            _fixture("guard", CAUSE_GUARD_OMISSION),
            _fixture("infeasible", CAUSE_INFEASIBLE_APPLIED_COMMAND),
            _fixture("route", CAUSE_ROUTE_TRAP),
            _fixture("unavoidable", CAUSE_ALREADY_UNAVOIDABLE_CONTACT, avoidable=False),
            _fixture("metric", CAUSE_METRIC_ARTIFACT),
            _ambiguous_fixture(),
            _negative_control(),
        ]
        with pytest.raises(CollisionCauseAttributionError, match="observation"):
            validate_fixture_manifest(fixtures)

    def test_accepts_mapping_inputs(self) -> None:
        """Manifest validation must accept JSON-style mappings."""
        fixtures = _full_manifest()
        mappings = [f.to_dict() for f in fixtures]
        resolved = validate_fixture_manifest(mappings)
        assert len(resolved) == len(fixtures)


class TestScoreAttribution:
    """Tests for score_attribution scoring logic."""

    def test_perfect_score_passes(self) -> None:
        """Perfect verdicts on a full manifest must produce a pass verdict."""
        fixtures = _full_manifest()
        verdicts = [
            _verdict(
                f.fixture_id, f.cause_class, step=f.activation_window[0], avoidable_pred=f.avoidable
            )
            for f in fixtures
            if f.ambiguity_status == AMBIGUITY_UNAMBIGUOUS
        ]
        verdicts.append(
            _verdict(
                "amb-1",
                CAUSE_INTERACTING_AMBIGUOUS,
                step=None,
                confidence=0.3,
                abstained=True,
                avoidable_pred=False,
            )
        )
        verdicts.append(
            _verdict(
                "neg-1", CAUSE_NONE, step=None, confidence=0.1, abstained=True, avoidable_pred=False
            )
        )
        report = score_attribution(fixtures, verdicts)
        assert report.verdict == VERDICT_PASS
        assert report.top_explanation_accuracy == 1.0
        assert report.n_fixtures == len(fixtures)

    def test_missing_verdict_raises(self) -> None:
        """A fixture without a verdict must raise."""
        fixtures = _full_manifest()
        with pytest.raises(CollisionCauseAttributionError, match="missing"):
            score_attribution(fixtures, [])

    def test_invalid_threshold_raises(self) -> None:
        """A threshold outside (0, 1] must raise."""
        fixtures = _full_manifest()
        verdicts = [
            _verdict(f.fixture_id, f.cause_class, avoidable_pred=f.avoidable) for f in fixtures
        ]
        with pytest.raises(CollisionCauseAttributionError, match="threshold"):
            score_attribution(fixtures, verdicts, high_confidence_threshold=0.0)

    def test_wrong_predictions_lower_accuracy(self) -> None:
        """Incorrect predictions must lower top_explanation_accuracy."""
        fixtures = [_fixture("f1", CAUSE_OBSERVATION_OMISSION)]
        verdicts = [_verdict("f1", CAUSE_BAD_SELECTION, avoidable_pred=True)]
        report = score_attribution(fixtures, verdicts)
        assert report.top_explanation_accuracy == 0.0
        assert report.verdict == VERDICT_REVISE

    def test_ambiguous_high_confidence_violation(self) -> None:
        """A high-confidence single-cause verdict on an ambiguous fixture is a violation."""
        fixtures = [_ambiguous_fixture()]
        verdicts = [
            _verdict("amb-1", CAUSE_OBSERVATION_OMISSION, confidence=0.95, avoidable_pred=False)
        ]
        report = score_attribution(fixtures, verdicts)
        assert "amb-1" in report.ambiguous_high_confidence_violations
        assert report.verdict == VERDICT_REVISE

    def test_negative_control_promotion(self) -> None:
        """A high-confidence concrete verdict on a negative control is a promotion."""
        fixtures = [_negative_control()]
        verdicts = [
            _verdict("neg-1", CAUSE_OBSERVATION_OMISSION, confidence=0.95, avoidable_pred=False)
        ]
        report = score_attribution(fixtures, verdicts)
        assert "neg-1" in report.negative_control_promotions
        assert report.verdict == VERDICT_REVISE

    def test_temporal_localization_error(self) -> None:
        """Temporal error must measure distance from predicted step to window."""
        fixtures = [_fixture("f1", window=(10, 20))]
        verdicts = [_verdict("f1", step=25, avoidable_pred=True)]
        report = score_attribution(fixtures, verdicts)
        assert report.median_temporal_localization_error == 5.0

    def test_temporal_error_zero_inside_window(self) -> None:
        """A prediction inside the activation window has zero temporal error."""
        fixtures = [_fixture("f1", window=(10, 20))]
        verdicts = [_verdict("f1", step=15, avoidable_pred=True)]
        report = score_attribution(fixtures, verdicts)
        assert report.median_temporal_localization_error == 0.0

    def test_avoidability_accuracy(self) -> None:
        """Avoidability accuracy must reflect correct avoidable_pred matches."""
        fixtures = [
            _fixture("f1", avoidable=True),
            _fixture("f2", CAUSE_PREDICTION_MISS, avoidable=False),
        ]
        verdicts = [
            _verdict("f1", avoidable_pred=True),
            _verdict("f2", CAUSE_PREDICTION_MISS, avoidable_pred=True),
        ]
        report = score_attribution(fixtures, verdicts)
        assert report.avoidability_accuracy == 0.5

    def test_report_to_dict_schema(self) -> None:
        """AttributionReport.to_dict must include the report schema version."""
        fixtures = [_fixture()]
        verdicts = [_verdict()]
        report = score_attribution(fixtures, verdicts)
        payload = report.to_dict()
        assert payload["schema_version"] == COLLISION_CAUSE_ATTRIBUTION_REPORT_SCHEMA
        assert "verdict" in payload
        assert "top_explanation_accuracy" in payload


class TestBuildValidationReport:
    """Tests for build_validation_report fail-closed behavior."""

    def test_no_verdicts_returns_analyser_unavailable(self) -> None:
        """Without verdicts the report must be analyser_unavailable, not scored."""
        fixtures = _full_manifest()
        report = build_validation_report(fixtures, verdicts=None)
        assert report.status == REPORT_STATUS_ANALYSER_UNAVAILABLE
        assert report.report is None
        assert report.covered_matrix is True

    def test_empty_verdicts_returns_analyser_unavailable(self) -> None:
        """An empty verdict list must also produce analyser_unavailable."""
        fixtures = _full_manifest()
        report = build_validation_report(fixtures, verdicts=[])
        assert report.status == REPORT_STATUS_ANALYSER_UNAVAILABLE

    def test_with_verdicts_returns_scored(self) -> None:
        """With verdicts the report must be scored."""
        fixtures = _full_manifest()
        verdicts = [
            _verdict(f.fixture_id, f.cause_class, avoidable_pred=f.avoidable) for f in fixtures
        ]
        report = build_validation_report(fixtures, verdicts)
        assert report.status == REPORT_STATUS_SCORED
        assert report.report is not None
        assert report.n_fixtures == len(fixtures)

    def test_validation_report_to_dict(self) -> None:
        """ValidationReport.to_dict must include status and coverage."""
        fixtures = _full_manifest()
        report = build_validation_report(fixtures, verdicts=None)
        payload = report.to_dict()
        assert payload["status"] == REPORT_STATUS_ANALYSER_UNAVAILABLE
        assert payload["covered_matrix"] is True
        assert payload["report"] is None
