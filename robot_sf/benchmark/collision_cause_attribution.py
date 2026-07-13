"""Frozen ground-truth contract and scoring for collision-cause attribution validation.

Issue #5443 (child of #5440; depends on the cause-report contract #5441 and the
last-avoidable-action counterfactual replay #5442) asks a *validation* question:

    How accurately and cautiously does the collision analyser recover known
    injected causes, locate their activation time, and abstain on ambiguous
    multi-cause interactions?

This module supplies the **validation-side** machinery so that the accuracy
measurement is fully specified and testable *before* the analyser under test
(#5441/#5442) exists:

* a frozen ground-truth fixture contract (:class:`GroundTruthFixture`) that pins
  the injected ``cause_class``, ``activation_window``, ``allowed_intervention``,
  ambiguity status, and avoidability *before* analysis (acceptance criterion 1);
* a manifest coverage check (:func:`validate_fixture_manifest`) that fails closed
  unless the predeclared validation matrix is represented;
* pure, deterministic scoring (:func:`score_attribution`) that reports class
  precision/recall, top-explanation accuracy, temporal-localization error,
  avoidability accuracy, abstention coverage, and confidence calibration
  (acceptance criterion 2);
* a fail-closed report builder (:func:`build_validation_report`) that returns
  ``analyser_unavailable`` when no analyser verdicts are supplied, rather than
  fabricating a passing result.

It is analysis tooling: pure, side-effect free, and it runs no simulation and
makes no benchmark or paper-grade claim. Attribution *verdicts* are consumed as
already-computed inputs; producing them is the analyser's job (#5441/#5442). The
accuracy RUN against a real analyser therefore remains blocked until those land,
exactly as a campaign RUN would.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from robot_sf.errors import RobotSfError

COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA = "collision_cause_attribution_fixture.v1"
COLLISION_CAUSE_ATTRIBUTION_REPORT_SCHEMA = "collision_cause_attribution_report.v1"

# Injected single-cause classes drawn from the issue's predeclared validation matrix.
# ``interacting_ambiguous`` is a multi-cause class; ``none`` is reserved for negative
# controls (a suspicious trace signal that does not change the counterfactual outcome).
CAUSE_OBSERVATION_OMISSION = "observation_omission"
CAUSE_OBSERVATION_DELAY = "observation_delay"
CAUSE_PREDICTION_MISS = "prediction_miss"
CAUSE_CANDIDATE_OMISSION = "candidate_omission"
CAUSE_BAD_SELECTION = "bad_selection"
CAUSE_GUARD_OMISSION = "guard_omission"
CAUSE_INFEASIBLE_APPLIED_COMMAND = "infeasible_applied_command"
CAUSE_ROUTE_TRAP = "route_trap"
CAUSE_ALREADY_UNAVOIDABLE_CONTACT = "already_unavoidable_contact"
CAUSE_METRIC_ARTIFACT = "metric_artifact"
CAUSE_INTERACTING_AMBIGUOUS = "interacting_ambiguous"
CAUSE_NONE = "none"

# Single, unambiguous injected causes that carry an accuracy target (criterion 3).
SIMPLE_CAUSE_CLASSES: frozenset[str] = frozenset(
    {
        CAUSE_OBSERVATION_OMISSION,
        CAUSE_OBSERVATION_DELAY,
        CAUSE_PREDICTION_MISS,
        CAUSE_CANDIDATE_OMISSION,
        CAUSE_BAD_SELECTION,
        CAUSE_GUARD_OMISSION,
        CAUSE_INFEASIBLE_APPLIED_COMMAND,
        CAUSE_ROUTE_TRAP,
        CAUSE_ALREADY_UNAVOIDABLE_CONTACT,
        CAUSE_METRIC_ARTIFACT,
    }
)

# Full set of ground-truth cause labels a fixture may declare.
CAUSE_CLASSES: frozenset[str] = SIMPLE_CAUSE_CLASSES | {CAUSE_INTERACTING_AMBIGUOUS, CAUSE_NONE}

AMBIGUITY_UNAMBIGUOUS = "unambiguous"
AMBIGUITY_AMBIGUOUS = "ambiguous"
AMBIGUITY_NEGATIVE_CONTROL = "negative_control"
AMBIGUITY_STATUSES: frozenset[str] = frozenset(
    {AMBIGUITY_UNAMBIGUOUS, AMBIGUITY_AMBIGUOUS, AMBIGUITY_NEGATIVE_CONTROL}
)

# Coverage the predeclared validation matrix requires. Observation is satisfied by either
# omission or delay; the remaining single causes must each appear at least once, plus at
# least one ambiguous fixture and at least one negative control.
_OBSERVATION_FAMILY: frozenset[str] = frozenset(
    {CAUSE_OBSERVATION_OMISSION, CAUSE_OBSERVATION_DELAY}
)
_REQUIRED_SINGLE_CAUSES: frozenset[str] = frozenset(
    {
        CAUSE_PREDICTION_MISS,
        CAUSE_CANDIDATE_OMISSION,
        CAUSE_BAD_SELECTION,
        CAUSE_GUARD_OMISSION,
        CAUSE_INFEASIBLE_APPLIED_COMMAND,
        CAUSE_ROUTE_TRAP,
        CAUSE_ALREADY_UNAVOIDABLE_CONTACT,
        CAUSE_METRIC_ARTIFACT,
    }
)

# Default confidence at or above which a single-cause verdict counts as "high confidence".
# Above this, an ambiguous fixture is a calibration failure and a negative control that is
# promoted to a concrete cause is a false attribution.
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.8

# Stop-rule thresholds from the issue (criteria 3): simple-fixture accuracy floor and the
# maximum acceptable median temporal-localization error, in control steps.
SIMPLE_ACCURACY_FLOOR = 0.90
MAX_MEDIAN_TEMPORAL_ERROR_STEPS = 1.0

REPORT_STATUS_ANALYSER_UNAVAILABLE = "analyser_unavailable"
REPORT_STATUS_SCORED = "scored"

VERDICT_PASS = "pass"
VERDICT_REVISE = "revise"


class CollisionCauseAttributionError(RobotSfError, ValueError):
    """Raised when a collision-cause attribution fixture or verdict violates the contract."""


def _validate_window_for_status(
    ambiguity_status: str, cause_class: str, avoidable: bool, start: int, end: int
) -> None:
    """Validate the activation window against the ambiguity status.

    Raises:
        CollisionCauseAttributionError: If the window or avoidability is inconsistent.
    """
    if ambiguity_status == AMBIGUITY_NEGATIVE_CONTROL:
        if cause_class != CAUSE_NONE:
            raise CollisionCauseAttributionError(
                "negative_control fixtures must declare cause_class 'none'"
            )
        if avoidable:
            raise CollisionCauseAttributionError(
                "negative_control fixtures have no actual cause and must be non-avoidable"
            )
        if (start, end) != (-1, -1):
            raise CollisionCauseAttributionError(
                "negative_control fixtures must use activation_window (-1, -1)"
            )
        return
    if cause_class == CAUSE_NONE:
        raise CollisionCauseAttributionError(
            "cause_class 'none' is reserved for negative_control fixtures"
        )
    if not (0 <= start <= end):
        raise CollisionCauseAttributionError(
            f"activation_window must satisfy 0 <= start <= end (got ({start}, {end}))"
        )


def _validate_class_status_agreement(
    ambiguity_status: str,
    cause_class: str,
    avoidable: bool,
    candidate_causes: tuple[str, ...],
) -> None:
    """Validate that the cause class agrees with the ambiguity status.

    Raises:
        CollisionCauseAttributionError: If the class and status disagree.
    """
    if ambiguity_status == AMBIGUITY_AMBIGUOUS:
        if cause_class != CAUSE_INTERACTING_AMBIGUOUS:
            raise CollisionCauseAttributionError(
                "ambiguous fixtures must declare cause_class 'interacting_ambiguous'"
            )
        if len(candidate_causes) < 2:
            raise CollisionCauseAttributionError(
                "ambiguous fixtures must list at least two candidate_causes"
            )
    if ambiguity_status == AMBIGUITY_UNAMBIGUOUS and cause_class in {
        CAUSE_INTERACTING_AMBIGUOUS,
        CAUSE_NONE,
    }:
        raise CollisionCauseAttributionError(
            f"unambiguous fixtures cannot declare cause_class {cause_class!r}"
        )
    if cause_class == CAUSE_ALREADY_UNAVOIDABLE_CONTACT and avoidable:
        raise CollisionCauseAttributionError(
            "already_unavoidable_contact fixtures must be non-avoidable"
        )


@dataclass(frozen=True)
class GroundTruthFixture:
    """A frozen ground-truth fixture for one injected (or negative-control) fault.

    The fields are the invariants that must be pinned *before* analysis so the
    attribution measurement cannot be tuned to the analyser's output.

    Attributes:
        fixture_id: Stable identifier, unique within a manifest.
        cause_class: Ground-truth injected cause, one of :data:`CAUSE_CLASSES`.
            ``interacting_ambiguous`` marks a genuine multi-cause interaction and
            ``none`` marks a negative control with no actual cause.
        activation_window: Inclusive ``(start, end)`` control-step window in which
            the injected cause is active. A single-step activation uses
            ``start == end``. Negative controls, which have no real activation,
            use ``(-1, -1)``.
        allowed_intervention: The single intervention permitted to test avoidability
            (e.g. ``"earlier_brake"``); ``"none"`` when the contact is unavoidable
            or there is no actual cause.
        ambiguity_status: ``unambiguous``, ``ambiguous``, or ``negative_control``.
        avoidable: Ground-truth avoidability of the collision.
        candidate_causes: Plausible causes for an ambiguous fixture (provenance only).
        notes: Free-text provenance for the injected fault.
    """

    fixture_id: str
    cause_class: str
    activation_window: tuple[int, int]
    allowed_intervention: str
    ambiguity_status: str
    avoidable: bool
    candidate_causes: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the frozen fixture invariants.

        Raises:
            CollisionCauseAttributionError: If any field violates the contract.
        """
        if not self.fixture_id.strip():
            raise CollisionCauseAttributionError("fixture_id must be a non-empty string")
        if self.cause_class not in CAUSE_CLASSES:
            raise CollisionCauseAttributionError(
                f"unsupported cause_class {self.cause_class!r}; expected one of {sorted(CAUSE_CLASSES)}"
            )
        if self.ambiguity_status not in AMBIGUITY_STATUSES:
            raise CollisionCauseAttributionError(
                f"unsupported ambiguity_status {self.ambiguity_status!r}; "
                f"expected one of {sorted(AMBIGUITY_STATUSES)}"
            )
        start, end = self.activation_window
        if not (isinstance(start, int) and isinstance(end, int)):
            raise CollisionCauseAttributionError("activation_window bounds must be integers")
        # Cross-field consistency: the ground-truth class and ambiguity status must agree,
        # so a fixture cannot silently mislabel its own difficulty.
        _validate_window_for_status(
            self.ambiguity_status, self.cause_class, self.avoidable, start, end
        )
        _validate_class_status_agreement(
            self.ambiguity_status, self.cause_class, self.avoidable, self.candidate_causes
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the schema-tagged JSON-safe fixture payload.

        Returns:
            Mapping with the schema version and frozen ground-truth fields.
        """
        return {
            "schema_version": COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA,
            "fixture_id": self.fixture_id,
            "cause_class": self.cause_class,
            "activation_window": list(self.activation_window),
            "allowed_intervention": self.allowed_intervention,
            "ambiguity_status": self.ambiguity_status,
            "avoidable": self.avoidable,
            "candidate_causes": list(self.candidate_causes),
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> GroundTruthFixture:
        """Build a fixture from a JSON-style mapping, validating the schema version.

        Args:
            record: Mapping with the fixture fields; ``schema_version`` must match
                :data:`COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA`.

        Returns:
            The validated :class:`GroundTruthFixture`.

        Raises:
            CollisionCauseAttributionError: If the schema version or window is malformed.
        """
        version = record.get("schema_version")
        if version != COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA:
            raise CollisionCauseAttributionError(
                f"schema_version must be {COLLISION_CAUSE_ATTRIBUTION_FIXTURE_SCHEMA!r} (got {version!r})"
            )
        window = record.get("activation_window")
        if not isinstance(window, Sequence) or isinstance(window, (str, bytes)) or len(window) != 2:
            raise CollisionCauseAttributionError(
                "activation_window must be a two-element [start, end] sequence"
            )
        return cls(
            fixture_id=str(record.get("fixture_id", "")),
            cause_class=str(record.get("cause_class", "")),
            activation_window=(int(window[0]), int(window[1])),
            allowed_intervention=str(record.get("allowed_intervention", "")),
            ambiguity_status=str(record.get("ambiguity_status", "")),
            avoidable=bool(record.get("avoidable", False)),
            candidate_causes=tuple(str(c) for c in record.get("candidate_causes", ())),
            notes=str(record.get("notes", "")),
        )


@dataclass(frozen=True)
class AttributionVerdict:
    """One analyser attribution output for a fixture (consumed, never produced here).

    Attributes:
        fixture_id: Fixture this verdict addresses.
        predicted_cause: Predicted cause label, one of :data:`CAUSE_CLASSES`.
        predicted_activation_step: Predicted activation step, or ``None`` when abstained.
        confidence: Confidence in ``[0, 1]`` for the single-cause verdict.
        avoidable_pred: Predicted avoidability.
        abstained: Whether the analyser abstained from a single-cause verdict.
    """

    fixture_id: str
    predicted_cause: str
    predicted_activation_step: int | None
    confidence: float
    avoidable_pred: bool
    abstained: bool = False

    def __post_init__(self) -> None:
        """Validate the verdict invariants.

        Raises:
            CollisionCauseAttributionError: If a field violates the contract.
        """
        if self.predicted_cause not in CAUSE_CLASSES:
            raise CollisionCauseAttributionError(
                f"unsupported predicted_cause {self.predicted_cause!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise CollisionCauseAttributionError(
                f"confidence must be in [0, 1] (got {self.confidence})"
            )

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> AttributionVerdict:
        """Build a verdict from a JSON-style mapping.

        Returns:
            The validated :class:`AttributionVerdict`.
        """
        step = record.get("predicted_activation_step")
        return cls(
            fixture_id=str(record.get("fixture_id", "")),
            predicted_cause=str(record.get("predicted_cause", "")),
            predicted_activation_step=None if step is None else int(step),
            confidence=float(record.get("confidence", 0.0)),
            avoidable_pred=bool(record.get("avoidable_pred", False)),
            abstained=bool(record.get("abstained", False)),
        )


@dataclass(frozen=True)
class AttributionReport:
    """Scored attribution accuracy and calibration over a fixture manifest."""

    n_fixtures: int
    n_scored: int
    top_explanation_accuracy: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    median_temporal_localization_error: float | None
    avoidability_accuracy: float
    abstention_coverage: float
    ambiguous_high_confidence_violations: tuple[str, ...]
    negative_control_promotions: tuple[str, ...]
    confidence_calibration_gap: float | None
    verdict: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the schema-tagged JSON-safe report payload.

        Returns:
            Mapping with the report schema version and all scored metrics.
        """
        return {
            "schema_version": COLLISION_CAUSE_ATTRIBUTION_REPORT_SCHEMA,
            "n_fixtures": self.n_fixtures,
            "n_scored": self.n_scored,
            "top_explanation_accuracy": self.top_explanation_accuracy,
            "per_class_precision": dict(self.per_class_precision),
            "per_class_recall": dict(self.per_class_recall),
            "median_temporal_localization_error": self.median_temporal_localization_error,
            "avoidability_accuracy": self.avoidability_accuracy,
            "abstention_coverage": self.abstention_coverage,
            "ambiguous_high_confidence_violations": list(self.ambiguous_high_confidence_violations),
            "negative_control_promotions": list(self.negative_control_promotions),
            "confidence_calibration_gap": self.confidence_calibration_gap,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
        }


def validate_fixture_manifest(
    fixtures: Iterable[GroundTruthFixture | Mapping[str, Any]],
) -> list[GroundTruthFixture]:
    """Validate a manifest and confirm it covers the predeclared validation matrix.

    Coverage requires: at least one observation-family fixture (omission or delay),
    at least one fixture for each remaining single cause, at least one ambiguous
    fixture, and at least one negative control (issue "Predeclared validation matrix").
    Fixture ids must be unique. Fails closed on any gap.

    Args:
        fixtures: Fixtures as :class:`GroundTruthFixture` or JSON-style mappings.

    Returns:
        The validated fixtures as a list of :class:`GroundTruthFixture`.

    Raises:
        CollisionCauseAttributionError: If ids collide or coverage is incomplete.
    """
    resolved: list[GroundTruthFixture] = [
        f if isinstance(f, GroundTruthFixture) else GroundTruthFixture.from_mapping(f)
        for f in fixtures
    ]
    if not resolved:
        raise CollisionCauseAttributionError("fixture manifest is empty")

    seen: set[str] = set()
    for fixture in resolved:
        if fixture.fixture_id in seen:
            raise CollisionCauseAttributionError(
                f"duplicate fixture_id {fixture.fixture_id!r} in manifest"
            )
        seen.add(fixture.fixture_id)

    present_causes = {f.cause_class for f in resolved}
    statuses = {f.ambiguity_status for f in resolved}

    missing: list[str] = []
    if present_causes.isdisjoint(_OBSERVATION_FAMILY):
        missing.append("observation_omission|observation_delay")
    missing.extend(sorted(_REQUIRED_SINGLE_CAUSES - present_causes))
    if AMBIGUITY_AMBIGUOUS not in statuses:
        missing.append("<ambiguous fixture>")
    if AMBIGUITY_NEGATIVE_CONTROL not in statuses:
        missing.append("<negative control>")
    if missing:
        raise CollisionCauseAttributionError(
            f"fixture manifest does not cover the predeclared validation matrix; missing: {missing}"
        )
    return resolved


def _index_verdicts(
    verdicts: Iterable[AttributionVerdict | Mapping[str, Any]],
) -> dict[str, AttributionVerdict]:
    """Index verdicts by fixture id, rejecting duplicates.

    Returns:
        Mapping of fixture id to its (single) verdict.
    """
    indexed: dict[str, AttributionVerdict] = {}
    for verdict in verdicts:
        resolved = (
            verdict
            if isinstance(verdict, AttributionVerdict)
            else AttributionVerdict.from_mapping(verdict)
        )
        if resolved.fixture_id in indexed:
            raise CollisionCauseAttributionError(
                f"duplicate verdict for fixture_id {resolved.fixture_id!r}"
            )
        indexed[resolved.fixture_id] = resolved
    return indexed


def _temporal_error(fixture: GroundTruthFixture, predicted_step: int) -> int:
    """Distance in control steps from a predicted step to the activation window.

    Returns:
        Zero inside the inclusive window, else the distance to the nearest bound.
    """
    start, end = fixture.activation_window
    if predicted_step < start:
        return start - predicted_step
    if predicted_step > end:
        return predicted_step - end
    return 0


@dataclass(frozen=True)
class _UnambiguousScore:
    """Confusion-derived metrics over the unambiguous fixtures."""

    top_explanation_accuracy: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    median_temporal: float | None
    calibration_gap: float | None


def _score_unambiguous(
    unambiguous: Sequence[GroundTruthFixture], indexed: Mapping[str, AttributionVerdict]
) -> _UnambiguousScore:
    """Score top-explanation accuracy, per-class P/R, localization, and calibration.

    Returns:
        The :class:`_UnambiguousScore` over the unambiguous fixtures.
    """
    correct_count = 0
    temporal_errors: list[int] = []
    correct_conf: list[float] = []
    incorrect_conf: list[float] = []
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    for fixture in unambiguous:
        verdict = indexed[fixture.fixture_id]
        predicted = None if verdict.abstained else verdict.predicted_cause
        actual = fixture.cause_class
        if predicted == actual:
            correct_count += 1
            correct_conf.append(verdict.confidence)
            tp[actual] = tp.get(actual, 0) + 1
            if verdict.predicted_activation_step is not None:
                temporal_errors.append(_temporal_error(fixture, verdict.predicted_activation_step))
        else:
            incorrect_conf.append(verdict.confidence)
            fn[actual] = fn.get(actual, 0) + 1
            if predicted is not None:
                fp[predicted] = fp.get(predicted, 0) + 1

    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for cls in sorted(set(tp) | set(fp) | set(fn)):
        tp_c = tp.get(cls, 0)
        precision_den = tp_c + fp.get(cls, 0)
        recall_den = tp_c + fn.get(cls, 0)
        per_class_precision[cls] = tp_c / precision_den if precision_den else 0.0
        per_class_recall[cls] = tp_c / recall_den if recall_den else 0.0

    return _UnambiguousScore(
        top_explanation_accuracy=correct_count / len(unambiguous) if unambiguous else 0.0,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        median_temporal=float(statistics.median(temporal_errors)) if temporal_errors else None,
        calibration_gap=(
            statistics.mean(correct_conf) - statistics.mean(incorrect_conf)
            if correct_conf and incorrect_conf
            else None
        ),
    )


def _is_high_confidence_single_cause(verdict: AttributionVerdict, threshold: float) -> bool:
    """Return whether a verdict is a confident, concrete single-cause attribution.

    Returns:
        ``True`` when the analyser did not abstain, named a concrete cause (not
        ``interacting_ambiguous`` or ``none``), and met the confidence threshold.
    """
    return (
        not verdict.abstained
        and verdict.predicted_cause not in {CAUSE_INTERACTING_AMBIGUOUS, CAUSE_NONE}
        and verdict.confidence >= threshold
    )


def _score_ambiguity_guard(
    ambiguous: Sequence[GroundTruthFixture],
    indexed: Mapping[str, AttributionVerdict],
    threshold: float,
) -> tuple[float, list[str]]:
    """Score abstention coverage and collect ambiguous honesty violations.

    Returns:
        ``(abstention_coverage, violations)`` where ``violations`` names ambiguous
        fixtures that received a high-confidence single-cause verdict.
    """
    violations = [
        f.fixture_id
        for f in ambiguous
        if _is_high_confidence_single_cause(indexed[f.fixture_id], threshold)
    ]
    if not ambiguous:
        return 1.0, violations
    coverage = (len(ambiguous) - len(violations)) / len(ambiguous)
    return coverage, violations


def _score_negative_controls(
    negatives: Sequence[GroundTruthFixture],
    indexed: Mapping[str, AttributionVerdict],
    threshold: float,
) -> list[str]:
    """Collect negative controls promoted to a concrete cause at high confidence.

    Returns:
        Fixture ids of negative controls whose suspicious signal was promoted.
    """
    return [
        f.fixture_id
        for f in negatives
        if _is_high_confidence_single_cause(indexed[f.fixture_id], threshold)
    ]


def _build_verdict_reasons(
    has_unambiguous: bool,
    top_explanation_accuracy: float,
    median_temporal: float | None,
    ambiguous_violations: Sequence[str],
    negative_promotions: Sequence[str],
) -> list[str]:
    """Assemble the stop-rule failure reasons for a scored report.

    Returns:
        A list of human-readable reasons; empty means the report passes.
    """
    reasons: list[str] = []
    if has_unambiguous and top_explanation_accuracy < SIMPLE_ACCURACY_FLOOR:
        reasons.append(
            f"top-explanation accuracy {top_explanation_accuracy:.3f} below floor "
            f"{SIMPLE_ACCURACY_FLOOR:.2f}"
        )
    if median_temporal is not None and median_temporal > MAX_MEDIAN_TEMPORAL_ERROR_STEPS:
        reasons.append(
            f"median temporal-localization error {median_temporal:.2f} exceeds "
            f"{MAX_MEDIAN_TEMPORAL_ERROR_STEPS:.0f} control step"
        )
    if ambiguous_violations:
        reasons.append(
            f"ambiguous fixtures received high-confidence single-cause verdicts: "
            f"{list(ambiguous_violations)}"
        )
    if negative_promotions:
        reasons.append(
            f"negative controls promoted to a concrete cause: {list(negative_promotions)}"
        )
    return reasons


def score_attribution(
    fixtures: Sequence[GroundTruthFixture],
    verdicts: Iterable[AttributionVerdict | Mapping[str, Any]],
    *,
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
) -> AttributionReport:
    """Score analyser verdicts against a frozen fixture manifest.

    Every fixture must have exactly one verdict; a missing verdict fails closed.

    Metrics (acceptance criterion 2):

    * ``top_explanation_accuracy`` over *unambiguous* fixtures: fraction whose
      predicted cause matches the ground-truth class without abstaining;
    * per-class precision/recall over unambiguous fixtures;
    * ``median_temporal_localization_error`` over correctly-classified unambiguous
      fixtures, in control steps;
    * ``avoidability_accuracy`` over all fixtures;
    * ``abstention_coverage``: fraction of ambiguous fixtures the analyser abstained
      on or held below the high-confidence threshold;
    * ``confidence_calibration_gap``: mean confidence on correct minus incorrect
      unambiguous classifications (higher is better-calibrated).

    Honesty guards (criteria 4, 5): ambiguous fixtures that receive a high-confidence
    single-cause verdict, and negative controls promoted to a concrete cause at high
    confidence, are collected as violations and force a ``revise`` verdict.

    Returns:
        The scored :class:`AttributionReport`.

    Raises:
        CollisionCauseAttributionError: If a fixture has no verdict or the threshold
            is out of range.
    """
    if not (0.0 < high_confidence_threshold <= 1.0):
        raise CollisionCauseAttributionError(
            f"high_confidence_threshold must be in (0, 1] (got {high_confidence_threshold})"
        )
    indexed = _index_verdicts(verdicts)
    missing_ids = [f.fixture_id for f in fixtures if f.fixture_id not in indexed]
    if missing_ids:
        raise CollisionCauseAttributionError(
            f"missing analyser verdict for fixtures: {missing_ids}"
        )

    unambiguous = [f for f in fixtures if f.ambiguity_status == AMBIGUITY_UNAMBIGUOUS]
    ambiguous = [f for f in fixtures if f.ambiguity_status == AMBIGUITY_AMBIGUOUS]
    negatives = [f for f in fixtures if f.ambiguity_status == AMBIGUITY_NEGATIVE_CONTROL]

    scored = _score_unambiguous(unambiguous, indexed)
    abstention_coverage, ambiguous_violations = _score_ambiguity_guard(
        ambiguous, indexed, high_confidence_threshold
    )
    negative_promotions = _score_negative_controls(negatives, indexed, high_confidence_threshold)

    avoidable_correct = sum(
        1 for f in fixtures if indexed[f.fixture_id].avoidable_pred == f.avoidable
    )
    avoidability_accuracy = avoidable_correct / len(fixtures) if fixtures else 0.0

    reasons = _build_verdict_reasons(
        bool(unambiguous),
        scored.top_explanation_accuracy,
        scored.median_temporal,
        ambiguous_violations,
        negative_promotions,
    )

    return AttributionReport(
        n_fixtures=len(fixtures),
        n_scored=len(indexed),
        top_explanation_accuracy=scored.top_explanation_accuracy,
        per_class_precision=scored.per_class_precision,
        per_class_recall=scored.per_class_recall,
        median_temporal_localization_error=scored.median_temporal,
        avoidability_accuracy=avoidability_accuracy,
        abstention_coverage=abstention_coverage,
        ambiguous_high_confidence_violations=tuple(ambiguous_violations),
        negative_control_promotions=tuple(negative_promotions),
        confidence_calibration_gap=scored.calibration_gap,
        verdict=VERDICT_PASS if not reasons else VERDICT_REVISE,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class ValidationReport:
    """Top-level validation report: manifest coverage plus optional scoring."""

    status: str
    n_fixtures: int
    covered_matrix: bool
    report: AttributionReport | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe validation report payload.

        Returns:
            Mapping with status, coverage, and the scored report when available.
        """
        return {
            "schema_version": COLLISION_CAUSE_ATTRIBUTION_REPORT_SCHEMA,
            "status": self.status,
            "n_fixtures": self.n_fixtures,
            "covered_matrix": self.covered_matrix,
            "report": self.report.to_dict() if self.report is not None else None,
            "note": self.note,
        }


def build_validation_report(
    fixtures: Iterable[GroundTruthFixture | Mapping[str, Any]],
    verdicts: Iterable[AttributionVerdict | Mapping[str, Any]] | None = None,
    *,
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
) -> ValidationReport:
    """Validate the manifest and, when verdicts exist, score attribution.

    Fails closed: with no analyser verdicts the report is
    :data:`REPORT_STATUS_ANALYSER_UNAVAILABLE` and carries no fabricated metrics.
    This is the expected state until the analyser (#5441/#5442) can emit verdicts.

    Args:
        fixtures: The frozen ground-truth manifest.
        verdicts: Analyser verdicts, or ``None``/empty when the analyser is unavailable.
        high_confidence_threshold: Threshold above which single-cause verdicts count
            as high confidence for the ambiguity and negative-control guards.

    Returns:
        A :class:`ValidationReport` describing coverage and (when scored) accuracy.
    """
    resolved = validate_fixture_manifest(fixtures)
    verdict_list = list(verdicts) if verdicts is not None else []
    if not verdict_list:
        return ValidationReport(
            status=REPORT_STATUS_ANALYSER_UNAVAILABLE,
            n_fixtures=len(resolved),
            covered_matrix=True,
            report=None,
            note=(
                "manifest frozen and matrix covered; no analyser verdicts supplied. "
                "Accuracy scoring is blocked until the cause analyser (#5441/#5442) "
                "emits verdicts."
            ),
        )
    report = score_attribution(
        resolved, verdict_list, high_confidence_threshold=high_confidence_threshold
    )
    return ValidationReport(
        status=REPORT_STATUS_SCORED,
        n_fixtures=len(resolved),
        covered_matrix=True,
        report=report,
        note="scored analyser verdicts against the frozen fixture manifest",
    )
