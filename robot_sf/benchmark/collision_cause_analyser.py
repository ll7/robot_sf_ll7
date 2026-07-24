"""Deterministic collision-cause analyser for controlled injected traces (#5443).

The scorer's manifest is an answer key and is never an analyser input. Fixtures
provide only low-level expected/observed pipeline events, kinematic state, and a
counterfactual repair. The analyser:

1. maps observable event patterns to a cause class;
2. derives activation from the first matching event step;
3. verifies decisiveness by replaying the repaired scenario;
4. detects metric artifacts only when no physical contact occurs over the full
   observation horizon; and
5. abstains on ambiguous or non-causal signals.

Evidence grade: controlled-fixture injected-fault validation only. This is not
a real-trace root-cause claim and assigns no legal or moral fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robot_sf.benchmark.collision_cause_attribution import (
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
    AttributionVerdict,
)
from robot_sf.benchmark.last_avoidable_fixtures import (
    CollisionCauseFixture,
    InjectedFault,
    KinematicCollisionModel,
    KinematicScenario,
    ObservableTraceEvent,
    TraceValue,
    find_contact_step,
)
from robot_sf.benchmark.last_avoidable_replay import (
    SUBSTITUTION_HOLD,
    VERDICT_ALREADY_UNAVOIDABLE,
    LastAvoidableReport,
    ReplayConfig,
    locate_last_avoidable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


HIGH_CONFIDENCE = 0.95
ABSTAIN_CONFIDENCE = 0.3
_DETERMINISM_REPLAYS = 5
_REPAIR_CONTACT_LOOKAHEAD = 120


@dataclass(frozen=True)
class SignatureRule:
    """Map one low-level expected/observed trace pattern to a cause class."""

    channel: str
    field: str
    expected: TraceValue
    observed: TraceValue
    cause: str

    def matches(self, event: ObservableTraceEvent) -> bool:
        """Return whether ``event`` exhibits this observable signature."""
        return (
            event.channel == self.channel
            and event.field == self.field
            and event.expected == self.expected
            and event.observed == self.observed
        )

    def as_mapping(self) -> dict[str, TraceValue]:
        """Return a JSON-safe representation for provenance hashing."""
        return {
            "channel": self.channel,
            "field": self.field,
            "expected": self.expected,
            "observed": self.observed,
            "cause": self.cause,
        }


_SIGNATURE_RULES = (
    SignatureRule(
        "observation",
        "detection_present",
        True,
        False,
        CAUSE_OBSERVATION_OMISSION,
    ),
    SignatureRule("observation", "age_steps", 0, 3, CAUSE_OBSERVATION_DELAY),
    SignatureRule(
        "prediction",
        "crossing_predicted",
        True,
        False,
        CAUSE_PREDICTION_MISS,
    ),
    SignatureRule(
        "candidate_generation",
        "evasive_candidate_present",
        True,
        False,
        CAUSE_CANDIDATE_OMISSION,
    ),
    SignatureRule(
        "selection",
        "selected_candidate_safe",
        True,
        False,
        CAUSE_BAD_SELECTION,
    ),
    SignatureRule(
        "safety_guard",
        "intervention_applied",
        True,
        False,
        CAUSE_GUARD_OMISSION,
    ),
    SignatureRule(
        "actuation",
        "command_feasible",
        True,
        False,
        CAUSE_INFEASIBLE_APPLIED_COMMAND,
    ),
    SignatureRule("route", "escape_available", True, False, CAUSE_ROUTE_TRAP),
)


def analyser_config() -> dict[str, object]:
    """Return the complete JSON-safe rule configuration used by this analyser."""
    return {
        "schema_version": "collision_cause_analyser_config.v1",
        "signature_rules": [rule.as_mapping() for rule in _SIGNATURE_RULES],
        "high_confidence": HIGH_CONFIDENCE,
        "abstain_confidence": ABSTAIN_CONFIDENCE,
        "determinism_replays": _DETERMINISM_REPLAYS,
        "repair_contact_lookahead": _REPAIR_CONTACT_LOOKAHEAD,
        "metric_artifact_requires_no_physical_contact_over_horizon": True,
    }


@dataclass(frozen=True)
class DetectedFault:
    """A label inferred from observable events, separate from fixture inputs."""

    fault: InjectedFault
    predicted_cause: str
    activation_step: int


@dataclass(frozen=True)
class AnalyserEvidence:
    """Computed evidence used by the rule chain for one fixture."""

    fixture_id: str
    replay: LastAvoidableReport
    decisive_faults: tuple[DetectedFault, ...]
    present_faults: tuple[DetectedFault, ...]
    metric_quirk_onset: int | None
    avoidable_pred: bool


def scenario_collides(
    scenario: KinematicScenario,
    *,
    lookahead: int = _REPAIR_CONTACT_LOOKAHEAD,
) -> bool:
    """Return whether maintain-speed contact occurs within ``lookahead`` steps."""
    return find_contact_step(scenario, max_steps=lookahead) is not None


def _reported_but_no_physical_contact(
    scenario: KinematicScenario,
    *,
    lookahead: int = _REPAIR_CONTACT_LOOKAHEAD,
) -> int | None:
    """Return first reported-contact step only if physical contact never occurs.

    The complete horizon is scanned before a metric artifact is returned. An
    inflated footprint that reports contact early but reaches physical contact
    later is a real collision, not a metric artifact.
    """
    model = KinematicCollisionModel(scenario)
    first_reported: int | None = None
    for step in range(lookahead + 1):
        if model.physical_collision():
            return None
        if first_reported is None and model.collision():
            first_reported = step
        if step < lookahead:
            model.step(0.0)
    return first_reported


def _derive_replay_config(scenario: KinematicScenario) -> ReplayConfig:
    """Build deterministic replay config from the first reported contact.

    Returns:
        Replay configuration covering the reported contact and repair horizon.
    """
    contact_step = find_contact_step(scenario)
    if contact_step is None or contact_step < 1:
        contact_step = 1
    return ReplayConfig(
        t_danger=0,
        t_contact=contact_step,
        horizon=contact_step + 8,
        substitution_mode=SUBSTITUTION_HOLD,
        determinism_replays=_DETERMINISM_REPLAYS,
        action_set_id="decel_lattice",
        feasibility_filter="all_admissible_decel",
        collision_predicate="euclidean_distance<=collision_radius",
        pedestrian_response=scenario.pedestrian_response,
    )


def run_replay(
    fixture: CollisionCauseFixture,
) -> tuple[LastAvoidableReport, KinematicCollisionModel]:
    """Run last-avoidable replay on one controlled fixture.

    Returns:
        Replay report and the initialized model used for that replay.
    """
    config = fixture.replay_config or _derive_replay_config(fixture.scenario)
    model = KinematicCollisionModel(fixture.scenario)
    baseline = [0.0] * (config.t_contact + config.horizon + 2)
    report = locate_last_avoidable(model, baseline, config)
    return report, model


def _detect_fault(fault: InjectedFault) -> DetectedFault | None:
    """Infer one cause and onset from label-free observable events.

    Returns:
        Detected cause and onset, or ``None`` for unmatched/mixed signatures.
    """
    matches = [
        (rule.cause, event.step)
        for event in fault.events
        for rule in _SIGNATURE_RULES
        if rule.matches(event)
    ]
    causes = {cause for cause, _step in matches}
    if len(causes) != 1:
        return None
    cause = causes.pop()
    activation_step = min(step for matched_cause, step in matches if matched_cause == cause)
    return DetectedFault(
        fault=fault,
        predicted_cause=cause,
        activation_step=activation_step,
    )


def _gather_evidence(fixture: CollisionCauseFixture) -> AnalyserEvidence:
    """Compute replay, trace-signature, repair, and metric evidence.

    Returns:
        Complete evidence consumed by the attribution rule chain.
    """
    replay, _model = run_replay(fixture)
    detected = tuple(
        detected_fault
        for fault in fixture.faults
        if (detected_fault := _detect_fault(fault)) is not None
    )
    decisive: list[DetectedFault] = []
    for detected_fault in detected:
        repair = detected_fault.fault.repair_scenario
        if repair is not None and not scenario_collides(repair(fixture.scenario)):
            decisive.append(detected_fault)
    return AnalyserEvidence(
        fixture_id=fixture.fixture_id,
        replay=replay,
        decisive_faults=tuple(decisive),
        present_faults=detected,
        metric_quirk_onset=_reported_but_no_physical_contact(fixture.scenario),
        avoidable_pred=replay.verdict != VERDICT_ALREADY_UNAVOIDABLE,
    )


def _verdict_from_evidence(evidence: AnalyserEvidence) -> AttributionVerdict:
    """Map computed evidence to one cautious attribution verdict.

    Returns:
        Schema-compatible cause attribution or low-confidence abstention.
    """
    non_causal_signals = [
        detected
        for detected in evidence.present_faults
        if detected not in evidence.decisive_faults and not detected.fault.gates_applied_command
    ]

    if evidence.metric_quirk_onset is not None:
        return AttributionVerdict(
            fixture_id=evidence.fixture_id,
            predicted_cause=CAUSE_METRIC_ARTIFACT,
            predicted_activation_step=evidence.metric_quirk_onset,
            confidence=HIGH_CONFIDENCE,
            avoidable_pred=False,
            abstained=False,
        )

    if len(evidence.present_faults) >= 2 and not evidence.decisive_faults:
        return AttributionVerdict(
            fixture_id=evidence.fixture_id,
            predicted_cause=CAUSE_INTERACTING_AMBIGUOUS,
            predicted_activation_step=None,
            confidence=ABSTAIN_CONFIDENCE,
            avoidable_pred=evidence.avoidable_pred,
            abstained=True,
        )

    if len(evidence.decisive_faults) == 1:
        detected = evidence.decisive_faults[0]
        return AttributionVerdict(
            fixture_id=evidence.fixture_id,
            predicted_cause=detected.predicted_cause,
            predicted_activation_step=detected.activation_step,
            confidence=HIGH_CONFIDENCE,
            avoidable_pred=evidence.avoidable_pred,
            abstained=False,
        )

    if non_causal_signals:
        return AttributionVerdict(
            fixture_id=evidence.fixture_id,
            predicted_cause=CAUSE_NONE,
            predicted_activation_step=None,
            confidence=ABSTAIN_CONFIDENCE,
            avoidable_pred=evidence.avoidable_pred,
            abstained=True,
        )

    if not evidence.present_faults and evidence.replay.t_inevitable is not None:
        activation_step = evidence.replay.t_inevitable
        if activation_step is None:
            activation_step = evidence.replay.config.t_contact
        return AttributionVerdict(
            fixture_id=evidence.fixture_id,
            predicted_cause=CAUSE_ALREADY_UNAVOIDABLE_CONTACT,
            predicted_activation_step=activation_step,
            confidence=HIGH_CONFIDENCE,
            avoidable_pred=False,
            abstained=False,
        )

    return AttributionVerdict(
        fixture_id=evidence.fixture_id,
        predicted_cause=CAUSE_NONE,
        predicted_activation_step=None,
        confidence=ABSTAIN_CONFIDENCE,
        avoidable_pred=evidence.avoidable_pred,
        abstained=True,
    )


def analyse_cause(fixture: CollisionCauseFixture) -> AttributionVerdict:
    """Attribute one fixture from observable evidence and counterfactual replay.

    Returns:
        Cause verdict derived without scorer answer-key input.
    """
    return _verdict_from_evidence(_gather_evidence(fixture))


@dataclass(frozen=True)
class AnalyserRunResult:
    """Analyser verdicts and transparent evidence for a fixture suite."""

    verdicts: tuple[AttributionVerdict, ...] = field(default_factory=tuple)
    evidence: tuple[AnalyserEvidence, ...] = field(default_factory=tuple)

    def verdict_mappings(self) -> list[dict[str, object]]:
        """Return verdicts as JSON-safe mappings for the scoring CLI.

        Returns:
            One JSON-safe mapping per verdict.
        """
        return [
            {
                "fixture_id": verdict.fixture_id,
                "predicted_cause": verdict.predicted_cause,
                "predicted_activation_step": verdict.predicted_activation_step,
                "confidence": verdict.confidence,
                "avoidable_pred": verdict.avoidable_pred,
                "abstained": verdict.abstained,
            }
            for verdict in self.verdicts
        ]


def analyse_suite(fixtures: Sequence[CollisionCauseFixture]) -> AnalyserRunResult:
    """Run the analyser over fixtures while retaining its computed evidence.

    Returns:
        Ordered verdicts and corresponding transparent evidence.
    """
    evidence = tuple(_gather_evidence(fixture) for fixture in fixtures)
    return AnalyserRunResult(
        verdicts=tuple(_verdict_from_evidence(item) for item in evidence),
        evidence=evidence,
    )
