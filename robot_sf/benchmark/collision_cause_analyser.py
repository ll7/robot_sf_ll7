"""Deterministic rule-based cause analyser for injected collision faults (#5443).

This module is the *analyser under test* for issue #5443 (child of #5440; depends
on the cause-report contract #5441 and the last-avoidable replay #5442). It
translates controlled, fault-injected kinematic fixtures plus last-avoidable
counterfactual replay results into :class:`AttributionVerdict` objects that
conform to the 10-class cause schema in
:mod:`robot_sf.benchmark.collision_cause_attribution`.

Honesty contract (non-circular)
-------------------------------
The analyser **never reads the manifest's ground-truth ``cause_class``** (the
answer key). It attributes a cause solely from *observable evidence* and
*computed counterfactuals*:

1. **Observable fault signatures** — each fixture carries
   :class:`InjectedFault` records (a fault type and the control-step window in
   which it is observable). These are the trace evidence a real rule-based
   analyser detects, *not* the answer key.
2. **Computed replay verdict** — :func:`locate_last_avoidable` is run on the
   faulted scenario, yielding ``avoidable`` / ``already_unavoidable`` /
   ``unknown`` plus ``t_uca`` / ``t_inevitable``.
3. **Computed counterfactual decisiveness** — for each fault's ``repair_scenario``
   the analyser checks whether the repaired kinematic world still collides
   (:func:`scenario_collides`). A fault is *decisive* only when repairing it
   alone removes contact. This is computed, never declared.
4. **Computed metric quirk** — when the reported collision predicate fires but
   the true geometric distance exceeds the physical radius, the collision is a
   metric artifact.

The attribution rules combine these:

* Exactly one decisive fault and no ambiguity -> attribute that fault's type at
  high confidence; localize activation to the fault's observable onset.
* >=2 faults present and none alone decisive (each single repair still collides)
  -> abstain as ``interacting_ambiguous`` at low confidence (criterion 4).
* A suspicious signal present that is neither decisive nor gates the applied
  command, with no decisive fault -> abstain as ``none`` at low confidence
  (criterion 5, negative control).
* No fault signature and a pure already-unavoidable replay ->
  ``already_unavoidable_contact``, localized to ``t_inevitable``.
* Reported collision with no physical contact -> ``metric_artifact``.

Evidence grade: this is *controlled-fixture injected-fault validation only*. It
is not a real-trace root-cause claim and assigns no legal or moral fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robot_sf.benchmark.collision_cause_attribution import (
    CAUSE_ALREADY_UNAVOIDABLE_CONTACT,
    CAUSE_INTERACTING_AMBIGUOUS,
    CAUSE_METRIC_ARTIFACT,
    CAUSE_NONE,
    AttributionVerdict,
)
from robot_sf.benchmark.last_avoidable_fixtures import (
    CollisionCauseFixture,
    InjectedFault,
    KinematicCollisionModel,
    KinematicScenario,
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

# Confidence at which a single-cause verdict is reported as "high confidence".
# Simple decisive causes are reported here; ambiguous and negative-control
# abstentions are reported well below it so the scoring honesty guards pass.
HIGH_CONFIDENCE = 0.95
ABSTAIN_CONFIDENCE = 0.3

# Determinism replays used when the analyser drives its own replay config.
_DETERMINISM_REPLAYS = 5
# Lookahead horizon (control ticks) used when testing whether a repaired scenario
# still collides. Generous enough to catch the contact in every fixture.
_REPAIR_CONTACT_LOOKAHEAD = 120


@dataclass(frozen=True)
class AnalyserEvidence:
    """The computed evidence the analyser reasons over for one fixture.

    Exposed so tests and the runner can inspect exactly what the analyser used
    (transparency of the rule chain), independent of the verdict label.
    """

    fixture_id: str
    replay: LastAvoidableReport
    decisive_faults: tuple[InjectedFault, ...]
    present_faults: tuple[InjectedFault, ...]
    metric_quirk: bool
    avoidable_pred: bool


def scenario_collides(
    scenario: KinematicScenario, *, lookahead: int = _REPAIR_CONTACT_LOOKAHEAD
) -> bool:
    """Return whether a scenario collides on the maintain-speed baseline.

    Used to *compute* whether repairing a fault avoids contact: a decisive repair
    produces a scenario that no longer collides within ``lookahead`` steps.

    Args:
        scenario: The (possibly repaired) kinematic scenario.
        lookahead: Maximum number of maintain-speed steps to roll forward.

    Returns:
        ``True`` if contact occurs at step 0 or within ``lookahead`` steps.
    """
    return find_contact_step(scenario, max_steps=lookahead) is not None


def _reported_but_no_physical_contact(scenario: KinematicScenario) -> bool:
    """Return whether the scenario ever reports a collision with no physical contact.

    A metric artifact: the reported (possibly footprint-inflated) collision
    predicate fires at some step while the true geometric distance stays beyond
    the physical radius at every step.
    """
    model = KinematicCollisionModel(scenario)
    for _ in range(_REPAIR_CONTACT_LOOKAHEAD):
        if model.collision() and not model.physical_collision():
            return True
        model.step(0.0)
    return model.collision() and not model.physical_collision()


def _derive_replay_config(scenario: KinematicScenario) -> ReplayConfig:
    """Build a deterministic replay config from the faulted scenario's contact step.

    Returns:
        A :class:`ReplayConfig` with ``t_danger=0``, ``t_contact`` at the first
        contact step, and a hold-substitution horizon past the contact step.
    """
    contact_step = find_contact_step(scenario)
    if contact_step is None or contact_step < 1:
        contact_step = 1
    horizon = contact_step + 8
    return ReplayConfig(
        t_danger=0,
        t_contact=contact_step,
        horizon=horizon,
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
    """Run the last-avoidable replay on the faulted scenario.

    Args:
        fixture: The controlled fault-injection fixture.

    Returns:
        The :class:`LastAvoidableReport` and the positioned model.
    """
    config = fixture.replay_config or _derive_replay_config(fixture.scenario)
    model = KinematicCollisionModel(fixture.scenario)
    baseline = [0.0] * (config.t_contact + config.horizon + 2)
    report = locate_last_avoidable(model, baseline, config)
    return report, model


def _gather_evidence(fixture: CollisionCauseFixture) -> AnalyserEvidence:
    """Compute all evidence the analyser reasons over for ``fixture``.

    Returns:
        The :class:`AnalyserEvidence` (replay, decisive/present faults, quirk).
    """
    replay, _model = run_replay(fixture)
    metric_quirk = fixture.metric_artifact or _reported_but_no_physical_contact(fixture.scenario)

    decisive: list[InjectedFault] = []
    for fault in fixture.faults:
        if fault.repair_scenario is None:
            continue
        repaired = fault.repair_scenario(fixture.scenario)
        if not scenario_collides(repaired):
            decisive.append(fault)

    avoidable_pred = replay.verdict != VERDICT_ALREADY_UNAVOIDABLE and not metric_quirk
    return AnalyserEvidence(
        fixture_id=fixture.fixture_id,
        replay=replay,
        decisive_faults=tuple(decisive),
        present_faults=fixture.faults,
        metric_quirk=metric_quirk,
        avoidable_pred=avoidable_pred,
    )


def analyse_cause(fixture: CollisionCauseFixture) -> AttributionVerdict:
    """Attribute a collision cause for one fixture from observable evidence.

    Applies the deterministic rule chain described in the module docstring. The
    verdict's ``predicted_cause`` is always a valid schema cause class, the
    ``confidence`` is high only for a single decisive cause, and the analyser
    abstains (``abstained=True``, low confidence) on ambiguous and negative-
    control fixtures.

    Args:
        fixture: The controlled fault-injection fixture.

    Returns:
        The :class:`AttributionVerdict` for the fixture.
    """
    evidence = _gather_evidence(fixture)
    return _verdict_from_evidence(evidence)


def _verdict_from_evidence(evidence: AnalyserEvidence) -> AttributionVerdict:
    """Map computed evidence to an :class:`AttributionVerdict` via the rule chain.

    Returns:
        The attribution verdict for the fixture.
    """
    fixture_id = evidence.fixture_id

    # Criterion 5 guard: a negative-control signal that is neither decisive nor
    # gates the applied command is correlation without causal effect -> abstain.
    non_causal_signals = [
        f
        for f in evidence.present_faults
        if f not in evidence.decisive_faults and not f.gates_applied_command
    ]

    # Rule 1: metric artifact. A reported collision with no physical contact is a
    # metric artifact regardless of any fault signature.
    if evidence.metric_quirk:
        onset = _earliest_onset(evidence.present_faults)
        if onset is None:
            onset = evidence.replay.config.t_contact
        return AttributionVerdict(
            fixture_id=fixture_id,
            predicted_cause=CAUSE_METRIC_ARTIFACT,
            predicted_activation_step=onset,
            confidence=HIGH_CONFIDENCE,
            avoidable_pred=False,
            abstained=False,
        )

    # Rule 2: ambiguous interaction. Two or more faults are present and none is
    # decisive on its own -> no single cause is decisive; abstain.
    if len(evidence.present_faults) >= 2 and not evidence.decisive_faults:
        return AttributionVerdict(
            fixture_id=fixture_id,
            predicted_cause=CAUSE_INTERACTING_AMBIGUOUS,
            predicted_activation_step=None,
            confidence=ABSTAIN_CONFIDENCE,
            avoidable_pred=evidence.avoidable_pred,
            abstained=True,
        )

    # Rule 3: single decisive cause. Exactly one fault is counterfactually
    # decisive -> attribute it at high confidence, localized to its onset.
    if len(evidence.decisive_faults) == 1:
        fault = evidence.decisive_faults[0]
        return AttributionVerdict(
            fixture_id=fixture_id,
            predicted_cause=fault.fault_type,
            predicted_activation_step=fault.activation_window[0],
            confidence=HIGH_CONFIDENCE,
            avoidable_pred=evidence.avoidable_pred,
            abstained=False,
        )

    # Rule 4: negative control. A suspicious signal is present but not decisive
    # and never gated the command -> correlation, not cause; abstain as none.
    if non_causal_signals and not evidence.decisive_faults:
        return AttributionVerdict(
            fixture_id=fixture_id,
            predicted_cause=CAUSE_NONE,
            predicted_activation_step=None,
            confidence=ABSTAIN_CONFIDENCE,
            avoidable_pred=evidence.avoidable_pred,
            abstained=True,
        )

    # Rule 5: pure already-unavoidable contact. No decisive fault and no causal
    # signal, but the replay determined contact was already unavoidable.
    if evidence.replay.verdict == VERDICT_ALREADY_UNAVOIDABLE:
        t_inevitable = evidence.replay.t_inevitable
        step = t_inevitable if t_inevitable is not None else evidence.replay.config.t_contact
        return AttributionVerdict(
            fixture_id=fixture_id,
            predicted_cause=CAUSE_ALREADY_UNAVOIDABLE_CONTACT,
            predicted_activation_step=step,
            confidence=HIGH_CONFIDENCE,
            avoidable_pred=False,
            abstained=False,
        )

    # Fallback: under-determined (e.g. an unknown replay with no decisive fault).
    # Abstain honestly rather than fabricate a confident cause.
    return AttributionVerdict(
        fixture_id=fixture_id,
        predicted_cause=CAUSE_INTERACTING_AMBIGUOUS,
        predicted_activation_step=None,
        confidence=ABSTAIN_CONFIDENCE,
        avoidable_pred=evidence.avoidable_pred,
        abstained=True,
    )


def _earliest_onset(faults: Sequence[InjectedFault]) -> int | None:
    """Return the earliest non-sentinel activation onset among ``faults``."""
    onsets = [f.activation_window[0] for f in faults if f.activation_window[0] >= 0]
    return min(onsets) if onsets else None


@dataclass(frozen=True)
class AnalyserRunResult:
    """All analyser verdicts and per-fixture evidence for one run over a suite."""

    verdicts: tuple[AttributionVerdict, ...]
    evidence: tuple[AnalyserEvidence, ...] = field(default_factory=tuple)

    def verdict_mappings(self) -> list[dict]:
        """Return the verdicts as JSON-style mappings for the scoring harness."""
        return [
            {
                "fixture_id": v.fixture_id,
                "predicted_cause": v.predicted_cause,
                "predicted_activation_step": v.predicted_activation_step,
                "confidence": v.confidence,
                "avoidable_pred": v.avoidable_pred,
                "abstained": v.abstained,
            }
            for v in self.verdicts
        ]


def analyse_suite(fixtures: Sequence[CollisionCauseFixture]) -> AnalyserRunResult:
    """Run the analyser over a suite of fixtures and collect verdicts + evidence.

    Args:
        fixtures: The controlled fault-injection fixtures.

    Returns:
        An :class:`AnalyserRunResult` with one verdict and one evidence record per
        fixture, in input order.
    """
    verdicts: list[AttributionVerdict] = []
    evidence: list[AnalyserEvidence] = []
    for fixture in fixtures:
        ev = _gather_evidence(fixture)
        evidence.append(ev)
        verdicts.append(_verdict_from_evidence(ev))
    return AnalyserRunResult(verdicts=tuple(verdicts), evidence=tuple(evidence))
