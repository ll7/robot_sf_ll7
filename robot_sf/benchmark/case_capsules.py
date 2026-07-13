"""Pinned Chapter 7 causal-trajectory *case-capsule* manifest builder (issue #5447).

This module assembles a **reproducible case-capsule manifest** for the Chapter 7
worked examples from a *validated* candidate manifest (the
``seed_flip_inversion_candidates.v1`` output of the issue #5446 miner,
:mod:`robot_sf.benchmark.seed_flip_mining`) plus optional causal / online-risk
reports (issues #5441-#5445).

Design contract (issue #5447)
-----------------------------
It is synthesis/selection tooling, **not** a benchmark metric and **not** a
figure renderer. It answers one question honestly: *given the candidates that
actually passed the evidence gates, which diverse worked-example archetypes can
Chapter 7 support, and what remains unavailable?* It deliberately avoids the two
failure modes the issue calls out:

1. **Never fabricate.** A capsule archetype whose source candidate or required
   causal/risk report is missing is emitted with ``status = "unavailable"`` and a
   concrete reason. Unavailable archetypes are labelled unavailable — never
   replaced post hoc by an attractive but unvalidated row.
2. **Descriptive-only unless a validated causal report exists.** Causal labels
   come only from a supplied validated causal report; otherwise the capsule is
   graded ``descriptive-only`` and its caption must say so.
3. **Fail closed.** An empty / wrong-schema / candidate-free manifest raises
   :class:`CaseCapsuleError` rather than emitting an empty capsule set. When
   fewer than ``min_capsules`` archetypes are admissible the manifest reports
   ``status = "insufficient_evidence"`` per the issue stop rule (stop at a
   smaller honest set; do not broaden claims).
4. **Author-pending, not fabricated, narrative.** Fields that require a human
   writer (competing explanation, what-failed, generalisation limits, the marked
   times and "why this time matters" text) are set to the :data:`AUTHOR_REQUIRED`
   sentinel and reported by the validator as ``author_pending`` — structurally
   valid, honestly incomplete. Mechanically derivable fields (source planner,
   seeds, archetype rationale) are filled from the candidate record.

The builder consumes a candidate manifest *dict* (already frozen at a pinned SHA
by the #5446 miner); it does not run benchmarks and does not read episode
artefacts. Actual vector-figure rendering from pinned episode trajectories is a
downstream step that reuses :mod:`robot_sf.benchmark.figures`; this module only
produces the machine-readable capsule manifest and its input-hash provenance so
that rendering step is reproducible.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

#: Schema tag for the emitted case-capsule manifest.
SCHEMA_VERSION = "ch7_case_capsule_manifest.v1"

#: Schema tag of the candidate manifest this builder consumes (issue #5446).
CANDIDATE_SCHEMA_VERSION = "seed_flip_inversion_candidates.v1"

#: Sentinel for a field that a human writer must supply before dissertation
#: integration. It is never fabricated by the builder; the validator reports every
#: remaining sentinel as ``author_pending`` (expected, not a structural failure).
AUTHOR_REQUIRED = "AUTHOR_REQUIRED"

#: Ordered evidence grades, strongest first. ``unavailable`` is reserved for
#: capsule slots that could not be sourced at all.
EVIDENCE_GRADES = ("causal", "descriptive-risk", "descriptive-only", "unavailable")

#: Default honest floor for a usable capsule set (issue #5447 stop rule: stop at
#: a smaller honest set if fewer than four archetypes pass the evidence gates).
DEFAULT_MIN_CAPSULES = 4

#: Optional ceiling on the number of capsules (issue #5447: four to six).
DEFAULT_MAX_CAPSULES = 6

#: Required keys of a pair-comparison shared-axis specification. Pair capsules
#: must render with identical axes / crop / palette / time markers / scales.
SHARED_AXIS_KEYS = (
    "axes_limits",
    "map_crop",
    "palette",
    "time_markers",
    "metric_risk_scale",
)

#: Required keys of a probability (online-risk) panel (issue #5447 acceptance).
PROBABILITY_PANEL_KEYS = (
    "horizon",
    "conditioning_action",
    "estimator_version",
    "calibration_status",
    "uncertainty",
)

#: Narrative fields every admitted capsule must carry (issue #5447 acceptance:
#: why selected, what went well, what failed, competing explanation, evidence
#: grade, generalisation limits).
NARRATIVE_KEYS = (
    "selection_rationale",
    "what_went_well",
    "what_failed",
    "competing_explanation",
    "evidence_grade",
    "generalization_limits",
)


class CaseCapsuleError(ValueError):
    """Raised when a defensible case-capsule manifest cannot be built."""


@dataclass(frozen=True)
class CapsuleArchetypeSpec:
    """Declarative spec for one Chapter 7 capsule archetype (issue #5447).

    Attributes:
        key: Stable archetype identifier used in the emitted manifest.
        title: Human-facing archetype description.
        source_archetypes: Candidate-manifest archetypes this capsule may draw
            from, in preference order.
        requires_causal: The archetype's defining claim is causal; without a
            validated causal report it is ``unavailable`` (not descriptive-only).
        requires_risk: The archetype needs an online-risk report; without one it
            is ``unavailable``.
        needs_disagreement: The source candidate must carry non-zero cross-planner
            disagreement entropy (recovery / critical-interval archetype).
        is_pair: The capsule is a paired comparison and needs a shared-axis spec.
        optional: The archetype is optional (issue #5447 sixth capsule); its
            absence does not by itself fail the set.
    """

    key: str
    title: str
    source_archetypes: tuple[str, ...]
    requires_causal: bool = False
    requires_risk: bool = False
    needs_disagreement: bool = False
    is_pair: bool = False
    optional: bool = False


#: The issue #5447 target capsule set, in order. Maps each desired worked-example
#: archetype to the candidate archetypes and reports it requires.
CH7_CAPSULE_ARCHETYPES: tuple[CapsuleArchetypeSpec, ...] = (
    CapsuleArchetypeSpec(
        key="hard_vs_easy_seed",
        title="hard versus easy seed for the same planner/scenario",
        source_archetypes=("seed_flip",),
        is_pair=True,
    ),
    CapsuleArchetypeSpec(
        key="strong_fail_weak_success",
        title="held-out strong-planner failure versus weak-planner success",
        source_archetypes=("planner_upset",),
        is_pair=True,
    ),
    CapsuleArchetypeSpec(
        key="paired_first_unsafe_action",
        title="paired approach with different first unsafe action and counterfactual",
        source_archetypes=("planner_upset", "seed_flip"),
        requires_causal=True,
        is_pair=True,
    ),
    CapsuleArchetypeSpec(
        key="near_miss_online_risk",
        title="near miss where online risk distinguishes candidate actions early",
        source_archetypes=("seed_flip", "planner_upset"),
        requires_risk=True,
    ),
    CapsuleArchetypeSpec(
        key="unexpected_recovery",
        title="unexpected recovery or critical-interval/aggregate disagreement",
        source_archetypes=("seed_flip", "planner_upset"),
        needs_disagreement=True,
    ),
    CapsuleArchetypeSpec(
        key="ambiguous_abstention",
        title="optional ambiguous multi-actor case whose correct causal output is abstention",
        source_archetypes=("planner_upset", "seed_flip"),
        requires_causal=True,
        optional=True,
    ),
)


def canonical_sha256(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable object.

    Uses sorted keys and compact separators so the same logical manifest always
    hashes identically regardless of dict insertion order.

    Returns:
        The hex digest string.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_candidate_manifest(manifest: Any) -> list[dict[str, Any]]:
    """Fail-closed check of the incoming candidate manifest; return its candidates.

    Returns:
        The list of candidate records.

    Raises:
        CaseCapsuleError: When the manifest is not a dict, carries the wrong
            schema version, or contains no candidate records.
    """
    if not isinstance(manifest, dict):
        raise CaseCapsuleError(f"candidate manifest must be a dict, got {type(manifest).__name__}")
    schema = manifest.get("schema_version")
    if schema != CANDIDATE_SCHEMA_VERSION:
        raise CaseCapsuleError(
            f"candidate manifest schema mismatch: expected {CANDIDATE_SCHEMA_VERSION!r}, "
            f"got {schema!r}. Build capsules only from a #5446 validated candidate manifest."
        )
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CaseCapsuleError(
            "candidate manifest has no candidates; refusing to build capsules from empty "
            "evidence (fail closed rather than fabricate a capsule set)"
        )
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise CaseCapsuleError(
            "candidate manifest contains non-dict candidate records; refusing to build capsules "
            "from malformed evidence"
        )
    return candidates


def _report_ref(reports: dict[str, Any] | None, *keys: str) -> Any:
    """Return the first matching validated report for ``keys`` or ``None``.

    ``reports`` maps a candidate id or scenario id to a validated report payload.
    A missing table or missing key yields ``None`` (never fabricated).

    Returns:
        The report payload, or ``None`` when unavailable.
    """
    if not isinstance(reports, dict):
        return None
    for key in keys:
        if key in reports and reports[key] is not None:
            return reports[key]
    return None


def _candidate_sort_key(cand: dict[str, Any]) -> tuple[int, int, float, float]:
    """Rank a candidate for capsule sourcing (higher is better, so negate).

    Prefers selected (Pareto-frontier) candidates, non-triage cells, larger
    disagreement entropy, and larger effective evidence.

    Returns:
        A tuple usable as a ``sorted`` key (ascending -> best first via negation).
    """
    flip = cand.get("seed_flip")
    if not isinstance(flip, dict):
        flip = {}
    return (
        0 if cand.get("selected") else 1,
        0 if not cand.get("triage_only") else 1,
        -float(cand.get("cross_planner_disagreement_entropy") or 0.0),
        -float(flip.get("effective_denominator") or 0.0),
    )


def _pick_candidate(
    spec: CapsuleArchetypeSpec,
    candidates: list[dict[str, Any]],
    used: set[str],
    *,
    allow_triage: bool,
) -> dict[str, Any] | None:
    """Choose the best unused candidate that can source ``spec`` or ``None``.

    Returns:
        The chosen candidate record, or ``None`` when no eligible source exists.
    """
    pool = [
        c
        for c in candidates
        if c.get("candidate_id") not in used
        and c.get("archetype") in spec.source_archetypes
        and (allow_triage or not c.get("triage_only"))
        and (
            not spec.needs_disagreement
            or float(c.get("cross_planner_disagreement_entropy") or 0.0) > 0.0
        )
    ]
    if not pool:
        return None
    return sorted(pool, key=_candidate_sort_key)[0]


def _shared_axis_spec() -> dict[str, str]:
    """Return an author-required shared-axis spec skeleton for pair capsules.

    Returns:
        A mapping whose keys are :data:`SHARED_AXIS_KEYS`, each set to the
        :data:`AUTHOR_REQUIRED` sentinel.
    """
    return dict.fromkeys(SHARED_AXIS_KEYS, AUTHOR_REQUIRED)


def _probability_panel() -> dict[str, str]:
    """Return an author-required online-risk probability-panel skeleton.

    Returns:
        A mapping whose keys are :data:`PROBABILITY_PANEL_KEYS`, each set to the
        :data:`AUTHOR_REQUIRED` sentinel.
    """
    return dict.fromkeys(PROBABILITY_PANEL_KEYS, AUTHOR_REQUIRED)


def _figure_spec(spec: CapsuleArchetypeSpec) -> dict[str, Any]:
    """Build the reproducible figure specification for a capsule.

    Every capsule requires a static map, metre scale, full robot/dynamic-object
    trajectories, common marked times, and actor footprints at those times
    (issue #5447 acceptance). Pair capsules additionally require a shared-axis
    spec so both panels render identically.

    Returns:
        The figure-spec mapping.
    """
    return {
        "static_map_required": True,
        "metre_scale_required": True,
        "trajectory_sources_required": True,
        "actor_footprints_at_marked_times_required": True,
        "marked_times": AUTHOR_REQUIRED,
        "why_this_time_matters": AUTHOR_REQUIRED,
        "shared_axis_spec": _shared_axis_spec() if spec.is_pair else None,
    }


def _selection_rationale(spec: CapsuleArchetypeSpec, cand: dict[str, Any]) -> str:
    """Mechanically derive a selection rationale from the source candidate.

    Returns:
        A one-line rationale string (data-derived, not fabricated narrative).
    """
    return (
        f"{spec.title}: sourced from {cand.get('archetype')} candidate "
        f"{cand.get('candidate_id')} (scenario={cand.get('scenario_id')}, "
        f"planner={cand.get('planner')}, "
        f"selected={bool(cand.get('selected'))}, triage_only={bool(cand.get('triage_only'))})."
    )


def _build_admitted_capsule(
    spec: CapsuleArchetypeSpec,
    cand: dict[str, Any],
    *,
    grade: str,
    causal_report: Any,
    risk_report: Any,
) -> dict[str, Any]:
    """Assemble an admitted capsule record from a resolved candidate.

    Returns:
        The capsule record dict.
    """
    narrative = {
        "selection_rationale": _selection_rationale(spec, cand),
        "what_went_well": AUTHOR_REQUIRED,
        "what_failed": AUTHOR_REQUIRED,
        "competing_explanation": AUTHOR_REQUIRED,
        "evidence_grade": grade,
        "generalization_limits": AUTHOR_REQUIRED,
    }
    return {
        "capsule_id": f"ch7::{spec.key}",
        "archetype": spec.key,
        "title": spec.title,
        "status": "admitted",
        "optional": spec.optional,
        "evidence_grade": grade,
        "is_pair": spec.is_pair,
        "source_candidate_id": cand.get("candidate_id"),
        "source_archetype": cand.get("archetype"),
        "scenario_id": cand.get("scenario_id"),
        "planner": cand.get("planner"),
        "triage_only": bool(cand.get("triage_only")),
        "figure_spec": _figure_spec(spec),
        "probability_panel": _probability_panel() if spec.requires_risk else None,
        # Causal labels only from a validated causal report; otherwise descriptive.
        "causal_label": (
            {"status": "validated", "report_ref": copy.deepcopy(causal_report)}
            if causal_report is not None
            else "descriptive-only"
        ),
        "risk_report_ref": copy.deepcopy(risk_report) if risk_report is not None else None,
        "narrative": narrative,
        "source_provenance": {
            "candidate_id": cand.get("candidate_id"),
            "scenario_id": cand.get("scenario_id"),
            "planner": cand.get("planner"),
            "reproducibility": copy.deepcopy(cand.get("reproducibility")),
            "seed_flip": copy.deepcopy(cand.get("seed_flip")),
            "upset_outcome": copy.deepcopy(cand.get("upset_outcome")),
        },
    }


def _unavailable_capsule(spec: CapsuleArchetypeSpec, reason: str) -> dict[str, Any]:
    """Build an ``unavailable`` capsule slot with a concrete reason.

    Returns:
        The unavailable-capsule record dict.
    """
    return {
        "capsule_id": f"ch7::{spec.key}",
        "archetype": spec.key,
        "title": spec.title,
        "status": "unavailable",
        "optional": spec.optional,
        "evidence_grade": "unavailable",
        "reason": reason,
    }


def _resolve_capsule(
    spec: CapsuleArchetypeSpec,
    candidates: list[dict[str, Any]],
    used: set[str],
    *,
    causal_reports: dict[str, Any] | None,
    risk_reports: dict[str, Any] | None,
    allow_triage: bool,
) -> dict[str, Any]:
    """Resolve one capsule archetype to an admitted or unavailable slot.

    Order of checks (fail closed): a source candidate must exist; a causal-defined
    archetype needs a validated causal report; a risk archetype needs a risk
    report. Any missing input yields an ``unavailable`` slot with a reason rather
    than a fabricated capsule.

    Returns:
        The resolved capsule record dict.
    """
    cand = _pick_candidate(spec, candidates, used, allow_triage=allow_triage)
    if cand is None:
        return _unavailable_capsule(spec, f"no_source_candidate:{'|'.join(spec.source_archetypes)}")

    causal_report = (
        _report_ref(causal_reports, str(cand.get("candidate_id")), str(cand.get("scenario_id")))
        if spec.requires_causal
        else None
    )
    risk_report = (
        _report_ref(risk_reports, str(cand.get("candidate_id")), str(cand.get("scenario_id")))
        if spec.requires_risk
        else None
    )

    if spec.requires_causal and causal_report is None:
        return _unavailable_capsule(spec, "causal_report_unavailable")
    if spec.requires_risk and risk_report is None:
        return _unavailable_capsule(spec, "risk_report_unavailable")

    if spec.requires_causal:
        grade = "causal"
    elif spec.requires_risk:
        grade = "descriptive-risk"
    else:
        grade = "descriptive-only"

    used.add(str(cand.get("candidate_id")))
    return _build_admitted_capsule(
        spec, cand, grade=grade, causal_report=causal_report, risk_report=risk_report
    )


def build_ch7_case_capsule_manifest(
    candidate_manifest: dict[str, Any],
    *,
    causal_reports: dict[str, Any] | None = None,
    risk_reports: dict[str, Any] | None = None,
    archetypes: tuple[CapsuleArchetypeSpec, ...] = CH7_CAPSULE_ARCHETYPES,
    min_capsules: int = DEFAULT_MIN_CAPSULES,
    max_capsules: int = DEFAULT_MAX_CAPSULES,
    allow_triage: bool = False,
) -> dict[str, Any]:
    """Build a pinned Chapter 7 case-capsule manifest from a candidate manifest.

    Args:
        candidate_manifest: A validated ``seed_flip_inversion_candidates.v1``
            manifest (issue #5446 miner output). Frozen at a pinned SHA by the
            caller; its canonical hash is recorded for provenance.
        causal_reports: Optional mapping from candidate id / scenario id to a
            *validated* causal report (issues #5441-#5445). Absent reports leave
            causal-requiring archetypes ``unavailable`` and non-causal capsules
            ``descriptive-only``.
        risk_reports: Optional mapping from candidate id / scenario id to a
            validated online-risk report. Absent reports leave risk-requiring
            archetypes ``unavailable``.
        archetypes: The ordered capsule archetype specs to resolve.
        min_capsules: Honest floor; below it the manifest is
            ``insufficient_evidence`` (issue #5447 stop rule).
        max_capsules: Ceiling; admitted capsules beyond it are flagged
            ``over_capacity`` rather than silently dropped.
        allow_triage: Permit triage-only (<= 3-seed) candidates as sources. Off by
            default so triage candidates never silently back a paper-facing capsule.

    Returns:
        A schema-versioned case-capsule manifest dict.

    Raises:
        CaseCapsuleError: When the candidate manifest fails the fail-closed gate
            (wrong type/schema or no candidates) or ``min_capsules`` is invalid.
    """
    if min_capsules < 1:
        raise CaseCapsuleError(f"min_capsules must be >= 1, got {min_capsules}")
    if causal_reports is not None and not isinstance(causal_reports, dict):
        raise CaseCapsuleError(
            f"causal_reports must be a dict, got {type(causal_reports).__name__}"
        )
    if risk_reports is not None and not isinstance(risk_reports, dict):
        raise CaseCapsuleError(f"risk_reports must be a dict, got {type(risk_reports).__name__}")
    candidates = _validate_candidate_manifest(candidate_manifest)

    used: set[str] = set()
    capsules = [
        _resolve_capsule(
            spec,
            candidates,
            used,
            causal_reports=causal_reports,
            risk_reports=risk_reports,
            allow_triage=allow_triage,
        )
        for spec in archetypes
    ]

    admitted = [c for c in capsules if c["status"] == "admitted"]
    # A capsule beyond the ceiling is a signal that the set is too large (issue
    # #5447 competing explanation: too many panels reduce explanation quality).
    for capsule in admitted[max_capsules:]:
        capsule["over_capacity"] = True

    n_admitted = len(admitted)
    if n_admitted < min_capsules:
        status = "insufficient_evidence"
    elif n_admitted > max_capsules:
        status = "over_capacity"
    else:
        status = "ok"

    unavailable = [c for c in capsules if c["status"] == "unavailable"]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": "#5447",
        "status": status,
        "claim_boundary": (
            "Synthesis/selection tooling: assembles Chapter 7 case-capsule slots from a "
            "validated #5446 candidate manifest. Not a benchmark metric, not a figure "
            "renderer, not a planner-ranking claim. Capsules are descriptive-only unless a "
            "validated causal report is supplied; unavailable archetypes are labelled "
            "unavailable, never substituted. Author-pending narrative/figure fields must be "
            "completed and an independent pinned-SHA review recorded before dissertation use."
        ),
        "stop_rule": (
            "Stop at a smaller honest set if fewer than "
            f"{min_capsules} archetypes pass the evidence gates; do not broaden claims or "
            "substitute unvalidated rows."
        ),
        "params": {
            "min_capsules": min_capsules,
            "max_capsules": max_capsules,
            "allow_triage": allow_triage,
            "n_archetypes_requested": len(archetypes),
        },
        "inputs": {
            "candidate_manifest_schema": candidate_manifest.get("schema_version"),
            "candidate_manifest_issue": candidate_manifest.get("issue"),
            "candidate_manifest_sha256": canonical_sha256(candidate_manifest),
            "n_candidates": len(candidates),
            "causal_reports_provided": bool(causal_reports),
            "risk_reports_provided": bool(risk_reports),
        },
        "summary": {
            "n_capsules_requested": len(archetypes),
            "n_admitted": n_admitted,
            "n_unavailable": len(unavailable),
            "admitted_archetypes": [c["archetype"] for c in admitted],
            "unavailable_archetypes": [c["archetype"] for c in unavailable],
            "evidence_grades": {c["archetype"]: c["evidence_grade"] for c in admitted},
            "meets_min_capsules": n_admitted >= min_capsules,
        },
        "capsules": capsules,
    }


@dataclass
class CaseCapsuleValidation:
    """Result of validating a case-capsule manifest.

    Attributes:
        structural_violations: Hard problems that make the manifest malformed.
        author_pending: Expected-but-incomplete author fields (sentinels) that
            must be filled before dissertation integration; not a structural fail.
    """

    structural_violations: list[str] = field(default_factory=list)
    author_pending: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the manifest is structurally valid (author fields may pend).

        Returns:
            ``True`` when there are no structural violations.
        """
        return not self.structural_violations

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the validation.

        Returns:
            Mapping with ``ok``, ``structural_violations``, and ``author_pending``.
        """
        return {
            "ok": self.ok,
            "structural_violations": list(self.structural_violations),
            "author_pending": list(self.author_pending),
        }


def _check_keys(
    container: Any,
    keys: tuple[str, ...],
    *,
    cid: str,
    label: str,
    violations: list[str],
    pending: list[str],
) -> None:
    """Check that ``container`` is a dict carrying ``keys``; sort missing vs pending.

    Missing keys are structural violations; keys still set to :data:`AUTHOR_REQUIRED`
    are recorded as author-pending (expected, not a failure).
    """
    if not isinstance(container, dict):
        violations.append(f"{cid}: missing {label} block")
        return
    for key in keys:
        if key not in container:
            violations.append(f"{cid}: {label} missing {key!r}")
        elif container[key] == AUTHOR_REQUIRED:
            pending.append(f"{cid}: {label}.{key} author-required")


def _validate_capsule_figure(
    capsule: dict[str, Any],
    cid: str,
    violations: list[str],
    pending: list[str],
) -> None:
    """Validate an admitted capsule's figure spec (map/scale/marked-times/pairs)."""
    figure = capsule.get("figure_spec")
    if not isinstance(figure, dict):
        violations.append(f"{cid}: missing figure_spec")
        return
    required_flags = (
        "static_map_required",
        "metre_scale_required",
        "trajectory_sources_required",
        "actor_footprints_at_marked_times_required",
    )
    if not all(figure.get(flag) for flag in required_flags):
        violations.append(f"{cid}: figure_spec must require map/scale/trajectories/footprints")
    for key in ("marked_times", "why_this_time_matters"):
        if figure.get(key) == AUTHOR_REQUIRED:
            pending.append(f"{cid}: figure_spec.{key} author-required")
    if capsule.get("is_pair"):
        _check_keys(
            figure.get("shared_axis_spec"),
            SHARED_AXIS_KEYS,
            cid=cid,
            label="shared_axis_spec",
            violations=violations,
            pending=pending,
        )


def _validate_admitted_capsule(
    capsule: dict[str, Any],
    violations: list[str],
    pending: list[str],
) -> None:
    """Validate one admitted capsule, appending structural / author-pending notes."""
    cid = capsule.get("capsule_id", "<unknown>")

    if capsule.get("evidence_grade") not in EVIDENCE_GRADES:
        violations.append(f"{cid}: invalid evidence_grade {capsule.get('evidence_grade')!r}")

    _check_keys(
        capsule.get("narrative"),
        NARRATIVE_KEYS,
        cid=cid,
        label="narrative",
        violations=violations,
        pending=pending,
    )
    _validate_capsule_figure(capsule, cid, violations, pending)

    if capsule.get("evidence_grade") == "descriptive-risk":
        _check_keys(
            capsule.get("probability_panel"),
            PROBABILITY_PANEL_KEYS,
            cid=cid,
            label="probability_panel",
            violations=violations,
            pending=pending,
        )

    if (
        capsule.get("evidence_grade") == "causal"
        and capsule.get("causal_label") == "descriptive-only"
    ):
        violations.append(f"{cid}: causal grade but no validated causal_label")


def validate_ch7_case_capsule_manifest(manifest: Any) -> CaseCapsuleValidation:
    """Structurally validate a case-capsule manifest against the issue #5447 gates.

    Structural violations mark a malformed manifest. ``author_pending`` entries
    are the expected sentinels for narrative / figure fields a human writer must
    complete; they do not make the manifest invalid.

    Returns:
        A :class:`CaseCapsuleValidation` with the two lists and an ``ok`` flag.
    """
    violations: list[str] = []
    pending: list[str] = []

    if not isinstance(manifest, dict):
        violations.append(f"manifest must be a dict, got {type(manifest).__name__}")
        return CaseCapsuleValidation(violations, pending)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        violations.append(
            f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
            f"got {manifest.get('schema_version')!r}"
        )

    capsules = manifest.get("capsules")
    if not isinstance(capsules, list) or not capsules:
        violations.append("manifest has no capsules")
        return CaseCapsuleValidation(violations, pending)

    for capsule in capsules:
        if not isinstance(capsule, dict):
            violations.append("capsules list contains non-dict records")
            continue
        status = capsule.get("status")
        if status == "unavailable":
            if not capsule.get("reason"):
                violations.append(
                    f"{capsule.get('capsule_id', '<unknown>')}: unavailable capsule needs a reason"
                )
            continue
        if status != "admitted":
            violations.append(
                f"{capsule.get('capsule_id', '<unknown>')}: unknown capsule status {status!r}"
            )
            continue
        _validate_admitted_capsule(capsule, violations, pending)

    return CaseCapsuleValidation(violations, pending)
