"""Diagnostic inventory for SNQI per-term normalization (issue #3699).

:func:`robot_sf.benchmark.snqi.compute.compute_snqi` assembles the SNQI terms on
**inconsistent scales**: the time and comfort terms enter the composite *raw*
(unbounded), while the count-type penalty terms (collisions, near-misses,
force-exceed events, jerk) are *baseline-normalized* to ``[0, 1]`` via
:func:`robot_sf.benchmark.snqi.compute.normalize_metric`. Because the raw terms
are unbounded, their weight coefficients are not comparable to the
baseline-normalized terms' coefficients, and an un-normalized term can dominate
the composite's variance independently of its weight (see issue #3699).

This module is **diagnostic only**. It does not change the SNQI formula, the
weights, ``normalize_metric``, or any emitted score. It exposes the per-term
scaling regime as a machine-checkable inventory so the mixed-scale condition is
explicit and a preflight can stay aware of it. It deliberately does **not**
choose between the two remedies (normalize the raw terms vs. clip-and-document
the asymmetry) -- that choice is tracked as ``decision-required`` on issue #3699
and would change emitted SNQI values, which is out of scope here.

The term table below mirrors ``compute_snqi`` exactly; the cross-check test
:mod:`tests.benchmark.test_snqi_normalization_inventory` reconstructs the SNQI
score from this table and asserts it equals ``compute_snqi`` so the inventory
cannot silently drift away from the scoring code it describes.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_sf.benchmark.snqi.compute import normalize_metric

# Scaling regimes. Kept as plain strings so inventory payloads are trivially
# JSON-serializable for preflight reports.
SCALING_RAW = "raw"
SCALING_BASELINE_NORMALIZED = "baseline_normalized"

#: Lower/upper bound of a baseline-normalized term's post-scaling contribution,
#: matching the clamp in ``normalize_metric``.
NORMALIZED_LOWER = 0.0
NORMALIZED_UPPER = 1.0

SNQI_LEGACY_SCORE_VERSION = "SNQI-v0"
SNQI_LEGACY_SCORE_STATUS = "legacy_mixed_basis_diagnostic_only"

# Static SNQI score-version contract. All values are immutable scalars, so a
# shallow copy is sufficient to isolate callers from mutating the shared source.
_SNQI_VERSION_CONTRACT: dict[str, object] = {
    "schema_version": "snqi_score_version_contract.v1",
    "score_version": SNQI_LEGACY_SCORE_VERSION,
    "status": SNQI_LEGACY_SCORE_STATUS,
    "diagnostic_only": True,
    "mixed_basis_preserved": True,
    "score_semantics_changed": False,
    "decision_required_issue": 3699,
    "future_bounded_contract": "SNQI-v1",
    "policy": (
        "Historical SNQI-v0 keeps the mixed raw/baseline-normalized basis for "
        "reproducibility and must not be used as a primary safety ranking."
    ),
}


def build_snqi_version_contract() -> dict[str, object]:
    """Return the current versioned SNQI normalization contract.

    Issue #3699's current product decision preserves historical SNQI as a
    legacy diagnostic while deferring any bounded/recalibrated score to a
    separately versioned contract.

    Returns a fresh shallow copy so callers cannot mutate the shared constant.
    """

    return _SNQI_VERSION_CONTRACT.copy()


@dataclass(frozen=True)
class TermScaling:
    """Static description of one SNQI term's scaling regime.

    Mirrors a single term as assembled in
    :func:`robot_sf.benchmark.snqi.compute.compute_snqi`.

    Attributes:
        term: Short SNQI term name (e.g. ``"time"``).
        weight_name: Weight coefficient key (e.g. ``"w_time"``).
        metric_key: Episode-metric key consumed for this term.
        scaling: One of :data:`SCALING_RAW` or :data:`SCALING_BASELINE_NORMALIZED`.
        measurement_basis: Native unit or normalization basis for the post-scaling value.
        sign: ``+1`` for the reward term (success), ``-1`` for penalty terms.
        bounded: Whether this term's post-scaling contribution is bounded.
        default: Value substituted when ``metric_key`` is absent from metrics
            (matches the ``metrics.get(..., default)`` calls in ``compute_snqi``).
        note: Short human-readable caveat about the term's boundedness.
    """

    term: str
    weight_name: str
    metric_key: str
    scaling: str
    measurement_basis: str
    sign: int
    bounded: bool
    default: float
    note: str

    @property
    def is_penalty(self) -> bool:
        """Whether this term subtracts from the SNQI composite."""
        return self.sign < 0

    @property
    def normalization_status(self) -> str:
        """Compact diagnostic status for preflight reports."""
        if self.scaling == SCALING_BASELINE_NORMALIZED:
            return (
                "baseline_normalized_bounded" if self.bounded else "baseline_normalized_unbounded"
            )
        return "raw_bounded" if self.bounded else "raw_unbounded"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of this term."""
        return {
            "term": self.term,
            "weight_name": self.weight_name,
            "metric_key": self.metric_key,
            "scaling": self.scaling,
            "measurement_basis": self.measurement_basis,
            "normalization_status": self.normalization_status,
            "sign": self.sign,
            "is_penalty": self.is_penalty,
            "bounded": self.bounded,
            "default": self.default,
            "note": self.note,
        }


#: Canonical term inventory, ordered as assembled in ``compute_snqi``. The
#: ``success`` term is raw but bounded by construction (a 0/1 indicator); the
#: ``time`` and ``comfort`` penalty terms are raw *and unbounded*, which is the
#: defect issue #3699 documents. The remaining penalty terms are
#: baseline-normalized to ``[0, 1]``.
SNQI_TERM_SCALING: tuple[TermScaling, ...] = (
    TermScaling(
        term="success",
        weight_name="w_success",
        metric_key="success",
        scaling=SCALING_RAW,
        measurement_basis="raw 0/1 episode success indicator",
        sign=+1,
        bounded=True,
        default=0.0,
        note="0/1 success indicator; bounded by construction, not by clamping.",
    ),
    TermScaling(
        term="time",
        weight_name="w_time",
        metric_key="time_to_goal_norm",
        scaling=SCALING_RAW,
        measurement_basis="raw time-to-goal ratio",
        sign=-1,
        bounded=False,
        default=1.0,
        note="Raw, unbounded time ratio; not routed through normalize_metric.",
    ),
    TermScaling(
        term="collisions",
        weight_name="w_collisions",
        metric_key="collisions",
        scaling=SCALING_BASELINE_NORMALIZED,
        measurement_basis="baseline-relative median/p95 clamped value",
        sign=-1,
        bounded=True,
        default=0.0,
        note="Baseline-normalized (median/p95) and clamped to [0, 1].",
    ),
    TermScaling(
        term="near",
        weight_name="w_near",
        metric_key="near_misses",
        scaling=SCALING_BASELINE_NORMALIZED,
        measurement_basis="baseline-relative median/p95 clamped value",
        sign=-1,
        bounded=True,
        default=0.0,
        note="Baseline-normalized (median/p95) and clamped to [0, 1].",
    ),
    TermScaling(
        term="comfort",
        weight_name="w_comfort",
        metric_key="comfort_exposure",
        scaling=SCALING_RAW,
        measurement_basis="raw accumulated comfort-exposure value",
        sign=-1,
        bounded=False,
        default=0.0,
        note="Raw, unbounded exposure accumulation; not routed through normalize_metric.",
    ),
    TermScaling(
        term="force_exceed",
        weight_name="w_force_exceed",
        metric_key="force_exceed_events",
        scaling=SCALING_BASELINE_NORMALIZED,
        measurement_basis="baseline-relative median/p95 clamped value",
        sign=-1,
        bounded=True,
        default=0.0,
        note="Baseline-normalized (median/p95) and clamped to [0, 1].",
    ),
    TermScaling(
        term="jerk",
        weight_name="w_jerk",
        metric_key="jerk_mean",
        scaling=SCALING_BASELINE_NORMALIZED,
        measurement_basis="baseline-relative median/p95 clamped value",
        sign=-1,
        bounded=True,
        default=0.0,
        note="Baseline-normalized (median/p95) and clamped to [0, 1].",
    ),
)


def scaled_term_value(
    term: TermScaling,
    metrics: dict[str, float | int | bool],
    baseline_stats: dict[str, dict[str, float]],
) -> float:
    """Return the post-scaling value ``compute_snqi`` uses for one term.

    This reuses the canonical primitives (:func:`normalize_metric` for
    baseline-normalized terms; a raw ``float`` read for raw terms) so it cannot
    diverge from the scoring code. It does **not** apply the weight or sign.

    Args:
        term: The term to evaluate.
        metrics: Episode metrics mapping.
        baseline_stats: Baseline normalization statistics (median/p95).

    Returns:
        The scalar value this term contributes (pre-weight, pre-sign).
    """
    raw = metrics.get(term.metric_key, term.default)
    if term.scaling == SCALING_BASELINE_NORMALIZED:
        return normalize_metric(term.metric_key, raw, baseline_stats)
    # Raw regime: ``compute_snqi`` coerces success via an explicit bool check;
    # for every raw term a plain float() reproduces its behavior.
    if term.term == "success":
        return 1.0 if isinstance(raw, bool) and raw else float(raw)
    return float(raw)


@dataclass(frozen=True)
class TermContribution:
    """Weighted SNQI component contribution for one metric row.

    The signed value mirrors ``compute_snqi`` exactly: reward terms are positive
    and penalty terms are negative. ``absolute_share`` is diagnostic only and is
    normalized by the sum of absolute weighted contributions for the row.
    """

    term: TermScaling
    raw_value: float | int | bool
    scaled_value: float
    weight: float
    signed_contribution: float
    absolute_share: float

    @property
    def exceeds_weight_bound(self) -> bool:
        """Whether this term contributes more than its nominal weight magnitude."""
        return abs(self.signed_contribution) > abs(self.weight)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable contribution diagnostics."""
        return {
            "term": self.term.term,
            "weight_name": self.term.weight_name,
            "metric_key": self.term.metric_key,
            "scaling": self.term.scaling,
            "normalization_status": self.term.normalization_status,
            "raw_value": self.raw_value,
            "scaled_value": self.scaled_value,
            "weight": self.weight,
            "signed_contribution": self.signed_contribution,
            "absolute_share": self.absolute_share,
            "exceeds_weight_bound": self.exceeds_weight_bound,
            "bounded": self.term.bounded,
            "measurement_basis": self.term.measurement_basis,
        }


def _build_normalization_contract(
    contributions: list[TermContribution],
    *,
    raw_penalty_share: float,
    normalized_penalty_share: float,
    weight_bound_exceedances: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize whether weighted SNQI penalties are comparable as configured.

    Returns:
        JSON-serializable diagnostic contract status for this contribution row.
    """
    raw_penalties = [
        c for c in contributions if c.term.is_penalty and c.term.scaling == SCALING_RAW
    ]
    normalized_penalties = [
        c
        for c in contributions
        if c.term.is_penalty and c.term.scaling == SCALING_BASELINE_NORMALIZED
    ]
    raw_unbounded = [c for c in raw_penalties if not c.term.bounded]

    comparable = not raw_unbounded and bool(normalized_penalties)
    status = "comparable_weight_basis" if comparable else "mixed_unbounded_penalty_basis"

    reasons: list[str] = []
    if raw_unbounded:
        reasons.append(
            "Raw unbounded penalty terms bypass normalize_metric, so their weights are not "
            "directly comparable with baseline-normalized penalty weights."
        )
    if raw_penalty_share > normalized_penalty_share:
        reasons.append(
            "Raw penalty terms contribute a larger absolute share than baseline-normalized "
            "penalty terms for this diagnostic row."
        )
    if weight_bound_exceedances:
        reasons.append("At least one weighted contribution exceeds its nominal weight magnitude.")

    return {
        "schema_version": "snqi_normalization_contract.v1",
        "diagnostic_only": True,
        "status": status,
        "weights_comparable": comparable,
        "raw_unbounded_penalty_terms": [c.term.term for c in raw_unbounded],
        "baseline_normalized_penalty_terms": [c.term.term for c in normalized_penalties],
        "raw_penalty_absolute_share": raw_penalty_share,
        "baseline_normalized_penalty_absolute_share": normalized_penalty_share,
        "weight_bound_exceedance_terms": [str(entry["term"]) for entry in weight_bound_exceedances],
        "decision_required_issue": 3699 if not comparable else None,
        "reasons": reasons,
    }


def build_snqi_contribution_diagnostics(
    metrics: dict[str, float | int | bool],
    weights: dict[str, float],
    baseline_stats: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Build read-only weighted contribution diagnostics for one SNQI row.

    This checker intentionally does not normalize the raw terms or alter any
    score. It exposes how much each mixed-basis term contributes under the
    current formula so issue #3699 can be reviewed without silently promoting a
    normalization redesign.

    Returns:
        JSON-serializable diagnostic payload with per-term contributions and
        aggregate raw-vs-baseline-normalized penalty shares.
    """
    preliminary: list[tuple[TermScaling, float | int | bool, float, float, float]] = []
    absolute_total = 0.0
    for term in SNQI_TERM_SCALING:
        raw_value = metrics.get(term.metric_key, term.default)
        scaled_value = scaled_term_value(term, metrics, baseline_stats)
        weight = float(weights.get(term.weight_name, 1.0))
        signed_contribution = term.sign * weight * scaled_value
        absolute_total += abs(signed_contribution)
        preliminary.append((term, raw_value, scaled_value, weight, signed_contribution))

    contributions = [
        TermContribution(
            term=term,
            raw_value=raw_value,
            scaled_value=scaled_value,
            weight=weight,
            signed_contribution=signed_contribution,
            absolute_share=(
                abs(signed_contribution) / absolute_total if absolute_total > 0.0 else 0.0
            ),
        )
        for term, raw_value, scaled_value, weight, signed_contribution in preliminary
    ]
    raw_penalty_share = sum(
        c.absolute_share
        for c in contributions
        if c.term.is_penalty and c.term.scaling == SCALING_RAW
    )
    normalized_penalty_share = sum(
        c.absolute_share
        for c in contributions
        if c.term.is_penalty and c.term.scaling == SCALING_BASELINE_NORMALIZED
    )
    weight_bound_exceedances = [
        contribution.to_dict()
        for contribution in contributions
        if contribution.term.is_penalty and contribution.exceeds_weight_bound
    ]
    normalization_contract = _build_normalization_contract(
        contributions,
        raw_penalty_share=raw_penalty_share,
        normalized_penalty_share=normalized_penalty_share,
        weight_bound_exceedances=weight_bound_exceedances,
    )
    return {
        "schema_version": "snqi_normalization_contributions.v1",
        "diagnostic_only": True,
        "mixed_basis": bool(raw_penalty_share and normalized_penalty_share),
        "absolute_contribution_total": absolute_total,
        "raw_penalty_absolute_share": raw_penalty_share,
        "baseline_normalized_penalty_absolute_share": normalized_penalty_share,
        "raw_penalty_terms_dominate": raw_penalty_share > normalized_penalty_share,
        "weight_bound_exceedances": weight_bound_exceedances,
        "has_weight_bound_exceedance": bool(weight_bound_exceedances),
        "normalization_contract": normalization_contract,
        "score_version_contract": build_snqi_version_contract(),
        "terms": [contribution.to_dict() for contribution in contributions],
    }


@dataclass(frozen=True)
class NormalizationInventory:
    """Structured inventory of SNQI per-term normalization status.

    Attributes:
        terms: Per-term scaling descriptors (mirrors ``compute_snqi``).
        baseline_covered: Map of metric key -> whether ``baseline_stats``
            provides median/p95 for it. Only meaningful for
            baseline-normalized terms.
    """

    terms: tuple[TermScaling, ...]
    baseline_covered: dict[str, bool]

    @property
    def penalty_terms(self) -> list[TermScaling]:
        """Penalty terms (everything subtracted from the composite)."""
        return [t for t in self.terms if t.is_penalty]

    @property
    def unbounded_terms(self) -> list[TermScaling]:
        """Terms whose post-scaling contribution is unbounded."""
        return [t for t in self.terms if not t.bounded]

    @property
    def raw_penalty_terms(self) -> list[TermScaling]:
        """Raw (un-normalized) penalty terms -- the mixed-scale offenders."""
        return [t for t in self.terms if t.is_penalty and t.scaling == SCALING_RAW]

    @property
    def normalized_penalty_terms(self) -> list[TermScaling]:
        """Baseline-normalized penalty terms."""
        return [t for t in self.terms if t.is_penalty and t.scaling == SCALING_BASELINE_NORMALIZED]

    @property
    def mixed_scale(self) -> bool:
        """Whether penalty terms span both raw and baseline-normalized scales.

        When True, penalty-term weight coefficients are not directly comparable
        as relative priorities (the issue #3699 defect).
        """
        return bool(self.raw_penalty_terms) and bool(self.normalized_penalty_terms)

    @property
    def missing_baseline_coverage(self) -> list[TermScaling]:
        """Baseline-normalized terms with no median/p95 in ``baseline_stats``.

        ``normalize_metric`` silently returns ``0.0`` for these, so the term
        contributes nothing regardless of its raw value -- a latent footgun.
        """
        return [
            t
            for t in self.normalized_penalty_terms
            if not self.baseline_covered.get(t.metric_key, False)
        ]

    @property
    def is_consistent(self) -> bool:
        """Whether all penalty terms enter on a comparable, bounded basis."""
        return not self.mixed_scale and not self.unbounded_terms

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view for preflight reporting."""
        return {
            "terms": [t.to_dict() for t in self.terms],
            "baseline_covered": dict(self.baseline_covered),
            "mixed_scale": self.mixed_scale,
            "raw_penalty_terms": [t.term for t in self.raw_penalty_terms],
            "normalized_penalty_terms": [t.term for t in self.normalized_penalty_terms],
            "unbounded_terms": [t.term for t in self.unbounded_terms],
            "missing_baseline_coverage": [t.metric_key for t in self.missing_baseline_coverage],
            "is_consistent": self.is_consistent,
            "score_version_contract": build_snqi_version_contract(),
        }


def build_snqi_normalization_inventory(
    baseline_stats: dict[str, dict[str, float]] | None = None,
) -> NormalizationInventory:
    """Inventory the per-term scaling status of the SNQI composite.

    Args:
        baseline_stats: Optional baseline normalization statistics. When given,
            each baseline-normalized term is checked for median/p95 coverage so
            silently-zeroed terms can be surfaced. When ``None``, coverage is
            reported as unknown (``False``) for every normalized term.

    Returns:
        A :class:`NormalizationInventory` describing the scaling of every term.
    """
    stats = baseline_stats or {}
    covered: dict[str, bool] = {}
    for term in SNQI_TERM_SCALING:
        if term.scaling == SCALING_BASELINE_NORMALIZED:
            entry = stats.get(term.metric_key)
            covered[term.metric_key] = isinstance(entry, dict) and "med" in entry and "p95" in entry
    return NormalizationInventory(terms=SNQI_TERM_SCALING, baseline_covered=covered)


def format_normalization_report(inventory: NormalizationInventory) -> str:
    """Render a compact human-readable normalization summary.

    Returns:
        A multi-line string suitable for preflight / CLI output.
    """
    lines = ["SNQI per-term normalization inventory (issue #3699)"]
    header = f"  {'term':<13}{'scaling':<22}{'bounded':<9}{'metric_key':<24}basis"
    lines.append(header)
    for term in inventory.terms:
        bounded = "yes" if term.bounded else "NO"
        lines.append(
            f"  {term.term:<13}{term.scaling:<22}{bounded:<9}"
            f"{term.metric_key:<24}{term.measurement_basis}"
        )
    lines.append(f"  mixed_scale            : {inventory.mixed_scale}")
    contract = build_snqi_version_contract()
    lines.append(f"  score version          : {contract['score_version']} ({contract['status']})")
    lines.append(
        "  raw penalty terms      : "
        + (", ".join(t.term for t in inventory.raw_penalty_terms) or "<none>")
    )
    lines.append(
        "  unbounded terms        : "
        + (", ".join(t.term for t in inventory.unbounded_terms) or "<none>")
    )
    missing = inventory.missing_baseline_coverage
    lines.append(
        "  missing baseline cover : " + (", ".join(t.metric_key for t in missing) or "<none>")
    )
    lines.append(f"  consistent basis       : {inventory.is_consistent}")
    return "\n".join(lines)


__all__ = [
    "NORMALIZED_LOWER",
    "NORMALIZED_UPPER",
    "SCALING_BASELINE_NORMALIZED",
    "SCALING_RAW",
    "SNQI_LEGACY_SCORE_STATUS",
    "SNQI_LEGACY_SCORE_VERSION",
    "SNQI_TERM_SCALING",
    "NormalizationInventory",
    "TermContribution",
    "TermScaling",
    "build_snqi_contribution_diagnostics",
    "build_snqi_normalization_inventory",
    "build_snqi_version_contract",
    "format_normalization_report",
    "scaled_term_value",
]
