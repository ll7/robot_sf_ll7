"""Generalize the uncertainty-drop safety effect across uncertainty sources (issue #3557).

#3471 found, using a single uncertainty source (existence-degradation), that dropping uncertain
agents is unsafe at the default gate. `ScenarioBelief` supports other sources too — visibility /
occlusion, covariance inflation, class-probability ambiguity, tracking noise. This module is the
pure **decision layer** that turns a per-source retained-vs-dropped safety contrast (run via the
#3471 harness parameterized by source) into the issue's deliverable: a per-source verdict plus
whether the unsafe-dropping finding **generalizes** across sources or is specific to
existence-degradation.

The parameterized harness runs are deferred (they need benchmark execution + the #3450 condition
builders); this layer is pure and side-effect free, mirroring the accepted decision layers in
``robot_sf/scenario_certification/failure_cause.py`` (#3484) and
``robot_sf/planner/stream_gap_gate_calibration.py`` (#3558).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_sf.benchmark.finite_checks import require_finite_fields

UNCERTAINTY_SOURCE_GENERALIZATION_SCHEMA = "uncertainty_source_generalization.v1"

_CONTRAST_METRIC_FIELDS = (
    "retained_unsafe_commit_rate",
    "dropped_unsafe_commit_rate",
    "min_separation_delta_m",
)

REPRODUCES = "reproduces_unsafe_dropping"
NO_EFFECT = "no_unsafe_dropping_effect"
INCONCLUSIVE = "inconclusive"

GENERALIZES = "generalizes"
SOURCE_SPECIFIC = "source_specific"
DOES_NOT_GENERALIZE = "does_not_generalize"


@dataclass(frozen=True, slots=True)
class SourceContrast:
    """Retained-vs-dropped safety contrast for one uncertainty source.

    Attributes:
        source: Uncertainty-source name (e.g. ``existence``, ``occlusion``, ``covariance``).
        retained_unsafe_commit_rate: Unsafe-commit rate under conservative retention.
        dropped_unsafe_commit_rate: Unsafe-commit rate when dropping uncertain agents.
        min_separation_delta_m: ``dropped − retained`` minimum separation (negative = dropping worse).
        n_episodes: Episodes per arm (used only to flag thin evidence).
    """

    source: str
    retained_unsafe_commit_rate: float
    dropped_unsafe_commit_rate: float
    min_separation_delta_m: float
    n_episodes: int


@dataclass(frozen=True, slots=True)
class EffectThresholds:
    """Predeclared effect-size thresholds for classifying a source contrast."""

    min_detectable_unsafe_delta: float = 0.02
    min_detectable_separation_delta_m: float = 0.05
    min_episodes: int = 1


def classify_source(contrast: SourceContrast, thresholds: EffectThresholds | None = None) -> str:
    """Classify whether a source reproduces the unsafe-dropping effect.

    Dropping is "unsafe" when it raises unsafe-commit or lowers separation beyond the detectable
    threshold; a difference below the threshold (or too-few episodes) is ``inconclusive``.

    Returns:
        str: ``reproduces_unsafe_dropping`` / ``no_unsafe_dropping_effect`` / ``inconclusive``.
    """
    require_finite_fields(
        f"contrast for source {contrast.source!r}", contrast, _CONTRAST_METRIC_FIELDS
    )
    thresholds = thresholds or EffectThresholds()
    unsafe_delta = contrast.dropped_unsafe_commit_rate - contrast.retained_unsafe_commit_rate
    sep_delta = contrast.min_separation_delta_m

    detectable = (
        abs(unsafe_delta) >= thresholds.min_detectable_unsafe_delta
        or abs(sep_delta) >= thresholds.min_detectable_separation_delta_m
    )
    if contrast.n_episodes < thresholds.min_episodes or not detectable:
        return INCONCLUSIVE
    dropping_worse = (
        unsafe_delta >= thresholds.min_detectable_unsafe_delta
        or sep_delta <= -thresholds.min_detectable_separation_delta_m
    )
    return REPRODUCES if dropping_worse else NO_EFFECT


def assess_source_generalization(
    contrasts: list[SourceContrast],
    thresholds: EffectThresholds | None = None,
) -> dict[str, Any]:
    """Assess whether the unsafe-dropping effect generalizes across uncertainty sources.

    Returns:
        dict[str, Any]: Versioned report with per-source verdicts and a generalization verdict —
        ``generalizes`` (all measurable sources reproduce), ``does_not_generalize`` (none do), or
        ``source_specific`` (mixed). Inconclusive sources are excluded from the generalization call.
    """
    if not contrasts:
        raise ValueError("at least one source contrast is required")
    thresholds = thresholds or EffectThresholds()
    per_source = [
        {
            "source": contrast.source,
            "unsafe_commit_delta": (
                contrast.dropped_unsafe_commit_rate - contrast.retained_unsafe_commit_rate
            ),
            "min_separation_delta_m": contrast.min_separation_delta_m,
            "n_episodes": contrast.n_episodes,
            "verdict": classify_source(contrast, thresholds),
        }
        for contrast in contrasts
    ]
    measurable = [row for row in per_source if row["verdict"] != INCONCLUSIVE]
    reproduces = [row for row in measurable if row["verdict"] == REPRODUCES]

    if not measurable:
        generalization = INCONCLUSIVE
    elif len(reproduces) == len(measurable):
        generalization = GENERALIZES
    elif not reproduces:
        generalization = DOES_NOT_GENERALIZE
    else:
        generalization = SOURCE_SPECIFIC

    return {
        "schema_version": UNCERTAINTY_SOURCE_GENERALIZATION_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "n_sources": len(contrasts),
        "per_source": per_source,
        "reproducing_sources": sorted({row["source"] for row in reproduces}),
        "inconclusive_sources": sorted(
            {row["source"] for row in per_source if row["verdict"] == INCONCLUSIVE}
        ),
        "generalization": generalization,
    }


__all__ = [
    "DOES_NOT_GENERALIZE",
    "GENERALIZES",
    "INCONCLUSIVE",
    "NO_EFFECT",
    "REPRODUCES",
    "SOURCE_SPECIFIC",
    "UNCERTAINTY_SOURCE_GENERALIZATION_SCHEMA",
    "EffectThresholds",
    "SourceContrast",
    "assess_source_generalization",
    "classify_source",
]
