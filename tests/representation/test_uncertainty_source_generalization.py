"""Tests for the uncertainty-source generalization decision layer (issue #3557)."""

from __future__ import annotations

import pytest

from robot_sf.representation.uncertainty_source_generalization import (
    DOES_NOT_GENERALIZE,
    GENERALIZES,
    INCONCLUSIVE,
    NO_EFFECT,
    REPRODUCES,
    SOURCE_SPECIFIC,
    UNCERTAINTY_SOURCE_GENERALIZATION_SCHEMA,
    EffectThresholds,
    SourceContrast,
    assess_source_generalization,
    classify_source,
)


def _contrast(source: str, retained: float, dropped: float, sep_delta: float, n: int = 50):
    """Build a per-source retained-vs-dropped safety contrast."""
    return SourceContrast(
        source=source,
        retained_unsafe_commit_rate=retained,
        dropped_unsafe_commit_rate=dropped,
        min_separation_delta_m=sep_delta,
        n_episodes=n,
    )


def test_dropping_worse_reproduces_effect() -> None:
    """A clear rise in unsafe commitment when dropping must reproduce the effect."""
    assert classify_source(_contrast("existence", 0.10, 0.25, -0.20)) == REPRODUCES


def test_dropping_at_least_as_safe_is_no_effect() -> None:
    """When dropping is at least as safe, the effect must not reproduce."""
    assert classify_source(_contrast("occlusion", 0.20, 0.18, 0.10)) == NO_EFFECT


def test_tiny_difference_is_inconclusive() -> None:
    """A difference below the detectable threshold must be inconclusive."""
    assert classify_source(_contrast("covariance", 0.10, 0.105, 0.01)) == INCONCLUSIVE


def test_thin_evidence_is_inconclusive() -> None:
    """Too-few episodes must be inconclusive regardless of the measured delta."""
    thresholds = EffectThresholds(min_episodes=30)
    assert classify_source(_contrast("class_prob", 0.1, 0.3, -0.2, n=5), thresholds) == INCONCLUSIVE


def test_effect_generalizes_when_all_measurable_sources_reproduce() -> None:
    """If every measurable source reproduces, the finding generalizes."""
    report = assess_source_generalization(
        [
            _contrast("existence", 0.10, 0.25, -0.20),
            _contrast("occlusion", 0.12, 0.22, -0.10),
            _contrast("covariance", 0.10, 0.105, 0.0),  # inconclusive, excluded
        ]
    )

    assert report["schema_version"] == UNCERTAINTY_SOURCE_GENERALIZATION_SCHEMA
    assert report["generalization"] == GENERALIZES
    assert set(report["reproducing_sources"]) == {"existence", "occlusion"}
    assert report["inconclusive_sources"] == ["covariance"]


def test_effect_is_source_specific_when_mixed() -> None:
    """When some sources reproduce and others do not, the effect is source-specific."""
    report = assess_source_generalization(
        [
            _contrast("existence", 0.10, 0.25, -0.20),  # reproduces
            _contrast("tracking_noise", 0.20, 0.18, 0.10),  # no effect
        ]
    )

    assert report["generalization"] == SOURCE_SPECIFIC


def test_effect_does_not_generalize_when_none_reproduce() -> None:
    """When no measurable source reproduces, the effect does not generalize."""
    report = assess_source_generalization(
        [
            _contrast("occlusion", 0.20, 0.18, 0.10),
            _contrast("tracking_noise", 0.15, 0.14, 0.08),
        ]
    )

    assert report["generalization"] == DOES_NOT_GENERALIZE


def test_all_inconclusive_yields_inconclusive_generalization() -> None:
    """If no source is measurable, the generalization verdict is inconclusive."""
    report = assess_source_generalization([_contrast("covariance", 0.10, 0.105, 0.0)])

    assert report["generalization"] == INCONCLUSIVE


def test_empty_input_is_rejected() -> None:
    """An empty set of source contrasts cannot be assessed."""
    with pytest.raises(ValueError):
        assess_source_generalization([])
