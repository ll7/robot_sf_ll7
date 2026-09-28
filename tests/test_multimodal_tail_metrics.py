"""Exact finite-distribution reference checks for optional arbitrator tails (#9853)."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from robot_sf.planner.multimodal_trajectory_arbitrator import (
    discrete_tail_metrics,
    discrete_upper_tail_cvar,
)


def _fraction_reference(
    losses: list[float], probabilities: list[float], alpha: float
) -> tuple[float, float, float, float]:
    """Use exact arithmetic on the binary floats passed to the production API."""
    weights = [Fraction.from_float(value) for value in probabilities]
    total = sum(weights)
    weights = [weight / total for weight in weights]
    atoms = [
        (Fraction.from_float(loss), weight) for loss, weight in zip(losses, weights, strict=True)
    ]
    confidence = Fraction.from_float(alpha)
    expected = sum(loss * weight for loss, weight in atoms)

    cumulative = Fraction(0)
    var = atoms[-1][0]
    for loss, weight in sorted(atoms):
        cumulative += weight
        if cumulative >= confidence:
            var = loss
            break

    tail_mass = 1 - confidence
    remaining = tail_mass
    tail_loss = Fraction(0)
    for loss, weight in sorted(atoms, reverse=True):
        taken = min(weight, remaining)
        tail_loss += loss * taken
        remaining -= taken
        if remaining == 0:
            break
    assert remaining == 0
    return float(expected), float(var), float(tail_loss / tail_mass), max(losses)


@pytest.mark.parametrize(
    ("losses", "probabilities", "alpha"),
    (
        ([1.0, 0.5], [0.0, 1.0], 0.9999999999),
        ([1.0, 0.5], [1.0e-12, 0.999999999999], 0.9999999999),
        ([0.0, 0.5, 1.0], [0.0, 1.0, 0.0], math.nextafter(1.0, 0.0)),
    ),
)
def test_near_one_tail_consumes_full_mass(
    losses: list[float], probabilities: list[float], alpha: float
) -> None:
    reference = _fraction_reference(losses, probabilities, alpha)
    observed = discrete_tail_metrics(losses, probabilities, alpha)
    scale = max(abs(loss) for loss in losses)
    assert observed[1] == reference[1]
    assert abs(observed[2] - reference[2]) <= 1.0e-9 * scale
    assert discrete_upper_tail_cvar(losses, probabilities, alpha) == observed[2]


@pytest.mark.parametrize(("alpha", "expected_cvar"), ((0.90, 0.60), (0.95, 1.0)))
def test_ordinary_confidence_results_are_preserved(alpha: float, expected_cvar: float) -> None:
    observed = discrete_upper_tail_cvar([0.0, 0.2, 1.0], [0.85, 0.10, 0.05], alpha)
    assert observed == pytest.approx(expected_cvar, abs=1.0e-12)


def test_random_finite_distributions_match_exact_reference_under_permutation() -> None:
    rng = random.Random(9853)
    alphas = (1.0e-12, 0.01, 0.9, 0.95, 0.9999999999, math.nextafter(1.0, 0.0))
    for _ in range(40):
        count = rng.randint(2, 7)
        losses = [float(rng.choice((-2, -1, 0, 0, 1, 2))) for _ in range(count)]
        if not any(losses):
            losses[0] = 1.0
        raw = [rng.randrange(1, 20) for _ in range(count)]
        raw[rng.randrange(count)] = 0
        if rng.random() < 0.5:
            raw[rng.randrange(count)] = 1.0e-12
        total = sum(raw)
        probabilities = [value / total for value in raw]
        for alpha in alphas:
            reference = _fraction_reference(losses, probabilities, alpha)
            observed = discrete_tail_metrics(losses, probabilities, alpha)
            scale = max(abs(loss) for loss in losses)
            for actual, exact in zip(observed, reference, strict=True):
                assert abs(actual - exact) <= 1.0e-9 * scale

            order = list(range(count))
            rng.shuffle(order)
            shuffled = discrete_tail_metrics(
                [losses[index] for index in order],
                [probabilities[index] for index in order],
                alpha,
            )
            for actual, original in zip(shuffled, observed, strict=True):
                assert abs(actual - original) <= 1.0e-9 * scale
