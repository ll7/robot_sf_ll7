"""Shared deterministic rank and rank-correlation helpers.

The helpers centralize the benchmark tooling convention used for Spearman and
Kendall comparisons: average ranks for tied metric values, deterministic key
ordering for reported rank lists, and caller-selected degenerate-input results
so legacy analysis surfaces keep their documented behavior.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence


def rank_by(
    values_by_key: Mapping[Hashable, float],
    *,
    higher_is_better: bool,
    tie_abs_tol: float = 0.0,
) -> dict[Hashable, float]:
    """Return one-based average ranks for keyed metric values.

    Rank ``1.0`` is the best value. Ties are averaged using exact equality by
    default; pass ``tie_abs_tol`` for legacy helpers that treated near-equal
    floating-point values as ties.
    """
    ordered = sorted(
        values_by_key.items(),
        key=lambda item: (-float(item[1]) if higher_is_better else float(item[1]), str(item[0])),
    )
    ranks: dict[Hashable, float] = {}
    index = 0
    while index < len(ordered):
        value = float(ordered[index][1])
        tie_end = index + 1
        while tie_end < len(ordered) and _tied(
            float(ordered[tie_end][1]), value, tie_abs_tol=tie_abs_tol
        ):
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for tied_key, _value in ordered[index:tie_end]:
            ranks[tied_key] = average_rank
        index = tie_end
    return ranks


def rank_order(
    values_by_key: Mapping[Hashable, float],
    *,
    higher_is_better: bool,
) -> list[Hashable]:
    """Return keys ordered from best to worst by metric value."""
    return [
        key
        for key, _value in sorted(
            values_by_key.items(),
            key=lambda item: (
                -float(item[1]) if higher_is_better else float(item[1]),
                str(item[0]),
            ),
        )
    ]


def top_tied(
    values_by_key: Mapping[Hashable, float],
    *,
    higher_is_better: bool,
    tie_abs_tol: float = 0.0,
) -> bool:
    """Return whether more than one key shares the best metric value."""
    if len(values_by_key) < 2:
        return False
    ordered = rank_order(values_by_key, higher_is_better=higher_is_better)
    best_value = float(values_by_key[ordered[0]])
    return (
        sum(
            _tied(float(value), best_value, tie_abs_tol=tie_abs_tol)
            for value in values_by_key.values()
        )
        > 1
    )


def spearman(
    left: Sequence[float],
    right: Sequence[float],
    *,
    degenerate: float | None = 0.0,
    tie_abs_tol: float = 0.0,
) -> float | None:
    """Return Spearman rank correlation for paired numeric sequences.

    Values are ranked ascending with average ranks for ties. ``degenerate`` is
    returned when the vectors are unequal length, contain fewer than two values,
    or one ranked vector has zero variance.
    """
    if len(left) != len(right) or len(left) < 2:
        return degenerate
    left_ranks = _rank_sequence(left, tie_abs_tol=tie_abs_tol)
    right_ranks = _rank_sequence(right, tie_abs_tol=tie_abs_tol)
    return _pearson(left_ranks, right_ranks, degenerate=degenerate)


def spearman_by_value(
    left_values: Mapping[Hashable, float],
    right_values: Mapping[Hashable, float],
    *,
    higher_is_better: bool,
    degenerate: float | None = None,
    tie_abs_tol: float = 0.0,
) -> float | None:
    """Return Spearman correlation over common keyed metric values."""
    common = sorted(set(left_values) & set(right_values), key=str)
    if len(common) < 2:
        return degenerate
    left_ranks = rank_by(
        {key: left_values[key] for key in common},
        higher_is_better=higher_is_better,
        tie_abs_tol=tie_abs_tol,
    )
    right_ranks = rank_by(
        {key: right_values[key] for key in common},
        higher_is_better=higher_is_better,
        tie_abs_tol=tie_abs_tol,
    )
    return _pearson(
        [left_ranks[key] for key in common],
        [right_ranks[key] for key in common],
        degenerate=degenerate,
    )


def spearman_from_order(
    order_a: Sequence[Hashable],
    order_b: Sequence[Hashable],
    *,
    degenerate: float | None = 0.0,
) -> float | None:
    """Return Spearman correlation between two explicit orderings."""
    if len(order_a) < 2 or set(order_a) != set(order_b):
        return degenerate
    index_b = {key: index for index, key in enumerate(order_b)}
    return spearman(
        list(range(len(order_a))),
        [index_b[key] for key in order_a],
        degenerate=degenerate,
    )


def spearman_from_rank_maps(
    left_ranks: Mapping[Hashable, float],
    right_ranks: Mapping[Hashable, float],
    *,
    degenerate: float | None = None,
) -> float | None:
    """Return Spearman rho from caller-provided rank positions.

    This preserves legacy callers that compare absolute rank positions over the
    overlapping keys instead of re-ranking the overlap.
    """
    common = sorted(set(left_ranks) & set(right_ranks), key=str)
    n = len(common)
    if n < 2:
        return degenerate
    d_squared = math.fsum((float(left_ranks[key]) - float(right_ranks[key])) ** 2 for key in common)
    return float(1.0 - ((6.0 * d_squared) / (n * (n * n - 1))))


def kendall_tau(
    order_a: Sequence[Hashable],
    order_b: Sequence[Hashable],
    *,
    degenerate: float | None = 0.0,
) -> float | None:
    """Return Kendall tau-a correlation between two explicit orderings."""
    if len(order_a) < 2 or set(order_a) != set(order_b):
        return degenerate
    index_a = {key: index for index, key in enumerate(order_a)}
    index_b = {key: index for index, key in enumerate(order_b)}
    concordant = 0
    discordant = 0
    for left_index, left in enumerate(order_a):
        for right in order_a[left_index + 1 :]:
            product = (index_a[left] - index_a[right]) * (index_b[left] - index_b[right])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return degenerate
    return float((concordant - discordant) / total)


def kendall_tau_by_value(
    left_values: Mapping[Hashable, float],
    right_values: Mapping[Hashable, float],
    *,
    higher_is_better: bool,
    degenerate: float | None = None,
    tie_abs_tol: float = 1e-12,
) -> float | None:
    """Return Kendall tau over common keyed metric values, skipping ties."""
    common = sorted(set(left_values) & set(right_values), key=str)
    if len(common) < 2:
        return degenerate
    concordant = 0
    discordant = 0
    multiplier = 1.0 if higher_is_better else -1.0
    for index, left in enumerate(common):
        for right in common[index + 1 :]:
            left_delta = multiplier * (float(left_values[left]) - float(left_values[right]))
            right_delta = multiplier * (float(right_values[left]) - float(right_values[right]))
            if abs(left_delta) <= tie_abs_tol or abs(right_delta) <= tie_abs_tol:
                continue
            if left_delta * right_delta > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return degenerate
    return float((concordant - discordant) / total)


def _rank_sequence(values: Sequence[float], *, tie_abs_tol: float) -> list[float]:
    """Return one-based average ranks in the original sequence order."""
    indexed = sorted(enumerate(values), key=lambda item: float(item[1]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        value = float(indexed[index][1])
        tie_end = index + 1
        while tie_end < len(indexed) and _tied(
            float(indexed[tie_end][1]), value, tie_abs_tol=tie_abs_tol
        ):
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for original_index, _value in indexed[index:tie_end]:
            ranks[original_index] = average_rank
        index = tie_end
    return ranks


def _pearson(
    left: Sequence[float],
    right: Sequence[float],
    *,
    degenerate: float | None,
) -> float | None:
    """Return Pearson correlation, or the caller-selected degenerate value."""
    if len(left) != len(right) or len(left) < 2:
        return degenerate
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    )
    left_norm = math.sqrt(math.fsum(value * value for value in left_centered))
    right_norm = math.sqrt(math.fsum(value * value for value in right_centered))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return degenerate
    return float(numerator / (left_norm * right_norm))


def _tied(left: float, right: float, *, tie_abs_tol: float) -> bool:
    """Return whether two values are tied under the requested tolerance."""
    if tie_abs_tol <= 0.0:
        return left == right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tie_abs_tol)
