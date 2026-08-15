"""Shared representative-run selection rule for multi-seed phenomenon campaigns.

This module is the single source of truth for the rule that decides *which*
seed of a multi-seed campaign cell is shown when a single run has to stand in
for the cell: figures, replay videos, and trace dossiers all pick their
exhibit with the same deterministic procedure.

The rule exists to make single-run exhibits non-cherry-picked:

1. restrict to the seeds whose per-seed verdict equals the cell's **majority
   verdict**, resolving a tied count toward the **weaker** label;
2. within that pool, take the seed with the **median primary order
   parameter** (the lower of the two middles for an even pool);
3. break an exact numeric tie toward the **lower seed**.

Before this module existed the rule was implemented twice -- once in
``robot_sf.research.emergent_phenomena_campaign`` (for the reported majority
verdict) and once in ``scripts/validation/render_issue_5149_emergent_phenomena_videos.py``
(for the rendered replay seed) -- along with two verbatim copies of the verdict
severity ordering and of the per-scenario primary-order-parameter map. A
guarantee that is stated twice can drift; this module states it once.

Relationship to :mod:`robot_sf.benchmark.trace_dossier_selection`
----------------------------------------------------------------
``trace_dossier_selection.select_representative`` implements the
``trace_dossier_selector.v1`` contract over a *caller-supplied numeric*
``label_strength``; :func:`verdict_label_strength` here is the mapping that
turns this campaign's verdict vocabulary into that number. The two selectors
agree for odd-sized pools but deliberately differ on an even-sized pool: the
campaign rule takes the lower of the two middle runs by rank, while the
``v1`` selector takes whichever middle run is closest to the interpolated
median value and breaks the resulting tie on seed *identity*. Reconciling the
two is a domain decision, not a refactor, and is tracked separately -- do not
"fix" one to match the other here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "PRIMARY_ORDER_PARAMETER",
    "VERDICT_SEVERITY",
    "RepresentativeCandidate",
    "majority_verdict",
    "primary_order_parameter",
    "select_representative_index",
    "verdict_label_strength",
]

# Verdict labels ordered weakest-first. Aggregation and exhibit selection both
# tie-break toward the weaker label so neither ever overclaims on a split seed
# population.
VERDICT_SEVERITY: tuple[str, ...] = (
    "absent_or_negligible",
    "weak_partial",
    "clearly_present",
)

# The order parameter that defines "median run" for each canonical scenario.
PRIMARY_ORDER_PARAMETER: Mapping[str, str] = MappingProxyType(
    {
        "bidirectional_corridor": "lane_segregation_index",
        "narrow_doorway": "oscillation_flips",
        "high_density_exit": "exit_density_ratio",
    }
)


@dataclass(frozen=True, slots=True)
class RepresentativeCandidate:
    """One campaign run reduced to the three fields the selection rule reads.

    Attributes:
        verdict: The run's per-seed phenomenon verdict label.
        primary_order: The scenario's primary order parameter for this run.
        seed: The run's seed, used as the final deterministic tie-break.
    """

    verdict: str
    primary_order: float
    seed: int


def verdict_label_strength(label: str, severity_order: Sequence[str] = VERDICT_SEVERITY) -> int:
    """Rank one verdict label weakest-first.

    Args:
        label: The verdict label to rank.
        severity_order: Verdict labels ordered weakest-first.

    Returns:
        The label's index in ``severity_order``; an unrecognized label ranks
        after every known one, so an unknown label never wins a
        "weaker label" tie-break.
    """
    try:
        return severity_order.index(label)
    except ValueError:
        return len(severity_order)


def primary_order_parameter(scenario: str) -> str:
    """Return the order parameter that defines the median run for one scenario.

    Args:
        scenario: Canonical scenario name.

    Returns:
        The name of the scenario's primary order parameter.

    Raises:
        KeyError: If the scenario has no declared primary order parameter.
    """
    try:
        return PRIMARY_ORDER_PARAMETER[scenario]
    except KeyError:
        raise KeyError(f"no primary order parameter declared for scenario {scenario!r}") from None


def majority_verdict(
    verdicts: Iterable[str] | Mapping[str, int],
    severity_order: Sequence[str] = VERDICT_SEVERITY,
) -> str:
    """Return the most common verdict, tie-breaking toward the weaker label.

    Args:
        verdicts: Either the per-seed verdict labels of one cell, or an
            already-counted mapping of label to count.
        severity_order: Verdict labels ordered weakest-first.

    Returns:
        The majority verdict label. A tied count resolves to the weaker label,
        and a tie between two equally-ranked (that is, both unrecognized)
        labels resolves to the lexicographically smaller label so the result
        never depends on input ordering.

    Raises:
        ValueError: If no verdicts were supplied.
    """
    if isinstance(verdicts, Mapping):
        counts = {str(label): int(count) for label, count in verdicts.items() if int(count) > 0}
    else:
        counts = {}
        for label in verdicts:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise ValueError("majority_verdict requires at least one verdict")

    def sort_key(item: tuple[str, int]) -> tuple[int, int, str]:
        label, count = item
        return (-count, verdict_label_strength(label, severity_order), label)

    return min(counts.items(), key=sort_key)[0]


def select_representative_index(
    candidates: Sequence[RepresentativeCandidate],
    severity_order: Sequence[str] = VERDICT_SEVERITY,
) -> int:
    """Select the representative run of one campaign cell.

    Restricts to the majority-verdict pool (weaker-label tie-break), then takes
    the median primary order parameter within that pool, taking the lower of
    the two middle runs for an even pool and the lower seed on an exact tie.

    Args:
        candidates: Every run of a single campaign cell.
        severity_order: Verdict labels ordered weakest-first.

    Returns:
        The index into ``candidates`` of the selected representative run.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("select_representative_index requires at least one candidate")

    majority = majority_verdict(
        (candidate.verdict for candidate in candidates), severity_order=severity_order
    )
    pool = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate.verdict == majority
    ]
    ordered = sorted(pool, key=lambda item: (item[1].primary_order, item[1].seed, item[0]))
    return ordered[(len(ordered) - 1) // 2][0]
