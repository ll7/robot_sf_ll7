"""Deterministic representative selection for trace dossiers.

The selector chooses one seed or episode from a single campaign cell.  It is
deliberately a pure, provenance-neutral helper: it does not compute a metric,
change a verdict, or admit a trace as benchmark evidence.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, TypeAlias

TRACE_DOSSIER_SELECTOR_SCHEMA_VERSION = "trace_dossier_selector.v1"
SelectionReason: TypeAlias = Literal[  # noqa: UP040
    "single_candidate",
    "majority_verdict",
    "weaker_verdict",
    "weakest_label",
    "median_primary_order",
    "seed_identity",
]


class TraceDossierSelectionError(ValueError):
    """Raised when representative selection cannot be justified deterministically."""


@dataclass(frozen=True, slots=True)
class TraceDossierCandidate:
    """Validated candidate input for one campaign cell."""

    cell_id: str
    verdict: str
    label_strength: float
    primary_order: float
    seed_id: str


@dataclass(frozen=True, slots=True)
class SelectionManifest:
    """Stable result of selecting one representative candidate."""

    schema_version: str
    cell_id: str
    selected_seed_id: str
    selected_verdict: str
    selected_label_strength: float
    selected_primary_order: float
    candidate_count: int
    majority_verdict: str
    majority_count: int
    selection_reason: SelectionReason

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest without clock-bearing fields."""

        return {
            "candidate_count": self.candidate_count,
            "cell_id": self.cell_id,
            "majority_count": self.majority_count,
            "majority_verdict": self.majority_verdict,
            "schema_version": self.schema_version,
            "selected_label_strength": self.selected_label_strength,
            "selected_primary_order": self.selected_primary_order,
            "selected_seed_id": self.selected_seed_id,
            "selected_verdict": self.selected_verdict,
            "selection_reason": self.selection_reason,
        }


CandidateInput: TypeAlias = TraceDossierCandidate | Mapping[str, Any]  # noqa: UP040


def select_representative(candidates: Sequence[CandidateInput]) -> SelectionManifest:
    """Select one deterministic representative from one campaign cell.

    The selection order is: majority verdict, weaker label strength (the
    smallest numeric value) for tied verdict counts and within the selected
    verdict pool, closest primary-order parameter to the median, then
    lexicographically smallest seed identity. A tied verdict count with no
    unique weakest label fails closed rather than using an arbitrary ordering.

    Raises:
        TraceDossierSelectionError: If inputs are empty, malformed, mixed-cell,
            duplicated, or do not provide a unique majority verdict.

    Returns:
        A frozen, JSON-serializable selection manifest.
    """

    if not candidates:
        raise TraceDossierSelectionError("candidates must be non-empty")

    normalized = tuple(
        _normalize_candidate(candidate, index) for index, candidate in enumerate(candidates)
    )
    cell_ids = {candidate.cell_id for candidate in normalized}
    if len(cell_ids) != 1:
        raise TraceDossierSelectionError("candidates must not contain mixed cell_id values")
    seed_ids = [candidate.seed_id for candidate in normalized]
    if len(set(seed_ids)) != len(seed_ids):
        raise TraceDossierSelectionError("seed_id values must be unique")

    verdict_counts = Counter(candidate.verdict for candidate in normalized)
    majority_verdict, tied_majority = _select_majority_verdict(normalized, verdict_counts)
    highest_count = verdict_counts[majority_verdict]
    majority_pool = tuple(
        candidate for candidate in normalized if candidate.verdict == majority_verdict
    )

    weakest_strength = min(candidate.label_strength for candidate in majority_pool)
    weakest_pool = tuple(
        candidate for candidate in majority_pool if candidate.label_strength == weakest_strength
    )

    median_order = _median_primary_order(weakest_pool)
    distances = {
        candidate.seed_id: abs(candidate.primary_order - median_order) for candidate in weakest_pool
    }
    closest_distance = min(distances.values())
    closest_pool = tuple(
        candidate for candidate in weakest_pool if distances[candidate.seed_id] == closest_distance
    )
    selected = min(closest_pool, key=lambda candidate: candidate.seed_id)

    if len(normalized) == 1:
        reason: SelectionReason = "single_candidate"
    elif tied_majority:
        reason = "weaker_verdict"
    elif len(majority_pool) < len(normalized):
        reason = "majority_verdict"
    elif len(weakest_pool) < len(majority_pool):
        reason = "weakest_label"
    elif len(closest_pool) > 1:
        reason = "seed_identity"
    else:
        reason = "median_primary_order"

    return SelectionManifest(
        schema_version=TRACE_DOSSIER_SELECTOR_SCHEMA_VERSION,
        cell_id=selected.cell_id,
        selected_seed_id=selected.seed_id,
        selected_verdict=selected.verdict,
        selected_label_strength=selected.label_strength,
        selected_primary_order=selected.primary_order,
        candidate_count=len(normalized),
        majority_verdict=majority_verdict,
        majority_count=highest_count,
        selection_reason=reason,
    )


def _select_majority_verdict(
    candidates: Sequence[TraceDossierCandidate], verdict_counts: Counter[str]
) -> tuple[str, bool]:
    """Select the highest-count verdict and resolve ties by weaker label.

    Returns:
        The selected verdict and whether a tied count required weaker-label resolution.
    """

    highest_count = max(verdict_counts.values())
    majority_verdicts = sorted(
        verdict for verdict, count in verdict_counts.items() if count == highest_count
    )
    if len(majority_verdicts) == 1:
        return majority_verdicts[0], False

    verdict_strengths = {
        verdict: min(
            candidate.label_strength for candidate in candidates if candidate.verdict == verdict
        )
        for verdict in majority_verdicts
    }
    weakest_strength = min(verdict_strengths.values())
    weakest_verdicts = sorted(
        verdict for verdict, strength in verdict_strengths.items() if strength == weakest_strength
    )
    if len(weakest_verdicts) != 1:
        raise TraceDossierSelectionError(
            "tied majority verdicts have no unique weaker label; refusing arbitrary tie-break"
        )
    return weakest_verdicts[0], True


def _normalize_candidate(candidate: CandidateInput, index: int) -> TraceDossierCandidate:
    """Validate and copy one candidate without mutating mapping inputs.

    Returns:
        A validated immutable candidate.
    """

    if isinstance(candidate, TraceDossierCandidate):
        values = {
            "cell_id": candidate.cell_id,
            "verdict": candidate.verdict,
            "label_strength": candidate.label_strength,
            "primary_order": candidate.primary_order,
            "seed_id": candidate.seed_id,
        }
    elif isinstance(candidate, Mapping):
        required = {"cell_id", "verdict", "label_strength", "primary_order", "seed_id"}
        missing = sorted(required - candidate.keys())
        if missing:
            raise TraceDossierSelectionError(
                f"candidate {index} is missing required fields: {', '.join(missing)}"
            )
        values = {field: candidate[field] for field in required}
    else:
        raise TraceDossierSelectionError(
            f"candidate {index} is not a TraceDossierCandidate or mapping"
        )

    cell_id = _required_text(values["cell_id"], "cell_id", index)
    verdict = _required_text(values["verdict"], "verdict", index)
    seed_id = _required_text(values["seed_id"], "seed_id", index)
    label_strength = _finite_number(values["label_strength"], "label_strength", index)
    primary_order = _finite_number(values["primary_order"], "primary_order", index)
    return TraceDossierCandidate(
        cell_id=cell_id,
        verdict=verdict,
        label_strength=label_strength,
        primary_order=primary_order,
        seed_id=seed_id,
    )


def _required_text(value: object, field: str, index: int) -> str:
    """Validate one stable textual identity field.

    Returns:
        The original validated text.
    """

    if not isinstance(value, str) or not value.strip():
        raise TraceDossierSelectionError(f"candidate {index} has invalid {field}")
    return value


def _finite_number(value: object, field: str, index: int) -> float:
    """Validate one finite numeric tie-break field.

    Returns:
        The validated finite number as a float.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TraceDossierSelectionError(f"candidate {index} has invalid {field}")
    try:
        number = float(value)
    except OverflowError as error:
        raise TraceDossierSelectionError(f"candidate {index} has out-of-range {field}") from error
    if not math.isfinite(number):
        raise TraceDossierSelectionError(f"candidate {index} has non-finite {field}")
    return number


def _median_primary_order(candidates: Sequence[TraceDossierCandidate]) -> float:
    """Return a finite-safe median for already validated primary-order values."""

    orders = sorted(candidate.primary_order for candidate in candidates)
    middle = len(orders) // 2
    if len(orders) % 2:
        return orders[middle]
    return orders[middle - 1] / 2 + orders[middle] / 2
