"""Disjoint fit/evaluation splitting, overlap provenance, and null tests.

These utilities make an adversarial proposal-vs-random comparison non-circular
(issue #3275). PR #3276 ranked candidates by distance to the failure archive and
evaluated them using the same distance to the same archive, which is circular.

This module provides the machinery a non-circular, held-out comparison requires:

* split a failure archive into fit/eval sets whose *scenario families* are
  disjoint, so a model fit on one family is evaluated on held-out families;
* compute explicit overlap provenance (scenario-family / seed / archive-id) plus
  archive hashes, so reviewers can verify disjointness;
* run permutation null tests (shuffled-outcome label test and a ranking
  permutation test) against an *independent* outcome vector;
* classify held-out evidence **fail-closed** — it is never ``eligible`` unless a
  disjoint split, independent (non-archive-nearness) outcomes, candidate
  certification, and a rejected null are all present.

None of these functions assert held-out yield on their own; they provide the
inputs a later evaluation step needs before any survived/falsified verdict.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Permutation-test float comparison tolerance.
_EPS = 1e-12


def scenario_family_key(entry: dict[str, Any]) -> str:
    """Return a stable scenario-family key for a failure-archive entry.

    Prefers the archive ``cluster_key`` (the archive's own grouping over policy,
    scenario template, and failure attribution). Falls back to failure and
    source-manifest fields, then to ``"unknown_family"``.

    Returns:
        A deterministic, JSON-comparable family key string.
    """
    cluster_key = entry.get("cluster_key")
    if isinstance(cluster_key, dict) and cluster_key:
        return json.dumps(cluster_key, sort_keys=True, separators=(",", ":"))
    if isinstance(cluster_key, str) and cluster_key.strip():
        return cluster_key

    parts: list[str] = []
    attribution = entry.get("failure_attribution")
    if isinstance(attribution, dict):
        primary = attribution.get("primary_failure")
        if isinstance(primary, str) and primary:
            parts.append(f"failure={primary}")
    source_manifest = entry.get("source_manifest")
    if isinstance(source_manifest, str) and source_manifest:
        parts.append(f"manifest={source_manifest}")
    return "|".join(parts) if parts else "unknown_family"


@dataclass(frozen=True)
class DisjointSplit:
    """A fit/evaluation split with non-overlapping scenario families."""

    fit_entries: list[dict[str, Any]]
    eval_entries: list[dict[str, Any]]
    fit_families: list[str]
    eval_families: list[str]
    is_disjoint_split: bool


def disjoint_family_split(
    entries: Sequence[dict[str, Any]],
    *,
    eval_fraction: float = 0.5,
    seed: int = 0,
) -> DisjointSplit:
    """Partition ``entries`` into fit/eval sets with disjoint scenario families.

    Each scenario family is assigned wholesale to either the fit or the eval
    side, so no family appears on both sides. The assignment is deterministic
    given ``seed``. When fewer than two families are present a disjoint split is
    impossible, so all entries go to the fit side and ``is_disjoint_split`` is
    ``False``.

    Args:
        entries: Failure-archive entries (dicts with optional ``cluster_key``).
        eval_fraction: Target fraction of *families* (not entries) held out for
            evaluation, clamped so both sides keep at least one family.
        seed: Deterministic shuffle seed for family assignment.

    Returns:
        A :class:`DisjointSplit`.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in the open interval (0, 1)")

    families: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        families.setdefault(scenario_family_key(entry), []).append(entry)

    family_keys = sorted(families)
    if len(family_keys) < 2:
        return DisjointSplit(
            fit_entries=list(entries),
            eval_entries=[],
            fit_families=family_keys,
            eval_families=[],
            is_disjoint_split=False,
        )

    shuffled = list(family_keys)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, min(len(shuffled) - 1, round(eval_fraction * len(shuffled))))
    eval_families = set(shuffled[:n_eval])

    fit_entries = [e for e in entries if scenario_family_key(e) not in eval_families]
    eval_entries = [e for e in entries if scenario_family_key(e) in eval_families]
    return DisjointSplit(
        fit_entries=fit_entries,
        eval_entries=eval_entries,
        fit_families=sorted(set(family_keys) - eval_families),
        eval_families=sorted(eval_families),
        is_disjoint_split=bool(fit_entries) and bool(eval_entries),
    )


def archive_sha256(data: Any) -> str:
    """Return a deterministic SHA-256 digest of JSON-serializable archive data."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distinct(entries: Sequence[dict[str, Any]], key: str) -> set[Any]:
    """Return the set of non-null ``candidate``/top-level values for ``key``."""
    values: set[Any] = set()
    for entry in entries:
        if key == "scenario_seed":
            candidate = entry.get("candidate", {})
            value = candidate.get("scenario_seed") if isinstance(candidate, dict) else None
        else:
            value = entry.get(key)
        if value is not None:
            values.add(value)
    return values


def compute_overlap_provenance(
    fit_entries: Sequence[dict[str, Any]],
    eval_entries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute fit/eval overlap provenance for a disjoint split.

    Reports scenario-family, scenario-seed, and archive-id overlaps between the
    fit and eval sets. ``disjointness_checks_passed`` is ``True`` only when both
    sides are non-empty and share no scenario family, scenario seed, or archive
    id. Seed overlap is invalid for this held-out-evidence gate unless a future
    paired/dependent-inference design defines a separate contract.

    Returns:
        A JSON-safe provenance dict.
    """
    fit_families = {scenario_family_key(e) for e in fit_entries}
    eval_families = {scenario_family_key(e) for e in eval_entries}
    family_overlap = sorted(fit_families & eval_families)

    seed_overlap = sorted(
        _distinct(fit_entries, "scenario_seed") & _distinct(eval_entries, "scenario_seed")
    )
    id_overlap = sorted(
        _distinct(fit_entries, "archive_id") & _distinct(eval_entries, "archive_id")
    )

    failure_reasons: list[str] = []
    if not fit_entries:
        failure_reasons.append("empty_fit")
    if not eval_entries:
        failure_reasons.append("empty_eval")
    if family_overlap:
        failure_reasons.append("scenario_family_overlap")
    if seed_overlap:
        failure_reasons.append("seed_overlap")
    if id_overlap:
        failure_reasons.append("archive_id_overlap")

    disjoint = not failure_reasons
    return {
        "split_policy": "disjoint_scenario_family",
        "fit_size": len(fit_entries),
        "eval_size": len(eval_entries),
        "fit_families": sorted(fit_families),
        "eval_families": sorted(eval_families),
        "scenario_family_overlap": family_overlap,
        "scenario_family_overlap_count": len(family_overlap),
        "seed_overlap": seed_overlap,
        "seed_overlap_count": len(seed_overlap),
        "archive_id_overlap": id_overlap,
        "archive_id_overlap_count": len(id_overlap),
        "disjointness_checks_passed": disjoint,
        "disjointness_failure_reasons": failure_reasons,
        "seed_overlap_invalidates_held_out_evidence": bool(seed_overlap),
    }


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or ``0.0`` for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def permutation_test_mean_difference(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-sided permutation test on the difference of group means (a - b).

    Returns the observed difference and a permutation p-value using the
    add-one estimator ``(count + 1) / (n_permutations + 1)``.

    Returns:
        A JSON-safe result dict; ``status`` is ``"not_available_empty_group"``
        when either group is empty.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    a = [float(x) for x in group_a]
    b = [float(x) for x in group_b]
    if not a or not b:
        return {
            "observed_difference": 0.0,
            "p_value": None,
            "n_permutations": 0,
            "status": "not_available_empty_group",
        }

    observed = _mean(a) - _mean(b)
    pooled = a + b
    n_a = len(a)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_diff = _mean(pooled[:n_a]) - _mean(pooled[n_a:])
        if abs(perm_diff) >= abs(observed) - _EPS:
            extreme += 1
    return {
        "observed_difference": round(observed, 6),
        "p_value": round((extreme + 1) / (n_permutations + 1), 6),
        "n_permutations": n_permutations,
        "status": "complete",
    }


def shuffled_outcome_null_test(
    proposal_outcomes: Sequence[float],
    random_outcomes: Sequence[float],
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Null test: are proposal and random *outcomes* exchangeable?

    If the proposal selection carries no real signal, the proposal-minus-random
    mean-outcome difference should be indistinguishable from permutations of the
    pooled outcome labels (large p-value).

    Returns:
        The :func:`permutation_test_mean_difference` result tagged with
        ``test = "shuffled_outcome_label_permutation"``.
    """
    result = permutation_test_mean_difference(
        proposal_outcomes,
        random_outcomes,
        n_permutations=n_permutations,
        seed=seed,
    )
    result["test"] = "shuffled_outcome_label_permutation"
    return result


def ranking_permutation_test(
    ranked_outcomes: Sequence[float],
    *,
    selection_size: int,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Null test: does the ranking concentrate high outcomes in its top-k?

    Compares the mean outcome of the top ``selection_size`` ranked items against
    the distribution of top-k means under random orderings of the same outcomes.
    A real ranking signal yields a small (one-sided) p-value.

    Returns:
        A JSON-safe result dict; ``status`` flags invalid selection sizes.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    outcomes = [float(x) for x in ranked_outcomes]
    n = len(outcomes)
    if n == 0 or selection_size <= 0 or selection_size > n:
        return {
            "test": "ranking_permutation",
            "p_value": None,
            "n_permutations": 0,
            "status": "not_available_invalid_selection",
        }

    observed_top_mean = _mean(outcomes[:selection_size])
    indices = list(range(n))
    rng = random.Random(seed)
    at_least_as_high = 0
    for _ in range(n_permutations):
        rng.shuffle(indices)
        sampled = [outcomes[i] for i in indices[:selection_size]]
        if _mean(sampled) >= observed_top_mean - _EPS:
            at_least_as_high += 1
    return {
        "test": "ranking_permutation",
        "observed_top_mean": round(observed_top_mean, 6),
        "p_value": round((at_least_as_high + 1) / (n_permutations + 1), 6),
        "n_permutations": n_permutations,
        "selection_size": selection_size,
        "status": "complete",
    }


def classify_held_out_evidence(
    *,
    disjointness_checks_passed: bool,
    independent_outcomes_available: bool,
    certification_available: bool,
    null_tests_reject_null: bool,
) -> str:
    """Fail-closed classification of held-out proposal-vs-random evidence.

    Returns ``"eligible_held_out_diagnostic"`` only when every precondition
    holds. Any missing precondition returns a precise ``not_available_*`` reason.
    This never returns eligible from circular archive-nearness outcomes, because
    ``independent_outcomes_available`` must be supplied by an evaluation step that
    does not reuse the ranking objective.

    Returns:
        An eligibility/``not_available_*`` string.
    """
    if not disjointness_checks_passed:
        return "not_available_no_disjoint_split"
    if not independent_outcomes_available:
        return "not_available_requires_independent_planner_outcomes"
    if not certification_available:
        return "not_available_requires_candidate_certification"
    if not null_tests_reject_null:
        return "not_available_null_tests_not_rejected"
    return "eligible_held_out_diagnostic"
