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
from dataclasses import dataclass, field
from pathlib import Path
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


# --- Archive-readiness / fail-closed input checker (issue #3275) ---------------
#
# Before the proposal-vs-random runner consumes a *real* certified failure
# archive, the archive must satisfy structural prerequisites or the downstream
# disjoint split, overlap provenance, certification, and null tests cannot be
# computed. The runner historically degraded a missing/malformed archive to a
# synthetic fixture (``run_proposal_vs_random_issue_2921.py``), which is fine for
# plumbing but hides whether a real archive is actually usable. These helpers
# provide a standalone, fail-closed readiness verdict over a real archive input.
# They never fabricate entries and never fall back to synthetic data: an archive
# that fails any prerequisite is reported ``ready=False`` with precise reasons.

#: Top-level schema tag emitted by ``robot_sf.adversarial.archive``.
ARCHIVE_SCHEMA_VERSION = "adversarial_failure_archive.v1"

#: Minimum scenario families required to form a disjoint fit/eval split. A
#: single family collapses both sides together and can never pass the held-out
#: disjointness gate (see :func:`classify_held_out_evidence`).
_MIN_DISJOINT_FAMILIES = 2


@dataclass(frozen=True)
class ArchiveReadinessReport:
    """Fail-closed readiness verdict for a certified failure-archive input.

    ``ready`` is ``True`` only when the archive can drive a non-circular,
    held-out proposal-vs-random comparison: it parses, carries entries with the
    fields the overlap-provenance and certification gates need, and admits a
    disjoint scenario-family split with non-empty fit/eval sides. Any failing
    prerequisite leaves ``ready=False`` with a precise entry in
    ``blocking_reasons``.
    """

    ready: bool
    status: str
    schema_ok: bool
    entry_count: int
    distinct_family_count: int
    disjoint_split_possible: bool
    overlap_metadata_ready: bool
    null_test_prerequisites_ready: bool
    entries_missing_archive_id: int
    entries_missing_scenario_seed: int
    entries_missing_failure_attribution: int
    entries_unknown_family: int
    archive_sha256: str | None = None
    blocking_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the readiness report."""
        return {
            "ready": self.ready,
            "status": self.status,
            "schema_ok": self.schema_ok,
            "entry_count": self.entry_count,
            "distinct_family_count": self.distinct_family_count,
            "disjoint_split_possible": self.disjoint_split_possible,
            "overlap_metadata_ready": self.overlap_metadata_ready,
            "null_test_prerequisites_ready": self.null_test_prerequisites_ready,
            "entries_missing_archive_id": self.entries_missing_archive_id,
            "entries_missing_scenario_seed": self.entries_missing_scenario_seed,
            "entries_missing_failure_attribution": self.entries_missing_failure_attribution,
            "entries_unknown_family": self.entries_unknown_family,
            "archive_sha256": self.archive_sha256,
            "blocking_reasons": list(self.blocking_reasons),
        }


def _not_ready(status: str, *reasons: str, **fields: Any) -> ArchiveReadinessReport:
    """Build a fail-closed ``not_ready`` report with sensible numeric defaults."""
    defaults: dict[str, Any] = {
        "schema_ok": False,
        "entry_count": 0,
        "distinct_family_count": 0,
        "disjoint_split_possible": False,
        "overlap_metadata_ready": False,
        "null_test_prerequisites_ready": False,
        "entries_missing_archive_id": 0,
        "entries_missing_scenario_seed": 0,
        "entries_missing_failure_attribution": 0,
        "entries_unknown_family": 0,
        "archive_sha256": None,
    }
    defaults.update(fields)
    return ArchiveReadinessReport(
        ready=False, status=status, blocking_reasons=list(reasons), **defaults
    )


def _entry_scenario_seed(entry: dict[str, Any]) -> Any:
    """Return the nested ``candidate.scenario_seed`` value, or ``None``."""
    candidate = entry.get("candidate")
    if isinstance(candidate, dict):
        return candidate.get("scenario_seed")
    return None


@dataclass(frozen=True)
class _EntryStats:
    """Aggregate structural statistics over a list of archive entries."""

    entry_count: int
    non_dict_count: int
    missing_archive_id: int
    missing_seed: int
    missing_attribution: int
    unknown_family: int
    distinct_family_count: int
    disjoint_split_possible: bool


def _collect_entry_stats(
    entries: list[Any], *, eval_fraction: float, split_seed: int
) -> _EntryStats:
    """Compute fail-closed structural statistics over raw archive entries."""
    dict_entries = [e for e in entries if isinstance(e, dict)]
    distinct_families = {scenario_family_key(e) for e in dict_entries}

    # A disjoint split must actually produce non-empty fit and eval sides; this
    # is the prerequisite for overlap provenance and the null tests downstream.
    disjoint_split_possible = False
    if len(distinct_families) >= _MIN_DISJOINT_FAMILIES:
        split = disjoint_family_split(dict_entries, eval_fraction=eval_fraction, seed=split_seed)
        disjoint_split_possible = split.is_disjoint_split

    return _EntryStats(
        entry_count=len(dict_entries),
        non_dict_count=len(entries) - len(dict_entries),
        missing_archive_id=sum(1 for e in dict_entries if not e.get("archive_id")),
        missing_seed=sum(1 for e in dict_entries if _entry_scenario_seed(e) is None),
        missing_attribution=sum(
            1 for e in dict_entries if not isinstance(e.get("failure_attribution"), dict)
        ),
        unknown_family=sum(1 for e in dict_entries if scenario_family_key(e) == "unknown_family"),
        distinct_family_count=len(distinct_families),
        disjoint_split_possible=disjoint_split_possible,
    )


def _readiness_blocking_reasons(
    stats: _EntryStats, *, schema_ok: bool, schema_version: Any, min_entries: int
) -> list[str]:
    """Collect precise fail-closed reasons an archive is not ready, in order."""
    reasons: list[str] = []
    if not schema_ok:
        reasons.append(f"unexpected_schema_version:{schema_version!r}")
    if stats.non_dict_count:
        reasons.append(f"non_object_entries:{stats.non_dict_count}")
    if stats.entry_count < min_entries:
        reasons.append(f"too_few_entries:{stats.entry_count}<{min_entries}")
    if stats.missing_archive_id:
        reasons.append(f"entries_missing_archive_id:{stats.missing_archive_id}")
    if stats.missing_seed:
        reasons.append(f"entries_missing_scenario_seed:{stats.missing_seed}")
    if stats.missing_attribution:
        reasons.append(f"entries_missing_failure_attribution:{stats.missing_attribution}")
    if stats.unknown_family:
        reasons.append(f"entries_unknown_family:{stats.unknown_family}")
    if stats.distinct_family_count < _MIN_DISJOINT_FAMILIES:
        reasons.append(f"insufficient_scenario_families:{stats.distinct_family_count}")
    if not stats.disjoint_split_possible:
        reasons.append("no_disjoint_split_possible")
    return reasons


def assess_archive_readiness(
    archive_data: Any,
    *,
    min_entries: int = _MIN_DISJOINT_FAMILIES,
    eval_fraction: float = 0.5,
    split_seed: int = 0,
) -> ArchiveReadinessReport:
    """Assess whether a loaded failure archive is ready for held-out evaluation.

    This is a pure, fail-closed structural check over already-parsed archive
    data. It does not execute planners, fabricate entries, or fall back to
    synthetic data. It composes :func:`scenario_family_key` and
    :func:`disjoint_family_split` so its notion of "splittable" matches the
    machinery the proposal runner actually uses.

    Args:
        archive_data: Parsed archive payload (expected to be a dict with a
            ``schema_version`` tag and a non-empty ``entries`` list).
        min_entries: Minimum number of entries required to attempt a split.
        eval_fraction: Eval-side family fraction forwarded to the trial split.
        split_seed: Deterministic seed forwarded to the trial split.

    Returns:
        An :class:`ArchiveReadinessReport`.
    """
    if not isinstance(archive_data, dict):
        return _not_ready("not_ready", "archive_payload_not_object")

    schema_version = archive_data.get("schema_version")
    schema_ok = schema_version == ARCHIVE_SCHEMA_VERSION
    archive_hash = archive_sha256(archive_data)

    entries = archive_data.get("entries")
    if not isinstance(entries, list) or not entries:
        return _not_ready(
            "not_ready",
            "archive_has_no_entries",
            schema_ok=schema_ok,
            archive_sha256=archive_hash,
        )

    stats = _collect_entry_stats(entries, eval_fraction=eval_fraction, split_seed=split_seed)
    reasons = _readiness_blocking_reasons(
        stats, schema_ok=schema_ok, schema_version=schema_version, min_entries=min_entries
    )

    # Overlap provenance needs disjoint families, unique archive ids, and seeds
    # to compare. Null tests additionally need a non-empty eval side, which the
    # disjoint-split check guarantees.
    overlap_metadata_ready = (
        stats.disjoint_split_possible and not stats.missing_archive_id and not stats.missing_seed
    )
    null_test_prerequisites_ready = stats.disjoint_split_possible and not stats.missing_attribution

    ready = not reasons
    return ArchiveReadinessReport(
        ready=ready,
        status="ready" if ready else "not_ready",
        schema_ok=schema_ok,
        entry_count=stats.entry_count,
        distinct_family_count=stats.distinct_family_count,
        disjoint_split_possible=stats.disjoint_split_possible,
        overlap_metadata_ready=overlap_metadata_ready,
        null_test_prerequisites_ready=null_test_prerequisites_ready,
        entries_missing_archive_id=stats.missing_archive_id,
        entries_missing_scenario_seed=stats.missing_seed,
        entries_missing_failure_attribution=stats.missing_attribution,
        entries_unknown_family=stats.unknown_family,
        archive_sha256=archive_hash,
        blocking_reasons=reasons,
    )


def assess_archive_file_readiness(path: Path | None) -> ArchiveReadinessReport:
    """Load an archive file fail-closed and assess its readiness.

    Unlike the proposal runner's loader, this never substitutes a synthetic
    fixture: a missing, empty, unreadable, or malformed input is reported
    ``ready=False`` with a precise ``not_ready`` reason. A real archive that
    parses is delegated to :func:`assess_archive_readiness`.

    Returns:
        An :class:`ArchiveReadinessReport`.
    """
    if path is None:
        return _not_ready("not_ready", "no_archive_path_provided")
    if not path.exists():
        return _not_ready("not_ready", f"archive_path_missing:{path}")
    if path.stat().st_size == 0:
        return _not_ready("not_ready", f"archive_file_empty:{path}")
    try:
        archive_data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return _not_ready("not_ready", f"archive_unreadable:{exc}")
    return assess_archive_readiness(archive_data)
