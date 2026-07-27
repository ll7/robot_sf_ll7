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
from math import comb
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.adversarial.config import SearchSpaceConfig

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

#: Null-test manifest key required before a certified archive can be treated as
#: ready for a proposal-vs-random rerun.
NULL_TEST_MANIFEST_KEY = "null_test_manifest"
REQUIRED_NULL_TESTS = frozenset({"shuffled_outcome_label_permutation", "ranking_permutation"})

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
    entries_missing_certification_status: int
    entries_not_certified: int
    entries_unknown_family: int
    scenario_family_overlap_count: int
    seed_overlap_count: int
    archive_id_overlap_count: int
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
            "entries_missing_certification_status": self.entries_missing_certification_status,
            "entries_not_certified": self.entries_not_certified,
            "entries_unknown_family": self.entries_unknown_family,
            "scenario_family_overlap_count": self.scenario_family_overlap_count,
            "seed_overlap_count": self.seed_overlap_count,
            "archive_id_overlap_count": self.archive_id_overlap_count,
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
        "entries_missing_certification_status": 0,
        "entries_not_certified": 0,
        "entries_unknown_family": 0,
        "scenario_family_overlap_count": 0,
        "seed_overlap_count": 0,
        "archive_id_overlap_count": 0,
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


def _entry_certification_status(entry: dict[str, Any]) -> str | None:
    """Return normalized per-entry candidate certification status, if present."""
    certification = entry.get("certification_status")
    if certification is None:
        certification = entry.get("candidate_certification")
    if not isinstance(certification, dict):
        return None
    status = certification.get("status")
    if not isinstance(status, str) or not status.strip():
        return None
    return status.strip().lower()


@dataclass(frozen=True)
class _EntryStats:
    """Aggregate structural statistics over a list of archive entries."""

    entry_count: int
    non_dict_count: int
    missing_archive_id: int
    missing_seed: int
    missing_attribution: int
    missing_certification_status: int
    not_certified: int
    unknown_family: int
    distinct_family_count: int
    disjoint_split_possible: bool
    scenario_family_overlap_count: int
    seed_overlap_count: int
    archive_id_overlap_count: int


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

    overlap = compute_overlap_provenance([], [])
    if disjoint_split_possible:
        overlap = compute_overlap_provenance(split.fit_entries, split.eval_entries)

    certification_statuses = [_entry_certification_status(e) for e in dict_entries]

    return _EntryStats(
        entry_count=len(dict_entries),
        non_dict_count=len(entries) - len(dict_entries),
        missing_archive_id=sum(1 for e in dict_entries if not e.get("archive_id")),
        missing_seed=sum(1 for e in dict_entries if _entry_scenario_seed(e) is None),
        missing_attribution=sum(
            1
            for e in dict_entries
            if not isinstance(e.get("failure_attribution"), dict) or not e["failure_attribution"]
        ),
        missing_certification_status=sum(1 for status in certification_statuses if status is None),
        not_certified=sum(
            1 for status in certification_statuses if status is not None and status != "passed"
        ),
        unknown_family=sum(1 for e in dict_entries if scenario_family_key(e) == "unknown_family"),
        distinct_family_count=len(distinct_families),
        disjoint_split_possible=disjoint_split_possible,
        scenario_family_overlap_count=len(overlap["scenario_family_overlap"]),
        seed_overlap_count=len(overlap["seed_overlap"]),
        archive_id_overlap_count=len(overlap["archive_id_overlap"]),
    )


def _readiness_blocking_reasons(
    stats: _EntryStats, *, schema_ok: bool, schema_version: Any, min_entries: int
) -> list[str]:
    """Collect precise fail-closed reasons an archive is not ready, in order."""
    reasons: list[str] = []
    if not schema_ok:
        reasons.append(f"unexpected_schema_version:{schema_version!r}")
    if stats.entry_count < min_entries:
        reasons.append(f"too_few_entries:{stats.entry_count}<{min_entries}")
    counted_blockers = (
        ("non_object_entries", stats.non_dict_count),
        ("entries_missing_archive_id", stats.missing_archive_id),
        ("entries_missing_scenario_seed", stats.missing_seed),
        ("entries_missing_failure_attribution", stats.missing_attribution),
        ("entries_missing_certification_status", stats.missing_certification_status),
        ("entries_not_certified", stats.not_certified),
        ("entries_unknown_family", stats.unknown_family),
        (
            "insufficient_scenario_families",
            stats.distinct_family_count
            if stats.distinct_family_count < _MIN_DISJOINT_FAMILIES
            else 0,
        ),
        ("scenario_family_overlap", stats.scenario_family_overlap_count),
        ("seed_overlap", stats.seed_overlap_count),
        ("archive_id_overlap", stats.archive_id_overlap_count),
    )
    reasons.extend(f"{name}:{count}" for name, count in counted_blockers if count)
    if not stats.disjoint_split_possible:
        reasons.append("no_disjoint_split_possible")
    return reasons


def _optional_summary_int(summary: dict[str, Any], key: str) -> tuple[int | None, str | None]:
    """Return optional integer summary value, rejecting bools and non-ints."""
    value = summary.get(key)
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"summary_{key}_not_int"
    return value, None


def _summary_consistency_blockers(archive_data: dict[str, Any], entries: list[Any]) -> list[str]:
    """Return blockers for stale optional archive summary metadata.

    The curator writes a compact top-level ``summary`` next to ``entries`` and
    ``clusters``. If a later packet edits entries without regenerating that
    metadata, downstream provenance can look cleaner than the archive input is.
    Missing summary metadata remains allowed for minimal fixtures, but present
    counts must agree with the payload they describe.
    """
    summary = archive_data.get("summary")
    if summary is None:
        return []
    if not isinstance(summary, dict):
        return ["summary_metadata_not_object"]

    blockers: list[str] = []
    archived_failure_count, count_blocker = _optional_summary_int(summary, "archived_failure_count")
    if count_blocker is not None:
        blockers.append(count_blocker)
    elif archived_failure_count is not None and archived_failure_count != len(entries):
        blockers.append(
            "summary_archived_failure_count_mismatch:"
            f"declared={archived_failure_count}:actual={len(entries)}"
        )

    cluster_count, count_blocker = _optional_summary_int(summary, "cluster_count")
    if count_blocker is not None:
        blockers.append(count_blocker)
        return blockers
    if cluster_count is None:
        return blockers

    clusters = archive_data.get("clusters")
    if clusters is None:
        blockers.append("summary_cluster_count_without_clusters")
    elif not isinstance(clusters, list):
        blockers.append("archive_clusters_not_list")
    elif cluster_count != len(clusters):
        blockers.append(
            f"summary_cluster_count_mismatch:declared={cluster_count}:actual={len(clusters)}"
        )
    return blockers


def _null_test_manifest_blockers(archive_data: dict[str, Any]) -> list[str]:
    """Validate explicit null-test prerequisites for held-out archive reruns."""
    manifest = archive_data.get(NULL_TEST_MANIFEST_KEY)
    if manifest is None:
        return ["null_test_manifest_missing"]
    if not isinstance(manifest, dict):
        return ["null_test_manifest_not_object"]

    blockers: list[str] = []
    required_tests = manifest.get("required_tests")
    if not isinstance(required_tests, list) or not required_tests:
        blockers.append("null_test_required_tests_missing")
    else:
        missing_tests = sorted(REQUIRED_NULL_TESTS - {str(test) for test in required_tests})
        if missing_tests:
            blockers.append(f"null_test_required_tests_missing:{','.join(missing_tests)}")

    n_permutations = manifest.get("n_permutations")
    if type(n_permutations) is not int or n_permutations < 1:
        blockers.append("null_test_n_permutations_invalid")

    return blockers


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
    reasons.extend(_summary_consistency_blockers(archive_data, entries))
    null_test_manifest_reasons = _null_test_manifest_blockers(archive_data)
    reasons.extend(null_test_manifest_reasons)

    # Overlap provenance needs disjoint families, unique archive ids, and seeds
    # to compare. Null tests additionally need a non-empty eval side, which the
    # disjoint-split check guarantees.
    overlap_metadata_ready = (
        stats.disjoint_split_possible
        and not stats.missing_archive_id
        and not stats.missing_seed
        and not stats.scenario_family_overlap_count
        and not stats.seed_overlap_count
        and not stats.archive_id_overlap_count
    )
    null_test_prerequisites_ready = (
        overlap_metadata_ready
        and not stats.missing_attribution
        and not null_test_manifest_reasons
        and not stats.missing_certification_status
        and not stats.not_certified
    )

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
        entries_missing_certification_status=stats.missing_certification_status,
        entries_not_certified=stats.not_certified,
        entries_unknown_family=stats.unknown_family,
        scenario_family_overlap_count=stats.scenario_family_overlap_count,
        seed_overlap_count=stats.seed_overlap_count,
        archive_id_overlap_count=stats.archive_id_overlap_count,
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


# --- Issue #3275 frozen same-planner contract primitives ----------------------
#
# The helpers below implement the frozen study contract for issue #6103 / parent
# #3275. They are pure and side-effect-free: they do not execute planners and
# produce no new empirical outcome. They provide the cross-family feature view,
# the deterministic held-out split, the arm-overlap policy, the binary-yield
# power/sensitivity calculation, and the ``continue | stop | inconclusive``
# decision rule that the proposal-vs-random contract freezes.

#: Frozen decision vocabulary for the #3275 contract. Post-outcome discretion
#: (``revise``) and generic ``blocked`` are intentionally absent: the only
#: terminal decisions are continue, stop, or inconclusive.
ISSUE_3275_DECISION_VOCABULARY = ("continue", "stop", "inconclusive")


def family_invariant_features(
    candidate: Any,
    search_space: SearchSpaceConfig,
) -> dict[str, float]:
    """Return a frozen, family-invariant feature view for one candidate.

    ``CandidateSpec.start`` and ``CandidateSpec.goal`` are robot-route endpoints:
    the materializer writes them to ``route_overrides.robot_routes``. The frozen
    view therefore keeps those endpoint coordinates as robot-route features and
    normalizes all seven candidate controls by the one pinned search space shared
    by the fit and held-out families. This preserves failure-anchor variation;
    projecting each archive route onto itself would collapse every anchor to the
    same spatial vector.

    Per-feature semantic argument:

    * ``robot_start_x/y_space_fraction`` and ``robot_goal_x/y_space_fraction``:
      the robot route endpoints within the pinned crossing/TTC search intervals.
      Both scenario families use metre-valued coordinates in the same 40x40 SVG
      frame at two cells per metre, and the exact same search-space file.
    * ``pedestrian_speed_space_fraction``,
      ``pedestrian_delay_space_fraction``, and ``spawn_time_space_fraction``:
      the remaining candidate controls, normalized by their pinned intervals.

    The transform is deterministic config normalization only. It consumes no
    outcome and does not use excluded cross-trap/goal failures for tuning.

    Args:
        candidate: A ``CandidateSpec`` or equivalent candidate dict.
        search_space: The raw-SHA-pinned shared search-space contract.

    Returns:
        A dict with the seven frozen family-invariant feature values.
    """

    def _value(name: str) -> float:
        if isinstance(candidate, dict):
            if name.startswith(("start_", "goal_")):
                pose_name, axis = name.split("_", maxsplit=1)
                return float(candidate[pose_name][axis])
            return float(candidate[name])
        if name.startswith(("start_", "goal_")):
            pose_name, axis = name.split("_", maxsplit=1)
            return float(getattr(getattr(candidate, pose_name), axis))
        return float(getattr(candidate, name))

    source_names = {
        "robot_start_x_space_fraction": "start_x",
        "robot_start_y_space_fraction": "start_y",
        "robot_goal_x_space_fraction": "goal_x",
        "robot_goal_y_space_fraction": "goal_y",
        "pedestrian_speed_space_fraction": "pedestrian_speed_mps",
        "pedestrian_delay_space_fraction": "pedestrian_delay_s",
        "spawn_time_space_fraction": "spawn_time_s",
    }
    features: dict[str, float] = {}
    for output_name, source_name in source_names.items():
        bounds = getattr(search_space, source_name)
        span = float(bounds.max) - float(bounds.min)
        if span <= 0.0:
            raise ValueError(f"frozen search-space range {source_name!r} must have positive span")
        features[output_name] = round((_value(source_name) - float(bounds.min)) / span, 9)
    return features


#: Feature names of the frozen family-invariant view, in canonical order.
FAMILY_INVARIANT_FEATURE_NAMES = (
    "robot_start_x_space_fraction",
    "robot_start_y_space_fraction",
    "robot_goal_x_space_fraction",
    "robot_goal_y_space_fraction",
    "pedestrian_speed_space_fraction",
    "pedestrian_delay_space_fraction",
    "spawn_time_space_fraction",
)


def family_invariant_distance(
    candidate: Any,
    anchor: Any,
    search_space: SearchSpaceConfig,
) -> float:
    """L1 distance in the frozen family-invariant feature space.

    Candidate and anchor controls are normalized by the same raw-SHA-pinned
    search-space intervals, so every dimension is directly comparable across
    the fit and held-out scenario families.
    """
    a = family_invariant_features(candidate, search_space)
    b = family_invariant_features(anchor, search_space)
    return sum(abs(float(a[name]) - float(b[name])) for name in FAMILY_INVARIANT_FEATURE_NAMES)


def frozen_held_out_family_split(
    entries: Sequence[dict[str, Any]],
    *,
    fit_family: str,
    eval_family: str,
    fit_entry_ids: Sequence[str] | None = None,
) -> DisjointSplit:
    """Deterministic held-out-family split frozen by the #3275 contract.

    Unlike :func:`disjoint_family_split` (which randomly assigns families), this
    assigns the frozen fit family and evaluation family explicitly. It is the
    same-planner held-out design: fit on ``fit_family`` (group crossing),
    evaluate on the disjoint ``eval_family`` (cross trap). Entries are matched by
    their ``scenario_family`` field (falling back to a string
    ``cluster_key.scenario_family``).

    Returns:
        A :class:`DisjointSplit` whose ``is_disjoint_split`` is ``True`` only
        when both sides are non-empty and the two families are distinct.
    """
    if fit_family == eval_family:
        raise ValueError("fit_family and eval_family must differ for a held-out split")

    def _family(entry: dict[str, Any]) -> str:
        fam = entry.get("scenario_family")
        if isinstance(fam, str) and fam:
            return fam
        cluster_key = entry.get("cluster_key")
        if isinstance(cluster_key, dict):
            fam = cluster_key.get("scenario_family")
            if isinstance(fam, str) and fam:
                return fam
        return scenario_family_key(entry)

    expected_fit_ids = {str(entry_id) for entry_id in fit_entry_ids} if fit_entry_ids else None
    if expected_fit_ids is not None and len(expected_fit_ids) != len(fit_entry_ids or ()):
        raise ValueError("fit_entry_ids must be unique for a frozen held-out split")
    fit_entries = [
        entry
        for entry in entries
        if _family(entry) == fit_family
        and (expected_fit_ids is None or str(entry.get("archive_id")) in expected_fit_ids)
    ]
    if expected_fit_ids is not None:
        observed_fit_ids = {str(entry.get("archive_id")) for entry in fit_entries}
        missing_fit_ids = sorted(expected_fit_ids - observed_fit_ids)
        if missing_fit_ids:
            raise ValueError(
                "frozen fit_entry_ids missing from held-out split source: "
                f"{', '.join(missing_fit_ids)}"
            )
    eval_entries = [e for e in entries if _family(e) == eval_family]
    other = [e for e in entries if _family(e) not in {fit_family, eval_family}]
    # Entries outside the two frozen families cannot be silently assigned; they
    # are reported via fit_families/eval_families so callers can detect drift.
    fit_families = sorted({_family(e) for e in fit_entries} | {_family(e) for e in other})
    return DisjointSplit(
        fit_entries=fit_entries,
        eval_entries=eval_entries,
        fit_families=fit_families,
        eval_families=[eval_family],
        is_disjoint_split=bool(fit_entries) and bool(eval_entries),
    )


@dataclass(frozen=True)
class ArmAssignment:
    """Result of the frozen disjoint-by-candidate arm-overlap policy."""

    proposal_ids: list[str]
    random_ids: list[str]
    overlap_ids: list[str]
    policy: str


def assign_arms_disjoint_by_candidate(
    ranked_pool_ids: Sequence[str],
    pool_ids: Sequence[str],
    *,
    budget_per_arm: int,
    rng_seed: int,
) -> ArmAssignment:
    """One deterministic, predeclared arm-overlap policy (issue #3275 gate #5).

    Each candidate (by manifest/archive id) belongs to exactly one arm. The
    proposal arm takes the top ``budget_per_arm`` from ``ranked_pool_ids``
    (model rank order). The random arm takes ``budget_per_arm`` ids from the
    remaining pool (``pool_ids`` minus the proposal arm's picks), drawn
    deterministically from ``rng_seed``. No candidate is ever counted in both
    arms, eliminating pseudoreplication. This single policy replaces any
    deduplicate-or-disjoint heuristic.

    Args:
        ranked_pool_ids: Pool ids ordered by model rank (best first).
        pool_ids: The full shared candidate pool id list.
        budget_per_arm: Identical budget for each arm.
        rng_seed: Deterministic seed for the random-arm draw.

    Returns:
        An :class:`ArmAssignment` whose ``overlap_ids`` is always empty.
    """
    if not isinstance(budget_per_arm, int) or isinstance(budget_per_arm, bool):
        raise ValueError("budget_per_arm must be an integer")
    if budget_per_arm < 0:
        raise ValueError("budget_per_arm must be >= 0")
    ranked = list(ranked_pool_ids)
    pool = list(pool_ids)
    if len(set(pool)) != len(pool):
        raise ValueError("pool_ids must contain unique stable candidate IDs")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked_pool_ids must contain unique stable candidate IDs")
    unknown_ranked_ids = sorted(set(ranked) - set(pool))
    if unknown_ranked_ids:
        raise ValueError(f"ranked_pool_ids contains IDs absent from pool_ids: {unknown_ranked_ids}")
    missing_ranked_ids = sorted(set(pool) - set(ranked))
    if missing_ranked_ids:
        raise ValueError(f"ranked_pool_ids omits pool IDs: {missing_ranked_ids}")
    if len(pool) < 2 * budget_per_arm:
        raise ValueError(
            "pool_ids must contain at least two disjoint arm budgets: "
            f"pool={len(pool)} budget_per_arm={budget_per_arm}"
        )
    proposal_ids = ranked[: min(budget_per_arm, len(ranked))]
    proposal_set = set(proposal_ids)
    remaining = [cid for cid in pool if cid not in proposal_set]
    random.Random(rng_seed).shuffle(remaining)
    random_ids = remaining[: min(budget_per_arm, len(remaining))]
    overlap = sorted(proposal_set & set(random_ids))
    return ArmAssignment(
        proposal_ids=proposal_ids,
        random_ids=random_ids,
        overlap_ids=overlap,
        policy="disjoint_by_candidate",
    )


def fisher_exact_two_sided(x: int, y: int, k: int) -> float:
    """Two-sided Fisher's exact p-value for two binomial samples of size ``k``.

    ``x`` and ``y`` are the failure (success) counts out of ``k`` trials in the
    proposal and random arms. Returns the sum of hypergeometric probabilities of
    all 2x2 tables with the same margins whose probability does not exceed the
    observed table's probability.
    """
    return fisher_exact_two_sided_table(x, k - x, y, k - y)


def _check_fisher_table_args(a: int, b: int, c: int, d: int) -> None:
    """Validate that the 2x2 table cells are non-negative integers."""
    for value, name in ((a, "a"), (b, "b"), (c, "c"), (d, "d")):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def fisher_exact_two_sided_table(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p-value for a general 2x2 contingency table.

    Table layout ``[[a, b], [c, d]]`` with row totals ``a+b`` and ``c+d`` and
    column totals ``a+c`` and ``b+d``. Used by the #3275 decision rule when the
    two arms may have different candidate counts.
    """
    _check_fisher_table_args(a, b, c, d)
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 1.0

    def _choose(nn: int, r: int) -> int:
        if r < 0 or r > nn:
            return 0
        return comb(nn, r)

    # Hypergeometric probability of table [[aa, row1-aa], [col1-aa, ...]] with
    # row totals (row1, row2) and column totals (col1, col2): choose which of the
    # ``row1`` row-1 units come from the ``col1`` column-1 units.
    denom = _choose(n, row1)
    if denom == 0:
        return 1.0

    def _prob(aa: int) -> float:
        return (_choose(col1, aa) * _choose(col2, row1 - aa)) / denom

    p_obs = _prob(a)
    lo = max(0, row1 - col2)
    hi = min(row1, col1)
    total = sum(_prob(aa) for aa in range(lo, hi + 1) if _prob(aa) <= p_obs + _EPS)
    return min(1.0, total)


def binary_yield_min_detectable_difference(k_per_arm: int, alpha: float = 0.05) -> float:
    """Smallest |yield difference| Fisher's exact test can reject at ``alpha``.

    Enumerates every pair of failure counts ``(x, y)`` out of ``k_per_arm`` and
    returns the minimum ``|x/k - y/k|`` whose two-sided Fisher p-value is at most
    ``alpha``. This is the boundary of the rejection region (not 80% power); it
    is the most optimistic detectable effect and is used only to record an
    honest power/sensitivity statement for the frozen budget.
    """
    if k_per_arm < 1:
        raise ValueError("k_per_arm must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    best = 1.0
    found = False
    for x in range(k_per_arm + 1):
        for y in range(k_per_arm + 1):
            if x == y:
                continue
            if fisher_exact_two_sided(x, y, k_per_arm) <= alpha:
                found = True
                diff = abs(x / k_per_arm - y / k_per_arm)
                best = min(best, diff)
    if not found:
        return 1.0
    return round(best, 6)


def classify_issue_3275_decision(
    *,
    proposal_yield: float,
    random_yield: float,
    minimally_important: float,
    null_rejected: bool,
    powered: bool,
    independent_available: bool,
) -> dict[str, Any]:
    """Frozen ``continue | stop | inconclusive`` decision for issue #3275.

    The decision follows independent native planner-execution outcomes only.
    Archive-nearness can never reach this function as a driver; it lives in a
    diagnostic-only namespace. The vocabulary is exactly
    :data:`ISSUE_3275_DECISION_VOCABULARY`: there is no ``revise`` and no
    generic ``blocked``.

    Decision table:

    * independent outcomes unavailable/fail-closed -> ``inconclusive``;
    * outcomes valid but underpowered for the minimally important effect ->
      ``inconclusive`` (before any continue/stop outcome sign is considered);
    * powered outcomes whose null is not rejected -> ``inconclusive``;
    * powered, null-rejected outcomes with
      ``proposal_yield - random_yield <= 0`` -> ``stop``;
    * otherwise (delta >= minimally important and null rejected and powered) ->
      ``continue``.

    Args:
        proposal_yield: Candidate-level certified failure yield in the proposal
            arm (fraction in [0, 1]).
        random_yield: Candidate-level certified failure yield in the random arm.
        minimally_important: Frozen minimally important absolute yield
            improvement.
        null_rejected: Whether the predeclared Fisher-exact null is rejected.
        powered: Whether the frozen budget can detect the minimally important
            effect.
        independent_available: Whether valid independent native execution
            outcomes are available (not fail-closed).

    Returns:
        A JSON-safe dict with ``status`` in the frozen vocabulary and a reason.
    """
    delta = float(proposal_yield) - float(random_yield)
    if not independent_available:
        status = "inconclusive"
        reason = "independent_outcomes_unavailable_or_fail_closed"
    elif not powered:
        status = "inconclusive"
        reason = "underpowered_for_minimally_important_effect"
    elif not null_rejected:
        status = "inconclusive"
        reason = "null_not_rejected"
    elif delta <= 0.0:
        status = "stop"
        reason = "proposal_does_not_beat_random"
    elif delta >= float(minimally_important):
        status = "continue"
        reason = "proposal_beats_random_beyond_minimally_important_effect"
    else:
        status = "inconclusive"
        reason = "positive_but_below_minimally_important_effect"
    if status not in ISSUE_3275_DECISION_VOCABULARY:  # defensive; should never fire
        raise AssertionError(f"non-frozen decision status: {status!r}")
    return {
        "status": status,
        "reason": reason,
        "vocabulary": list(ISSUE_3275_DECISION_VOCABULARY),
        "evidence_tier": "diagnostic_only",
        "follows": "independent_planner_execution_outcomes",
        "claim_boundary": (
            "issue #3275/#2921 decision from independent planner-execution rows only; "
            "not benchmark, paper, or planner-performance evidence"
        ),
    }
