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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Permutation-test float comparison tolerance.
_EPS = 1e-12
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


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


def _verify_contract_hashes(
    contract_data: dict[str, Any],
    archive_data: dict[str, Any],
    archive_raw_content: bytes | None,
) -> list[str]:
    """Check raw file and archive payload hashes against contract."""
    reasons: list[str] = []
    if archive_raw_content is not None and "archive_raw_sha256" in contract_data:
        raw_hash = hashlib.sha256(archive_raw_content).hexdigest()
        if raw_hash != contract_data["archive_raw_sha256"]:
            reasons.append(
                f"raw_archive_hash_mismatch:expected={contract_data['archive_raw_sha256']}:actual={raw_hash}"
            )
    if "archive_payload_sha256" in contract_data:
        payload_hash = archive_sha256(archive_data)
        if payload_hash != contract_data["archive_payload_sha256"]:
            reasons.append(
                f"archive_payload_hash_mismatch:expected={contract_data['archive_payload_sha256']}:actual={payload_hash}"
            )
    return reasons


def _load_recertification_lineage(
    lineage: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load a content-addressed recertification artifact, or return blockers."""
    if not isinstance(lineage, dict):
        return None, ["recertification_lineage_missing"]
    path_value = lineage.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None, ["recertification_path_missing"]
    path = Path(path_value)
    if not path.is_file():
        return None, [f"recertification_path_missing:{path}"]

    raw_content = path.read_bytes()
    reasons = []
    if hashlib.sha256(raw_content).hexdigest() != lineage.get("file_sha256"):
        reasons.append("recertification_file_hash_mismatch")
    try:
        recertification = json.loads(raw_content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, reasons + ["recertification_payload_unreadable"]
    if not isinstance(recertification, dict):
        return None, reasons + ["recertification_payload_not_object"]
    return recertification, reasons


def _verify_recertification_identity(
    recertification: dict[str, Any], lineage: dict[str, Any], contract_data: dict[str, Any]
) -> list[str]:
    """Verify recertification identity, archive binding, and correction status."""
    reasons: list[str] = []
    if recertification.get("recertification_sha256") != lineage.get("payload_sha256"):
        reasons.append("recertification_payload_hash_mismatch")
    if recertification.get("archive_sha256") != lineage.get("archive_sha256"):
        reasons.append("recertification_archive_hash_mismatch")
    if recertification.get("archive_sha256") != contract_data.get("archive_raw_sha256"):
        reasons.append("recertification_archive_not_bound_to_contract")
    if recertification.get("issue") != lineage.get("issue"):
        reasons.append("recertification_issue_mismatch")
    if recertification.get("schema_version") != "issue_6139_recertification.v1":
        reasons.append("recertification_schema_invalid")
    correction = recertification.get("correction")
    if not isinstance(correction, dict) or correction.get("accepted_archive_modified") is not False:
        reasons.append("recertification_archive_modification_status_invalid")
    return reasons


def _verify_recertified_fit_eligibility(
    recertification: dict[str, Any], lineage: dict[str, Any], contract_data: dict[str, Any]
) -> list[str]:
    """Require every fit ID to remain benchmark-eligible after recertification."""
    reasons: list[str] = []
    if lineage.get("fit_entry_eligibility_policy") != "eligible_only":
        return ["fit_entry_eligibility_policy_invalid"]
    records = recertification.get("records")
    if not isinstance(records, list):
        return ["recertification_records_missing"]
    records_by_id = {
        record.get("archive_id"): record for record in records if isinstance(record, dict)
    }
    for fit_entry_id in contract_data.get("fit_entry_ids", []):
        record = records_by_id.get(fit_entry_id)
        if not isinstance(record, dict):
            reasons.append(f"recertification_fit_entry_missing:{fit_entry_id}")
            continue
        if record.get("status") != "unchanged":
            reasons.append(f"recertification_fit_entry_status_invalid:{fit_entry_id}")
        after = record.get("after")
        eligibility = after.get("benchmark_eligibility") if isinstance(after, dict) else None
        if eligibility != "eligible":
            reasons.append(f"fit_entry_not_benchmark_eligible:{fit_entry_id}:{eligibility}")
    return reasons


def _verify_recertification_lineage(contract_data: dict[str, Any]) -> list[str]:
    """Bind fit eligibility to the merged #6139 recertification artifact."""
    lineage = contract_data.get("recertification")
    recertification, reasons = _load_recertification_lineage(lineage)
    if recertification is None or not isinstance(lineage, dict):
        return reasons
    return (
        reasons
        + _verify_recertification_identity(recertification, lineage, contract_data)
        + _verify_recertified_fit_eligibility(recertification, lineage, contract_data)
    )


def _is_sha256(value: Any) -> bool:
    """Return whether ``value`` is a non-placeholder SHA-256 digest."""
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _verify_contract_families_and_hashes(contract_data: dict[str, Any]) -> list[str]:
    """Validate held-out family identity and planner/search-space provenance."""
    reasons: list[str] = []
    fit_family = contract_data.get("fit_scenario_family")
    eval_family = contract_data.get("eval_scenario_family")
    if eval_family != "classic_cross_trap_medium":
        reasons.append(
            f"eval_scenario_family_mismatch:expected=classic_cross_trap_medium:actual={eval_family}"
        )
    if not isinstance(fit_family, str) or not fit_family.strip():
        reasons.append("fit_scenario_family_missing")
    elif eval_family == fit_family:
        reasons.append("held_out_scenario_family_not_distinct")

    planner_hash = contract_data.get("target_planner_config_sha256")
    if not _is_sha256(planner_hash):
        reasons.append("target_planner_config_sha256_invalid")

    search_space_path = contract_data.get("search_space_path")
    search_space_hash = contract_data.get("search_space_sha256")
    if not isinstance(search_space_path, str) or not search_space_path.strip():
        reasons.append("search_space_path_missing")
    elif not _is_sha256(search_space_hash):
        reasons.append("search_space_sha256_invalid")
    else:
        path = Path(search_space_path)
        if not path.is_file():
            reasons.append(f"search_space_path_missing:{path}")
        else:
            observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed_hash != search_space_hash:
                reasons.append(
                    "search_space_hash_mismatch:"
                    f"expected={search_space_hash}:actual={observed_hash}"
                )

    return reasons


def _verify_power_sensitivity(power: Any, budget: Any, effect_margin: Any) -> list[str]:
    """Validate a predeclared power/sensitivity calculation when available."""
    if not isinstance(power, dict):
        return ["power_sensitivity_missing"]
    if power.get("status") != "computed":
        return [f"power_sensitivity_not_computed:{power.get('status', 'missing')}"]

    reasons: list[str] = []
    if power.get("candidate_budget_per_arm") != budget:
        reasons.append("power_sensitivity_budget_mismatch")
    if power.get("minimum_effect_margin") != effect_margin:
        reasons.append("power_sensitivity_effect_margin_mismatch")
    estimated_power = power.get("estimated_power")
    if not isinstance(estimated_power, (int, float)) or not 0.0 <= estimated_power <= 1.0:
        reasons.append("power_sensitivity_estimated_power_invalid")
    return reasons


def _verify_study_parameters(contract_data: dict[str, Any]) -> list[str]:
    """Validate budget, effect-size, overlap, and power contract fields."""
    reasons: list[str] = []
    parameters = contract_data.get("study_parameters")
    if not isinstance(parameters, dict):
        reasons.append("study_parameters_missing")
        parameters = {}

    budget = parameters.get("candidate_budget_per_arm")
    pool_size = parameters.get("candidate_pool_size")
    confirmations = parameters.get("confirmation_seeds_per_candidate")
    effect_margin = parameters.get("minimally_important_effect_margin")
    if not isinstance(budget, int) or budget < 1:
        reasons.append("candidate_budget_per_arm_invalid")
    if not isinstance(pool_size, int) or not isinstance(budget, int) or pool_size < 2 * budget:
        reasons.append("candidate_pool_size_invalid")
    if not isinstance(confirmations, int) or confirmations < 1:
        reasons.append("confirmation_seeds_per_candidate_invalid")
    if not isinstance(effect_margin, (int, float)) or not 0.0 < float(effect_margin) <= 1.0:
        reasons.append("minimally_important_effect_margin_invalid")
    if parameters.get("overlapping_candidate_policy") != "deterministic_disjoint_assignment":
        reasons.append("overlapping_candidate_policy_not_deterministic_disjoint_assignment")

    reasons.extend(
        _verify_power_sensitivity(contract_data.get("power_sensitivity"), budget, effect_margin)
    )
    return reasons


def _verify_admission_shape(admission: Any) -> tuple[list[str], dict[str, Any]]:
    """Validate required outcome-admission fields and return a mapping."""
    reasons: list[str] = []
    if not isinstance(admission, dict):
        reasons.append("outcome_admission_missing")
        admission = {}
    if admission.get("schema_version") != "adversarial_independent_outcomes.v2":
        reasons.append("outcome_admission_schema_invalid")
    if admission.get("execution_status") != "native":
        reasons.append("outcome_admission_requires_native_execution")
    if admission.get("candidate_aggregation") != "candidate_level_binary_yield":
        reasons.append("outcome_admission_candidate_aggregation_invalid")
    if admission.get("fallback_degraded_policy") != "exclude":
        reasons.append("outcome_admission_fallback_policy_invalid")
    if admission.get("stable_failure_attribution_required") is not True:
        reasons.append("outcome_admission_stable_attribution_not_required")
    return reasons, admission


def _verify_admission_confirmation(admission: dict[str, Any], confirmations: Any) -> list[str]:
    """Validate deterministic replay and the predeclared confirmation threshold."""
    reasons: list[str] = []

    replay = admission.get("deterministic_replay")
    if not isinstance(replay, dict) or replay.get("exact_signature_match_required") is not True:
        reasons.append("outcome_admission_deterministic_replay_invalid")
    confirmation = admission.get("independent_seed_confirmation")
    if not isinstance(confirmation, dict):
        reasons.append("outcome_admission_independent_confirmation_missing")
    else:
        required_count = confirmation.get("minimum_confirmed_count")
        if not isinstance(required_count, int) or not isinstance(confirmations, int):
            reasons.append("outcome_admission_confirmation_threshold_invalid")
        elif not 1 <= required_count <= confirmations:
            reasons.append("outcome_admission_confirmation_threshold_invalid")
    return reasons


def _verify_outcome_admission(contract_data: dict[str, Any]) -> list[str]:
    """Validate native, candidate-level admission and frozen-manifest requirements."""
    parameters = contract_data.get("study_parameters")
    confirmations = (
        parameters.get("confirmation_seeds_per_candidate") if isinstance(parameters, dict) else None
    )
    reasons, admission = _verify_admission_shape(contract_data.get("outcome_admission"))
    reasons.extend(_verify_admission_confirmation(admission, confirmations))

    manifest = admission.get("candidate_manifest")
    if not isinstance(manifest, dict) or manifest.get("status") != "frozen":
        manifest_status = (
            manifest.get("status", "missing") if isinstance(manifest, dict) else "missing"
        )
        reasons.append(f"candidate_manifest_not_frozen:{manifest_status}")
    return reasons


def _verify_decision_and_state(contract_data: dict[str, Any]) -> list[str]:
    """Validate the exact decision vocabulary and external gate state."""
    reasons: list[str] = []

    decision_rule = contract_data.get("decision_rule")
    expected_decisions = {
        "positive_outcome": "continue",
        "negative_outcome": "stop",
        "neutral_outcome": "inconclusive",
        "invalid_or_fallback_outcome": "inconclusive",
    }
    if not isinstance(decision_rule, dict):
        reasons.append("decision_rule_missing")
    else:
        for key, expected in expected_decisions.items():
            if decision_rule.get(key) != expected:
                reasons.append(
                    f"decision_rule_mismatch:{key}:expected={expected}:actual={decision_rule.get(key)}"
                )

    if contract_data.get("contract_status") != "frozen":
        reasons.append(f"contract_not_frozen:{contract_data.get('contract_status', 'missing')}")
    prerequisites = contract_data.get("external_prerequisites")
    if not isinstance(prerequisites, list) or not prerequisites:
        reasons.append("external_prerequisites_missing")
    else:
        for prerequisite in prerequisites:
            if not isinstance(prerequisite, dict):
                reasons.append("external_prerequisite_invalid")
                continue
            if prerequisite.get("status") != "satisfied":
                reasons.append(
                    "external_prerequisite_unsatisfied:"
                    f"issue={prerequisite.get('issue', 'unknown')}:"
                    f"status={prerequisite.get('status', 'missing')}"
                )
    return reasons


def _verify_same_planner_design(contract_data: dict[str, Any]) -> list[str]:
    """Validate the non-archive fields that define the #3275 study contract.

    This check is deliberately strict: a contract may be used by the normal
    runner only after it defines the held-out cross-trap arm, reproducible
    planner/search-space provenance, deterministic arm handling, row admission,
    and the predeclared decision vocabulary.  A provisional contract reports
    its unmet external prerequisites rather than looking final merely because
    its source archive still parses.
    """
    return (
        _verify_contract_families_and_hashes(contract_data)
        + _verify_study_parameters(contract_data)
        + _verify_outcome_admission(contract_data)
        + _verify_decision_and_state(contract_data)
    )


def _verify_contract_entries(
    contract_data: dict[str, Any],
    entries: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Check fit/excluded entry IDs, planners, and scenario families against contract."""
    reasons: list[str] = []
    expected_target_planner = contract_data.get("target_planner")
    expected_planner_config_hash = contract_data.get("target_planner_config_sha256")
    expected_fit_family = contract_data.get("fit_scenario_family")
    expected_fit_ids = set(contract_data.get("fit_entry_ids", []))
    expected_excluded_ids = set(contract_data.get("excluded_entry_ids", []))
    expected_fit_count = contract_data.get("fit_entry_count", 12)

    actual_entry_ids = {e.get("archive_id") for e in entries if isinstance(e, dict)}
    missing_fit = sorted(expected_fit_ids - actual_entry_ids)
    if missing_fit:
        reasons.append(f"missing_fit_entries:{missing_fit}")

    missing_excluded = sorted(expected_excluded_ids - actual_entry_ids)
    if missing_excluded:
        reasons.append(f"missing_excluded_entries:{missing_excluded}")

    observed_fit_entries = [
        e for e in entries if isinstance(e, dict) and e.get("archive_id") in expected_fit_ids
    ]
    if len(observed_fit_entries) != expected_fit_count:
        reasons.append(
            f"fit_entry_count_mismatch:expected={expected_fit_count}:actual={len(observed_fit_entries)}"
        )

    for entry in observed_fit_entries:
        planner = (
            entry.get("target_planner")
            or (entry.get("provenance") or {}).get("target_planner")
            or (entry.get("mechanism_cluster_key") or {}).get("policy")
        )
        if planner != expected_target_planner:
            reasons.append(
                f"fit_entry_planner_mismatch:{entry.get('archive_id')}:expected={expected_target_planner}:actual={planner}"
            )
        provenance = entry.get("provenance")
        observed_config_hash = (
            provenance.get("config_sha256") if isinstance(provenance, dict) else None
        )
        if observed_config_hash != expected_planner_config_hash:
            reasons.append(
                "fit_entry_planner_config_hash_mismatch:"
                f"{entry.get('archive_id')}:expected={expected_planner_config_hash}:"
                f"actual={observed_config_hash}"
            )
        fam = entry.get("scenario_family") or entry.get("cluster_key")
        if isinstance(fam, dict):
            fam = fam.get("scenario_family")
        if fam != expected_fit_family:
            reasons.append(
                f"fit_entry_family_mismatch:{entry.get('archive_id')}:expected={expected_fit_family}:actual={fam}"
            )

    return reasons, observed_fit_entries


def verify_same_planner_contract(
    contract_data: dict[str, Any],
    archive_data: dict[str, Any],
    archive_raw_content: bytes | None = None,
) -> dict[str, Any]:
    """Verify that a loaded archive complies with the frozen same-planner contract."""
    blocking_reasons: list[str] = []
    blocking_reasons.extend(
        _verify_contract_hashes(contract_data, archive_data, archive_raw_content)
    )
    blocking_reasons.extend(_verify_recertification_lineage(contract_data))
    blocking_reasons.extend(_verify_same_planner_design(contract_data))

    entries = archive_data.get("entries", []) if isinstance(archive_data, dict) else []
    if not isinstance(entries, list):
        blocking_reasons.append("archive_entries_not_list")
        entries = []

    entry_reasons, observed_fit_entries = _verify_contract_entries(contract_data, entries)
    blocking_reasons.extend(entry_reasons)

    fit_payload_hash = archive_sha256(observed_fit_entries)
    if (
        "fit_entries_payload_sha256" in contract_data
        and fit_payload_hash != contract_data["fit_entries_payload_sha256"]
    ):
        blocking_reasons.append(
            f"fit_entries_payload_hash_mismatch:expected={contract_data['fit_entries_payload_sha256']}:actual={fit_payload_hash}"
        )

    passed = not blocking_reasons
    return {
        "schema_version": "issue_3275_contract_verification.v1",
        "status": "passed" if passed else "failed",
        "contract_id": contract_data.get("study_id", "issue_3275_same_planner_contract"),
        "target_planner": contract_data.get("target_planner"),
        "target_planner_config_sha256": contract_data.get("target_planner_config_sha256"),
        "fit_scenario_family": contract_data.get("fit_scenario_family"),
        "eval_scenario_family": contract_data.get("eval_scenario_family"),
        "search_space_sha256": contract_data.get("search_space_sha256"),
        "fit_entry_count": len(observed_fit_entries),
        "excluded_entry_count": contract_data.get("excluded_entry_count", 5),
        "fit_entries_payload_sha256": fit_payload_hash,
        "checks_passed": passed,
        "blocking_reasons": blocking_reasons,
    }
