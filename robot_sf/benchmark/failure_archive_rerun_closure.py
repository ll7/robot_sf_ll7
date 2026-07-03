"""Consolidated rerun closure packet for proposal-model failure-archive reruns.

Issue #3275 accumulated many bounded fail-closed readiness guards (archive
lineage, archive-ID leakage, duplicate IDs, certification metadata, overlap
metadata, null-test prerequisites). Each guard answers one narrow question and
they now all funnel into one canonical owner:

    ``robot_sf.benchmark.failure_archive_rerun_readiness`` — the pair gate that
    decides whether a source archive and a disjoint rerun archive can drive a
    non-circular rerun without leakage or missing certification.

This module adds no new gate. It *consolidates* that pair gate into one durable
closure packet so a reviewer can read a single artifact instead of re-deriving
the aggregate verdict from a stream of micro-guards. For a real disjoint
source/rerun archive pair the packet records:

- a single ``disposition`` (ready for rerun, blocked, or diagnostic-only),
- the consolidated blocker list drawn from the pair gate, and
- the explicit *next empirical action* required before any held-out yield claim.

The packet is intentionally fail-closed and never runs planners, benchmarks,
proposal sampling, or artifact publication. A missing or malformed archive
input produces a ``fail_closed_blocked`` packet, not a synthetic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.failure_archive_rerun_readiness import (
    BLOCKED,
    DIAGNOSTIC_ONLY,
    READY,
    classify_failure_archive_rerun_readiness,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = "failure_archive_rerun_closure_packet.v1"
CLAIM_BOUNDARY = (
    "consolidated readiness/leakage closure packet only; no benchmark campaign run, "
    "no proposal-model inference, no held-out yield claim, no dissertation or "
    "paper-facing claim promotion"
)

# The closure disposition is a 1:1 renaming of the pair-gate verdict into
# rerun-facing language. The pair gate is the sole authority; the packet only
# aggregates and annotates it with a next empirical action.
READY_FOR_RERUN = "ready_for_rerun"
FAIL_CLOSED_BLOCKED = "fail_closed_blocked"
DIAGNOSTIC_ONLY_DISPOSITION = "diagnostic_only"

_DISPOSITION_BY_PAIR_STATUS = {
    READY: READY_FOR_RERUN,
    BLOCKED: FAIL_CLOSED_BLOCKED,
    DIAGNOSTIC_ONLY: DIAGNOSTIC_ONLY_DISPOSITION,
}

# Next-action guidance, keyed by the dominant remaining blocker category. Order
# matters: the first matching category (by this tuple order) wins so the
# recommended action is deterministic for a given blocker set. Needles are
# substrings of the pair gate's blocker strings (see
# ``failure_archive_rerun_readiness``).
_NEXT_ACTION_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "missing_archive_input",
        (
            "path_missing",
            "file_empty",
            "unreadable",
            "payload_not_object",
            "archive_has_no_entries",
        ),
        "Provide a real populated failure archive for each side; the adversarial "
        "search input is absent or malformed. Build it with "
        "scripts/tools/curate_adversarial_failure_archive.py from #1488/#2920 search outputs.",
    ),
    (
        "archive_leakage",
        ("archive_id_overlap", "source_manifest_overlap", "duplicate_archive_ids"),
        "Provide a genuinely disjoint rerun archive: no shared archive IDs, no duplicate "
        "IDs within a side, and no shared config.source_manifests lineage between the "
        "source and rerun archives.",
    ),
    (
        "missing_certification",
        ("certification_metadata", "certification_status", "certification_lineage"),
        "Certify every emitted candidate with "
        "scripts/tools/certify_adversarial_candidate_batch.py so each source and rerun "
        "entry carries passing certification metadata before the rerun.",
    ),
    (
        "missing_overlap_metadata",
        ("missing_overlap_metadata", "missing_archive_lineage", "unexpected_schema_version"),
        "Repair archive entry and top-level metadata (archive_id, scenario family, "
        "candidate.scenario_seed, config.source_manifests) so a scenario-family-disjoint "
        "split is well defined on both sides.",
    ),
    (
        "missing_null_tests",
        ("null_test",),
        "Attach a valid null-test prerequisite report (shuffled-outcome and "
        "ranking-permutation p-values bound to the source/rerun archive SHA-256 pair) before "
        "any held-out yield interpretation.",
    ),
)

_READY_NEXT_ACTION = (
    "Inputs are disjoint and certified. Run "
    "scripts/adversarial/run_proposal_vs_random_issue_2921.py with independent "
    "planner-execution outcomes on the rerun side, then classify the held-out "
    "proposal-vs-random result under the issue #2921 continue/revise/stop rule."
)

_DIAGNOSTIC_NEXT_ACTION = (
    "Inputs pass leakage/certification checks, but the supplied rerun output is marked "
    "diagnostic-only. Re-run with a non-diagnostic proposal-vs-random report and "
    "independent planner-execution outcomes before treating the result as held-out evidence."
)

_FALLBACK_NEXT_ACTION = (
    "Resolve the reported blockers, then re-run the closure packet. Do not promote a "
    "held-out yield claim while any blocker remains."
)


@dataclass(frozen=True)
class RerunClosurePacket:
    """Consolidated fail-closed closure packet for a disjoint archive rerun."""

    disposition: str
    source_archive: str
    rerun_archive: str
    pair_readiness: dict[str, Any]
    consolidated_blockers: list[str] = field(default_factory=list)
    next_empirical_action: str = ""

    @property
    def ready(self) -> bool:
        """Return whether the pair is ready for an independent-outcome rerun."""

        return self.disposition == READY_FOR_RERUN

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable closure packet payload."""

        return {
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": CLAIM_BOUNDARY,
            "disposition": self.disposition,
            "ready": self.ready,
            "source_archive": self.source_archive,
            "rerun_archive": self.rerun_archive,
            "consolidated_blockers": list(self.consolidated_blockers),
            "next_empirical_action": self.next_empirical_action,
            "pair_readiness": dict(self.pair_readiness),
        }


def _derive_next_action(disposition: str, consolidated_blockers: list[str]) -> str:
    """Return the deterministic next empirical action for the packet.

    For a ready or diagnostic-only disposition the action is fixed. For a
    blocked disposition the first matching blocker category (in
    :data:`_NEXT_ACTION_RULES` order) selects the guidance so the recommendation
    is stable for a given blocker set rather than order-dependent on discovery.
    """

    if disposition == READY_FOR_RERUN:
        return _READY_NEXT_ACTION
    if disposition == DIAGNOSTIC_ONLY_DISPOSITION:
        return _DIAGNOSTIC_NEXT_ACTION

    joined = " ".join(consolidated_blockers)
    for _category, needles, action in _NEXT_ACTION_RULES:
        if any(needle in joined for needle in needles):
            return action
    return _FALLBACK_NEXT_ACTION


def build_rerun_closure_packet(
    source_archive: str | Path,
    rerun_archive: str | Path,
    *,
    rerun_output: str | Path | None = None,
    null_test_prerequisites: str | Path | dict[str, Any] | None = None,
) -> RerunClosurePacket:
    """Assemble a consolidated closure packet for a disjoint archive rerun.

    This composes the canonical pair gate
    (:func:`classify_failure_archive_rerun_readiness`). No new gate logic is
    introduced; the packet only aggregates the verdict and annotates it with a
    deterministic next empirical action.

    Args:
        source_archive: Archive used to fit/select/rank proposal-model candidates.
        rerun_archive: Separate archive intended for the rerun/evaluation slice.
        rerun_output: Optional rerun report JSON; diagnostic-only markers cap the
            disposition at ``diagnostic_only`` when inputs otherwise pass.
        null_test_prerequisites: Optional null-test prerequisite JSON path or
            payload required before any held-out claim.

    Returns:
        A fail-closed :class:`RerunClosurePacket`. The function never runs
        planners, benchmarks, proposal sampling, or artifact publication.
    """

    pair = classify_failure_archive_rerun_readiness(
        source_archive,
        rerun_archive,
        rerun_output=rerun_output,
        null_test_prerequisites=null_test_prerequisites,
    )

    consolidated = list(pair.blockers)
    disposition = _DISPOSITION_BY_PAIR_STATUS.get(pair.status, FAIL_CLOSED_BLOCKED)
    next_action = _derive_next_action(disposition, consolidated)

    return RerunClosurePacket(
        disposition=disposition,
        source_archive=str(source_archive),
        rerun_archive=str(rerun_archive),
        pair_readiness=pair.to_payload(),
        consolidated_blockers=consolidated,
        next_empirical_action=next_action,
    )
