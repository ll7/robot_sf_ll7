"""Pre-planner certification gate for adversarial candidate batches (issue #2920).

Rejects invalid, degenerate, or duplicate generated candidates *before* planner
execution, so they are never counted as useful stress tests. This reuses the
per-candidate quality records computed by
:mod:`robot_sf.adversarial.manifest_quality` (``status`` plus
``normalized_control_hash``) rather than recomputing any metrics.

Boundary: this gate covers **pre-execution** rejection categories only —
schema/parse *invalidity*, *degeneracy* (no effective perturbation), and
*intra-batch duplicates*. The post-execution categories the issue also asks to
distinguish — simulation failure, fallback, and degraded benchmark rows — are
produced by running the planner and are classified by the fail-closed benchmark
row-status policy, not here. Emitting them would require planner execution,
which this gate exists to avoid for rejected candidates.

This is quality-signal-only evidence: it measures generator/candidate health and
makes no planner benchmark claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from robot_sf.adversarial.manifest_quality import (
    EVIDENCE_BOUNDARY,
    load_adversarial_manifest_quality_records,
    summarize_adversarial_manifest_quality_records,
)

if TYPE_CHECKING:
    from robot_sf.adversarial.manifest_quality import (
        ManifestQualityRecord,
        ManifestsQualitySummary,
    )


ADVERSARIAL_CANDIDATE_QUALITY_SCHEMA = "adversarial_candidate_quality.v1"

# Pre-planner rejection reason codes.
REASON_INVALID = "invalid"
REASON_DEGENERATE = "degenerate"
REASON_DUPLICATE = "duplicate"


@dataclass(frozen=True)
class BatchCertificationPolicy:
    """Policy controlling which candidates a batch certification rejects.

    Attributes:
        reject_statuses: Manifest statuses that fail certification (default
            ``invalid`` and ``degenerate``).
        reject_duplicates: Reject later candidates that repeat an earlier
            candidate's normalized control hash within the batch.
        min_batch_validity_rate: Optional batch-level gate; when set, the whole
            batch is rejected if its validity rate falls below this threshold.
    """

    reject_statuses: tuple[str, ...] = (REASON_INVALID, REASON_DEGENERATE)
    reject_duplicates: bool = True
    min_batch_validity_rate: float | None = None

    def __post_init__(self) -> None:
        """Validate policy values that affect emitted certification payloads."""
        if self.min_batch_validity_rate is None:
            return
        if not math.isfinite(self.min_batch_validity_rate):
            raise ValueError("min_batch_validity_rate must be finite")
        if not 0.0 <= self.min_batch_validity_rate <= 1.0:
            raise ValueError("min_batch_validity_rate must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe policy settings used by a certification run."""
        return {
            "reject_statuses": list(self.reject_statuses),
            "reject_duplicates": self.reject_duplicates,
            "min_batch_validity_rate": self.min_batch_validity_rate,
        }


@dataclass(frozen=True)
class CandidateCertification:
    """Per-candidate certification decision."""

    index: int
    path: str
    status: str
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation.

        Returns:
            Mapping with the candidate index, path, status, decision, and reasons.
        """
        return {
            "index": self.index,
            "path": self.path,
            "status": self.status,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class BatchCertification:
    """Outcome of certifying a batch of candidates before planner execution."""

    accepted: bool
    total: int
    accepted_count: int
    rejected_count: int
    rejection_counts: dict[str, int]
    validity_rate: float
    candidates: list[CandidateCertification]
    policy: BatchCertificationPolicy
    quality_summary: ManifestsQualitySummary | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the ``adversarial_candidate_quality.v1`` JSON-safe payload.

        Returns:
            Mapping with schema version, evidence boundary, batch decision,
            counts, and per-candidate decisions.
        """
        payload: dict[str, object] = {
            "schema_version": ADVERSARIAL_CANDIDATE_QUALITY_SCHEMA,
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "batch_accepted": self.accepted,
            "total": self.total,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "validity_rate": self.validity_rate,
            "rejection_counts": dict(self.rejection_counts),
            "policy": self.policy.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        if self.quality_summary is not None:
            payload["quality_summary"] = self.quality_summary.to_dict()
        return payload


def certify_records(
    records: list[ManifestQualityRecord],
    policy: BatchCertificationPolicy | None = None,
) -> BatchCertification:
    """Apply the pre-planner gate to already-parsed quality records.

    Pure function (no IO): the first occurrence of a control hash is kept; later
    repeats are rejected as duplicates when the policy enables it. Returns the
    full batch decision.

    Returns:
        A :class:`BatchCertification` with per-candidate and batch-level results.
    """
    policy = policy or BatchCertificationPolicy()
    seen_hashes: set[str] = set()
    candidates: list[CandidateCertification] = []
    rejection_counts: Counter[str] = Counter()
    accepted_count = 0
    valid_count = 0

    for index, record in enumerate(records):
        reasons: list[str] = []
        if record.status == "valid":
            valid_count += 1
        if record.status in policy.reject_statuses:
            reasons.append(record.status)
        if policy.reject_duplicates and record.normalized_control_hash is not None:
            if record.normalized_control_hash in seen_hashes:
                reasons.append(REASON_DUPLICATE)
            else:
                seen_hashes.add(record.normalized_control_hash)

        accepted = not reasons
        accepted_count += int(accepted)
        for reason in reasons:
            rejection_counts[reason] += 1
        candidates.append(
            CandidateCertification(
                index=index,
                path=record.path,
                status=record.status,
                accepted=accepted,
                reasons=tuple(reasons),
            )
        )

    total = len(records)
    validity_rate = valid_count / total if total else 0.0
    batch_accepted = total > 0 and accepted_count > 0
    if policy.min_batch_validity_rate is not None:
        batch_accepted = batch_accepted and validity_rate >= policy.min_batch_validity_rate

    return BatchCertification(
        accepted=batch_accepted,
        total=total,
        accepted_count=accepted_count,
        rejected_count=total - accepted_count,
        rejection_counts=dict(rejection_counts),
        validity_rate=validity_rate,
        candidates=candidates,
        policy=policy,
    )


def certify_candidate_batch(
    manifest_inputs: list[str | Path],
    policy: BatchCertificationPolicy | None = None,
    *,
    reference_manifest: str | Path | None = None,
    include_quality_summary: bool = True,
) -> BatchCertification:
    """Load a batch of candidate manifests and certify them before planner runs.

    Reuses the shared manifest parser for status/hash and, when requested, embeds
    the full :func:`summarize_adversarial_manifest_quality` summary for provenance.

    Returns:
        A :class:`BatchCertification` (optionally carrying the quality summary).
    """
    records = load_adversarial_manifest_quality_records(
        manifest_inputs,
        reference_manifest=reference_manifest,
    )
    certification = certify_records(records, policy)
    if not include_quality_summary:
        return certification
    summary = summarize_adversarial_manifest_quality_records(
        records,
        reference_manifest=reference_manifest,
    )
    return replace(certification, quality_summary=summary)


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI parser for the pre-planner batch-certification gate."""
    parser = argparse.ArgumentParser(
        description="Certify adversarial candidate batches before planner execution."
    )
    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="Candidate manifest YAML files and/or directories.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=None,
        help="Optional reference manifest used for perturbation-distance provenance.",
    )
    parser.add_argument(
        "--min-batch-validity-rate",
        type=float,
        default=None,
        help="Optional batch-level minimum validity rate required to accept the batch.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep later duplicate control hashes instead of rejecting them.",
    )
    parser.add_argument(
        "--skip-quality-summary",
        action="store_true",
        help="Skip embedding the manifest-quality provenance summary in the output payload.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON output to this path; if omitted, print to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes:
        0: batch accepted.
        1: input/IO/parse error.
        2: batch rejected by the certification gate.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    policy = BatchCertificationPolicy(
        reject_duplicates=not args.allow_duplicates,
        min_batch_validity_rate=args.min_batch_validity_rate,
    )
    try:
        certification = certify_candidate_batch(
            manifest_inputs=[*args.manifests],
            policy=policy,
            reference_manifest=args.reference_manifest,
            include_quality_summary=not args.skip_quality_summary,
        )
        payload = certification.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True)
        output = text
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(text + "\n", encoding="utf-8")
            output = args.output_json.as_posix()
        sys.stdout.write(f"{output}\n")
        return 0 if certification.accepted else 2
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
