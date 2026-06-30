#!/usr/bin/env python3
"""Extract diagnostic readiness packet for issue #2557 replica evidence.

This helper reads already-tracked public evidence only. It never submits Slurm
work, reads private credentials, promotes raw artifacts, or claims benchmark
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "issue-2557-replica-readiness-packet.v1"
DEFAULT_SUMMARY = Path(
    "docs/context/evidence/issue_2557_reward_curriculum_partial_2026-06-08/seed_summary.json"
)
DEFAULT_RECOVERED_NOTE = Path("docs/context/issue_2557_recovered_diagnostic_seeds.md")
DEFAULT_OUTPUT = Path(
    "docs/context/evidence/issue_2557_replica_readiness_packet_2026-06-29/packet.json"
)
DEFAULT_GENERATED_AT = "2026-06-29T00:00:00+02:00"

CLAIM_BOUNDARY = (
    "diagnostic-only readiness packet for issue #2557 fixed-seed queue-fill "
    "replica evidence; no Slurm submission, no benchmark-success claim, no "
    "paper or dissertation claim promotion"
)

RECOVERED_DIAGNOSTIC_ROWS = [
    {
        "job_id": 13024,
        "issue": 3266,
        "seed": 509,
        "success_rate_ci95": "0.831 [0.810, 0.851]",
        "collision_rate": 0.167,
        "snqi": -0.0692,
        "status": "diagnostic_manifest_recovered",
    },
    {
        "job_id": 12949,
        "issue": 2919,
        "seed": 506,
        "success_rate_ci95": "0.851 [0.831, 0.870]",
        "collision_rate": 0.144,
        "snqi": 0.0169,
        "status": "diagnostic_manifest_recovered",
    },
    {
        "job_id": 12950,
        "issue": 3203,
        "seed": 508,
        "success_rate_ci95": "0.810 [0.790, 0.831]",
        "collision_rate": 0.187,
        "snqi": -0.1115,
        "status": "diagnostic_manifest_recovered",
    },
]


def build_packet(
    summary_path: Path = DEFAULT_SUMMARY,
    recovered_note: Path = DEFAULT_RECOVERED_NOTE,
    *,
    generated_at: str = DEFAULT_GENERATED_AT,
) -> dict[str, Any]:
    """Build issue #2557 readiness packet from tracked evidence surfaces."""

    summary = _load_seed_summary(summary_path)
    rows = sorted(summary["rows"], key=lambda item: int(item["seed"]))
    retrieved_seeds = [int(item["seed"]) for item in rows]
    retrieved_jobs = [
        {
            "job_id": int(item["job_id"]),
            "seed": int(item["seed"]),
            "partition": item.get("partition"),
            "status": "completed_retrieved_compact_summary",
            "success_rate": item.get("success_rate"),
            "collision_rate": item.get("collision_rate"),
            "snqi": item.get("snqi"),
            "wandb_url": item.get("wandb_url"),
        }
        for item in rows
    ]

    recovered_note_present = recovered_note.exists()
    recovered_seeds = [int(item["seed"]) for item in RECOVERED_DIAGNOSTIC_ROWS]
    original_missing = [int(seed) for seed in summary.get("incomplete_or_pending_seeds", [])]
    still_missing_or_unpromoted = sorted(set(original_missing) - set(recovered_seeds))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "issue": 2557,
        "status": "diagnostic_only_blocked_artifact_promotion",
        "claim_boundary": CLAIM_BOUNDARY,
        "scheduler_snapshot": {
            "latest_public_source": (
                "GitHub issue #2557 comment 2026-06-23T15:24:03Z; all known "
                "jobs were reported terminal, with no running or pending jobs."
            ),
            "running_jobs": [],
            "pending_jobs": [],
            "terminal_status_basis": "issue_comment_not_live_squeue",
            "terminal_job_groups": [
                {
                    "jobs": "12769-12782",
                    "status": "terminal",
                    "source": "issue #2557 comment 2026-06-23",
                },
                {
                    "jobs": "12917,12931,12932",
                    "status": "reached_10m_then_manifest_serialization_failed",
                    "source": "issue #2557 comment 2026-06-23",
                },
                {
                    "jobs": "12916",
                    "status": "diagnostic_artifacts_only_no_final_metrics",
                    "source": "issue #2557 comment 2026-06-23",
                },
            ],
        },
        "completed_jobs": retrieved_jobs,
        "retrieved_evidence": {
            "compact_seed_summary": str(summary_path),
            "compact_seed_summary_generated_utc": summary.get("generated_utc"),
            "compact_seed_count": len(retrieved_seeds),
            "compact_seeds": retrieved_seeds,
            "compact_aggregate": summary.get("aggregate", {}),
            "recovered_note": str(recovered_note),
            "recovered_note_present": recovered_note_present,
            "recovered_diagnostic_rows": RECOVERED_DIAGNOSTIC_ROWS,
        },
        "evidence_gap": {
            "unpromoted_or_missing_seeds": still_missing_or_unpromoted,
            "artifact_promotion_missing": True,
            "durable_pointer_gap": (
                "Raw Slurm logs, checkpoints, W&B payloads, episode logs, and "
                "per-scenario evaluations are not mirrored here; compact public "
                "evidence lacks finalizer manifests or durable artifact URI "
                "pointers for every terminal job."
            ),
            "claim_blockers": [
                "Tracked evidence is explicitly partial or diagnostic-only.",
                "Some reported terminal jobs have no compact public metrics or finalizer manifest.",
                "Recovered diagnostic rows have marginal/non-positive SNQI and elevated collisions.",
                "No fresh live Slurm state is collected by this CPU-only packet.",
                "No benchmark, ranking, paper, or dissertation claim should be promoted from this packet.",
            ],
        },
        "candidate_queue_entry": {
            "kind": "local_artifact_promotion_finalizer_audit",
            "submission_recommendation": "no_new_slurm_queue_fill",
            "state": "blocked_until_raw_artifacts_or_durable_pointers_available",
            "next_queue_item": (
                "Run a local artifact-finalization audit against existing #2557 "
                "raw outputs or durable artifact pointers; do not submit more replicas."
            ),
        },
        "cost_risk": {
            "local_packet_generation": "low; CPU-only tracked-file read",
            "local_artifact_finalizer_audit": (
                "moderate; needs access to retained raw output or durable artifact pointers"
            ),
            "additional_slurm_submission": (
                "high and not authorized here; duplicates likely because issue "
                "comments report known jobs terminal"
            ),
            "claim_promotion": "high; blocked by diagnostic status and evidence gaps",
        },
        "go_no_go": {
            "new_slurm_submission": "NO-GO",
            "new_slurm_reason": (
                "Known #2557 jobs are reported terminal and the next useful work "
                "is compact artifact refresh/promotion, not another queue-fill run."
            ),
            "local_public_packet": "GO",
            "exact_go_command": (
                "uv run python scripts/validation/"
                "extract_issue_2557_replica_readiness_packet.py --markdown"
            ),
            "exact_write_command": (
                "uv run python scripts/validation/"
                "extract_issue_2557_replica_readiness_packet.py --write-json "
                "docs/context/evidence/issue_2557_replica_readiness_packet_2026-06-29/"
                "packet.json"
            ),
        },
    }


def render_markdown(packet: dict[str, Any]) -> str:
    """Render a compact public Markdown packet."""

    lines = [
        "# Issue #2557 Replica Readiness Packet",
        "",
        "This packet summarizes the public tracked state for fixed-seed queue-fill "
        "replica evidence and keeps the result diagnostic-only.",
        "",
        "## Claim Boundary",
        "",
        f"- Status: `{packet['status']}`",
        f"- Boundary: {packet['claim_boundary']}",
        "",
        "## Completed / Running Jobs",
        "",
        f"- Running jobs: `{packet['scheduler_snapshot']['running_jobs']}`",
        f"- Pending jobs: `{packet['scheduler_snapshot']['pending_jobs']}`",
        f"- Latest scheduler source: {packet['scheduler_snapshot']['latest_public_source']}",
        f"- Compact completed-job rows: {len(packet['completed_jobs'])}",
        "",
        "## Retrieved Evidence",
        "",
        f"- Compact summary: `{packet['retrieved_evidence']['compact_seed_summary']}`",
        f"- Compact seeds: `{packet['retrieved_evidence']['compact_seeds']}`",
        f"- Aggregate: `{packet['retrieved_evidence']['compact_aggregate']}`",
        f"- Recovered diagnostic note: `{packet['retrieved_evidence']['recovered_note']}`",
        "",
        "## Evidence Gap",
        "",
        f"- Unpromoted or missing seeds: `{packet['evidence_gap']['unpromoted_or_missing_seeds']}`",
        f"- Durable pointer gap: {packet['evidence_gap']['durable_pointer_gap']}",
    ]
    for blocker in packet["evidence_gap"]["claim_blockers"]:
        lines.append(f"- {blocker}")

    queue = packet["candidate_queue_entry"]
    risk = packet["cost_risk"]
    decision = packet["go_no_go"]
    lines.extend(
        [
            "",
            "## Candidate Queue Entry",
            "",
            f"- Kind: `{queue['kind']}`",
            f"- State: `{queue['state']}`",
            f"- Recommendation: `{queue['submission_recommendation']}`",
            f"- Next item: {queue['next_queue_item']}",
            "",
            "## Cost / Risk",
            "",
            f"- Local packet generation: {risk['local_packet_generation']}",
            f"- Local artifact-finalizer audit: {risk['local_artifact_finalizer_audit']}",
            f"- Additional Slurm submission: {risk['additional_slurm_submission']}",
            f"- Claim promotion: {risk['claim_promotion']}",
            "",
            "## Go / No-Go",
            "",
            f"- New Slurm submission: `{decision['new_slurm_submission']}`",
            f"- Reason: {decision['new_slurm_reason']}",
            f"- Local public packet: `{decision['local_public_packet']}`",
            f"- Exact command: `{decision['exact_go_command']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_seed_summary(path: Path) -> dict[str, Any]:
    """Load and lightly validate the tracked #2557 seed summary."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "issue_2557_reward_curriculum_partial.v2":
        raise ValueError(f"{path} does not look like issue #2557 seed summary v2")
    if not isinstance(payload.get("rows"), list) or not payload["rows"]:
        raise ValueError(f"{path} must contain non-empty rows")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--recovered-note", type=Path, default=DEFAULT_RECOVERED_NOTE)
    parser.add_argument("--generated-at", default=DEFAULT_GENERATED_AT)
    parser.add_argument("--markdown", action="store_true", help="Print Markdown instead of JSON.")
    parser.add_argument("--write-json", type=Path, help="Write packet JSON to path.")
    parser.add_argument("--write-markdown", type=Path, help="Write packet Markdown to path.")
    args = parser.parse_args(argv)

    packet = build_packet(
        args.summary,
        args.recovered_note,
        generated_at=args.generated_at,
    )
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    if args.write_markdown:
        args.write_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.write_markdown.write_text(render_markdown(packet), encoding="utf-8")
    if args.markdown:
        print(render_markdown(packet), end="")
    else:
        print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
