#!/usr/bin/env python3
"""Build the issue #1554 Slurm evidence synthesis packet.

The input is a small public fixture distilled from completed Slurm job metadata and
retrieved benchmark summaries. It intentionally does not read private queue state,
submit jobs, or promote paper/dissertation claims.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLAIM_BLOCKING_WARNINGS = ("SNQI", "contract")


@dataclass(frozen=True, slots=True)
class JobEvidence:
    """Public, non-secret metadata for one completed Slurm job."""

    job_id: int
    campaign: str
    config: str
    slurm_state: str
    exit_code: str
    public_commit: str
    finding: str
    evidence_role: str
    artifact_summary: dict[str, Any]
    limitations: list[str]

    @property
    def completed_successfully(self) -> bool:
        """Whether Slurm completed without a process-level failure."""
        return self.slurm_state == "COMPLETED" and self.exit_code == "0:0"

    @property
    def has_claim_blocker(self) -> bool:
        """Whether the row carries a warning that blocks paper-grade interpretation."""
        text = " ".join(self.limitations + self.artifact_summary.get("warnings", []))
        return any(fragment in text for fragment in CLAIM_BLOCKING_WARNINGS)


def _load_jobs(path: Path) -> list[JobEvidence]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", payload)
    if not isinstance(jobs, list):
        raise ValueError(f"{path} must contain a JSON list or an object with a 'jobs' list")
    return [JobEvidence(**job) for job in jobs]


def build_packet(jobs: list[JobEvidence]) -> dict[str, Any]:
    """Build a deterministic next-decision packet from completed job evidence."""
    job_ids = [job.job_id for job in jobs]
    if job_ids != [13192, 13198, 13203]:
        raise ValueError(f"issue #1554 packet expects jobs [13192, 13198, 13203], got {job_ids}")

    successful_jobs = [job.job_id for job in jobs if job.completed_successfully]
    claim_blockers = [
        {"job": job.job_id, "limitations": job.limitations} for job in jobs if job.has_claim_blocker
    ]
    result_jobs = [
        job.job_id
        for job in jobs
        if job.evidence_role in {"result_matrix", "comparison_run"} and job.completed_successfully
    ]

    if 13198 in result_jobs:
        recommendation = (
            "Do not enqueue another duplicate S20/H500 planner-family run yet. Analyze job 13198 "
            "first, including the soft SNQI contract warning, then submit only a targeted follow-up "
            "for an explicit analysis gap or validated config/schema unblock."
        )
        status = "analysis_before_submit"
    else:
        recommendation = (
            "No result-producing S20/H500 matrix is available; prepare the smallest public config or "
            "checker unblock before another queue entry."
        )
        status = "blocked_until_result_matrix"

    return {
        "schema_version": "issue1554-slurm-evidence-packet.v1",
        "issue": 1554,
        "jobs_synthesized": job_ids,
        "status": status,
        "successful_jobs": successful_jobs,
        "claim_blockers": claim_blockers,
        "evidence": [
            {
                "job": job.job_id,
                "campaign": job.campaign,
                "config": job.config,
                "public_commit": job.public_commit,
                "slurm_state": job.slurm_state,
                "exit_code": job.exit_code,
                "role": job.evidence_role,
                "finding": job.finding,
                "artifact_summary": job.artifact_summary,
                "limitations": job.limitations,
            }
            for job in jobs
        ],
        "next_slurm_queue_recommendation": recommendation,
        "forbidden_actions_confirmed": {
            "compute_submit": False,
            "artifact_deletion": False,
            "paper_or_dissertation_claim_edits": False,
        },
    }


def render_markdown(packet: dict[str, Any]) -> str:
    """Render the synthesis packet for review in docs/context/evidence."""
    lines = [
        "# Issue #1554 Slurm Evidence Packet",
        "",
        "This packet synthesizes completed jobs 13192, 13198, and 13203. It is a "
        "queue-decision artifact only; it does not edit paper or dissertation claims.",
        "",
        f"- status: `{packet['status']}`",
        f"- jobs synthesized: `{', '.join(str(job) for job in packet['jobs_synthesized'])}`",
        f"- next queue recommendation: {packet['next_slurm_queue_recommendation']}",
        "",
        "## Job Findings",
        "",
    ]
    for item in packet["evidence"]:
        summary = item["artifact_summary"]
        lines.extend(
            [
                f"### Job {item['job']}",
                "",
                f"- campaign: `{item['campaign']}`",
                f"- config: `{item['config']}`",
                f"- public commit: `{item['public_commit']}`",
                f"- Slurm result: `{item['slurm_state']}` / `{item['exit_code']}`",
                f"- role: `{item['role']}`",
                f"- finding: {item['finding']}",
                f"- artifact summary: {json.dumps(summary, sort_keys=True)}",
                f"- limitations: {'; '.join(item['limitations']) or 'none recorded'}",
                "",
            ]
        )
    if packet["claim_blockers"]:
        lines.extend(["## Claim Blockers", ""])
        for blocker in packet["claim_blockers"]:
            lines.append(f"- job `{blocker['job']}`: {'; '.join(blocker['limitations'])}")
        lines.append("")
    lines.extend(
        [
            "## Forbidden Actions",
            "",
            "- no Slurm/GPU submission",
            "- no artifact deletion",
            "- no paper/dissertation claim edits",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Run the command-line packet builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Job evidence fixture JSON.")
    parser.add_argument("--output-json", type=Path, required=True, help="Packet JSON output path.")
    parser.add_argument(
        "--output-md", type=Path, required=True, help="Packet Markdown output path."
    )
    args = parser.parse_args()

    jobs = _load_jobs(args.input)
    packet = build_packet(jobs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(packet), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
