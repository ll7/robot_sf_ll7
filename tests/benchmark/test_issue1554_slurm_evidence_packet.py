"""Tests for the issue #1554 Slurm evidence synthesis packet builder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "benchmark" / "build_issue1554_slurm_evidence_packet.py"
)

spec = importlib.util.spec_from_file_location("issue1554_slurm_evidence_packet", SCRIPT)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

build_packet = module.build_packet
render_markdown = module.render_markdown
_load_jobs = module._load_jobs


FIXTURE = (
    Path(__file__).parent / "fixtures" / "issue1554_slurm_evidence" / "jobs_13192_13198_13203.json"
)


def test_issue1554_packet_recommends_analysis_before_duplicate_submit() -> None:
    """The completed S20/H500 result should route to analysis, not duplicate submission."""
    jobs = _load_jobs(FIXTURE)

    packet = build_packet(jobs)

    assert packet["jobs_synthesized"] == [13192, 13198, 13203]
    assert packet["status"] == "analysis_before_submit"
    assert packet["successful_jobs"] == [13192, 13198, 13203]
    assert (
        "Do not enqueue another duplicate S20/H500 planner-family run yet"
        in packet["next_slurm_queue_recommendation"]
    )
    assert packet["claim_blockers"] == [
        {
            "job": 13198,
            "limitations": [
                "SNQI contract warning blocks paper-grade interpretation until analyzed.",
                "Analyze this result before scheduling another duplicate S20/H500 planner-family run.",
            ],
        }
    ]
    assert packet["forbidden_actions_confirmed"] == {
        "compute_submit": False,
        "artifact_deletion": False,
        "paper_or_dissertation_claim_edits": False,
    }


def test_issue1554_packet_markdown_names_jobs_and_forbidden_actions() -> None:
    """The Markdown packet names the evidence set and forbidden actions."""
    packet = build_packet(_load_jobs(FIXTURE))

    rendered = render_markdown(packet)

    assert "completed jobs 13192, 13198, and 13203" in rendered
    assert "Job 13198" in rendered
    assert "no Slurm/GPU submission" in rendered
    assert "no paper/dissertation claim edits" in rendered


def test_issue1554_packet_json_is_serializable() -> None:
    """The packet remains plain JSON for durable evidence review."""
    packet = build_packet(_load_jobs(FIXTURE))

    encoded = json.dumps(packet, sort_keys=True)

    assert "issue1554-slurm-evidence-packet.v1" in encoded


def test_job13198_constraints_first_artifact_bundle_is_boundary_only() -> None:
    """The job 13198 bundle preserves bounded claims without paper-grade promotion."""
    bundle = (
        Path(__file__).parents[2]
        / "docs"
        / "context"
        / "evidence"
        / "issue_1554_job_13198_constraints_first_analysis"
    )
    packet = json.loads((bundle / "packet.json").read_text(encoding="utf-8"))
    inventory = json.loads((bundle / "artifact_inventory.json").read_text(encoding="utf-8"))

    assert packet["schema_version"] == "issue1554-job13198-constraints-first-analysis.v1"
    assert inventory["accepted_packet_id"] == (
        "issue1554-job13198-constraints-first-analysis-20260701"
    )
    assert packet["job_id"] == 13198
    assert packet["campaign"] == "2026-06-issue1554-s20-h500-split-mem180-run"
    assert packet["public_commit"] == "12a188de7246aad3b9088ea76e6a25a20029f976"
    assert packet["row_status"]["benchmark_row_status"] == "successful_evidence"
    assert packet["row_status"]["planner_rows"] == 9
    assert packet["row_status"]["episode_rows"] == 8640
    assert packet["claim_counts"] == {
        "ci_separable": 5,
        "diagnostic_only": 8,
        "not_statistically_distinguishable_budget": 3,
    }
    assert packet["decision"]["more_seed_budget_compute"] == "conditional"
    assert packet["snqi"]["contract_status"] == "fail"
    assert packet["snqi"]["decision_role"] == "explanatory_only"
    assert inventory["forbidden_actions_confirmed"] == {
        "artifact_deletion": False,
        "compute_submit": False,
        "paper_or_dissertation_claim_edits": False,
    }
    assert {
        "README.md",
        "packet.json",
        "constraints_first_metrics.csv",
        "adjacent_rank_claims.csv",
        "claim_decision.md",
    }.issubset({artifact["path"] for artifact in inventory["files"]})
