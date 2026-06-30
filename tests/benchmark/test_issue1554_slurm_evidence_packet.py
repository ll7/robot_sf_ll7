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
