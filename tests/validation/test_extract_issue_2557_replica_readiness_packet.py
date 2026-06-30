"""Tests issue #2557 fixed-seed replica readiness packet extractor."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.validation import extract_issue_2557_replica_readiness_packet as extractor

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/extract_issue_2557_replica_readiness_packet.py"


def test_build_packet_from_tracked_evidence_keeps_diagnostic_boundary() -> None:
    """Tracked #2557 evidence produces no-submit diagnostic readiness packet."""

    packet = extractor.build_packet(generated_at="2026-06-29T00:00:00+02:00")

    assert packet["status"] == "diagnostic_only_blocked_artifact_promotion"
    assert packet["scheduler_snapshot"]["running_jobs"] == []
    assert packet["scheduler_snapshot"]["pending_jobs"] == []
    assert len(packet["completed_jobs"]) == 14
    assert packet["retrieved_evidence"]["compact_seed_count"] == 14
    assert 501 in packet["retrieved_evidence"]["compact_seeds"]
    assert 522 in packet["retrieved_evidence"]["compact_seeds"]
    assert packet["retrieved_evidence"]["recovered_note_present"] is True
    assert packet["evidence_gap"]["unpromoted_or_missing_seeds"] == [
        510,
        511,
        512,
        513,
        514,
        515,
        516,
        523,
        524,
    ]
    assert packet["candidate_queue_entry"]["submission_recommendation"] == (
        "no_new_slurm_queue_fill"
    )
    assert packet["go_no_go"]["new_slurm_submission"] == "NO-GO"
    assert packet["go_no_go"]["local_public_packet"] == "GO"


def test_cli_writes_json_and_renders_required_markdown_sections(tmp_path: Path) -> None:
    """CLI can write deterministic JSON and print public Markdown packet."""

    output = tmp_path / "packet.json"
    markdown = tmp_path / "README.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--generated-at",
            "2026-06-29T00:00:00+02:00",
            "--write-json",
            str(output),
            "--write-markdown",
            str(markdown),
            "--markdown",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "## Completed / Running Jobs" in result.stdout
    assert "## Evidence Gap" in result.stdout
    assert "## Candidate Queue Entry" in result.stdout
    assert "## Cost / Risk" in result.stdout
    assert "## Go / No-Go" in result.stdout
    assert "New Slurm submission: `NO-GO`" in result.stdout

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == extractor.SCHEMA_VERSION
    assert payload["generated_at"] == "2026-06-29T00:00:00+02:00"
    assert markdown.read_text(encoding="utf-8") == result.stdout
