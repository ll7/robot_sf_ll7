"""Tests for SLURM-to-claim trace checklist helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256
from scripts.tools import slurm_to_claim_trace_checklist

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> Path:
    """Write compact JSON fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_evidence_dir(repo_root: Path) -> Path:
    """Create a compact tracked-evidence-style directory with checksums."""

    evidence_dir = repo_root / "docs/context/evidence/issue_3425_trace"
    readme = evidence_dir / "README.md"
    summary = evidence_dir / "compact_summary.csv"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("# Trace\n", encoding="utf-8")
    summary.write_text("job_id,status\n13268,diagnostic\n", encoding="utf-8")
    checksum_lines = []
    for artifact in (readme, summary):
        relpath = artifact.relative_to(repo_root).as_posix()
        checksum_lines.append(f"{_sha256(artifact)}  {relpath}")
    (evidence_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return evidence_dir


def _write_source_manifest(repo_root: Path) -> Path:
    """Write h600-style source manifest fixture."""

    return _write_json(
        repo_root / "docs/context/evidence/issue_3425_trace/source_manifest.json",
        {
            "schema_version": "issue_4195_h600_aggregation.v1.source_manifest",
            "runs": [
                {
                    "job_id": "13268",
                    "run_label": "confirm",
                    "campaign": {"campaign_id": "issue3810_h600_confirm"},
                    "planner_keys": ["goal", "social_force"],
                    "source_sha256": {"campaign_summary.json": "abc123"},
                }
            ],
        },
    )


def test_complete_trace_checklist_passes(tmp_path: Path) -> None:
    """A source, evidence, finalizer, reconciliation, and spine pointer pass."""

    evidence_dir = _write_evidence_dir(tmp_path)
    source_manifest = _write_source_manifest(tmp_path)
    finalizer = _write_json(
        tmp_path / "docs/context/evidence/issue_3425_trace/finalization_13268.json",
        {
            "schema_version": "robot-sf-slurm-job-finalization.v1",
            "issue_number": 3425,
            "job_id": "13268",
            "classification": "success",
            "artifact_status": "all_required_present",
            "durable_uri": "wandb://robot-sf/h600/job-13268:v0",
            "claim_boundary": "workflow trace only",
            "claim_decision": "keep diagnostic",
            "artifacts": [
                {
                    "path": "output/h600/13268/reports/campaign_summary.json",
                    "exists": True,
                    "required": True,
                    "kind": "file",
                    "sha256": "cafebabe",
                    "size_bytes": 10,
                }
            ],
        },
    )
    reconciliation = _write_json(
        tmp_path / "docs/context/evidence/issue_3425_trace/reconciliation_13268.json",
        {
            "schema_version": "slurm-evidence-reconciler.v1",
            "errors": [],
            "finalizer_bridge": {
                "schema_version": "slurm-job-finalizer-bridge.v1",
                "rows": [
                    {
                        "issue": 3425,
                        "job_id": "13268",
                        "claim_decision": "keep_diagnostic",
                        "claim_boundary": "workflow trace only",
                    }
                ],
            },
        },
    )

    report = slurm_to_claim_trace_checklist.build_checklist(
        job_id="13268",
        issue=3425,
        source_manifest=source_manifest,
        evidence_dir=evidence_dir,
        finalizer_manifest=finalizer,
        reconciliation=reconciliation,
        spine_pointer="docs/context/evidence/issue_3425_trace/README.md",
        repo_root=tmp_path,
        generated_at="2026-07-03T00:00:00+00:00",
    )

    assert report["status"] == "pass"
    assert report["claim_decision"] == "keep_diagnostic"
    assert report["blockers"] == []
    assert report["retrieval"]["run"]["job_id"] == "13268"


def test_sha256sums_comment_markers_are_ignored(tmp_path: Path) -> None:
    """SHA256SUMS with # comment header does not produce a synthetic missing-artifact blocker.

    Regression for issue #5197: PR #5187 adds '# AI-GENERATED NEEDS-REVIEW' headers
    to SHA256SUMS files; the parser must skip them rather than treating them as entries.
    """

    evidence_dir = tmp_path / "docs/context/evidence/issue_3425_trace"
    readme = evidence_dir / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("# Trace\n", encoding="utf-8")
    relpath = readme.relative_to(tmp_path).as_posix()
    digest = _sha256(readme)
    # Write SHA256SUMS with a leading comment line (as added by PR #5187)
    (evidence_dir / "SHA256SUMS").write_text(
        f"# AI-GENERATED NEEDS-REVIEW\n{digest}  {relpath}\n",
        encoding="utf-8",
    )

    entries = slurm_to_claim_trace_checklist._parse_sha256s(evidence_dir / "SHA256SUMS")

    assert "AI-GENERATED NEEDS-REVIEW" not in entries
    assert relpath in entries
    assert entries[relpath] == digest


def test_sha256sums_comment_markers_do_not_block_checklist(tmp_path: Path) -> None:
    """Evidence dir whose SHA256SUMS has a comment header passes checksum verification."""

    evidence_dir = tmp_path / "docs/context/evidence/issue_3425_trace"
    readme = evidence_dir / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("# Trace\n", encoding="utf-8")
    relpath = readme.relative_to(tmp_path).as_posix()
    digest = _sha256(readme)
    (evidence_dir / "SHA256SUMS").write_text(
        f"# AI-GENERATED NEEDS-REVIEW\n{digest}  {relpath}\n",
        encoding="utf-8",
    )
    source_manifest = _write_source_manifest(tmp_path)

    report = slurm_to_claim_trace_checklist.build_checklist(
        job_id="13268",
        issue=3425,
        source_manifest=source_manifest,
        evidence_dir=evidence_dir,
        finalizer_manifest=None,
        reconciliation=None,
        spine_pointer="docs/context/evidence/issue_3425_trace/README.md",
        claim_boundary="workflow trace only; no benchmark claim",
        repo_root=tmp_path,
        generated_at="2026-07-11T00:00:00+00:00",
    )

    # The comment should not appear as a synthetic missing-artifact blocker
    blocker_details = " ".join(b["detail"] for b in report["blockers"])
    assert "AI-GENERATED" not in blocker_details
    assert "NEEDS-REVIEW" not in blocker_details
    # Evidence checksums should pass (comment line is skipped, real entry verified)
    check_statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert check_statuses.get("evidence_checksums") == "pass"


def test_missing_finalizer_and_reconciliation_block_without_losing_retrieval(
    tmp_path: Path,
) -> None:
    """Current live gaps become explicit blockers while preserving source trace."""

    evidence_dir = _write_evidence_dir(tmp_path)
    source_manifest = _write_source_manifest(tmp_path)

    report = slurm_to_claim_trace_checklist.build_checklist(
        job_id="13268",
        issue=3425,
        source_manifest=source_manifest,
        evidence_dir=evidence_dir,
        finalizer_manifest=None,
        reconciliation=None,
        spine_pointer="docs/context/evidence/issue_3425_trace/README.md",
        claim_boundary="workflow trace only; no benchmark claim",
        repo_root=tmp_path,
        generated_at="2026-07-03T00:00:00+00:00",
    )

    assert report["status"] == "blocked"
    assert report["claim_decision"] == "block"
    blocker_names = {blocker["name"] for blocker in report["blockers"]}
    assert "finalizer_manifest" in blocker_names
    assert "reconciliation_output" in blocker_names
    assert report["retrieval"]["run"]["job_id"] == "13268"
