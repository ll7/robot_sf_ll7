"""Regression tests for the claim-decision contract added by issue #3425.

These assertions are derived directly from the issue Phase 1.3 acceptance
criteria: normalizer correctness, CLI integration, idempotence, and
reconciler bridge preservation. Each test is CPU-only and deterministic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools.slurm_job_finalize import (
    ALLOWED_CLAIM_DECISIONS,
    build_finalization_report,
    normalize_claim_decision,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


class TestNormalizeClaimDecision:
    """Direct unit tests for the claim-decision normalizer."""

    def test_valid_decisions_are_accepted(self) -> None:
        """All four allowed decisions pass through unchanged."""
        for decision in ALLOWED_CLAIM_DECISIONS:
            assert normalize_claim_decision(decision) == decision

    def test_none_returns_none(self) -> None:
        """None input yields None output without raising."""
        assert normalize_claim_decision(None) is None

    def test_blank_returns_none(self) -> None:
        """Whitespace-only input is treated as absent."""
        assert normalize_claim_decision("   ") is None

    def test_space_variant_normalizes_to_snake_case(self) -> None:
        """Issue 3425 requires both CLI spelling forms to be accepted."""
        assert normalize_claim_decision("keep diagnostic") == "keep_diagnostic"

    def test_hyphen_variant_normalizes_to_snake_case(self) -> None:
        """Hyphen spelling normalizes to underscore form."""
        assert normalize_claim_decision("keep-diagnostic") == "keep_diagnostic"

    def test_case_insensitive_normalization(self) -> None:
        """Mixed case is normalized to lowercase."""
        assert normalize_claim_decision("KEEP_DIAGNOSTIC") == "keep_diagnostic"
        assert normalize_claim_decision("Promote") == "promote"

    def test_invalid_decision_raises_value_error(self) -> None:
        """Invalid decisions must raise a usage-level error, not a traceback."""
        with pytest.raises(ValueError, match="must be one of"):
            normalize_claim_decision("publish paper claim")

    def test_invalid_decision_error_lists_allowed_values(self) -> None:
        """Error message includes the allowed values for self-service correction."""
        with pytest.raises(ValueError, match="promote"):
            normalize_claim_decision("nonsense")


class TestBuildFinalizationReportClaimDecision:
    """Assertions that build_finalization_report emits claim_decision correctly."""

    def test_space_variant_normalized_in_report(self, tmp_path: Path) -> None:
        """build_finalization_report with 'keep diagnostic' emits 'keep_diagnostic'."""
        artifact = tmp_path / "output" / "job-900" / "summary.json"
        _write(artifact, "{}\n")

        report = build_finalization_report(
            issue_number=3425,
            job_id="900",
            job_state="COMPLETED",
            expected_artifacts=["output/job-900/summary.json"],
            repo_root=tmp_path,
            claim_decision="keep diagnostic",
        )

        assert report["claim_decision"] == "keep_diagnostic"

    def test_hyphen_variant_normalized_in_report(self, tmp_path: Path) -> None:
        """Hyphen CLI spelling normalizes to snake_case in report output."""
        artifact = tmp_path / "output" / "job-901" / "summary.json"
        _write(artifact, "{}\n")

        report = build_finalization_report(
            issue_number=3425,
            job_id="901",
            job_state="COMPLETED",
            expected_artifacts=["output/job-901/summary.json"],
            repo_root=tmp_path,
            claim_decision="keep-diagnostic",
        )

        assert report["claim_decision"] == "keep_diagnostic"

    def test_claim_decision_in_json_output(self, tmp_path: Path) -> None:
        """CLI --claim-decision writes the field to JSON."""
        artifact = tmp_path / "output" / "job-902" / "summary.json"
        _write(artifact, "{}\n")
        output = tmp_path / "finalization.json"
        markdown = tmp_path / "finalization.md"

        from scripts.tools.slurm_job_finalize import main

        exit_code = main(
            [
                "--repo-root",
                str(tmp_path),
                "--issue",
                "3425",
                "--job-id",
                "902",
                "--job-state",
                "COMPLETED",
                "--expected-artifact",
                "output/job-902/summary.json",
                "--claim-decision",
                "keep-diagnostic",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
            ]
        )

        assert exit_code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["claim_decision"] == "keep_diagnostic"

    def test_claim_decision_in_markdown_output(self, tmp_path: Path) -> None:
        """CLI --claim-decision writes the field to Markdown."""
        artifact = tmp_path / "output" / "job-903" / "summary.json"
        _write(artifact, "{}\n")
        output = tmp_path / "finalization.json"
        markdown = tmp_path / "finalization.md"

        from scripts.tools.slurm_job_finalize import main

        main(
            [
                "--repo-root",
                str(tmp_path),
                "--issue",
                "3425",
                "--job-id",
                "903",
                "--job-state",
                "COMPLETED",
                "--expected-artifact",
                "output/job-903/summary.json",
                "--claim-decision",
                "keep-diagnostic",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
            ]
        )

        md_content = markdown.read_text(encoding="utf-8")
        assert "claim decision" in md_content
        assert "keep_diagnostic" in md_content

    def test_invalid_claim_decision_exits_cleanly(self, tmp_path: Path) -> None:
        """Invalid decisions produce a CLI error, not a traceback."""
        output = tmp_path / "rejected.json"

        from scripts.tools.slurm_job_finalize import main

        with pytest.raises(SystemExit):
            main(
                [
                    "--repo-root",
                    str(tmp_path),
                    "--issue",
                    "3425",
                    "--job-id",
                    "904",
                    "--job-state",
                    "COMPLETED",
                    "--expected-artifact",
                    "output/job-904/summary.json",
                    "--claim-decision",
                    "invalid-choice",
                    "--output",
                    str(output),
                ]
            )

    def test_idempotence_holds_with_claim_decision(self, tmp_path: Path) -> None:
        """Re-running the finalizer with claim_decision must not change semantics."""
        artifact = tmp_path / "output" / "job-905" / "summary.json"
        _write(artifact, '{"ok": true}\n')
        kwargs = {
            "issue_number": 3425,
            "job_id": "905",
            "job_state": "COMPLETED",
            "expected_artifacts": ["output/job-905/summary.json"],
            "repo_root": tmp_path,
            "claim_decision": "keep_diagnostic",
        }

        first = build_finalization_report(**kwargs)
        second = build_finalization_report(**kwargs)

        first.pop("generated_at")
        second.pop("generated_at")
        assert first == second

    def test_claim_decision_absent_when_none(self, tmp_path: Path) -> None:
        """When no claim_decision is passed, the field should be absent from the report."""
        artifact = tmp_path / "output" / "job-906" / "summary.json"
        _write(artifact, "{}\n")

        report = build_finalization_report(
            issue_number=3425,
            job_id="906",
            job_state="COMPLETED",
            expected_artifacts=["output/job-906/summary.json"],
            repo_root=tmp_path,
        )

        assert "claim_decision" not in report


class TestReconcilerClaimDecisionPreservation:
    """Assertions that the reconciler preserves claim_decision in bridge rows."""

    def test_bridge_row_preserves_claim_decision(self, tmp_path: Path) -> None:
        """Reconciler bridge rows must carry the claim_decision from the finalizer."""
        import yaml

        from scripts.tools.reconcile_slurm_evidence import reconcile

        queue_path = tmp_path / "queue.yaml"
        queue_path.write_text(
            yaml.safe_dump(
                {
                    "entries": [
                        {
                            "id": "slice-3425",
                            "seeds": [501],
                            "status": "completed",
                            "issue": 3425,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "jobs": [
                        {
                            "queue_id": "slice-3425",
                            "status": "completed",
                            "slurm_job_id": "13042",
                            "seeds": [501],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        finalizer_path = tmp_path / "finalization.json"
        finalizer_path.write_text(
            json.dumps(
                {
                    "schema_version": "robot-sf-slurm-job-finalization.v1",
                    "issue_number": 3425,
                    "job_id": "13042",
                    "classification": "success",
                    "artifact_status": "all_required_present",
                    "durable_uri": "wandb-artifact://robot-sf/issue3425/run:v0",
                    "claim_boundary": "smoke evidence only",
                    "claim_decision": "keep diagnostic",
                    "artifacts": [],
                }
            ),
            encoding="utf-8",
        )
        evidence_path = tmp_path / "evidence" / "seed_summary.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "queue_id": "slice-3425",
                            "seed": 501,
                            "job_id": "13042",
                            "wandb_url": "https://wandb.ai/ll7/robot_sf/runs/3425",
                            "claim_boundary": "smoke evidence only",
                            "run_summary_sha256": "0123456789abcdef0123456789abcdef",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        report = reconcile(
            queue_path=queue_path,
            submission_manifests=[manifest_path],
            evidence_root=tmp_path / "evidence",
            finalizer_manifests=[finalizer_path],
            generated_at="2026-07-04T00:00:00+00:00",
        )

        assert report["errors"] == []
        bridge = report["finalizer_bridge"]
        assert bridge["schema_version"] == "slurm-job-finalizer-bridge.v1"
        row = bridge["rows"][0]
        assert row["job_id"] == "13042"
        assert row["claim_decision"] == "keep_diagnostic"
        assert row["claim_boundary"] == "smoke evidence only"
        assert row["durable_pointer"] == "wandb-artifact://robot-sf/issue3425/run:v0"
