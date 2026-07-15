"""Reconstruction regression for the job-13274 exit-code remap (issue #5244).

This reproduces, with a minimal fixture, the exact defect that job 13274
exhibited: a camera-ready campaign that finished all planner rows but exited 5
under ``snqi_contract.enforcement=warn``. The responsible layer is the
post-campaign report/analysis stage (the job-13274 exit-5 candidate: an optional
dep miss or a missing report artifact under ``set -e``) chained *after* the
campaign completed; under the legacy wrapper its exit propagated as the job exit,
relabeling a complete campaign as a failed scheduler job.

The test drives the adopted production boundary (the
``robot-sf-post-campaign-stage-status.v1`` envelope plus ``run_post_campaign_stage``
and ``slurm_job_finalize``) against that scenario and asserts the corrected
behavior: the campaign lane (and ``job_exit_code``) stays 0, while the stage
failure is isolated in its own lane and the legacy mapping is reconstructed for
contrast. This is the issue's own validation item "reproduce or reconstruct the
job-13274 exit mapping with a minimal fixture" and uses the production
dispatch/serialization boundary, not only a synthetic returned payload.

This module also hardens the reconstruction harness per follow-up issue #5702:
the finalizer must receive the campaign summary as the expected campaign artifact
and the status envelope separately, the harness must reject exit codes outside
``0..255`` before producing fixtures, and a hard (nonzero) campaign exit must stay
nonzero while the post-campaign stage failure remains isolated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.tools import reconstruct_job13274_exit_mapping, run_post_campaign_stage

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestJob13274ExitMappingReconstruction:
    """Reconstruct the job-13274 before/after exit mapping at the production boundary."""

    def _completed_campaign_summary(self, tmp_path: Path, *, soft_warning: bool) -> Path:
        """Write a minimal completed ``enforcement=warn`` campaign summary fixture."""
        summary = tmp_path / "cid" / "reports" / "campaign_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps(
                {
                    "soft_contract_warning": soft_warning,
                    "warnings": (
                        [
                            "SNQI contract status=warn with snqi_contract.enforcement=warn; "
                            "campaign marked with soft contract warning."
                        ]
                        if soft_warning
                        else []
                    ),
                    "benchmark_success": True,
                    "status": "benchmark_success",
                }
            ),
            encoding="utf-8",
        )
        return summary

    def test_before_fix_relabels_completed_campaign_as_failed(self, tmp_path: Path) -> None:
        """The legacy ``set -e`` wrapper propagated the stage exit 5 as the job exit.

        This is exactly the job-13274 orphan: a campaign that completed with exit 0
        was relabeled as a failed job because the chained report stage exited 5.
        """
        campaign_root = tmp_path / "cid"
        before = reconstruct_job13274_exit_mapping.reconstruct_before_fix(
            campaign_root, stage_exit_code=5
        )
        assert before["legacy_job_exit_code"] == 5
        assert before["legacy_classification"] == "failed"

    def test_after_fix_preserves_campaign_exit_zero_through_production_boundary(
        self, tmp_path: Path, capsys
    ) -> None:
        """The adopted boundary keeps job exit 0 while isolating the stage exit 5.

        Drives the real reconstruction harness (which runs the live
        ``run_post_campaign_stage`` + ``slurm_job_finalize`` primitives) so the
        completed-warn campaign is not relabeled by the downstream stage. The
        finalizer receives the campaign summary as the expected campaign artifact and
        the status envelope as a separate required artifact (issue #5702).
        """
        self._completed_campaign_summary(tmp_path, soft_warning=True)
        campaign_root = tmp_path / "cid"

        exit_code = reconstruct_job13274_exit_mapping.main(
            [
                "--campaign-root",
                str(campaign_root),
                "--stage-exit-code",
                "5",
                "--emit-side-by-side",
            ]
        )
        assert exit_code == 0

        finalize_output = campaign_root / "finalize.json"
        report = json.loads(finalize_output.read_text(encoding="utf-8"))
        assert report["classification"] == "success"
        envelope = report["exit_code_lanes"]
        assert envelope["job_exit_code"] == 0
        assert envelope["campaign"]["exit_code"] == 0
        assert envelope["post_campaign_stage"]["exit_code"] == 5
        assert "not benchmark evidence" in report["claim_boundary"].lower()

    def test_reconstruction_cli_emits_side_by_side(self, tmp_path: Path, capsys) -> None:
        """The runnable harness reproduces the scenario and prints the mapping."""
        campaign_root = tmp_path / "cid"
        exit_code = reconstruct_job13274_exit_mapping.main(
            [
                "--campaign-root",
                str(campaign_root),
                "--stage-exit-code",
                "5",
                "--emit-side-by-side",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "job 13274 exit-code reconstruction" in out
        assert "BEFORE" in out
        assert "AFTER" in out
        assert "job_exit_code = 5" in out
        assert "job_exit_code             = 0" in out

    def test_hard_success_campaign_without_soft_warning_also_preserved(
        self, tmp_path: Path
    ) -> None:
        """The boundary preserves a non-warning completed campaign just the same."""
        summary = self._completed_campaign_summary(tmp_path, soft_warning=False)
        campaign_root = tmp_path / "cid"
        envelope_path = campaign_root / "reports" / "post_campaign_stage_status.json"
        stage_script = campaign_root / "stage_fail.sh"
        stage_script.parent.mkdir(parents=True, exist_ok=True)
        stage_script.write_text("#!/usr/bin/env bash\nexit 5\n", encoding="utf-8")
        stage_script.chmod(0o755)

        run_post_campaign_stage.main(
            [
                "--campaign-summary",
                str(summary),
                "--campaign-exit-code",
                "0",
                "--stage-name",
                "headline_ci_rank_stability_report",
                "--output",
                str(envelope_path),
                "--stage-command",
                str(stage_script),
            ]
        )
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        assert envelope["campaign"]["exit_code"] == 0
        assert envelope["campaign"]["soft_contract_warning"] is False
        assert envelope["job_exit_code"] == 0
        assert envelope["post_campaign_stage"]["exit_code"] == 5

    def test_hard_campaign_exit_stays_nonzero_while_stage_failure_isolated(
        self, tmp_path: Path
    ) -> None:
        """A hard (nonzero) campaign exit must remain nonzero through the boundary.

        The post-campaign stage may also fail, but the harness must never mask the
        campaign failure: ``job_exit_code`` follows the campaign lane, so a failed
        campaign stays failed (regression guard from issue #5702).
        """
        self._completed_campaign_summary(tmp_path, soft_warning=False)
        campaign_root = tmp_path / "cid"

        exit_code = reconstruct_job13274_exit_mapping.main(
            [
                "--campaign-root",
                str(campaign_root),
                "--campaign-exit-code",
                "3",
                "--stage-exit-code",
                "5",
            ]
        )
        assert exit_code == 3

        finalize_output = campaign_root / "finalize.json"
        report = json.loads(finalize_output.read_text(encoding="utf-8"))
        assert report["classification"] == "failed"
        envelope = report["exit_code_lanes"]
        assert envelope["job_exit_code"] == 3
        assert envelope["campaign"]["exit_code"] == 3
        assert envelope["post_campaign_stage"]["exit_code"] == 5

    def test_invalid_stage_exit_code_rejected_before_fixture_creation(self, tmp_path: Path) -> None:
        """Exit codes outside ``0..255`` must fail closed at the harness boundary.

        The production boundary accepts only shell exit codes; the harness must reject
        an impossible value (e.g. 256) before writing any fixture, rather than letting a
        nested production parser raise mid-run (issue #5702).
        """
        campaign_root = tmp_path / "cid"
        with pytest.raises(ValueError, match=r"0\.\.255"):
            reconstruct_job13274_exit_mapping.main(
                [
                    "--campaign-root",
                    str(campaign_root),
                    "--stage-exit-code",
                    "256",
                ]
            )
        # No fixture output should have been produced.
        assert not (campaign_root / "reports" / "campaign_summary.json").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
