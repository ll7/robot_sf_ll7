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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """The adopted boundary keeps job exit 0 while isolating the stage exit 5.

        Drives the real reconstruction harness (which runs the live
        ``run_post_campaign_stage`` + ``slurm_job_finalize`` primitives) so the
        completed-warn campaign is not relabeled by the downstream stage.
        """
        summary = self._completed_campaign_summary(tmp_path, soft_warning=True)
        campaign_root = tmp_path / "cid"

        monkeypatch.setattr(sys, "argv", ["reconstruct_job13274_exit_mapping.py"])
        after = reconstruct_job13274_exit_mapping.reconstruct_after_fix(
            summary,
            campaign_root,
            campaign_exit_code=0,
            stage_exit_code=5,
        )

        assert after["envelope_schema"] == "robot-sf-post-campaign-stage-status.v1"
        assert after["campaign_exit_code"] == 0
        assert after["job_exit_code"] == 0
        assert after["finalize_exit"] == 0
        assert after["finalize_classification"] == "success"
        assert after["post_campaign_stage_exit_code"] == 5
        assert after["post_campaign_stage_status"] == "report_stage_failed"
        assert "not benchmark evidence" in after["claim_boundary"].lower()

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

    def test_cli_rejects_out_of_range_stage_exit_code(self, tmp_path: Path) -> None:
        """The harness boundary rejects impossible shell exit codes before any output.

        The reproduction in issue #5702 shows the merged head accepted an integer
        such as 256 at its own parser and only failed later inside the nested
        production parser. The harness must reject it at the CLI boundary.
        """
        campaign_root = tmp_path / "cid"
        with pytest.raises(SystemExit) as exc_info:
            reconstruct_job13274_exit_mapping.main(
                [
                    "--campaign-root",
                    str(campaign_root),
                    "--stage-exit-code",
                    "256",
                ]
            )
        assert exc_info.value.code == 2
        assert not campaign_root.exists()

    def test_after_fix_finalizer_expects_campaign_summary_as_artifact(self, tmp_path: Path) -> None:
        """The finalizer receives the campaign summary as the campaign artifact.

        Issue #5702 gap 1: the finalizer's required-artifact check must represent
        the campaign summary/report boundary, not only the status envelope.
        """
        summary = self._completed_campaign_summary(tmp_path, soft_warning=True)
        campaign_root = tmp_path / "cid"
        after = reconstruct_job13274_exit_mapping.reconstruct_after_fix(
            summary,
            campaign_root,
            campaign_exit_code=0,
            stage_exit_code=5,
        )
        assert after["campaign_exit_code"] == 0
        assert after["job_exit_code"] == 0
        # The report must record the campaign summary as a required expected artifact
        # and the status envelope separately, so artifact loading is observable.
        report_path = campaign_root / "finalize.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_paths = {artifact["path"] for artifact in report["artifacts"]}
        assert summary.resolve().as_posix() in artifact_paths
        stage_artifact = next(
            artifact
            for artifact in report["artifacts"]
            if artifact["role"] == "post_campaign_stage_status"
        )
        assert stage_artifact["exists"] is True
        assert report["post_campaign_stage_status"]["load_status"] == "loaded"

    def test_hard_campaign_exit_preserved_through_production_boundary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuinely failed campaign exit is preserved, not relabeled by the stage.

        Hard-campaign-exit regression for issue #5702: a nonzero campaign exit must
        stay the job exit code and classify the run as failed, while a coincident
        stage failure does not hide the campaign failure behind a success.
        """
        summary = self._completed_campaign_summary(tmp_path, soft_warning=True)
        campaign_root = tmp_path / "cid"
        monkeypatch.setattr(sys, "argv", ["reconstruct_job13274_exit_mapping.py"])
        after = reconstruct_job13274_exit_mapping.reconstruct_after_fix(
            summary,
            campaign_root,
            campaign_exit_code=3,
            stage_exit_code=5,
        )
        assert after["campaign_exit_code"] == 3
        assert after["job_exit_code"] == 3
        assert after["post_campaign_stage_exit_code"] == 5
        assert after["finalize_classification"] == "failed"
        assert "benchmark evidence" in after["claim_boundary"].lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
