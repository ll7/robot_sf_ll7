"""Production-boundary handoff regression for issue #5244.

This drives the *real* production dispatch/serialization boundary of the
issue-3216 headline campaign pipeline, not only a synthetic returned payload:

1. The Python launcher ``run_camera_ready_benchmark.main`` runs a completed
   ``enforcement=warn`` campaign and writes the
   ``robot-sf-post-campaign-stage-status.v1`` envelope to
   ``<campaign_root>/reports/post_campaign_stage_status.json`` (campaign exit 0,
   ``soft_contract_warning: true``).
2. The adopted post-campaign stage primitive ``run_post_campaign_stage.main``
   runs a chained stage that exits 5 (the job-13274 candidate: an optional-dep
   miss / missing report artifact) and rewrites the SAME on-disk envelope so the
   failing stage is recorded in ``post_campaign_stage`` while the campaign lane
   stays 0.
3. The finalizer ``slurm_job_finalize.main`` consumes that exact on-disk envelope
   and must classify the job as ``success`` (job_exit_code 0) while surfacing the
   separate report-stage failure lane and the "not benchmark evidence" boundary.

This proves the adopted shell wrapper cannot re-relabel a completed campaign as a
nonzero scheduler job through the public production path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.tools import run_camera_ready_benchmark, run_post_campaign_stage, slurm_job_finalize

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestIssue3216CampaignStageHandoff:
    """Launcher -> adopted post-campaign stage -> finalizer, real on-disk envelope."""

    def _run_launcher(self, tmp_path: Path, monkeypatch, *, soft_warning: bool) -> Path:
        """Run the real launcher with a faked campaign that writes a summary file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        campaign_root = tmp_path / "out" / "cid"
        summary_path = campaign_root / "reports" / "campaign_summary.json"

        def _fake_run_campaign(cfg, **kwargs):
            del cfg, kwargs
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {"soft_contract_warning": soft_warning, "warnings": ["SNQI warn"]}
                    if soft_warning
                    else {}
                ),
                encoding="utf-8",
            )
            return {
                "campaign_id": "cid",
                "campaign_root": str(campaign_root),
                "summary_json": str(summary_path),
                "benchmark_success": True,
                "campaign_execution_status": "completed",
                "evidence_status": "valid",
                "row_status_summary": {
                    "successful_evidence_rows": 1,
                    "accepted_unavailable_rows": 0,
                    "unexpected_failed_rows": 0,
                    "fallback_or_degraded_rows": 0,
                },
            }

        monkeypatch.setattr(
            run_camera_ready_benchmark, "load_campaign_config", lambda path: object()
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "prepare_campaign_preflight",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
        )
        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)

        exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])
        assert exit_code == 0
        return campaign_root

    def test_completed_warn_campaign_stage_exit_five_stays_job_exit_zero(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Full public pipeline preserves job exit 0 for a completed warn campaign."""
        campaign_root = self._run_launcher(tmp_path, monkeypatch, soft_warning=True)
        envelope_path = campaign_root / "reports" / "post_campaign_stage_status.json"
        assert envelope_path.is_file()

        # Launcher envelope: campaign completed (exit 0, soft warning), stage lane = campaign.
        launcher_payload = json.loads(envelope_path.read_text(encoding="utf-8"))
        assert launcher_payload["campaign"]["exit_code"] == 0
        assert launcher_payload["campaign"]["soft_contract_warning"] is True
        assert launcher_payload["job_exit_code"] == 0
        assert launcher_payload["post_campaign_stage"]["exit_code"] == 0

        # Adopted production step: chained stage exits 5 (job-13274 candidate).
        stage_script = tmp_path / "stage_fail.sh"
        stage_script.write_text("#!/usr/bin/env bash\nexit 5\n", encoding="utf-8")
        stage_script.chmod(0o755)
        stage_exit = run_post_campaign_stage.main(
            [
                "--campaign-summary",
                str(campaign_root / "reports" / "campaign_summary.json"),
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
        assert stage_exit == 0

        # The on-disk envelope now records the report-stage failure separately.
        rewritten = json.loads(envelope_path.read_text(encoding="utf-8"))
        assert rewritten["campaign"]["exit_code"] == 0
        assert rewritten["job_exit_code"] == 0
        assert rewritten["post_campaign_stage"]["exit_code"] == 5
        assert rewritten["post_campaign_stage"]["status"] == "report_stage_failed"

        # Finalizer consumes the exact rewritten envelope from disk.
        capsys.readouterr()
        finalize_output = tmp_path / "finalize.json"
        finalize_exit = slurm_job_finalize.main(
            [
                "--issue",
                "5244",
                "--job-id",
                "13274",
                "--job-state",
                "COMPLETED",
                "--expected-artifact",
                str(envelope_path),
                "--post-campaign-stage-status",
                str(envelope_path),
                "--output",
                str(finalize_output),
            ]
        )
        report = json.loads(finalize_output.read_text(encoding="utf-8"))
        assert finalize_exit == 0
        assert report["classification"] == "success"
        lanes = report["exit_code_lanes"]
        assert lanes["campaign"]["exit_code"] == 0
        assert lanes["job_exit_code"] == 0
        assert lanes["post_campaign_stage"]["exit_code"] == 5
        assert "not benchmark evidence" in report["claim_boundary"].lower()

    def test_hard_campaign_failure_not_laundered_by_stage_success(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A hard campaign failure must finalize as failed even if the stage passes."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        campaign_root = tmp_path / "out" / "cid"
        summary_path = campaign_root / "reports" / "campaign_summary.json"

        def _fake_run_campaign(cfg, **kwargs):
            del cfg, kwargs
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("{}", encoding="utf-8")
            return {
                "campaign_id": "cid",
                "campaign_root": str(campaign_root),
                "summary_json": str(summary_path),
                "benchmark_success": False,
                "campaign_execution_status": "failed",
                "evidence_status": "invalid",
                "row_status_summary": {
                    "successful_evidence_rows": 0,
                    "accepted_unavailable_rows": 0,
                    "unexpected_failed_rows": 1,
                    "fallback_or_degraded_rows": 0,
                },
            }

        monkeypatch.setattr(
            run_camera_ready_benchmark, "load_campaign_config", lambda path: object()
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "prepare_campaign_preflight",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
        )
        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)

        launcher_exit = run_camera_ready_benchmark.main(["--config", str(config_path)])
        assert launcher_exit == 2
        envelope_path = campaign_root / "reports" / "post_campaign_stage_status.json"
        assert envelope_path.is_file()

        # Successful report stage must not launder a failed campaign to success.
        stage_ok = tmp_path / "stage_ok.sh"
        stage_ok.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        stage_ok.chmod(0o755)
        run_post_campaign_stage.main(
            [
                "--campaign-summary",
                str(summary_path),
                "--campaign-exit-code",
                "2",
                "--stage-name",
                "headline_ci_rank_stability_report",
                "--output",
                str(envelope_path),
                "--stage-command",
                str(stage_ok),
            ]
        )
        payload = json.loads(envelope_path.read_text(encoding="utf-8"))
        assert payload["campaign"]["exit_code"] == 2
        assert payload["job_exit_code"] == 2
        assert payload["post_campaign_stage"]["exit_code"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
