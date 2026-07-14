#!/usr/bin/env python3
"""Regression tests for the post-campaign stage runner (issue #5244).

These exercise the real subprocess dispatch + serialization boundary: a chained
SNQI-style step that exits 5 (optional-deps-missing, the job-13274 candidate) must be
recorded in the ``post_campaign_stage`` lane without remapping a completed campaign's
exit 0. The hard-fail path must stay nonzero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.tools import run_post_campaign_stage

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPostCampaignStageRunner:
    """Issue #5244: post-campaign stage exit 5 must not remap a completed campaign."""

    def test_completed_campaign_keeps_exit_zero_when_stage_exits_five(self, tmp_path: Path) -> None:
        """A chained stage exit 5 is recorded separately; process exit stays 0."""
        summary = tmp_path / "campaign" / "reports" / "campaign_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps({"soft_contract_warning": True, "warnings": ["SNQI warn"]}),
            encoding="utf-8",
        )
        output = tmp_path / "stage_status.json"
        stage_script = tmp_path / "stage_fail.sh"
        stage_script.write_text("#!/usr/bin/env bash\nexit 5\n", encoding="utf-8")
        stage_script.chmod(0o755)

        exit_code, payload = run_post_campaign_stage.build_post_campaign_stage_payload(
            campaign_summary_path=summary,
            campaign_exit_code=0,
            stage_name="snqi_recompute_weights",
            stage_command=[str(stage_script)],
            output=output,
        )

        assert exit_code == 0
        assert payload["schema_version"] == "robot-sf-post-campaign-stage-status.v1"
        assert payload["campaign"]["exit_code"] == 0
        assert payload["campaign"]["status"] == "completed"
        assert payload["campaign"]["soft_contract_warning"] is True
        assert payload["job_exit_code"] == 0
        assert payload["post_campaign_stage"]["name"] == "snqi_recompute_weights"
        assert payload["post_campaign_stage"]["exit_code"] == 5
        assert payload["post_campaign_stage"]["status"] == "report_stage_failed"
        assert output.is_file()

    def test_hard_campaign_failure_stays_nonzero_when_stage_succeeds(self, tmp_path: Path) -> None:
        """A hard campaign exit must remain nonzero even if the stage succeeds."""
        summary = tmp_path / "campaign" / "reports" / "campaign_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        output = tmp_path / "stage_status.json"
        stage_script = tmp_path / "stage_ok.sh"
        stage_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        stage_script.chmod(0o755)

        exit_code, payload = run_post_campaign_stage.build_post_campaign_stage_payload(
            campaign_summary_path=summary,
            campaign_exit_code=2,
            stage_name="snqi_recompute_weights",
            stage_command=[str(stage_script)],
            output=output,
        )

        assert exit_code == 2
        assert payload["campaign"]["exit_code"] == 2
        assert payload["campaign"]["status"] == "failed"
        assert payload["job_exit_code"] == 2
        assert payload["post_campaign_stage"]["exit_code"] == 0
        assert payload["post_campaign_stage"]["status"] == "completed"

    def test_missing_stage_command_records_optional_deps_missing_lane(self, tmp_path: Path) -> None:
        """A missing chained command is reported as the optional-deps-missing lane."""
        summary = tmp_path / "campaign" / "reports" / "campaign_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        output = tmp_path / "stage_status.json"

        exit_code, payload = run_post_campaign_stage.build_post_campaign_stage_payload(
            campaign_summary_path=summary,
            campaign_exit_code=0,
            stage_name="snqi_sensitivity_analysis",
            stage_command=["/nonexistent/command_5244"],
            output=output,
        )

        assert exit_code == 0
        assert payload["post_campaign_stage"]["exit_code"] == 5
        assert payload["post_campaign_stage"]["status"] == "report_stage_failed"

    def test_cli_runs_real_subprocess_and_writes_envelope(self, tmp_path: Path) -> None:
        """The CLI entry point drives a real subprocess at the dispatch boundary."""
        summary = tmp_path / "campaign" / "reports" / "campaign_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps({"soft_contract_warning": True, "warnings": []}),
            encoding="utf-8",
        )
        output = tmp_path / "stage_status.json"
        stage_script = tmp_path / "stage_fail.sh"
        stage_script.write_text("#!/usr/bin/env bash\nexit 5\n", encoding="utf-8")
        stage_script.chmod(0o755)

        exit_code = run_post_campaign_stage.main(
            [
                "--campaign-summary",
                str(summary),
                "--campaign-exit-code",
                "0",
                "--stage-name",
                "snqi_recompute_weights",
                "--output",
                str(output),
                "--stage-command",
                str(stage_script),
            ]
        )

        assert exit_code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["job_exit_code"] == 0
        assert payload["post_campaign_stage"]["exit_code"] == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
