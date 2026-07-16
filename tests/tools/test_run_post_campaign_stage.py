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

from scripts.tools import run_post_campaign_stage, slurm_job_finalize

REPO_ROOT = Path(__file__).resolve().parents[2]

# The only public SNQI tool that emits job-13274's observed exit 5
# (EXIT_OPTIONAL_DEPS_MISSING, e.g. matplotlib missing on a compute node). Driven
# here as a real post-campaign step at the production boundary.
SNQI_SENSITIVITY_ANALYSIS = REPO_ROOT / "scripts" / "snqi_sensitivity_analysis.py"


def _write_minimal_snqi_fixtures(base: Path) -> Path:
    """Write a minimal, self-consistent SNQI sensitivity-analysis fixture set.

    Returns the episode JSONL path; weights/baseline paths are derived alongside it.
    """
    (base / "reports").mkdir(parents=True, exist_ok=True)
    episodes = base / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "planner_key": "static",
                "metrics": {
                    "collision_rate": 0.1,
                    "ttc_violation_rate": 0.2,
                    "path_deviation": 0.3,
                    "comfort_jerk": 0.4,
                    "energy_efficiency": 0.5,
                    "progress_smoothness": 0.6,
                    "waiting_time": 0.7,
                    "goal_deviation": 0.8,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fields = [
        "collision_rate",
        "ttc_violation_rate",
        "path_deviation",
        "comfort_jerk",
        "energy_efficiency",
        "progress_smoothness",
        "waiting_time",
        "goal_deviation",
    ]
    baseline = {f: {"mean": 0.1 + 0.1 * i, "std": 0.01} for i, f in enumerate(fields)}
    (base / "baseline.json").write_text(json.dumps({"static": baseline}), encoding="utf-8")
    (base / "weights.json").write_text(json.dumps(dict.fromkeys(fields, 1.0)), encoding="utf-8")
    return episodes


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

    def test_real_snqi_analysis_stage_through_production_boundary(self, tmp_path: Path) -> None:
        """The real ``snqi_sensitivity_analysis.py`` post-campaign step must run through
        ``run_post_campaign_stage`` (the job-13274 exit-5 candidate tool) and preserve the
        completed campaign's exit 0 all the way to ``slurm_job_finalize``.

        Prior slices only exercised synthetic ``exit 5`` bash scripts; this drives the
        actual public SNQI tool at the production dispatch/serialization boundary so a
        completed ``enforcement=warn`` campaign cannot be relabeled by a downstream
        analysis step.
        """
        campaign_root = tmp_path / "campaign"
        summary = campaign_root / "reports" / "campaign_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps({"soft_contract_warning": True, "warnings": ["SNQI warn"]}),
            encoding="utf-8",
        )
        episodes = _write_minimal_snqi_fixtures(tmp_path)
        envelope_path = tmp_path / "stage_status.json"
        snqi_out = tmp_path / "snqi_out"

        primitive_exit = run_post_campaign_stage.main(
            [
                "--campaign-summary",
                str(summary),
                "--campaign-exit-code",
                "0",
                "--stage-name",
                "snqi_sensitivity_analysis",
                "--output",
                str(envelope_path),
                "--stage-command",
                sys.executable,
                str(SNQI_SENSITIVITY_ANALYSIS),
                "--episodes",
                str(episodes),
                "--baseline",
                str(tmp_path / "baseline.json"),
                "--weights",
                str(tmp_path / "weights.json"),
                "--output",
                str(snqi_out),
                "--skip-visualizations",
            ]
        )
        assert primitive_exit == 0

        payload = json.loads(envelope_path.read_text(encoding="utf-8"))
        assert payload["campaign"]["exit_code"] == 0
        assert payload["campaign"]["soft_contract_warning"] is True
        assert payload["job_exit_code"] == 0
        assert payload["post_campaign_stage"]["exit_code"] == 0
        assert payload["post_campaign_stage"]["name"] == "snqi_sensitivity_analysis"
        # The real tool wrote its own artifacts (proof it actually ran).
        assert (snqi_out / "sensitivity_analysis_results.json").is_file()

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
        assert lanes["post_campaign_stage"]["exit_code"] == 0


class TestWrapperArgvContractIssue5707:
    """Issue #5707: the issue-3216 wrapper must dispatch a ``run_post_campaign_stage``
    argv shape that argparse accepts without ambiguity.

    A prior regression passed ``--campaign`` (an option the report builder owns) in a
    position where ``run_post_campaign_stage``'s parser rejected it as ambiguous against
    its own ``--campaign-summary`` / ``--campaign-exit-code`` options. The wrapper now
    emits ``--campaign`` only inside ``--stage-command`` (argparse.REMAINDER), so it must
    parse cleanly. This test guards the exact dispatch contract so the optional
    predictive readiness lane cannot silently regress again.
    """

    def test_argparse_accepts_wrapper_dispatch_shape(self, tmp_path: Path) -> None:
        """The wrapper's exact argv parses without an ambiguous-option error."""
        campaign_root = tmp_path / "issue3216_s20_headline_ci"
        config = tmp_path / "benchmark.yaml"
        stage_status = campaign_root / "reports" / "post_campaign_stage_status.json"
        summary = campaign_root / "reports" / "campaign_summary.json"

        argv = [
            "--campaign-summary",
            str(summary),
            "--campaign-exit-code",
            "0",
            "--stage-name",
            "headline_ci_rank_stability_report",
            "--output",
            str(stage_status),
            "--stage-command",
            "uv",
            "run",
            "python",
            "scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py",
            "--campaign",
            str(campaign_root),
            "--rank-metric",
            "snqi",
            "--expected-planners-from-config",
            str(config),
            "--output-dir",
            str(tmp_path / "report"),
        ]

        # Must not raise argparse.ArgumentError / SystemExit (ambiguous option).
        args = run_post_campaign_stage._parse_args(argv)

        assert args.campaign_summary == summary
        assert args.campaign_exit_code == 0
        assert args.stage_name == "headline_ci_rank_stability_report"
        assert args.output == stage_status
        # The nested ``--campaign`` stays inside the REMAINDER-based stage command.
        assert "--campaign" in args.stage_command
        assert str(campaign_root) in args.stage_command
        # ``--campaign`` must NOT leak into the primitive's own top-level options.
        assert not hasattr(args, "campaign")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
