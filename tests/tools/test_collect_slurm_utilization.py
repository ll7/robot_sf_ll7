"""Tests for Slurm utilization evidence collection."""

from __future__ import annotations

import json
import subprocess

from scripts.tools.collect_slurm_utilization import collect_job_reports, write_reports


def test_collect_job_reports_runs_accounting_commands(monkeypatch) -> None:
    """Collector should query live, accounting, and efficiency views per job."""
    calls: list[tuple[str, ...]] = []

    def fake_run(command, *, capture_output, text, check):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = collect_job_reports(["12345"], include_seff=True)

    assert report["jobs"][0]["job_id"] == "12345"
    assert ("sstat", "-j", "12345.batch", "--format=JobID,AveCPU,MaxRSS,MaxVMSize,AveRSS") in calls
    assert (
        "sacct",
        "-j",
        "12345",
        "--format=JobID,JobName%40,Partition,State,Elapsed,AllocCPUS,AveCPU,TotalCPU,MaxRSS,ReqMem,MaxVMSize,ExitCode",
        "-P",
    ) in calls
    assert ("seff", "12345") in calls
    assert all(result["returncode"] == 0 for result in report["jobs"][0]["commands"].values())


def test_collect_job_reports_records_missing_tools(monkeypatch) -> None:
    """Missing Slurm helpers should be explicit evidence instead of hard crashes."""

    def fake_run(command, *, capture_output, text, check):
        if command[0] == "sstat":
            raise FileNotFoundError("sstat")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = collect_job_reports(["12345"], include_seff=False)

    sstat = report["jobs"][0]["commands"]["sstat"]
    assert sstat["returncode"] is None
    assert sstat["missing_tool"] == "sstat"
    assert "seff" not in report["jobs"][0]["commands"]


def test_write_reports_emits_json_and_markdown(tmp_path) -> None:
    """Reports should be durable and easy to paste into issue follow-ups."""
    report = {
        "schema_version": "slurm_utilization_evidence.v1",
        "created_at_utc": "2026-05-02T00:00:00+00:00",
        "job_ids": ["12345"],
        "jobs": [
            {
                "job_id": "12345",
                "commands": {
                    "sacct": {
                        "command": ["sacct", "-j", "12345"],
                        "returncode": 0,
                        "stdout": "JobID|State|AllocCPUS|AveCPU\n12345|COMPLETED|24|00:10:00\n",
                        "stderr": "",
                    }
                },
            }
        ],
    }

    json_path = tmp_path / "slurm.json"
    md_path = tmp_path / "slurm.md"
    write_reports(report, output_json=json_path, output_md=md_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["job_ids"] == ["12345"]
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Slurm Utilization Evidence" in markdown
    assert "12345" in markdown
    assert "sacct -j 12345" in markdown
