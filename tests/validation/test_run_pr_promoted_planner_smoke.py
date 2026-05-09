"""Tests for the PR promoted planner smoke runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.validation import run_pr_promoted_planner_smoke as smoke


def _episode(algorithm: str, *, success: bool = True) -> dict[str, object]:
    return {
        "episode_id": f"{algorithm}-1",
        "scenario_id": "pr_promoted_planner_smoke",
        "seed": 101,
        "algo": algorithm,
        "scenario_params": {"algo": algorithm},
        "metrics": {
            "success": success,
            "collisions": 0,
            "near_misses": 0,
            "time_to_goal_norm": 0.4,
            "path_efficiency": 1.0,
        },
    }


def test_runner_writes_summary_and_markdown(monkeypatch, tmp_path: Path) -> None:
    """Runner writes machine and Markdown summaries for a passing planner."""
    baseline_path = tmp_path / "baseline.json"
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text("scenarios: []\n", encoding="utf-8")
    baseline_path.write_text(
        json.dumps(
            {
                "planners": {
                    "goal": {
                        "minimum_success_mean": 1.0,
                        "maximum_collisions_mean": 0.0,
                        "maximum_near_misses_mean": 0.0,
                        "reference_metrics": {
                            "success.mean": 1.0,
                            "collisions.mean": 0.0,
                            "near_misses.mean": 0.0,
                            "time_to_goal_norm.mean": 0.5,
                            "path_efficiency.mean": 1.0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_run(
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        text: bool,
        stdout: int,
        stderr: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, text, stdout, stderr, check
        out_arg = command[command.index("--out") + 1]
        Path(out_arg).write_text(json.dumps(_episode("goal")) + "\n", encoding="utf-8")
        payload = {
            "event": "benchmark.run.summary",
            "benchmark_availability": {
                "availability_status": "available",
                "readiness_status": "native",
                "benchmark_success": True,
            },
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload) + "\n")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "robot_sf_bench")

    output_root = tmp_path / "out"
    summary_path = tmp_path / "step-summary.md"
    exit_code = smoke.main(
        [
            "--matrix",
            str(matrix_path),
            "--baseline",
            str(baseline_path),
            "--output-root",
            str(output_root),
            "--algorithms",
            "goal",
            "--github-step-summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["planners"][0]["deltas"]["time_to_goal_norm.mean"] == -0.09999999999999998
    assert (output_root / "summary.md").exists()
    assert "PR promoted planner smoke" in summary_path.read_text(encoding="utf-8")


def test_runner_fails_on_degraded_readiness(monkeypatch, tmp_path: Path) -> None:
    """Runner fails closed when benchmark readiness is degraded."""
    baseline_path = tmp_path / "baseline.json"
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text("scenarios: []\n", encoding="utf-8")
    baseline_path.write_text(
        json.dumps(
            {
                "planners": {
                    "goal": {
                        "minimum_success_mean": 1.0,
                        "maximum_collisions_mean": 0.0,
                        "maximum_near_misses_mean": 0.0,
                        "reference_metrics": {"success.mean": 1.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_run(
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        text: bool,
        stdout: int,
        stderr: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, text, stdout, stderr, check
        out_arg = command[command.index("--out") + 1]
        Path(out_arg).write_text(json.dumps(_episode("goal")) + "\n", encoding="utf-8")
        payload = {
            "event": "benchmark.run.summary",
            "benchmark_availability": {
                "availability_status": "available",
                "readiness_status": "degraded",
                "benchmark_success": True,
            },
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload) + "\n")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "robot_sf_bench")

    exit_code = smoke.main(
        [
            "--matrix",
            str(matrix_path),
            "--baseline",
            str(baseline_path),
            "--output-root",
            str(tmp_path / "out"),
            "--algorithms",
            "goal",
        ]
    )

    assert exit_code == 1
    payload = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "readiness_status='degraded'" in payload["planners"][0]["failures"]


def test_workflow_invokes_runner_and_uploads_summary() -> None:
    """Workflow exposes the runner, step summary, and artifact outputs."""
    workflow = Path(".github/workflows/pr-promoted-planner-smoke.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "scripts/validation/run_pr_promoted_planner_smoke.py" in workflow
    assert "--github-step-summary" in workflow
    assert "output/benchmarks/pr_promoted_planner_smoke/summary.json" in workflow
    assert "output/benchmarks/pr_promoted_planner_smoke/summary.md" in workflow
