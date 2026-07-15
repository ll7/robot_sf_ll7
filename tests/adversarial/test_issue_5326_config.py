"""Validation tests for issue #5326 temporal-robustness objective comparison config.

These tests verify the config and runner work correctly in synthetic mode.
They do NOT run actual campaigns - that requires SLURM and is out of scope for CPU validation.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestIssue5326ConfigValidation:
    """Validate issue #5326 objective comparison configuration."""

    @pytest.fixture
    def repo_root(self) -> Path:
        """Return repository root path."""
        return Path(__file__).resolve().parents[2]

    @pytest.fixture
    def config_path(self, repo_root: Path) -> Path:
        """Return path to issue #5326 config file."""
        return repo_root / "configs" / "adversarial" / "issue_5326_objective_comparison.yaml"

    @pytest.fixture
    def runner_path(self, repo_root: Path) -> Path:
        """Return path to comparison runner script."""
        return repo_root / "scripts" / "tools" / "compare_adversarial_samplers.py"

    def test_config_file_exists(self, config_path: Path) -> None:
        """Verify the issue #5326 config file exists."""
        assert config_path.exists(), f"Config not found: {config_path}"

    def test_config_yaml_valid(self, config_path: Path) -> None:
        """Verify the config YAML is parseable."""
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "schema_version" in config
        assert config["issue"] == 5326
        assert "objectives" in config
        assert "worst_case_snqi" in config["objectives"]
        assert "temporal_robustness" in config["objectives"]
        assert "budget_grid" in config
        assert config["budget_grid"] == [16, 32, 64]
        assert "repeated_seeds" in config
        assert config["repeated_seeds"] == [1101, 2202, 3303]

    def test_durable_state_matches_comparison_contract(
        self, repo_root: Path, config_path: Path
    ) -> None:
        """Keep the high-churn integration report aligned with the executable manifest."""
        import yaml

        state_path = repo_root / "docs" / "context" / "issue_5326_state.yaml"
        state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        contract = state["comparison_contract"]

        assert state["issue"] == config["issue"] == 5326
        assert contract["manifest"] == "configs/adversarial/issue_5326_objective_comparison.yaml"
        assert contract["objectives"] == config["objectives"]
        assert contract["budget_grid"] == config["budget_grid"]
        assert contract["repeated_seeds"] == config["repeated_seeds"]
        assert contract["samplers"] == config["samplers"]
        assert contract["reporting_contract"] == config["reporting_contract"]
        assert state["claim_boundary"].find("not matched-budget benchmark evidence") >= 0
        assert state["remaining_gate"]["blockers_new"] == []
        assert state["forbidden_claims"]["paper_or_dissertation_claim"] is False

    def test_runner_script_exists(self, runner_path: Path) -> None:
        """Verify the comparison runner script exists."""
        assert runner_path.exists(), f"Runner not found: {runner_path}"

    def test_temporal_robustness_objective_registered(self, repo_root: Path) -> None:
        """Verify temporal_robustness objective is registered and importable."""
        from robot_sf.adversarial.objectives import get_objective

        obj = get_objective("temporal_robustness")
        assert obj is not None
        assert callable(obj)

    def test_synthetic_smoke_single_objective(self, repo_root: Path, runner_path: Path) -> None:
        """Run a minimal synthetic smoke test with one objective.

        This validates the runner can execute with the temporal_robustness objective
        without running actual campaign episodes. Uses synthetic evaluator for speed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_json = Path(tmpdir) / "report.json"

            cmd = [
                "uv",
                "run",
                "python",
                str(runner_path),
                "--objective",
                "temporal_robustness",
                "--budget",
                "2",
                "--seed",
                "42",
                "--sampler",
                "random",
                "--output-dir",
                str(output_dir),
                "--out-json",
                str(output_json),
                "--synthetic",
            ]

            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            assert result.returncode == 0, f"Runner failed: {result.stderr}"
            assert output_json.exists(), "Output JSON not written"

            with output_json.open("r", encoding="utf-8") as f:
                report = json.load(f)

            assert report["schema_version"] == "adversarial-sampler-comparison.v3"
            assert "temporal_robustness" in report["objectives"]
            assert len(report["rows"]) == 1
            row = report["rows"][0]
            assert row["objective"] == "temporal_robustness"
            assert row["sampler"] == "random"
            assert row["budget"] == 2

    def test_synthetic_smoke_both_objectives(self, repo_root: Path, runner_path: Path) -> None:
        """Run synthetic smoke test comparing both objectives.

        Validates the multi-objective comparison path works.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_json = Path(tmpdir) / "report.json"

            cmd = [
                "uv",
                "run",
                "python",
                str(runner_path),
                "--objective",
                "worst_case_snqi",
                "--objective",
                "temporal_robustness",
                "--budget",
                "2",
                "--seed",
                "42",
                "--sampler",
                "random",
                "--output-dir",
                str(output_dir),
                "--out-json",
                str(output_json),
                "--synthetic",
            ]

            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            assert result.returncode == 0, f"Runner failed: {result.stderr}"
            assert output_json.exists(), "Output JSON not written"

            with output_json.open("r", encoding="utf-8") as f:
                report = json.load(f)

            assert len(report["rows"]) == 2
            objectives = {row["objective"] for row in report["rows"]}
            assert objectives == {"worst_case_snqi", "temporal_robustness"}

    def test_submission_script_exists(self, repo_root: Path) -> None:
        """Verify the SLURM submission wrapper script exists."""
        submission_script = repo_root / "scripts" / "benchmark" / "submit_issue_5326_campaign.sh"
        assert submission_script.exists(), f"Submission script not found: {submission_script}"

    def test_submission_script_executable(self, repo_root: Path) -> None:
        """Verify the submission script is executable."""
        submission_script = repo_root / "scripts" / "benchmark" / "submit_issue_5326_campaign.sh"
        assert submission_script.stat().st_mode & 0o111, "Submission script not executable"

    def test_submission_script_dry_run(self, repo_root: Path) -> None:
        """Verify the submission script dry-run mode works."""
        submission_script = repo_root / "scripts" / "benchmark" / "submit_issue_5326_campaign.sh"

        result = subprocess.run(
            [str(submission_script), "--dry-run"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "DRY RUN MODE" in result.stdout
        assert "--manifest" in result.stdout
        assert str(repo_root / "configs/adversarial/issue_5326_objective_comparison.yaml") in (
            result.stdout
        )
        assert "--out-md" in result.stdout

    def test_submission_script_rejects_unknown_argument(self, repo_root: Path) -> None:
        """Verify typos cannot silently turn a real submission into a dry run."""
        submission_script = repo_root / "scripts" / "benchmark" / "submit_issue_5326_campaign.sh"

        result = subprocess.run(
            [str(submission_script), "--not-a-real-option"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 2
        assert "Usage:" in result.stderr
