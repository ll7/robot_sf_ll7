"""Integration test for end-to-end report generation."""

import json

import pytest

from robot_sf.research.orchestrator import ReportOrchestrator


@pytest.fixture
def sample_metric_records():
    """Sample metric records."""
    return [
        {
            "policy_type": "baseline",
            "seed": 42,
            "total_timesteps": 500000,
            "final_reward": 0.85,
        },
        {
            "policy_type": "baseline",
            "seed": 43,
            "total_timesteps": 480000,
            "final_reward": 0.87,
        },
        {
            "policy_type": "baseline",
            "seed": 44,
            "total_timesteps": 520000,
            "final_reward": 0.83,
        },
        {
            "policy_type": "pretrained",
            "seed": 42,
            "total_timesteps": 280000,
            "final_reward": 0.84,
        },
        {
            "policy_type": "pretrained",
            "seed": 43,
            "total_timesteps": 270000,
            "final_reward": 0.86,
        },
        {
            "policy_type": "pretrained",
            "seed": 44,
            "total_timesteps": 290000,
            "final_reward": 0.82,
        },
    ]


@pytest.fixture
def baseline_timesteps():
    """Baseline timesteps to convergence."""
    return [500000.0, 480000.0, 520000.0]


@pytest.fixture
def pretrained_timesteps():
    """Pretrained timesteps to convergence."""
    return [280000.0, 270000.0, 290000.0]


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    output_path = tmp_path / "test_report"
    return output_path


def test_generate_full_report(
    sample_metric_records, baseline_timesteps, pretrained_timesteps, output_dir
):
    """Test end-to-end report generation."""
    orchestrator = ReportOrchestrator(output_dir=output_dir)

    # Generate report
    report_path = orchestrator.generate_report(
        experiment_name="Test Imitation Learning",
        metric_records=sample_metric_records,
        run_id="test_run_001",
        seeds=[42, 43, 44],
        baseline_timesteps=baseline_timesteps,
        pretrained_timesteps=pretrained_timesteps,
        threshold=40.0,
    )

    # Verify report file created
    assert report_path.exists()
    assert report_path.name == "report.md"

    # Verify output structure
    assert (output_dir / "figures").exists()
    assert (output_dir / "data").exists()
    assert (output_dir / "configs").exists()

    # Verify figures created (at minimum sample_efficiency should exist)
    figures_dir = output_dir / "figures"
    assert (figures_dir / "fig-sample-efficiency.pdf").exists()
    assert (figures_dir / "fig-sample-efficiency.png").exists()

    # Verify data exports
    data_dir = output_dir / "data"
    assert (data_dir / "metrics.json").exists()
    assert (data_dir / "metrics.csv").exists()

    # Verify hypothesis export
    assert (data_dir / "hypothesis.json").exists()

    # Verify report content
    report_content = report_path.read_text()
    assert "Test Imitation Learning" in report_content
    assert "Hypothesis" in report_content
    assert "Results" in report_content


def test_generate_report_pass_hypothesis(
    sample_metric_records, baseline_timesteps, pretrained_timesteps, output_dir
):
    """Test report generation with passing hypothesis."""
    orchestrator = ReportOrchestrator(output_dir=output_dir)

    _ = orchestrator.generate_report(
        experiment_name="Pass Hypothesis Test",
        metric_records=sample_metric_records,
        run_id="test_run_002",
        seeds=[42, 43, 44],
        baseline_timesteps=baseline_timesteps,
        pretrained_timesteps=pretrained_timesteps,
        threshold=40.0,
    )

    # Check hypothesis result
    with open(output_dir / "data" / "hypothesis.json", encoding="utf-8") as f:
        hypothesis_data = json.load(f)

    # Should pass with ~44% reduction
    hypothesis = hypothesis_data["hypotheses"][0]
    assert hypothesis["decision"] == "PASS"
    assert hypothesis["measured_value"] >= 40.0


def test_generate_report_fail_hypothesis(
    sample_metric_records, baseline_timesteps, pretrained_timesteps, output_dir
):
    """Test report generation with failing hypothesis."""
    orchestrator = ReportOrchestrator(output_dir=output_dir)

    # High threshold should fail
    _ = orchestrator.generate_report(
        experiment_name="Fail Hypothesis Test",
        metric_records=sample_metric_records,
        run_id="test_run_003",
        seeds=[42, 43, 44],
        baseline_timesteps=baseline_timesteps,
        pretrained_timesteps=pretrained_timesteps,
        threshold=90.0,
    )

    with open(output_dir / "data" / "hypothesis.json", encoding="utf-8") as f:
        hypothesis_data = json.load(f)

    # Should fail with only ~44% reduction
    hypothesis = hypothesis_data["hypotheses"][0]
    assert hypothesis["decision"] == "FAIL"
    assert hypothesis["measured_value"] < 90.0


def test_generate_report_incomplete_data(sample_metric_records, output_dir):
    """Test report generation with incomplete data."""
    orchestrator = ReportOrchestrator(output_dir=output_dir)

    # No timesteps data
    _ = orchestrator.generate_report(
        experiment_name="Incomplete Data Test",
        metric_records=sample_metric_records,
        run_id="test_run_004",
        seeds=[42],
        threshold=40.0,
    )

    with open(output_dir / "data" / "hypothesis.json", encoding="utf-8") as f:
        hypothesis_data = json.load(f)

    # Should mark incomplete
    hypothesis = hypothesis_data["hypotheses"][0]
    assert hypothesis["decision"] == "INCOMPLETE"


def test_metadata_collection(output_dir):
    """Test metadata collection."""
    orchestrator = ReportOrchestrator(output_dir=output_dir)
    metadata = orchestrator.collect_metadata()

    # Verify required fields
    assert "git_commit" in metadata
    assert "git_branch" in metadata
    assert "git_dirty" in metadata
    assert "python_version" in metadata
    assert "hardware" in metadata
    assert "timestamp" in metadata

    # Verify git metadata format
    assert len(metadata["git_commit"]) == 40  # SHA hash length

    # Verify Python version format
    assert "." in metadata["python_version"]

    # Verify hardware metadata
    hardware = metadata["hardware"]
    assert "cpu_cores" in hardware
    assert "memory_gb" in hardware
