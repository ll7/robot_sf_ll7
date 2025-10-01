"""
Tests for visual artifact validation.

These tests verify that the validate_visual_artifacts function correctly
identifies real visualizations vs placeholders and malformed artifacts.
"""

from datetime import datetime

from robot_sf.benchmark.visualization import VisualArtifact, validate_visual_artifacts


class TestValidateVisualArtifacts:
    """Test suite for validate_visual_artifacts function."""

    def test_empty_artifacts_list(self):
        """Test validation of empty artifacts list."""
        result = validate_visual_artifacts([])

        assert result.passed is True
        assert result.failed_artifacts == []
        assert result.details == {"message": "No artifacts to validate"}

    def test_valid_plot_artifact(self):
        """Test validation of a properly generated plot artifact."""
        artifact = VisualArtifact(
            artifact_id="test_plot_001",
            artifact_type="plot",
            format="pdf",
            filename="metrics_distribution.pdf",
            source_data="benchmark_episodes.jsonl",
            generation_time=datetime.now(),
            file_size=102400,  # 100KB
            status="generated",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is True
        assert result.failed_artifacts == []
        assert result.details is not None
        assert result.details["total_artifacts"] == 1
        assert result.details["passed_count"] == 1
        assert result.details["failed_count"] == 0

    def test_valid_video_artifact(self):
        """Test validation of a properly generated video artifact."""
        artifact = VisualArtifact(
            artifact_id="test_video_001",
            artifact_type="video",
            format="mp4",
            filename="scenario_socialforce.mp4",
            source_data="episode_12345",
            generation_time=datetime.now(),
            file_size=5242880,  # 5MB
            status="generated",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is True
        assert result.failed_artifacts == []
        assert result.details is not None
        assert result.details["total_artifacts"] == 1

    def test_failed_status_artifact(self):
        """Test that artifacts with failed status are rejected."""
        artifact = VisualArtifact(
            artifact_id="failed_plot",
            artifact_type="plot",
            format="pdf",
            filename="failed_metrics.pdf",
            source_data="benchmark_episodes.jsonl",
            generation_time=datetime.now(),
            file_size=1024,
            status="failed",
            error_message="Plot generation failed",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is False
        assert len(result.failed_artifacts) == 1
        assert result.failed_artifacts[0].artifact_id == "failed_plot"
        assert result.details is not None
        assert result.details["failed_count"] == 1

    def test_zero_file_size_artifact(self):
        """Test that artifacts with zero file size are rejected."""
        artifact = VisualArtifact(
            artifact_id="empty_plot",
            artifact_type="plot",
            format="pdf",
            filename="empty_plot.pdf",
            source_data="benchmark_episodes.jsonl",
            generation_time=datetime.now(),
            file_size=0,  # Empty file
            status="generated",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is False
        assert len(result.failed_artifacts) == 1
        assert result.failed_artifacts[0].artifact_id == "empty_plot"

    def test_wrong_file_extension_plot(self):
        """Test that plot artifacts with wrong extension are rejected."""
        artifact = VisualArtifact(
            artifact_id="wrong_ext_plot",
            artifact_type="plot",
            format="pdf",
            filename="metrics_distribution.png",  # Wrong extension
            source_data="benchmark_episodes.jsonl",
            generation_time=datetime.now(),
            file_size=102400,
            status="generated",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is False
        assert len(result.failed_artifacts) == 1

    def test_wrong_file_extension_video(self):
        """Test that video artifacts with wrong extension are rejected."""
        artifact = VisualArtifact(
            artifact_id="wrong_ext_video",
            artifact_type="video",
            format="mp4",
            filename="scenario_socialforce.avi",  # Wrong extension
            source_data="episode_12345",
            generation_time=datetime.now(),
            file_size=5242880,
            status="generated",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is False
        assert len(result.failed_artifacts) == 1

    def test_placeholder_in_source_data(self):
        """Test that artifacts with placeholder indicators in source_data are rejected."""
        artifact = VisualArtifact(
            artifact_id="placeholder_source",
            artifact_type="plot",
            format="pdf",
            filename="metrics.pdf",
            source_data="PLACEHOLDER: Real data not available",  # Contains placeholder
            generation_time=datetime.now(),
            file_size=102400,
            status="generated",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is False
        assert len(result.failed_artifacts) == 1

    def test_placeholder_in_filename(self):
        """Test that artifacts with placeholder indicators in filename are rejected."""
        artifact = VisualArtifact(
            artifact_id="placeholder_filename",
            artifact_type="plot",
            format="pdf",
            filename="TODO_implement_real_plots.pdf",  # Contains TODO
            source_data="benchmark_episodes.jsonl",
            generation_time=datetime.now(),
            file_size=102400,
            status="generated",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is False
        assert len(result.failed_artifacts) == 1

    def test_mixed_valid_invalid_artifacts(self):
        """Test validation of a mix of valid and invalid artifacts."""
        valid_artifact = VisualArtifact(
            artifact_id="valid_plot",
            artifact_type="plot",
            format="pdf",
            filename="valid_metrics.pdf",
            source_data="benchmark_episodes.jsonl",
            generation_time=datetime.now(),
            file_size=102400,
            status="generated",
        )

        invalid_artifact = VisualArtifact(
            artifact_id="invalid_plot",
            artifact_type="plot",
            format="pdf",
            filename="placeholder.pdf",  # Contains placeholder
            source_data="benchmark_episodes.jsonl",
            generation_time=datetime.now(),
            file_size=102400,
            status="generated",
        )

        result = validate_visual_artifacts([valid_artifact, invalid_artifact])

        assert result.passed is False
        assert len(result.failed_artifacts) == 1
        assert result.failed_artifacts[0].artifact_id == "invalid_plot"
        assert result.details is not None
        assert result.details["total_artifacts"] == 2
        assert result.details["passed_count"] == 1
        assert result.details["failed_count"] == 1

    def test_case_insensitive_placeholder_detection(self):
        """Test that placeholder detection is case-insensitive."""
        artifact = VisualArtifact(
            artifact_id="case_insensitive",
            artifact_type="video",
            format="mp4",
            filename="Placeholder_Video.mp4",  # Mixed case
            source_data="episode_data",
            generation_time=datetime.now(),
            file_size=5242880,
            status="generated",
        )

        result = validate_visual_artifacts([artifact])

        assert result.passed is False
        assert len(result.failed_artifacts) == 1
