"""
Contract tests for validate_visual_artifacts function.

These tests define the expected behavior and will fail until the implementation is complete.
"""

import tempfile
from datetime import datetime
from pathlib import Path

from robot_sf.benchmark.visualization import (
    ValidationResult,
    VisualArtifact,
    validate_visual_artifacts,
)


def test_validate_visual_artifacts_contract():
    """Test that validate_visual_artifacts meets its contract."""

    # Given: Mock VisualArtifact objects
    artifacts = [
        VisualArtifact(
            artifact_id="plot_001",
            artifact_type="plot",
            format="pdf",
            filename="test_plot.pdf",
            source_data="Real plot data",
            generation_time=datetime.now(),
            file_size=1000,
            status="generated",
        )
    ]

    # When: Function is called
    result = validate_visual_artifacts(artifacts)

    # Then: Returns ValidationResult
    assert isinstance(result, ValidationResult)
    assert hasattr(result, "passed")
    assert hasattr(result, "failed_artifacts")
    assert isinstance(result.failed_artifacts, list)


def test_validate_visual_artifacts_with_real_files():
    """Test validate_visual_artifacts with actual generated files."""

    # Given: Create some fake artifact files to simulate real outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        plots_dir = Path(tmp_dir) / "plots"
        plots_dir.mkdir()
        videos_dir = Path(tmp_dir) / "videos"
        videos_dir.mkdir()

        # Create fake PDF and MP4 files
        fake_plot = plots_dir / "test_plot.pdf"
        fake_plot.write_bytes(b"fake pdf content")

        fake_video = videos_dir / "test_video.mp4"
        fake_video.write_bytes(b"fake mp4 content")

        # Create VisualArtifact objects
        artifacts = [
            VisualArtifact(
                artifact_id="plot_001",
                artifact_type="plot",
                format="pdf",
                filename=str(fake_plot),
                source_data="test data",
                generation_time=datetime.now(),
                file_size=fake_plot.stat().st_size,
                status="generated",
            ),
            VisualArtifact(
                artifact_id="video_001",
                artifact_type="video",
                format="mp4",
                filename=str(fake_video),
                source_data="test data",
                generation_time=datetime.now(),
                file_size=fake_video.stat().st_size,
                status="generated",
            ),
        ]

        # When: Function is called
        result = validate_visual_artifacts(artifacts)

        # Then: Validation passes for real-looking files
        assert result.passed
        assert len(result.failed_artifacts) == 0


def test_validate_visual_artifacts_detects_placeholders():
    """Test that validate_visual_artifacts detects placeholder/fake files."""

    # Given: Create obviously fake/placeholder files
    with tempfile.TemporaryDirectory() as tmp_dir:
        plots_dir = Path(tmp_dir) / "plots"
        plots_dir.mkdir()

        # Create a clearly placeholder file
        fake_plot = plots_dir / "placeholder.pdf"
        fake_plot.write_text("This is a placeholder - TODO: implement real plots")

        artifacts = [
            VisualArtifact(
                artifact_id="placeholder_001",
                artifact_type="plot",
                format="pdf",
                filename=str(fake_plot),
                source_data="placeholder data",
                generation_time=datetime.now(),
                file_size=fake_plot.stat().st_size,
                status="generated",
            )
        ]

        # When: Function is called
        result = validate_visual_artifacts(artifacts)

        # Then: Validation fails for placeholder files
        assert not result.passed
        assert len(result.failed_artifacts) == 1


def test_validate_visual_artifacts_empty_list():
    """Test validate_visual_artifacts handles empty artifact list."""

    # When: Function called with empty list
    result = validate_visual_artifacts([])

    # Then: Returns valid result for empty list
    assert result.passed
    assert len(result.failed_artifacts) == 0
