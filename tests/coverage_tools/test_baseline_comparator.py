"""
Tests for baseline coverage comparison functionality.

Validates the core comparison logic, delta calculation, and warning generation
for CI/CD coverage monitoring.
"""

import json

import pytest

from robot_sf.coverage_tools.baseline_comparator import (
    CoverageBaseline,
    CoverageDelta,
    CoverageSnapshot,
    compare,
    generate_warning,
    load_baseline,
)


def test_coverage_snapshot_from_json(sample_coverage_data):
    """Test CoverageSnapshot creation from coverage.json structure."""
    snapshot = CoverageSnapshot.from_coverage_json(
        sample_coverage_data, timestamp="2025-10-23T12:00:00"
    )

    assert snapshot.total_coverage == 66.67
    assert len(snapshot.file_coverage) == 2
    assert snapshot.file_coverage["robot_sf/gym_env/environment.py"] == 70.0
    assert snapshot.file_coverage["robot_sf/sim/simulator.py"] == 62.5
    assert snapshot.timestamp == "2025-10-23T12:00:00"


def test_load_baseline_missing_file(tmp_path):
    """Test load_baseline with missing baseline file."""
    baseline = load_baseline(tmp_path / "nonexistent.json")
    assert baseline is None


def test_load_baseline_valid_file(tmp_path, sample_coverage_data):
    """Test load_baseline with valid baseline file."""
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(sample_coverage_data))

    baseline = load_baseline(baseline_path)

    assert baseline is not None
    assert baseline.snapshot.total_coverage == 66.67
    assert str(baseline_path) in baseline.source


def test_load_baseline_invalid_json(tmp_path):
    """Test load_baseline with invalid JSON."""
    baseline_path = tmp_path / "invalid.json"
    baseline_path.write_text("not valid json {")

    baseline = load_baseline(baseline_path)
    assert baseline is None


def test_compare_no_baseline(tmp_path, sample_coverage_data):
    """Test compare with no baseline (first run scenario)."""
    current_path = tmp_path / "coverage.json"
    current_path.write_text(json.dumps(sample_coverage_data))

    delta = compare(current_path, baseline=None, threshold=1.0)

    assert delta.current_coverage == 66.67
    assert delta.baseline_coverage == 66.67
    assert delta.delta == 0.0
    assert not delta.has_decrease
    assert "No baseline" in delta.warnings[0]


def test_compare_coverage_decreased(tmp_path, sample_coverage_data):
    """Test compare when coverage has decreased."""
    # Create baseline with higher coverage (deep copy to avoid mutation)
    import copy

    baseline_data = copy.deepcopy(sample_coverage_data)
    baseline_data["totals"]["percent_covered"] = 75.0

    baseline_snapshot = CoverageSnapshot.from_coverage_json(
        baseline_data, timestamp="2025-10-22T12:00:00"
    )
    baseline = CoverageBaseline(snapshot=baseline_snapshot, source="main")

    # Current coverage is lower (66.67%)
    current_path = tmp_path / "coverage.json"
    current_path.write_text(json.dumps(sample_coverage_data))

    delta = compare(current_path, baseline, threshold=1.0)

    assert delta.current_coverage == 66.67
    assert delta.baseline_coverage == 75.0
    assert delta.delta == pytest.approx(-8.33, abs=0.01)
    assert delta.has_decrease
    assert len(delta.warnings) > 0
    assert "decreased" in delta.warnings[0].lower()


def test_compare_coverage_increased(tmp_path, sample_coverage_data):
    """Test compare when coverage has increased."""
    # Create baseline with lower coverage (deep copy to avoid mutation)
    import copy

    baseline_data = copy.deepcopy(sample_coverage_data)
    baseline_data["totals"]["percent_covered"] = 60.0

    baseline_snapshot = CoverageSnapshot.from_coverage_json(
        baseline_data, timestamp="2025-10-22T12:00:00"
    )
    baseline = CoverageBaseline(snapshot=baseline_snapshot, source="main")

    # Current coverage is higher (66.67%)
    current_path = tmp_path / "coverage.json"
    current_path.write_text(json.dumps(sample_coverage_data))

    delta = compare(current_path, baseline, threshold=1.0)

    assert delta.current_coverage == 66.67
    assert delta.baseline_coverage == 60.0
    assert delta.delta == pytest.approx(6.67, abs=0.01)
    assert delta.has_increase
    assert not delta.has_decrease


def test_compare_missing_current_file(tmp_path):
    """Test compare with missing current coverage file."""
    with pytest.raises(FileNotFoundError):
        compare(tmp_path / "missing.json", baseline=None)


def test_coverage_delta_threshold_logic():
    """Test CoverageDelta threshold logic."""
    # Decrease beyond threshold
    delta = CoverageDelta(
        current_coverage=65.0,
        baseline_coverage=70.0,
        delta=-5.0,
        threshold=1.0,
        changed_files=[],
        warnings=[],
    )
    assert delta.has_decrease
    assert not delta.has_increase

    # Increase
    delta_increase = CoverageDelta(
        current_coverage=75.0,
        baseline_coverage=70.0,
        delta=5.0,
        threshold=1.0,
        changed_files=[],
        warnings=[],
    )
    assert not delta_increase.has_decrease
    assert delta_increase.has_increase

    # Small change within threshold
    delta_stable = CoverageDelta(
        current_coverage=70.5,
        baseline_coverage=70.0,
        delta=0.5,
        threshold=1.0,
        changed_files=[],
        warnings=[],
    )
    assert not delta_stable.has_decrease
    assert not delta_stable.has_increase


def test_generate_warning_terminal_format():
    """Test terminal format warning generation."""
    delta = CoverageDelta(
        current_coverage=65.0,
        baseline_coverage=70.0,
        delta=-5.0,
        threshold=1.0,
        changed_files=[
            {
                "file": "robot_sf/gym_env/environment.py",
                "current": 60.0,
                "baseline": 70.0,
                "delta": -10.0,
            }
        ],
        warnings=["Coverage decreased"],
    )

    warning = generate_warning(delta, format_type="terminal")

    assert "WARNING" in warning
    assert "65.00%" in warning
    assert "70.00%" in warning
    assert "-5.00%" in warning
    assert "robot_sf/gym_env/environment.py" in warning


def test_generate_warning_github_format():
    """Test GitHub Actions annotation format."""
    delta = CoverageDelta(
        current_coverage=65.0,
        baseline_coverage=70.0,
        delta=-5.0,
        threshold=1.0,
        changed_files=[
            {
                "file": "robot_sf/gym_env/environment.py",
                "current": 60.0,
                "baseline": 70.0,
                "delta": -10.0,
            }
        ],
        warnings=["Coverage decreased"],
    )

    warning = generate_warning(delta, format_type="github")

    assert "::warning" in warning
    assert "Coverage Decreased" in warning or "Coverage dropped" in warning


def test_generate_warning_json_format():
    """Test JSON format warning generation."""
    delta = CoverageDelta(
        current_coverage=65.0,
        baseline_coverage=70.0,
        delta=-5.0,
        threshold=1.0,
        changed_files=[],
        warnings=["Coverage decreased"],
    )

    warning = generate_warning(delta, format_type="json")
    data = json.loads(warning)

    assert data["has_decrease"] is True
    assert data["current_coverage"] == 65.0
    assert data["baseline_coverage"] == 70.0
    assert data["delta"] == -5.0


def test_generate_warning_no_decrease():
    """Test warning generation when coverage is stable."""
    delta = CoverageDelta(
        current_coverage=70.0,
        baseline_coverage=70.0,
        delta=0.0,
        threshold=1.0,
        changed_files=[],
        warnings=[],
    )

    warning_terminal = generate_warning(delta, format_type="terminal")
    assert "maintained" in warning_terminal.lower() or "improved" in warning_terminal.lower()

    warning_github = generate_warning(delta, format_type="github")
    assert warning_github == ""  # No GitHub annotation for stable coverage
