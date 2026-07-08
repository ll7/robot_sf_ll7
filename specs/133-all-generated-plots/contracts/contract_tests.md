# Contract Tests: Visualization Generation

**Date**: 2025-09-24
**Feature**: 133-all-generated-plots

## Test Structure
Contract tests that validate the visualization API contracts. These tests will fail until implementation is complete.

## Test Cases

### test_generate_benchmark_plots_contract
```python
def test_generate_benchmark_plots_contract(tmp_path, sample_episodes_jsonl):
    """Test that generate_benchmark_plots meets its contract."""

    # Given: Valid episode data
    episodes_path = sample_episodes_jsonl
    output_dir = tmp_path / "output"

    # When: Function is called
    artifacts = generate_benchmark_plots(episodes_path, output_dir)

    # Then: Contract is satisfied
    assert isinstance(artifacts, list)
    assert len(artifacts) > 0

    for artifact in artifacts:
        assert artifact.status == "generated"
        assert artifact.file_size > 0
        assert (output_dir / "plots" / artifact.filename).exists()
        assert artifact.filename.endswith(".pdf")
```

### test_generate_benchmark_videos_contract
```python
def test_generate_benchmark_videos_contract(tmp_path, sample_episodes_jsonl):
    """Test that generate_benchmark_videos meets its contract."""

    # Given: Episode data with trajectories
    episodes_path = sample_episodes_jsonl
    output_dir = tmp_path / "output"

    # When: Function is called
    artifacts = generate_benchmark_videos(episodes_path, output_dir)

    # Then: Contract is satisfied
    assert isinstance(artifacts, list)

    for artifact in artifacts:
        assert artifact.status == "generated"
        assert artifact.file_size > 0
        assert (output_dir / "videos" / artifact.filename).exists()
        assert artifact.filename.endswith(".mp4")
```

### test_visualization_error_handling
```python
def test_visualization_error_handling(tmp_path):
    """Test error handling for missing dependencies or invalid data."""

    # Given: Missing matplotlib
    episodes_path = "nonexistent.jsonl"

    # When/Then: Appropriate exceptions raised
    with pytest.raises(FileNotFoundError):
        generate_benchmark_plots(episodes_path, tmp_path)

    # Given: Missing MoviePy
    with patch.dict('sys.modules', {'moviepy': None}):
        with pytest.raises(DependencyError):
            generate_benchmark_videos("dummy.jsonl", tmp_path)
```

### test_validate_visual_artifacts_contract
```python
def test_validate_visual_artifacts_contract(tmp_path, real_artifacts, fake_artifacts):
    """Test that artifact validation correctly identifies real vs fake outputs."""

    # Given: Mix of real and placeholder artifacts
    artifacts = real_artifacts + fake_artifacts

    # When: Validation is run
    result = validate_visual_artifacts(artifacts)

    # Then: Real artifacts pass, fakes fail
    assert result.passed == False  # Some artifacts are fake
    assert len(result.failed_artifacts) == len(fake_artifacts)
    assert len(result.passed_artifacts) == len(real_artifacts)
```

### test_visualization_integration_contract
```python
def test_visualization_integration_contract(tmp_path, benchmark_config):
    """Test that visualization integrates properly with benchmark orchestrator."""

    # Given: Complete benchmark run
    config = benchmark_config
    output_dir = tmp_path / "benchmark_output"

    # When: Full benchmark with visualization is run
    run_full_benchmark(config, output_dir)

    # Then: Both plots and videos are generated
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"

    assert plots_dir.exists()
    assert videos_dir.exists()

    # And: Contain real files, not placeholders
    plot_files = list(plots_dir.glob("*.pdf"))
    video_files = list(videos_dir.glob("*.mp4"))

    assert len(plot_files) > 0
    assert len(video_files) > 0

    # Validate no placeholders
    validation = validate_visual_artifacts_from_dir(output_dir)
    assert validation.all_real == True
```

## Fixtures Required

### sample_episodes_jsonl
```python
@pytest.fixture
def sample_episodes_jsonl(tmp_path):
    """Create sample episode data for testing."""
    episodes = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "metrics": {"collisions": 0, "success": True, "snqi": 0.95},
            "trajectory_data": [[0.0, 0.0], [1.0, 1.0]]  # Simplified trajectory
        }
    ]

    file_path = tmp_path / "episodes.jsonl"
    with open(file_path, 'w') as f:
        for ep in episodes:
            f.write(json.dumps(ep) + '\n')

    return file_path
```

### benchmark_config
```python
@pytest.fixture
def benchmark_config():
    """Provide benchmark configuration for integration tests."""
    return {
        "scenarios": ["classic_interactions.yaml"],
        "baselines": ["socialforce", "random"],
        "episodes_per_scenario": 5
    }
```

## Test Execution Notes

- **Expected to fail initially**: These are contract tests that define the expected behavior
- **Run after implementation**: Use these to validate that contracts are met
- **Integration testing**: The final test validates end-to-end integration
- **Performance bounds**: Tests should complete within reasonable time limits (< 30s for plots, < 60s for videos)