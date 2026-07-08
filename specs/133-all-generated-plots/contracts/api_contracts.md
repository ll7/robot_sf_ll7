# API Contracts: Visualization Generation

**Date**: 2025-09-24
**Feature**: 133-all-generated-plots

## Overview
Contracts for the visualization generation functions that will replace placeholder outputs in the benchmark system.

## Function Contracts

### generate_benchmark_plots
**Purpose**: Generate real PDF plots from benchmark episode data
**Location**: `robot_sf/benchmark/visualization.py` (new module)

**Signature**:
```python
def generate_benchmark_plots(
    episodes_path: str,
    output_dir: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None
) -> List[VisualArtifact]:
    """
    Generate comprehensive PDF plots from benchmark episode data.

    Args:
        episodes_path: Path to JSONL file containing episode records
        output_dir: Directory to save generated PDF plots
        scenario_filter: Optional scenario ID to filter plots
        baseline_filter: Optional baseline name to filter plots

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If plotting fails due to data or dependency issues
        FileNotFoundError: If episodes_path doesn't exist
    """
```

**Preconditions**:
- `episodes_path` exists and contains valid JSONL episode data
- `output_dir` is writable
- Matplotlib is available and importable

**Postconditions**:
- PDF files are created in `output_dir/plots/`
- All artifacts have status "generated" or raise exception
- No placeholder images are generated

### generate_benchmark_videos
**Purpose**: Generate real MP4 videos from benchmark episode trajectories
**Location**: `robot_sf/benchmark/visualization.py` (new module)

**Signature**:
```python
def generate_benchmark_videos(
    episodes_path: str,
    output_dir: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None,
    fps: int = 30,
    max_duration: float = 10.0
) -> List[VisualArtifact]:
    """
    Generate MP4 videos by replaying benchmark episodes.

    Args:
        episodes_path: Path to JSONL file containing episode records
        output_dir: Directory to save generated MP4 videos
        scenario_filter: Optional scenario ID to filter videos
        baseline_filter: Optional baseline name to filter videos
        fps: Frames per second for video rendering
        max_duration: Maximum video duration in seconds

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If video rendering fails
        DependencyError: If MoviePy or rendering dependencies missing
    """
```

**Preconditions**:
- `episodes_path` contains episodes with trajectory data
- Environment factory functions are available
- MoviePy is installed and importable

**Postconditions**:
- MP4 files are created in `output_dir/videos/`
- Videos show actual simulation replay, not placeholders
- All artifacts have valid file sizes > 0

### validate_visual_artifacts
**Purpose**: Validate that generated artifacts are real and not placeholders
**Location**: `robot_sf/benchmark/visualization.py`

**Signature**:
```python
def validate_visual_artifacts(artifacts: List[VisualArtifact]) -> ValidationResult:
    """
    Validate that visual artifacts contain real data.

    Args:
        artifacts: List of VisualArtifact objects to validate

    Returns:
        ValidationResult with pass/fail status and details
    """
```

**Preconditions**:
- Artifacts exist on disk
- File sizes are reasonable for content type

**Postconditions**:
- Returns validation result with specific failure reasons
- No false positives for real artifacts

## Data Contracts

### VisualArtifact
**Purpose**: Metadata about generated visual outputs

```python
@dataclass
class VisualArtifact:
    artifact_id: str
    artifact_type: Literal["plot", "video"]
    format: Literal["pdf", "mp4"]
    filename: str
    source_data: str
    generation_time: datetime
    file_size: int
    status: Literal["generated", "failed", "placeholder"]
    error_message: Optional[str] = None
```

### VisualizationError
**Purpose**: Custom exception for visualization failures

```python
class VisualizationError(Exception):
    """Raised when visualization generation fails."""

    def __init__(self, message: str, artifact_type: str, details: Optional[dict] = None):
        super().__init__(message)
        self.artifact_type = artifact_type
        self.details = details or {}
```

## Integration Contracts

### Benchmark Orchestrator Integration
**Contract**: The benchmark orchestrator will call visualization functions after episode execution:

```python
# In benchmark orchestrator
def run_full_benchmark(...):
    # ... existing episode execution ...

    # NEW: Generate real visualizations
    try:
        plots = generate_benchmark_plots(episodes_path, output_dir)
        videos = generate_benchmark_videos(episodes_path, output_dir)
        validation = validate_visual_artifacts(plots + videos)

        if not validation.passed:
            logger.warning(f"Some artifacts failed validation: {validation.details}")

    except (VisualizationError, DependencyError) as e:
        logger.error(f"Visualization generation failed: {e}")
        # Continue with placeholder fallback or clear error

    # ... existing completion logic ...
```

### Environment Factory Integration
**Contract**: Video generation will use existing factory functions:

```python
# Video generation will call:
env = make_robot_env(debug=False)  # For replay
# or
env = make_image_robot_env(debug=False)  # If image observations needed
```

## Error Handling Contracts

### Dependency Checking
**Contract**: Functions must check dependencies at runtime:

```python
def _check_dependencies():
    """Check that required packages are available."""
    try:
        import matplotlib
        import moviepy
    except ImportError as e:
        raise DependencyError(f"Missing visualization dependency: {e}")
```

### Graceful Degradation
**Contract**: On failure, provide clear error messages and fallback options:

- Log specific failure reasons
- Suggest remediation steps (install dependencies, check data)
- Allow benchmark to complete with warnings rather than hard failures