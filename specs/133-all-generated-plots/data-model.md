# Data Model: Fix Benchmark Placeholder Outputs

**Date**: 2025-09-24
**Feature**: 133-all-generated-plots

## Overview
This feature extends the benchmark system to generate real visual outputs instead of placeholders. The data model focuses on the artifacts produced by benchmark execution and their relationships to episode data.

## Core Entities

### EpisodeRecord
**Purpose**: Raw benchmark execution data that serves as input for visualization
**Fields**:
- `episode_id`: string (unique identifier)
- `seed`: int (reproducibility seed)
- `scenario_id`: string (scenario configuration identifier)
- `scenario_params`: dict (scenario parameters)
- `metrics`: dict (computed performance metrics)
- `timing`: dict (execution timing data)
- `status`: string (completion status)
- `trajectory_data`: array (optional robot/pedestrian positions over time)

**Validation Rules**:
- `episode_id` must be unique across all episodes
- `metrics` must contain required fields (collisions, success, etc.)
- `status` must be one of: "completed", "failed", "aborted"

### BenchmarkRun
**Purpose**: Metadata about a complete benchmark execution
**Fields**:
- `run_id`: string (timestamp-based identifier)
- `scenarios`: array (list of scenario configurations used)
- `baselines`: array (list of baseline algorithms executed)
- `output_dir`: string (directory containing all outputs)
- `start_time`: datetime
- `end_time`: datetime
- `status`: string (overall run status)

**Relationships**:
- Contains multiple EpisodeRecord instances
- Produces VisualArtifact instances

### VisualArtifact
**Purpose**: Generated plots and videos from benchmark data
**Fields**:
- `artifact_id`: string (unique identifier)
- `artifact_type`: string ("plot" or "video")
- `format`: string ("pdf" for plots, "mp4" for videos)
- `filename`: string (output filename)
- `source_data`: string (description of data used to generate)
- `generation_time`: datetime
- `file_size`: int (bytes)
- `status`: string ("generated", "failed", "placeholder")

**Validation Rules**:
- `status` must be "generated" for successful completion
- `file_size` must be > 0 for generated artifacts
- `filename` must follow naming convention: `{type}_{scenario}_{baseline}.{ext}`

## Data Flow

### Input Processing
1. **Episode Data Loading**: Parse JSONL files from benchmark execution
2. **Data Validation**: Ensure required metrics and trajectory data present
3. **Aggregation**: Group episodes by scenario and baseline for analysis

### Visualization Generation
1. **Plot Generation**: Extract metrics → create matplotlib figures → save as PDF
2. **Video Generation**: Extract trajectories → replay in environment → render frames → encode MP4
3. **Artifact Validation**: Verify outputs are real data, not placeholders

### Output Structure
```
results/{run_id}/
├── episodes.jsonl          # Raw episode data
├── plots/                  # Generated PDF plots
│   ├── metrics_distribution.pdf
│   ├── trajectory_comparison.pdf
│   └── performance_summary.pdf
└── videos/                 # Generated MP4 videos
    ├── scenario_001_baseline_socialforce.mp4
    └── scenario_002_baseline_random.mp4
```

## State Transitions

### BenchmarkRun States
- `initializing` → `running` → `processing_visuals` → `completed`
- Any state can transition to `failed` on error

### VisualArtifact States
- `pending` → `generating` → `generated`
- `pending` → `failed` (with error details)

## Error Handling

### Data Validation Errors
- **Missing Metrics**: Episode lacks required metric fields
- **Invalid Trajectories**: Trajectory data malformed or incomplete
- **Corrupt Files**: JSONL parsing failures

### Generation Errors
- **Dependency Missing**: Matplotlib/MoviePy not available
- **Rendering Failure**: Video generation crashes
- **Disk Space**: Insufficient space for outputs

### Recovery Strategies
- **Fallback to Placeholders**: Generate placeholder with clear warning
- **Partial Success**: Complete run with some artifacts failed
- **Detailed Logging**: Record specific failure reasons for debugging

## Performance Considerations

### Memory Usage
- Load episode data in batches for large benchmark runs
- Clean up intermediate rendering data after video generation

### Processing Time
- Plot generation: < 30 seconds for typical benchmark sizes
- Video generation: < 60 seconds per scenario (configurable)

### Scalability
- Support parallel video rendering for multiple scenarios
- Implement progress tracking for long-running generations