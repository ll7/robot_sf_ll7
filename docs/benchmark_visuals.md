# Benchmark Visual Artifacts

> **UPDATED**: As of feature `133-all-generated-plots`, benchmark outputs now generate **real statistical plots and simulation videos** instead of placeholders. The visualization system produces actual data-driven PDFs and MP4s from episode metrics and trajectories.

> Previous documentation for feature branch `127-enhance-benchmark-visual` – unified generation of plot & video artifacts for the *Full Classic Benchmark* with SimulationView rendering, synthetic fallback, schema validation, and performance instrumentation.

## Table of Contents
- [Goals](#goals)
- [Lifecycle Overview](#lifecycle-overview)
- [Renderer Selection Flow](#renderer-selection-flow)
- [Replay Data Requirements](#replay-data-requirements)
- [Skip / Status Notes](#skip--status-notes)
- [Performance Metrics](#performance-metrics)
- [Dependency Matrix](#dependency-matrix)
- [Configuration Flags](#configuration-flags)
- [Schema Validation](#schema-validation)
- [Memory Sampling](#memory-sampling)
- [Extending / Future Work](#extending--future-work)

## Real Visualization Generation (Feature 133)

As of feature `133-all-generated-plots`, the benchmark system now generates **real visualizations** instead of placeholder outputs:

### Plot Generation
- **Function**: `robot_sf.benchmark.visualization.generate_benchmark_plots()`
- **Output**: Statistical PDF plots showing actual metric distributions from episode data
- **Types**: Metrics distribution plots, scenario comparison plots
- **Format**: Vector PDFs for publication-quality figures
- **Data Source**: Episode metrics (collisions, success rates, SNQI scores, etc.)

### Video Generation  
- **Function**: `robot_sf.benchmark.visualization.generate_benchmark_videos()`
- **Output**: MP4 videos replaying actual simulation episodes
- **Content**: Robot navigation trajectories with pedestrian interactions
- **Fallback**: Placeholder videos when trajectory data unavailable
- **Data Source**: Episode trajectory data (robot/pedestrian positions over time)

### Validation
- **Function**: `robot_sf.benchmark.visualization.validate_visual_artifacts()`
- **Purpose**: Ensures generated artifacts contain real data, not placeholders
- **Checks**: File existence, non-zero size, content validation

### Integration
The visualization functions are automatically called during benchmark execution:
```python
# In benchmark orchestrator
plots = generate_benchmark_plots(episodes_path, output_dir)
videos = generate_benchmark_videos(episodes_path, output_dir)  
validation = validate_visual_artifacts(plots + videos)
```

### Performance
- Plot generation: < 30 seconds for typical benchmark sizes
- Video generation: < 60 seconds per scenario
- Memory usage: Scales with episode count and trajectory length

## Troubleshooting

### Common Issues

#### Plots Not Generating
**Symptoms**: No PDF files created, or empty/placeholder PDFs
**Causes**:
- Missing matplotlib: `pip install matplotlib`
- Invalid episode data: Check JSONL format and metric fields
- Permission issues: Ensure output directory is writable

**Solutions**:
```bash
# Install dependencies
uv add matplotlib

# Check episode data
uv run python -c "import json; print(json.load(open('episodes.jsonl')))"
```

#### Videos Not Generating
**Symptoms**: No MP4 files created, or placeholder videos
**Causes**:
- Missing moviepy: `pip install moviepy`
- No trajectory data: Episodes lack position/time data
- Environment issues: Factory functions unavailable

**Solutions**:
```bash
# Install dependencies  
uv add moviepy

# Check trajectory data
uv run python -c "
import json
ep = json.loads(open('episodes.jsonl').readline())
print('Trajectory data:', 'trajectory_data' in ep)
"
```

#### Validation Failures
**Symptoms**: `validate_visual_artifacts()` returns failed artifacts
**Causes**:
- Files corrupted during generation
- Insufficient disk space
- Permission issues during file creation

**Solutions**:
```python
from robot_sf.benchmark.visualization import validate_visual_artifacts

result = validate_visual_artifacts(artifacts)
for failed in result.failed_artifacts:
    print(f"Failed: {failed.filename} - {failed.error_message}")
```

#### Performance Issues
**Symptoms**: Generation takes longer than expected
**Causes**:
- Large episode datasets
- Complex trajectory data
- Insufficient memory

**Solutions**:
- Use filters to reduce data: `scenario_filter="scenario1"`
- Increase memory limits
- Run on more powerful hardware

### Error Messages

#### "Missing visualization dependencies"
```
VisualizationError: Missing visualization dependencies: matplotlib
```
**Solution**: Install missing packages with `uv add matplotlib moviepy`

#### "No episode data found"
```
VisualizationError: No episode data found
```
**Solution**: Check that episodes file exists and contains valid JSONL data

#### "Failed to load episode data"
```
VisualizationError: Failed to load episode data: [Errno 2] No such file
```
**Solution**: Verify episodes file path and permissions

### Getting Help

If issues persist:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed: `uv sync`
3. Test with minimal data: single episode, no filters
4. Report issues with episode data samples and error traces

---

## Legacy Documentation (Feature 127)

## Lifecycle Overview
```
records (episodes) --> select subset --> (optional) extract replay episodes
        |                                 |
        v                                 v
   generate_plots()                 attempt SimulationView
        |                                 | success
        | failure/skip                     v
        |----------------------> encode_frames() --> video artifact entry
        |                                           (status, note, timings)
        |                                 failure / unavailable
        |                                        v
        |                                synthetic fallback
        v
write plot_artifacts.json + video_artifacts.json + performance_visuals.json
(optional) validate schemas (env flag)
```

## Renderer Selection Flow
Modes via `cfg.video_renderer`:
- `auto` (default): try SimulationView → if no success, synthetic fallback. If replay captured but insufficient (<2 steps), artifact reclassified as skipped SimulationView with `insufficient-replay-state`.
- `sim-view`: only SimulationView path. If unavailable or encode blocked (e.g. moviepy/ffmpeg missing), outputs skipped SimulationView artifacts with an explanatory note (either `simulation-view-missing` or `moviepy-missing`). No synthetic fallback.
- `synthetic`: force synthetic path (never reclassified) – useful for headless/CI or speed-focused runs.

Decision pseudocode (simplified):
```python
if mode == 'synthetic':
    synthetic()
elif mode == 'sim-view':
    attempt = simulation_view()
    if not attempt: skipped(reason=sim_view_or_moviepy_note)
else:  # auto
    attempt = simulation_view()
    if attempt: return attempt
    synthetic = synthetic()
    maybe_reclassify_insufficient_replay(synthetic)
```

## Replay Data Requirements
A `ReplayEpisode` is valid if:
- ≥ 2 steps.
- Non‑decreasing timestamps.
- Each step supplies `(t, x, y, heading)`; enrichment optionally adds `ped_positions`, `action`, `speed`.

Minimum for SimulationView video: two valid steps.
Optional enrichment (T039) improves visual fidelity (trajectory continuity, ped overlays).

## Skip / Status Notes
| Note | Meaning | Typical Cause | Renderer Field |
|------|---------|---------------|----------------|
| `disabled` | Videos disabled explicitly | `cfg.disable_videos` | sim-view (if available) else synthetic |
| `smoke-mode` | Smoke run skipped heavy artifacts | `cfg.smoke=True` | sim-view or synthetic |
| `simulation-view-missing` | SimulationView unavailable (import or probe failed) | Missing pygame / view init | simulation_view |
| `moviepy-missing` | Encoding stack unavailable | moviepy or ffmpeg missing | simulation_view |
| `insufficient-replay-state` | Replay captured but invalid (<2 steps) | capture enabled, too few steps | simulation_view |
| `render-error:<Type>` | Exception during SimulationView frame/encode attempt | runtime error | simulation_view (failed) |

Synthetic fallback artifacts normally have `renderer = synthetic`; only reclassified in `auto` mode when insufficient replay is detected.

## Performance Metrics
File: `performance_visuals.json`
- `plots_time_s` – wall time for plot generation.
- `videos_time_s` – wall time for first video phase (selection + render + encode attempt).
- `first_video_time_s` / `first_video_encode_time_s` – encoding duration for first successful video (legacy alias).
- `first_video_render_time_s` – (approx) total video phase minus encode time for SimulationView (render overhead).
- `first_video_peak_rss_mb` – peak RSS sampled during encode (if psutil present).
- `plots_over_budget` (>2.0s), `video_over_budget` (encode >5.0s), `memory_over_budget` (>100MB).

## Dependency Matrix
| Capability | pygame | moviepy | ffmpeg bin | jsonschema | psutil |
|------------|--------|---------|------------|------------|--------|
| SimulationView rendering | ✔ | ✔ (encode) | ✔ | optional | optional |
| Synthetic fallback | ✖ | ✖ (for static gradient) | ✖ | optional | optional |
| Schema validation | ✖ | ✖ | ✖ | ✔ | ✖ |
| Memory sampling | ✖ | ✖ | ✖ | ✖ | ✔ |
| Renderer toggle flag | ✖ | ✖ | ✖ | ✖ | ✖ |

## Configuration Flags
| Flag | Purpose | Default |
|------|---------|---------|
| `video_renderer` | `auto|synthetic|sim-view` renderer selection | `auto` |
| `capture_replay` | Enable replay capture & SimulationView eligibility | False / config dependent |
| `disable_videos` | Skip all video generation | False |
| `smoke` | Force fast path (skip videos) | False |
| `max_videos` | Limit number of videos encoded | 1 |
| `sim_view_max_frames` | Cap frames per SimulationView video | 0 (no cap) |
| `video_fps` | Output FPS | 10 |
| `ROBOT_SF_VALIDATE_VISUALS` (env) | Enable JSON Schema validation | unset/0 |

## Schema Validation
When `ROBOT_SF_VALIDATE_VISUALS=1`, manifests are validated against JSON Schemas under `specs/127-enhance-benchmark-visual/contracts/`. Validation failures raise (test mode) to catch drift.

## Memory Sampling
If `psutil` is available, a lightweight background sampler records peak RSS during encode. Missing `psutil` leaves `peak_rss_mb` as null.

## Extending / Future Work
- Richer SimulationView overlays (pedestrian intentions, action vectors).
- Frame-by-frame metric logging (e.g., curvature) synchronized with video timeline.
- Optional downsampling for long episodes.
- Structured error taxonomy replacing free-form `render-error:<Type>` strings.

---
*Generated as part of feature documentation tasks (T060/T060A/T063A).*