# Quickstart: Classic Benchmark Visual Artifacts

## Goal
Run the Full Classic Interaction Benchmark and obtain plot & video artifacts (or informative skip manifests) for qualitative review.

## Prerequisites
```
uv sync --all-extras  # ensure optional matplotlib/moviepy available (optional)
```

## Basic Run (default visuals)
```
uv run python scripts/classic_benchmark_full.py \
  --scenarios configs/scenarios/classic_interactions.yaml \
  --output results/full_classic_visual_demo \
  --workers 1 --seed 123 --algo ppo \
  --initial-episodes 5 --max-episodes 10 --batch-size 5
```

## Expected Outputs
```
results/full_classic_visual_demo/
  episodes/episodes.jsonl
  aggregates/summary.json
  reports/effect_sizes.json
  reports/statistical_sufficiency.json
  reports/plot_artifacts.json
  reports/video_artifacts.json
  plots/*.pdf
  videos/*.mp4   (<= max_videos, may be absent if skipped)
```

## Disabling Videos
```
--disable-videos
```
Video manifest still emitted with status=skipped entries.

## Smoke Mode (fast, minimal videos skipped)
```
--smoke
```
Plots generated (placeholders). Videos all skipped with note "smoke mode".

## Interpreting Manifests
- `status=generated`: File should open (PDF/MP4)
- `status=skipped`: Check `note` (dependency missing, disabled, smoke)
- `status=error`: Non-fatal; investigate note and logs

## Performance Guidance
If plots or videos exceed target soft budgets (2s / 5s), consider:
- Reducing `max_videos`
- Running with `--disable-videos` in CI
- Ensuring ffmpeg installed to avoid slow fallbacks

