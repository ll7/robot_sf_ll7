# Quickstart: Enhanced Benchmark Visual Artifacts

This guide shows how to run the full classic benchmark and obtain real SimulationView MP4 videos plus validated manifests.

## Prerequisites
```
uv sync
uv run pip install moviepy psutil jsonschema pygame  # if not already present
```
Ensure ffmpeg is available in PATH (moviepy uses it). On macOS (Homebrew):
```
brew install ffmpeg
```

## Run Benchmark
```
uv run python scripts/classic_benchmark_full.py \
  --scenarios configs/scenarios/classic_interactions.yaml \
  --output results/full_classic_visual_demo \
  --workers 1 --seed 123 --algo ppo \
  --initial-episodes 5 --max-episodes 10 --batch-size 5
```

## Outputs
Inside the output directory after completion:
- `plot_artifacts.json` (structure validated if jsonschema installed)
- `video_artifacts.json` (renderer = `simulation_view` when dependencies present; otherwise skipped or synthetic)
- `performance_visuals.json` (timings + over_budget flags)
- `video_<episode_id>.mp4` files for each successful episode

## Degradation Cases
| Missing Component | Outcome | Manifest Note |
|-------------------|---------|---------------|
| pygame | Synthetic fallback videos | simulation-view-missing |
| moviepy (or ffmpeg) | No videos | moviepy-missing |
| Replay state insufficient | Skipped | insufficient-replay-state |
| Smoke mode flag (future) | Skipped | smoke-mode |

## Validate Manifests Manually
```
uv run python - <<'PY'
import json, jsonschema, pathlib
base = pathlib.Path('results/full_classic_visual_demo')
video_schema = json.load(open('specs/127-enhance-benchmark-visual/contracts/video_artifacts.schema.json'))
manifest = json.load(open(base/'video_artifacts.json'))
jsonschema.validate(manifest, video_schema)
print('Video manifest valid')
PY
```

## Force Synthetic Fallback (Testing)
Uninstall/remove pygame or set an environment variable (to be implemented) before running.

## Memory & Performance
`performance_visuals.json` will add `memory_over_budget` if sampling collected a peak > 100 MB during first SimulationView encode.

## Next Steps
- Inspect MP4 outputs.
- Integrate figures into papers (vector plots already produced in plots directory).
