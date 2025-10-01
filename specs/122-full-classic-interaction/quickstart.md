# Quickstart: Full Classic Interaction Benchmark

This guide shows how to run the full benchmark and a fast smoke mode for CI / local validation.

## 1. Environment Prerequisites
```
git submodule update --init --recursive
uv sync && source .venv/bin/activate
```

## 2. Smoke Mode (≈ <2 minutes)
```
uv run python scripts/classic_benchmark_full.py \
  --scenarios configs/scenarios/classic_interactions.yaml \
  --output results/classic_full_smoke \
  --smoke --workers 2 --seed 123
```
Outputs (structure kept minimal):
```
classic_full_smoke/
  manifest.json
  episodes/episodes.jsonl
  aggregates/summary.json
  reports/statistical_sufficiency.json
  plots/ (subset)
  reports/effect_sizes.json
  logs/benchmark.log
```
Videos are skipped in smoke mode (status noted in statistical_sufficiency.json).

## 3. Full Run
```
uv run python scripts/classic_benchmark_full.py \
  --scenarios configs/scenarios/classic_interactions.yaml \
  --output results/classic_full_run \
  --workers 6 --bootstrap-samples 1000 --seed 123
```
Optional flags:
- `--algo ppo` (default) or other baseline
- `--snqi-weights model/snqi_canonical_weights_v1.json`
- `--initial-episodes 150 --batch-size 30 --max-episodes 250`
- `--collision-ci 0.02 --success-ci 0.03 --time-mean-ci 0.05 --efficiency-mean-ci 0.05 --snqi-mean-ci 0.05`

## 4. Resume Behavior
Re-run the same command; previously completed episode ids are skipped. If scenario matrix changes, a hash mismatch warning will appear; rerun with `--force-continue` to reuse prior episodes.

## 5. Key Artifacts
| Path | Description |
|------|-------------|
| manifest.json | Run metadata (hashes, counts, config). |
| episodes/episodes.jsonl | Raw episode records. |
| aggregates/summary.json | Aggregated metrics + CIs. |
| reports/effect_sizes.json | Effect size entries per archetype. |
| reports/statistical_sufficiency.json | Precision evaluation, early stop notes. |
| plots/*.pdf | Generated plots (also PNG for convenience). |
| videos/*.mp4 | Annotated representative episodes. |

## 6. Validating Success
1. Ensure `final_pass`: true in `statistical_sufficiency.json` for full run.
2. Inspect a plot (e.g., distributions_collision_rate.pdf) opens correctly.
3. Confirm at least one video per archetype exists and is playable.
4. Run unit tests (`uv run pytest tests`) — new benchmark tests must pass.

## 7. Minimal Programmatic Use
```python
from robot_sf.benchmark.full_classic import run_full_benchmark, BenchmarkConfig
cfg = BenchmarkConfig(output_root="results/classic_prog", scenario_matrix_path="configs/scenarios/classic_interactions.yaml")
run_full_benchmark(cfg)
```

## 8. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Missing videos | ffmpeg not available | Install ffmpeg or ignore (report marks skipped). |
| CI bounds not met | Too few episodes | Increase max-episodes or relax thresholds. |
| Slow runtime | Worker underutilization | Increase --workers (up to 8) and monitor efficiency. |
| Hash mismatch warning | Scenario matrix changed | Use --force-continue or move old results aside. |

## 9. Reproducibility Note
Given identical git hash, scenario matrix, seed, and config parameters, reruns must produce identical episode ids and aggregate statistics within stochastic CI variation.
