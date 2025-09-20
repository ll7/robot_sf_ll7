# Examples

- `snqi_figures_example.py`
  - Generate publication-ready figures with optional SNQI integration via the orchestrator.
  - Usage:
    ```bash
    uv run python examples/snqi_figures_example.py \
      --episodes results/episodes_sf_long_fix1.jsonl \
      --weights examples/snqi_weights_example.json \
      --baseline results/baseline_stats.json
    ```
  - Produces a canonical figures folder under `docs/figures/<stem>__<sha>__v<schema>/` and updates `docs/figures/_latest.txt`.
  - If `--baseline` is omitted, SNQI normalization defaults will be used for missing metrics.

Other demos:
- See `examples/demo_*` and `examples/plot_*` for environment and plotting walkthroughs.

### Full Classic Benchmark Demo

Run the adaptive full classic benchmark end-to-end (episodes + aggregates + visual artifacts):

```bash
uv run python examples/demo_full_classic_benchmark.py
```

Fast defaults inside the script:
- initial_episodes=2, max_episodes=4, batch_size=2
- workers=1 (deterministic order)
- max_videos=1 (synthetic fallback unless SimulationView implemented)

Generated artifacts (timestamped results directory):
- episodes/episodes.jsonl
- aggregates/summary.json
- reports/plot_artifacts.json, video_artifacts.json, performance_visuals.json
- plots/*.pdf (if matplotlib installed)
- videos/*.mp4

Toggle flags in the script:
- Set `smoke=True` for even faster placeholders
- Set `disable_videos=True` to skip video generation

For full CLI flag coverage see `scripts/classic_benchmark_full.py`.

## Full SNQI Flow (Episodes → Baseline → Figures)

The script `snqi_full_flow.py` automates an end-to-end reproducible pipeline:

1. (Optional) Generate episodes from a scenario matrix if the `--episodes` JSONL does not yet exist.
2. Compute baseline median / p95 statistics used for SNQI normalization.
3. Ensure a weights file exists (creates a default if missing; you can later replace with optimized weights).
4. Invoke `scripts/generate_figures.py` injecting SNQI and producing Pareto plot, distributions, table, force‑field figure, canonical output directory, and `_latest` pointer.

Minimal (episodes already exist):
```bash
uv run python examples/snqi_full_flow.py \
  --episodes results/episodes_sf_long_fix1.jsonl \
  --baseline-json results/baseline_stats.json \
  --weights-json examples/snqi_weights_example.json
```

Generate episodes first from a matrix:
```bash
uv run python examples/snqi_full_flow.py \
  --episodes results/episodes_small.jsonl \
  --matrix configs/baselines/example_matrix.yaml \
  --baseline-json results/baseline_stats.json \
  --weights-json examples/snqi_weights_example.json
```

Artifacts: `docs/figures/<episodes-stem>__<gitsha7>__v<schema>/` plus pointer `docs/figures/_latest.txt`.

Replace the default weights by supplying your own optimized weights via `--weights-json`.
