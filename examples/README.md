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
