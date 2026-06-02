# docs/figures/

This directory stores tracked visual assets for documentation, LaTeX integration, and
paper-facing figures. All figures here must be **reproducible from code**.

## Contents

- **Scenario thumbnails**: per-scenario PNG/PDF montages under `scenarios/`
- **Episode-run figures**: per-campaign figure folders named
  `<episodes-stem>__<gitsha7>__<schema>/` with Pareto plots, distributions, tables, etc.
- **Standalone reference figures**: `fig-<short-description>.pdf` for individual
  LaTeX-ready figures (force field visualizations, heatmaps, etc.)
- **`_latest.txt`**: alias file pointing to the preferred campaign folder for the
  current draft

## Conventions

- Each figure = one standalone script in `robot_sf/benchmark/figures/` or
  `scripts/figures/` (tracked).
- Script reads data, generates plot, and saves output here.
- Export vector PDFs for LaTeX inclusion; optionally export PNG (300 dpi).
- Names follow the pattern `fig-<short-description>.pdf`.
- Subdirectory names for campaign output:
  `<episodes-stem>__<gitsha7>__<schema>`

## How figures get here

```bash
# Scenario thumbnails
robot_sf_bench plot-scenarios \
  --matrix configs/scenarios/classic_interactions.yaml \
  --out-dir docs/figures/scenarios

# Distribution plots
uv run python scripts/figures/plot_distributions.py \
  --data output/episodes/some_run \
  --out-dir docs/figures

# Force field heatmap
uv run python scripts/figures/plot_force_field.py \
  --pdf docs/figures/fig-force-field.pdf
```

## Related

- `docs/img/` - static images (screenshots, diagrams)
- `docs/video/` - animated GIF recordings
- `output/figures/` - intermediate data exports for figure scripts
