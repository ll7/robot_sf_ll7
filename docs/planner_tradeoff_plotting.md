# Planner Tradeoff Plotting

Create a paper-ready safety-efficiency scatter plot from a Robot SF publication bundle. The plot
places collision rate on the x-axis and success rate on the y-axis, with optional bootstrap
confidence intervals computed over per-seed planner means.

## Input Contract

The command expects a publication bundle with:

- `payload/reports/campaign_table.csv`
- `payload/runs/<planner>*/episodes.jsonl`
- optional `publication_manifest.json` for run-id metadata

The campaign table supplies planner-level means. Episode JSONL files supply seed-level bootstrap
confidence intervals. Collision indicators are read from `outcome.collision_event` when present, so
the figure matches camera-ready bundles where `metrics.collisions` is a stale or non-indicator count.

## CLI Usage

```bash
robot_sf_bench plot-planner-tradeoff \
  --bundle-path output/benchmarks/camera_ready/<campaign_id>_publication_bundle \
  --out output/figures/planner_tradeoff_safety_efficiency.png \
  --out-pdf output/figures/planner_tradeoff_safety_efficiency.pdf \
  --metadata-out output/figures/planner_tradeoff_safety_efficiency.json
```

Options:

- `--bootstrap-samples` controls resamples over seed-level means. Default: `400`.
- `--ci-confidence` controls the interval level. Default: `0.95`.
- `--bootstrap-seed` makes bootstrap intervals reproducible. Default: `42`.
- `--title ""` suppresses the default preferred-region title.

## Programmatic Usage

```python
from pathlib import Path

from robot_sf.benchmark.planner_tradeoff_plot import save_planner_tradeoff_figure

meta = save_planner_tradeoff_figure(
    Path("output/benchmarks/camera_ready/my_publication_bundle"),
    out_png=Path("output/figures/planner_tradeoff_safety_efficiency.png"),
    out_pdf=Path("output/figures/planner_tradeoff_safety_efficiency.pdf"),
)
print(meta["run_id"])
```

## Artifact Policy

Generated figures should live under `output/` during iteration. Promote only small, reviewable
metadata or evidence copies into `docs/context/evidence/` when a PR, issue, or release needs a
durable proof artifact.
