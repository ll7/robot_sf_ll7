# Pareto plotting for benchmark metrics

Create Pareto scatter plots to visualize trade‑offs (e.g., collisions vs. comfort).

Why
- Understand competing objectives across algorithms/groups
- Identify non‑dominated (Pareto‑optimal) groups

## CLI usage

Generate a PNG plot grouped by algorithm (default):

```bash
robot_sf_bench plot-pareto \
  --in results/episodes.jsonl \
  --out results/pareto.png \
  --x-metric collisions \
  --y-metric comfort_exposure
```

Options
- --group-by (default scenario_params.algo)
- --fallback-group-by (default scenario_id)
- --agg mean|median (default mean)
- --x-higher-better / --y-higher-better (flip orientation for higher‑is‑better metrics)
- --title optional plot title

## Programmatic usage

```python
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.plots import save_pareto_png

records = read_jsonl("results/episodes.jsonl")
meta = save_pareto_png(
    records,
    out_path="results/pareto.png",
    x_metric="collisions",
    y_metric="comfort_exposure",
    title="Collisions vs Comfort"
)
print(meta)
```

`meta` includes keys like `count`, `front_size`, and `front_labels`.

## Tips
- Lower‑is‑better is assumed by default. Use the `--*-higher-better` flags to invert when needed (e.g., SNQI).
- If you see “No points available” errors, check your metric names and grouping keys.
- Headless environments are supported via MPLBACKEND=Agg (set automatically when using seed utils).
