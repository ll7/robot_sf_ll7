# Distribution Plots

This page documents the per-metric distribution plots for benchmark episodes.

> **Hands-on examples**: See `examples/plotting/snqi_figures_example.py` and
> `examples/plotting/plot_pareto.py` (catalogued in
> [`examples/README.md`](../examples/README.md)) for ready-to-run figure scripts.

- Input: JSONL of episode records produced by `robot_sf_bench run`
- Output: PNG histograms per metric, optionally vector PDFs for LaTeX
- Grouping: values are collected per group using a dotted key

## CLI usage

Basic usage:

```
robot_sf_bench plot-distributions \
  --in results/episodes.jsonl \
  --out-dir docs/figures \
  --metrics collisions,comfort_exposure
```

Options:
- `--group-by` (default `scenario_params.algo`)
- `--fallback-group-by` (default `scenario_id`)
- `--bins` (default 30)
- `--kde` overlay a kernel density estimate when SciPy is available
- `--out-pdf` also export LaTeX-friendly PDFs (rcParams set per dev_guide)

## Programmatic

```
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.distributions import collect_grouped_values, save_distributions

records = read_jsonl("results/episodes.jsonl")
grouped = collect_grouped_values(
    records,
    metrics=["collisions", "comfort_exposure"],
    group_by="scenario_params.algo",
    fallback_group_by="scenario_id",
)
meta = save_distributions(grouped, out_dir="docs/figures", bins=30, kde=True, out_pdf=True)
print(meta)
```

## Notes

- When SciPy is not installed, the KDE overlay is silently skipped.
- PDFs are exported with LaTeX-compatible settings: `savefig.bbox=tight`, `pdf.fonttype=42`, consistent fonts.
