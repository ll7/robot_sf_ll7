# Scenario Thumbnails and Montage

This tool renders small, consistent thumbnails for each scenario in a matrix and can compose a montage grid. PNGs are suitable for docs/web; PDFs follow our LaTeX-friendly settings (tight bbox, pdf.fonttype=42).

## CLI

- Render thumbnails for all unique scenarios and a montage:

```
uv run robot_sf_bench plot-scenarios \
  --matrix configs/baselines/example_matrix.yaml \
  --out-dir docs/figures/scenarios \
  --base-seed 0 \
  --pdf \
  --montage --cols 3
```

Outputs
- docs/figures/scenarios/<scenario_id>.png (+ .pdf when --pdf)
- docs/figures/scenarios/montage.png (+ montage.pdf when --pdf)

## Programmatic API

```python
from robot_sf.benchmark.scenario_thumbnails import save_scenario_thumbnails, save_montage
from robot_sf.benchmark.runner import load_scenario_matrix

scenarios = load_scenario_matrix("configs/baselines/example_matrix.yaml")
metas = save_scenario_thumbnails(scenarios, out_dir="docs/figures/scenarios", base_seed=0, out_pdf=True)
_ = save_montage(metas, out_png="docs/figures/scenarios/montage.png", cols=3, out_pdf="docs/figures/scenarios/montage.pdf")
```

## Notes
- Deterministic thumbnails: seed = base_seed + hash(scenario_id).
- Repeats in the matrix are ignored for thumbnails.
- Headless-safe via MPLBACKEND=Agg (set in seed_utils).
