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

Scenario ID resolution for file naming is explicit:
- `id` -> `name` -> `scenario_id` -> stable hash fallback
- labels are sanitized for filesystem safety
- collisions after sanitization are deterministically disambiguated with suffixes (`__2`, `__3`, ...)

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

## Scenario Atlas

Use `scripts/tools/generate_scenario_atlas.py` when you need a compact
discoverability pack that combines scenario rows, thumbnails, mechanism cards,
coverage gaps, and a checksum manifest:

```bash
uv run python scripts/tools/generate_scenario_atlas.py \
  --matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --output output/scenario_atlas/verified_simple_subset_v1 \
  --run-id verified_simple_subset_v1 \
  --base-seed 0
```

Expected output tree:

```text
scenario_atlas.md
scenario_atlas.csv
coverage_gaps.json
atlas_manifest.json
thumbnails/
  <scenario_id>.png
mechanism_cards/
  <scenario_id>.md
```

The atlas is a discovery and inspection artifact, not benchmark evidence.
Rows distinguish authored intent, certification status, executed evidence, and
missing optional contract/certificate/hazard mappings. Thumbnails use the same
deterministic ID resolution and rendering helper as `robot_sf_bench
plot-scenarios`, so they are useful visual cues but not route-clearance,
certification, or planner-performance proof.
