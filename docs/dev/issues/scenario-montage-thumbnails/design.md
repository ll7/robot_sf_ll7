# Scenario Montage Thumbnails — Design

Goal
- Produce small, reproducible visual thumbnails for each scenario in a scenario matrix, plus an optional montage grid suitable for papers/slides.
- Export PNG for web/docs, and LaTeX-friendly PDF (fonttype=42, tight bbox) for publications.

Inputs
- Scenario matrix YAML (list of scenario dicts; supports `repeats` which is ignored for thumbnails)
- Base seed for determinism (thumbnail seed = base_seed + stable hash of scenario id)

Outputs
- One PNG per scenario: `<out_dir>/<scenario_id>.png`
- Optional one PDF per scenario: `<out_dir>/<scenario_id>.pdf`
- Optional montage image: `<out_dir>/montage.png` and optional `<out_dir>/montage.pdf`

Visualization
- 2D map 10m x 6m (matching generator constants)
- Obstacles drawn as line segments
- Initial pedestrian positions as faint circles; arrows to their goals (optional thin lines)
- Robot start/goal markers (using runner defaults when unspecified)
- Minimal legend; consistent style; tight layout

API Sketch
- render_scenario_thumbnail(params, seed, out_png, out_pdf=None, figsize=(3.2, 2.0)) -> dict
- save_scenario_thumbnails(matrix, out_dir, base_seed=0, out_pdf=False) -> list[str]
- save_montage(out_dir, scenario_ids, out_png, cols=3, out_pdf=None) -> dict

CLI
- robot_sf_bench plot-scenarios --matrix configs/...yaml --out-dir docs/figures/scenarios \
  --base-seed 0 --pdf --montage --cols 4

Edge Cases
- Unknown/duplicate scenario ids → sanitize and dedupe filenames
- No obstacles → skip drawing them
- Large number of scenarios → montage should paginate or just write all in one with auto rows

Testing
- Use `configs/baselines/example_matrix.yaml`; assert per-scenario PNGs exist; when `--pdf`, PDFs too; montage written.

Notes
- Use seed_utils to set MPLBACKEND=Agg for headless.
- Do not require pysocialforce; rendering uses generator metadata (state, obstacles) only.
