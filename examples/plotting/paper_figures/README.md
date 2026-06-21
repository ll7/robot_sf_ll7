# Paper Figure Scripts (AMV Local-Planning Benchmark)

These scripts reproduce the figures and tables of the AMV local-planning
benchmark paper directly from a `robot_sf_ll7` benchmark **publication bundle**
(the archived release artifacts: raw episode JSONL, aggregate CSV/JSON reports,
and manifests). They are standalone (no `robot_sf` import) and read only the
public bundle outputs, so a third party can regenerate the manuscript figures
from the released artifacts without re-running the campaign.

They are registered with `ci_enabled: false` in `examples_manifest.yaml` because
each one requires a publication-bundle path; there is no default CI input.

## Scripts

| Script | Produces |
| --- | --- |
| `build_planner_tradeoff_figure.py` | Safety–efficiency trade-off scatter (success vs collision rate, main rows + seed-bootstrap bands) |
| `build_scenario_coverage_matrix.py` | Scenario coverage matrix LaTeX table |
| `build_svg_scenario_overview.py` | Source-SVG static scenario overview panel grid |
| `build_runtime_scene_panels.py` | Runtime scene-context panel grid (from rendered frames) |
| `build_observation_grid_figure.py` | PPO planner-facing occupancy-grid observation frame |

## Usage

Each script takes a bundle/source path; run with `--help` for the exact flags.
For example, the trade-off figure:

```bash
.venv/bin/python examples/plotting/paper_figures/build_planner_tradeoff_figure.py \
    --bundle-path output/benchmarks/publication/<your_publication_bundle>
```

The default paths in some scripts point at the paper repository's artifact layout;
pass the explicit bundle/source path to run them against any `robot_sf_ll7`
publication bundle.

## Provenance

These scripts are mirrored from the manuscript repository so that the
release-to-figure rendering path is publicly inspectable alongside the benchmark
release. They are figure/table renderers only and do not constitute benchmark
evidence; the numeric evidence lives in the bundle's episode records and
aggregate reports.
