# Scenario/Planner Gallery

`robot-sf gallery build` generates a **browsable, self-contained static HTML
gallery** of the scenarios in a scenario matrix, with one card per scenario.
It aggregates the *existing* thumbnail/render tooling and scenario metadata —
it does not introduce new rendering. This is a discoverability and inspection
artifact, **not benchmark evidence**.

## Build the gallery

```bash
uv run robot-sf gallery build \
  --matrix configs/baselines/example_matrix.yaml \
  --out-dir output/gallery
```

By default this writes, under `output/gallery/`:

- `index.html` — the gallery page. Thumbnails are embedded as base64 data URIs,
  so the single HTML file is fully portable (open it directly in a browser, copy
  it anywhere, host it statically).
- `gallery_manifest.json` — a machine-readable sidecar listing every card with
  its metadata (schema `scenario_gallery_html.v1`).
- `thumbnails/<scenario_id>.png` — the rendered per-scenario thumbnails.

## What each card shows

Every scenario in the matrix gets a card with:

- a **thumbnail** rendered deterministically (same renderer/IDs as
  `robot_sf_bench plot-scenarios`, seeded per scenario);
- the **scenario ID**, **map name**, **pedestrian count** (derived from
  `density`), and a deterministic **difficulty** band/score;
- **supported planners** — a documented constant set of canonical planner
  families (`simple_policy`, `baseline_sf`, `dwa`, `orca`, `classic_global`).
  This is a discoverability aid, **not** a per-scenario measured capability;
- an **expected runtime** — a deterministic order-of-magnitude CPU estimate
  (`peds × horizon × per-step cost`), labeled `est.` and shown in italics. It is
  explicitly **not** a benchmark result;
- a **"run this" command** — a copy-pasteable `uv run robot_sf_bench run ...`
  invocation for the scenario;
- a **"view sample rollout"** link, when a sample rollout is discoverable under
  `--sample-rollout-root` (matched by `<scenario_id>.{mp4,webm,jsonl,html}`);
- the **raw scenario params** in a collapsible `<details>` block for transparency.

## Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--matrix` | `configs/baselines/example_matrix.yaml` | Source scenario matrix YAML |
| `--out-dir` | `output/gallery` | Output directory |
| `--base-seed` | `0` | Base seed for deterministic thumbnails |
| `--horizon` | `100` | Horizon (steps) for the runtime estimate |
| `--no-thumbnails` | off | Skip thumbnail rendering (cards show a placeholder) |
| `--link-thumbnails` | off | Reference thumbnails by relative path instead of embedding as data URIs |
| `--sample-rollout-root` | none | Directory searched for per-scenario sample rollouts |
| `--title` | derived from matrix | Optional page title |

## Headless / no-render mode

In environments without matplotlib, pass `--no-thumbnails` to build the page
without rendering (cards show a "no thumbnail rendered" placeholder). The page
and manifest are still complete and reference every scenario.

## Programmatic API

```python
from robot_sf.benchmark.runner import load_scenario_matrix
from robot_sf.gallery.builder import build_gallery

scenarios = load_scenario_matrix("configs/baselines/example_matrix.yaml")
result = build_gallery(
    scenarios,
    matrix_path="configs/baselines/example_matrix.yaml",
    out_dir="output/gallery",
)
print(result.html_path, len(result.cards))
```

## Relationship to other tooling

- Thumbnails use the same deterministic ID resolution and renderer as
  [`robot_sf_bench plot-scenarios`](scenario_thumbnails.md).
- The scenario list comes from the same `load_scenario_matrix` loader used by
  the benchmark runner.
- Benchmark/UX epic: #5791. Gallery feature: #5796.
