# Full Classic Interaction Benchmark

> Documentation page (T047) summarizing usage, configuration flags, artifacts, and extensibility.
>
> Status: Initial implementation complete (T001–T041 plus hardening tests T042–T044). This page will evolve; treat as living document.

## Overview
The Full Classic Interaction Benchmark provides a reproducible evaluation harness over a canonical set of interaction archetypes (e.g., crossing, head‑on, overtaking). It produces:
- Line‑oriented episode records (`episodes/episodes.jsonl`)
- Aggregated metric summaries with bootstrap/Wilson CIs (`aggregates/summary.json`)
- Effect size reports (`reports/effect_sizes.json`)
- Statistical precision report with per‑metric CI half‑widths (`reports/statistical_sufficiency.json`)
- Plot artifacts (distribution, trajectory, KDE placeholder, Pareto placeholder, force heatmap placeholder) under `plots/`
- Optional annotated videos of representative episodes under `videos/`
- A manifest capturing configuration, git hash, scenario matrix hash, timing & scaling efficiency (`manifest.json`)

All outputs are deterministic given the scenario matrix file and master seed.

## Quick Start
```bash
uv run python scripts/classic_benchmark_full.py \
  --scenarios configs/scenarios/classic_interactions.yaml \
  --output results/full_classic_run_01 \
  --workers 2 --seed 123 --algo ppo \
  --initial-episodes 2 --max-episodes 4 --batch-size 2 \
  --target-collision-half-width 0.05 \
  --target-success-half-width 0.05 \
  --target-snqi-half-width 0.05
```
Smoke mode (fast placeholder artifacts):
```bash
uv run python scripts/classic_benchmark_full.py \
  --scenarios configs/scenarios/classic_interactions.yaml \
  --output results/full_classic_smoke \
  --smoke --initial-episodes 1 --max-episodes 1
```

## CLI Flags
| Flag | Purpose |
|------|---------|
| `--scenarios PATH` | Scenario matrix YAML (required) |
| `--output DIR` | Output root directory (required) |
| `--workers N` | Parallel process workers (>=1) |
| `--seed N` | Master seed for deterministic planning |
| `--algo NAME` | Algorithm label stored in records |
| `--initial-episodes N` | Planned seeds per scenario for first pass |
| `--max-episodes N` | Per-scenario cap before stopping (0 = only precision criteria) |
| `--batch-size N` | Episodes scheduled per adaptive iteration |
| `--horizon N` | Override horizon for all episodes (0 = scenario default) |
| `--smoke` | Enable fast placeholder run (skips heavy video generation) |
| `--target-collision-half-width F` | CI half-width target for collision_rate |
| `--target-success-half-width F` | CI half-width target for success_rate |
| `--target-snqi-half-width F` | CI half-width target for snqi (placeholder until SNQI integrated) |
| `--disable-videos` | Skip video artifact generation |
| `--max-videos N` | Max representative videos to render |

## Adaptive Precision Loop
Each iteration performs:
1. Aggregate metrics (bootstrap + Wilson for rate metrics)
2. Compute effect sizes (Cohen's h, Glass Δ approximation)
3. Evaluate precision vs configured CI half‑width targets
4. Early stop if all targets satisfied or `max_episodes` boundary reached

Episodes are added via `adaptive_sampling_iteration` in batches (default 1) until criteria met.

## Output Schema Highlights
`episodes/episodes.jsonl` (one JSON line per episode):
```json
{
  "episode_id": "crossing__low__crossing-123456:...",
  "scenario_id": "crossing__low__crossing",
  "seed": 123456,
  "archetype": "crossing",
  "density": "low",
  "status": "success",
  "metrics": {"collision_rate": 0.0, "success_rate": 1.0, ...},
  "steps": 110,
  "wall_time_sec": 0.0021,
  "algo": "ppo",
  "created_at": 1737423423.5123
}
```

`manifest.json` adds (excerpt):
```json
{
  "git_hash": "abc12def3456",
  "scenario_matrix_hash": "4f9c1e2d3a10",
  "runtime_sec": 0.73,
  "episodes_per_second": 52.1,
  "scaling_efficiency": {
    "runtime_sec": 0.73,
    "executed_jobs": 8,
    "skipped_jobs": 0,
    "episodes_per_second": 52.1,
    "workers": 2,
    "parallel_efficiency_placeholder": 0.5,
    "finalized": true
  }
}
```

## Scaling & Efficiency
Current efficiency metric is a placeholder (episodes_per_second / (workers * episodes_per_second)) and will be replaced with a more meaningful comparison vs. sequential baseline timing in a future optimization task.

## Reproducibility Guarantees
Determinism hinges on two persisted identifiers:
- `git_hash`: Short commit hash captured at run start for exact code provenance.
- `scenario_matrix_hash`: SHA1 (first 12 chars) of a canonical JSON dump of the scenario matrix file.

If these two values plus `master_seed` and the scenario matrix file content are the same, the benchmark will produce identical episode IDs and (with the current synthetic metrics path) identical aggregates/effect sizes. The manifest stores all three, enabling downstream verification scripts to compare runs. Future extensions integrating real simulations must retain these fields to preserve this guarantee.

## Plots & Videos
- Plots always created (placeholders if data limited) unless matplotlib missing.
- Videos require matplotlib + moviepy; gracefully skipped otherwise. Smoke mode always skips.

## Extensibility Roadmap
| Area | Next Step |
|------|-----------|
| Episode Execution | Replace synthetic record generator with real simulation integration |
| Metrics | Integrate SNQI weights + additional continuous metrics |
| Effect Sizes | Include confidence intervals via bootstrap on standardized metrics |
| Precision | Relative half‑width targets for continuous metrics |
| Scaling | True parallel efficiency (compare against timed sequential baseline) |
| Visualization | Replace placeholders (KDE, Pareto, force heatmap) with real data-driven plots |

## Related Spec Files
- `specs/122-full-classic-interaction/tasks.md`
- `specs/122-full-classic-interaction/quickstart.md`
- `specs/122-full-classic-interaction/data-model.md`

## CI Integration (Planned)
A lightweight smoke validation (`scripts/validation/test_classic_benchmark_full.sh`) can be integrated into CI as a separate job or appended to an existing validation stage:
```yaml
  classic-benchmark-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - name: Install deps
        run: uv sync
      - name: Run classic benchmark smoke
        run: ./scripts/validation/test_classic_benchmark_full.sh
```
This job should finish in a few seconds (synthetic metrics path). Future real simulation integration may require marking it optional or adding a time budget.

## Validation
Run the performance smoke test:
```bash
uv run pytest tests/benchmark_full/test_integration_performance_smoke.py::test_performance_smoke -q
```

Run the resume test:
```bash
uv run pytest tests/benchmark_full/test_integration_resume.py::test_resume_skips_existing -q
```

Validation shell smoke (added T054):
```bash
./scripts/validation/test_classic_benchmark_full.sh
```

Example artifact tree (smoke run):
```
full_classic_smoke/
  episodes/
    episodes.jsonl
  aggregates/
    summary.json
  reports/
    effect_sizes.json
    statistical_sufficiency.json
  plots/
    distribution.pdf
    trajectory.pdf
    kde_placeholder.pdf
    pareto_placeholder.pdf
    force_heatmap_placeholder.pdf
  manifest.json
```

## Changelog
See `CHANGELOG.md` (pending T048 entry for initial release).
