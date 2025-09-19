# Evaluation Scripts (Episodes JSONL)

This folder documents small, programmatic helpers to analyze benchmark episode outputs (JSONL).
Each script works on the JSONL produced by the benchmark runner and writes summaries to CSV/JSON/Markdown.

- Location: `scripts/`
- Input: episodes JSONL (`results/episodes.jsonl` or similar)
- Grouping keys use dotted paths into the episode record (e.g., `scenario_params.algo`).

## 1) Seed variance (SNQI)

File: `scripts/seed_variance.py`

Purpose: Compute the variability of SNQI across random seeds for each group (algorithm by default).
- Aggregation: mean SNQI per seed first, then std/CV across seed means.
- Output: JSON with per-group statistics (seeds, snqi_mean, snqi_std, snqi_cv)

Example:

```bash
uv run python scripts/seed_variance.py \
  --episodes results/episodes.jsonl \
  --out results/seed_variance.json \
  --group-by scenario_params.algo
```

## 2) Ranking table by metric

File: `scripts/ranking_table.py`

Purpose: Produce a ranking table (CSV/Markdown) for a given metric (defaults to SNQI).
- Uses `robot_sf.benchmark.aggregate.compute_aggregates` (mean/median/p95)
- Sorts by mean value (descending for SNQI)

Example:

```bash
uv run python scripts/ranking_table.py \
  --episodes results/episodes.jsonl \
  --out-csv results/ranking_snqi.csv \
  --out-md results/ranking_snqi.md \
  --metric snqi \
  --group-by scenario_params.algo
```

## 3) Failure case extractor

File: `scripts/failure_extractor.py`

Purpose: Select the top-k worst episodes by a given metric (e.g., lowest SNQI or highest collision count).
- Metric path is dotted (e.g., `metrics.snqi`, `metrics.collisions`)
- Direction `min|max` controls worst-episode selection
- Output: JSON list with score, episode_id, scenario_id, seed, algo, and full metrics

Examples:

```bash
# Lowest SNQI (worst), top-20
uv run python scripts/failure_extractor.py \
  --episodes results/episodes.jsonl \
  --out results/worst_snqi.json \
  --metric metrics.snqi \
  --direction min \
  --top-k 20

# Highest collision count, top-50
uv run python scripts/failure_extractor.py \
  --episodes results/episodes.jsonl \
  --out results/most_collisions.json \
  --metric metrics.collisions \
  --direction max \
  --top-k 50
```

## Notes
- Group keys commonly useful: `scenario_params.algo`, `scenario_id`.
- If SNQI is missing in episodes, aggregate utilities can recompute it when weights/baseline are provided (see `robot_sf/benchmark/aggregate.py`).
- All scripts are small and composable—feel free to adapt them for project-specific reports.
