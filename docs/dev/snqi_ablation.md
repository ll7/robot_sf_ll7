# SNQI Component Ablation

This tool measures how removing each SNQI component (setting its weight to 0) affects algorithm rankings.

What it does
- Computes a base ranking by mean SNQI per group (default: per algorithm).
- For each weight in the SNQI formula, recomputes the ranking with that weight set to 0.
- Reports per-group rank shifts (positive = moved down/worse since SNQI is higher-is-better).

CLI usage

```bash
robot_sf_bench snqi-ablate \
  --in results/episodes.jsonl \
  --out results/ablation.md \
  --snqi-weights model/snqi_canonical_weights_v1.json \
  --snqi-baseline results/baseline_stats.json \
  --format md
```

Options
- --group-by: grouping key (default: scenario_params.algo)
- --fallback-group-by: fallback key (default: scenario_id)
- --format: md | csv | json

API usage

```python
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.ablation import compute_snqi_ablation, format_markdown

records = read_jsonl("results/episodes.jsonl")
weights = {...}
baseline = {...}
rows = compute_snqi_ablation(records, weights=weights, baseline=baseline)
print(format_markdown(rows))
```

Notes
- Only weights present in the provided weight mapping are ablated.
- Groups missing after ablation (e.g., no valid episodes) keep their base position.
- SNQI is computed with the canonical implementation used across tools.
