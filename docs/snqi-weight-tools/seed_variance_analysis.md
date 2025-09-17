## Seed variance analysis

Quantify variability across seeds using coefficient of variation (CV = std/mean) per metric and group.

CLI usage

- Compute seed variance grouped by scenario id:
  - robot_sf_bench seed-variance --in results/episodes.jsonl --out results/seed_var.json --group-by scenario_id
- Select metrics only:
  - robot_sf_bench seed-variance --in results/episodes.jsonl --out results/seed_var_a.json --metrics success,collisions,curvature_mean

Output format

- For each group and metric: keys mean, std, cv, count.
- NaNs are used when insufficient data (e.g., count < 2 for std/cv).

Programmatic usage

```python
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.seed_variance import compute_seed_variance

records = read_jsonl("results/episodes.jsonl")
sv = compute_seed_variance(records, group_by="scenario_id", metrics=["success", "collisions"]) 
```
