# Ranking generator

Produce a ranked table of groups (e.g., algorithms or scenarios) by the mean of a selected metric.

## CLI usage

- Markdown table (default):

  robot_sf_bench rank --in results/episodes.jsonl --out results/ranking.md --metric collisions

- CSV output:

  robot_sf_bench rank --in results/episodes.jsonl --out results/ranking.csv --metric snqi --format csv --descending

- JSON output (raw rows):

  robot_sf_bench rank --in results/episodes.jsonl --out results/ranking.json --metric comfort_exposure --format json --top 10

### Options

- --group-by: grouping key (default: scenario_params.algo)
- --fallback-group-by: fallback grouping key when group-by missing (default: scenario_id)
- --metric: metric name under metrics.<name> (default: collisions)
- --ascending / --descending: sort direction (default: ascending)
- --top: limit to top N rows
- --format: md | csv | json (default: md)

## Programmatic usage

```python
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.ranking import compute_ranking, format_markdown

records = read_jsonl("results/episodes.jsonl")
rows = compute_ranking(records, metric="collisions", group_by="scenario_params.algo")
print(format_markdown(rows, "collisions"))
```

Notes
- Missing or non-numeric values for the metric are ignored per group.
- Groups with no valid values are omitted.
- For lower-is-better metrics (e.g., collisions), keep ascending order. For higher-is-better, use --descending.
