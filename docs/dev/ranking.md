# Ranking generator

Produce a ranked table of groups (e.g., algorithms or scenarios) by the mean of a selected metric.

## CLI usage

- Markdown table (default):

  robot_sf_bench rank --in output/benchmarks/episodes.jsonl --out output/benchmarks/ranking.md --metric collisions

- CSV output:

  robot_sf_bench rank --in output/benchmarks/episodes.jsonl --out output/benchmarks/ranking.csv --metric snqi --format csv --descending

- JSON output (raw rows):

  robot_sf_bench rank --in output/benchmarks/episodes.jsonl --out output/benchmarks/ranking.json --metric comfort_exposure --format json --top 10

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

records = read_jsonl("output/benchmarks/episodes.jsonl")
rows = compute_ranking(records, metric="collisions", group_by="scenario_params.algo")
print(format_markdown(rows, "collisions"))
```

Notes
- Before grouping, records with the explicit
  `algorithm_metadata.foresight_prediction.evidence_eligible=false` marker are excluded by the
  canonical benchmark evidence-admission filter. The ranking rows do not carry an exclusion count;
  use aggregate output metadata when evidence-custody accounting is required.
- Missing, non-numeric, non-finite, and float-conversion-overflow values for the metric are ignored
  per group. Means are calculated with overflow-safe scaling, so finite inputs cannot emit a
  non-finite ranking mean.
- Groups with no valid values are omitted.
- For lower-is-better metrics (e.g., collisions), keep ascending order. For higher-is-better, use --descending.
- Ranking output is implementation/diagnostic output until downstream domain review authorizes any
  benchmark or paper-facing interpretation.
