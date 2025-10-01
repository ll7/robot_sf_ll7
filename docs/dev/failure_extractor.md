# Failure case extractor

Find and export episodes that meet failure-like criteria across metrics.

## CLI usage

- Write JSON with episode IDs only:

  robot_sf_bench extract-failures --in results/episodes.jsonl --out results/fail_ids.json --collision-threshold 1 --comfort-threshold 0.2 --near-miss-threshold 0 --ids-only

- Write matching full records as JSONL:

  robot_sf_bench extract-failures --in results/episodes.jsonl --out results/failures.jsonl --collision-threshold 1 --comfort-threshold 0.2

## Options

- --collision-threshold: minimum collisions to flag (default 1)
- --comfort-threshold: minimum comfort_exposure to flag (default 0.2)
- --near-miss-threshold: flags when near_misses > threshold (strictly greater-than; default 0)
- --snqi-below: optional threshold; if provided and metric present, flags when snqi < value
- --max-count: optional cap on number of failures to output
- --ids-only: output JSON with {episode_ids:[...]} instead of JSONL records

## Programmatic usage

```python
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.failure_extractor import extract_failures

records = read_jsonl("results/episodes.jsonl")
fails = extract_failures(records, collision_threshold=1, comfort_threshold=0.2)
```
