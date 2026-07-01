# Issue #3949: Benchmark Result Provenance Manifest

## Related

- Issue: [#3949](https://github.com/ll7/robot_sf_ll7/issues/3949)
- Predecessor: Issue #3000 added top-level summary provenance blocks. This issue
  extends to row-level provenance with a fail-closed manifest.
- Provenance reference: Ortega et al.,
  [Replicable Simulation-Based Robot Validation through Provenance](https://arxiv.org/abs/2605.29973)
- Modules:
  - `robot_sf/benchmark/result_provenance.py` — manifest builder + validator
  - `robot_sf/benchmark/map_runner_episode.py` — row-level `result_provenance` field
  - `robot_sf/benchmark/map_runner.py` — manifest emission in `run_map_batch`
  - `scripts/validation/check_benchmark_result_provenance.py` — fail-closed checker CLI
- Tests:
  - `tests/benchmark/test_benchmark_result_provenance.py`

## Scope

Add a row-level provenance manifest (`benchmark_result_provenance.v1`) emitted
alongside `episodes.jsonl` as `episodes.jsonl.provenance.json` for map-runner
batches.

- **First target**: Map-based benchmark runner (`run_map_batch`).
- **No campaign execution**, **no metric-semantics changes**.
- **No JSON-LD/FAIR publication** — deferred per issue scope.

## Manifest Path Convention

```
episodes.jsonl → episodes.jsonl.provenance.json
```

The resume sidecar (`episodes.jsonl.manifest.json`) is untouched.

## Schema: `benchmark_result_provenance.v1`

The manifest links each emitted row to:

- Row identity (`episode_id`, `scenario_id`, `seed`)
- Config hash, repo commit
- Simulator settings (`horizon`, `dt`, `record_forces`, `observation_mode`, etc.)
- Raw artifact path (episodes JSONL) with SHA256
- Post-processing steps (`compute_all_metrics`, `post_process_metrics`)

Skipped/preflight batches produce a manifest with `completeness.status:
"not_applicable"`.

## Fail-Closed Checker

```bash
uv run python scripts/validation/check_benchmark_result_provenance.py \
  --manifest output/path/episodes.jsonl.provenance.json
```

Exit 0 on valid, exit 2 on missing required field, invalid schema version,
missing artifact SHA256, missing row link, or malformed postprocessing entry.

## Validation

```bash
uv run pytest tests/benchmark/test_benchmark_result_provenance.py -q
uv run ruff check robot_sf/benchmark/result_provenance.py robot_sf/benchmark/map_runner.py robot_sf/benchmark/map_runner_episode.py scripts/validation/check_benchmark_result_provenance.py tests/benchmark/test_benchmark_result_provenance.py
uv run ruff format --check robot_sf/benchmark/result_provenance.py robot_sf/benchmark/map_runner.py robot_sf/benchmark/map_runner_episode.py scripts/validation/check_benchmark_result_provenance.py tests/benchmark/test_benchmark_result_provenance.py
git diff --check
```

## Evidence Boundary

- Classification: `infrastructure/metadata`.
- No benchmark-strength evidence claims. No metric semantics changes.
- Manifest format is a bridge to future JSON-LD/FAIR publication.
