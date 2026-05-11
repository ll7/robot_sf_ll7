# Issue #1087 Planner Adapter Starter Template

Issue: [#1087](https://github.com/ll7/robot_sf_ll7/issues/1087)

## Goal

Reduce local-planner onboarding friction by documenting the minimum adapter contract and shipping a
runnable reference adapter that exercises the same map-runner seams contributors must use.

## Decision

Use a copy-and-adapt documentation path rather than a code generator. The repository does not have
a single adapter registry; real benchmark execution currently crosses these surfaces:

* adapter implementation under `robot_sf/planner/`
* map-runner dispatch in `robot_sf/benchmark/map_runner.py`
* readiness gating in `robot_sf/benchmark/algorithm_readiness.py`
* benchmark metadata in `robot_sf/benchmark/algorithm_metadata.py`
* targeted unit and dispatch tests

The reference adapter is `TrivialReferencePlannerAdapter` with benchmark key `trivial_reference`
and alias `reference_adapter`. It is intentionally diagnostic-only and requires
`allow_testing_algorithms: true`.

## Scope Boundary

Implemented:

* deterministic reference adapter with `plan`, `reset`, `close`, and `diagnostics` hooks
* map-runner dispatch and metadata/readiness catalog entries
* stable opt-in config at `configs/algos/reference_adapter.yaml`
* contributor template docs at `docs/dev/planner_adapter_template.md`
* targeted adapter, dispatch, metadata, and readiness tests

Not implemented:

* cookiecutter or file generation CLI
* third-party planner code
* promotion of the reference adapter as a benchmark planner

## Validation Plan

Required proof:

* targeted adapter/metadata/readiness/map-runner tests
* reference adapter smoke through `robot_sf_bench run`
* full PR readiness after syncing with `origin/main`

Generated smoke outputs under `output/benchmarks/reference_adapter_smoke/` are local,
reproducible, and ignored.

## 2026-05-09 Validation

Observed locally:

* Focused adapter, map-runner dispatch, metadata, and readiness tests passed.
* Ruff check/format passed for changed Python files.
* The documented reference smoke ran through `robot_sf_bench run`:

```bash
rtk uv run robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --out output/benchmarks/reference_adapter_smoke/episodes.jsonl \
  --algo reference_adapter \
  --algo-config configs/algos/reference_adapter.yaml \
  --repeats 1 \
  --horizon 300 \
  --workers 1 \
  --no-video \
  --benchmark-profile experimental \
  --no-resume \
  --fail-fast \
  --external-log-noise suppress \
  --structured-output json
```

The smoke wrote three episode records. Aggregation reported `success.mean=1.0` and
`collisions.mean=0.0` for the diagnostic reference adapter.
