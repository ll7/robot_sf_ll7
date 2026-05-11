# Issue #1085 Pedestrian-Impact Aggregate Metrics

Issue: [#1085](https://github.com/ll7/robot_sf_ll7/issues/1085)

Status date: 2026-05-09

## Goal

Promote the existing opt-in pedestrian-impact signals from flat experimental values into a
schema-backed, aggregate-ready metric surface. The change keeps the metrics exploratory and
opt-in, but removes the need for custom paper-side parsing once a benchmark run emits them.

## Contract

Enable the metrics for benchmark JSONL output with:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --out output/benchmarks/issue_1085/ped_impact_smoke/episodes.jsonl \
  --experimental-ped-impact
```

The episode record keeps legacy flat `ped_impact_*` keys and adds:

- `metrics.pedestrian_impact.schema_version == "pedestrian-impact.v1"`
- `metrics.pedestrian_impact.parameters` for radius/window provenance
- `metrics.pedestrian_impact.units` for acceleration, turn-rate, radius, and sample counts
- `metrics.pedestrian_impact.sample_counts` for near/far support
- `metrics.pedestrian_impact.canonical_reductions` for aggregate-ready near-minus-far deltas

Aggregation flattens the structured block into `ped_impact_*` numeric columns before computing the
standard mean/median/p95 summaries. The canonical comparison reductions are:

- `ped_impact_accel_delta_mean`
- `ped_impact_accel_delta_median`
- `ped_impact_turn_rate_delta_mean`
- `ped_impact_turn_rate_delta_median`

Validity counters must be reported beside those reductions because deltas are undefined when a
pedestrian has only near or only far support.

## Boundaries

This does not calibrate a composite social-impact score, change SNQI weights, or add new pedestrian
behavior models. The metrics remain opt-in until a later paper or benchmark policy decides whether
they should be emitted in every canonical campaign.

## Validation Plan

- Schema contract test accepts a valid `pedestrian-impact.v1` block and rejects an invalid schema
  version.
- Metric post-processing test verifies the structured block is attached while preserving flat
  `ped_impact_*` keys.
- Aggregation test verifies records with only the structured block produce `ped_impact_*`
  aggregate reductions.
- CLI smoke should run `robot_sf_bench run --experimental-ped-impact` on a small representative
  slice, followed by `robot_sf_bench aggregate` on the emitted JSONL.

Observed local smoke:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/sets/classic_crossing_subset.yaml \
  --out output/benchmarks/issue_1085/ped_impact_crossing/episodes.jsonl \
  --algo goal \
  --horizon 40 \
  --workers 1 \
  --experimental-ped-impact

uv run robot_sf_bench aggregate \
  --in output/benchmarks/issue_1085/ped_impact_crossing/episodes.jsonl \
  --out output/benchmarks/issue_1085/ped_impact_crossing/summary.json
```

The smoke wrote 6 schema-valid episode records. The representative crossing slice produced
pedestrian-impact blocks with pedestrian counts (`ped_impact_ped_count` aggregate mean `7.5`) and
far-sample support (`ped_impact_far_samples` aggregate mean `209.0`). Near-sample support was zero
for this short horizon, so delta-valid counters remained zero; the synthetic unit tests cover the
positive near/far delta path directly.
