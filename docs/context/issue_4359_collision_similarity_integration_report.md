# Issue #4359 Collision-Similarity Integration Report

Plain-language status: collision-scenario similarity can now summarize related collision or near-miss records in tracked artifacts. It remains a diagnostic analysis aid because the representative durable artifact does not include raw per-step trajectory arrays for replay-level validation.

## Current Contract

| Surface | Status | Contract boundary |
| --- | --- | --- |
| `robot_sf_bench collision-scenario-similarity` | Available from PR #4371 and extended by the current issue #4359 progression slice. | Preserves the legacy `collision_scenario_similarity.v1` nearest-neighbor/groups output and optionally compares legacy summary, raw trajectory/action, and combined-context feature sets on one raw-trajectory cohort. |
| Durable issue #1470 artifact report | Available from PR #4386 and refreshed by PR #4410. | Shows two selected unsafe records, singleton groups, available external-label context, and available trajectory-derived metric-field context. |
| Labeled trace fixture report | Available from PR #4386 and refreshed by PR #4410. | Proves the report surfaces labels and raw trajectory fields when the source artifact contains them. |
| Raw per-step trajectory-array boundary | Recorded by PR #4434. | Durable issue #1470 source has summary trajectory-derived metrics but no raw trajectory arrays for the selected records, so replay-level validation remains blocked. |

## Reproducible Raw-Trajectory Contract Probe

This deterministic fixture proves the optional descriptor and report path without making a
failure-family or benchmark claim:

```bash
uv run robot_sf_bench collision-scenario-similarity \
  --episodes-jsonl tests/fixtures/benchmark/collision_scenario_similarity_trajectory.jsonl \
  --out-json output/diagnostics/collision_similarity/trajectory_fixture.json \
  --out-markdown output/diagnostics/collision_similarity/trajectory_fixture.md \
  --nearest-k 1 \
  --require-trajectory-comparison
```

The strict flag fails closed unless at least two selected collision or near-collision records have
complete robot and pedestrian position arrays. The synthetic fixture is implementation-integrity
proof only; it is not representative replay validation or externally labeled evidence.

## Integration Result

The merged slices form one coherent diagnostic contract:

- collision similarity may group and compare unsafe scenario records using logged descriptor fields;
- optional validation blocks are descriptive context, not benchmark truth or planner-ranking evidence;
- unavailable raw trajectory fields must remain explicit instead of being imputed from summary metrics;
- candidate feature sets must use the same raw-trajectory cohort so coverage differences do not
  masquerade as descriptor differences;
- representative replay-level validation requires a durable source with raw robot and pedestrian trajectory arrays for the same records.

## Remaining Blockers

| Blocker | Why it remains | Next empirical action |
| --- | --- | --- |
| Raw per-step trajectory-array replay validation | `durable_issue1470_similarity_report.json` reports `trajectory_fields.status: unavailable` while `trajectory_metric_fields.status: available`, meaning summary metrics exist but the source artifact cannot replay the selected records. | Identify or publish a durable artifact with aligned raw per-step robot and pedestrian position arrays for the same records, then rerun with `--require-trajectory-comparison`; do not treat the synthetic contract fixture as representative evidence. |
| External-label calibration beyond fixture proof | The tracked fixture proves label plumbing but is not a representative calibration set. | Use a representative labeled source only after its provenance is durable; otherwise keep label checks descriptive. |

## Explicit Non-Claims

- No full benchmark campaign was run for this integration report.
- No Slurm, GPU, or compute submission was performed.
- No planner ranking, leaderboard, paper, dissertation, or safety claim is changed.
- No raw trajectory arrays were reconstructed from summary metrics.
- No synthetic nearest-neighbor grouping was promoted as an observed failure mode.
