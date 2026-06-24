# Issue #3254 Predictive Crossing-Conflict Rerun 13042

## Scope

This bundle preserves the compact public-safe interpretation of Slurm job `13042` for
issue #3254. The run fixed the predictive ego-schema launch mismatch and completed training, but
the resulting planner failed the final evaluation quality gate.

## Evidence Status

- `schema`: `issue_3254_predictive_crossing_conflict_13042.v1`
- `claim_boundary`: `negative_training_result_not_benchmark_promotion`
- `result_classification`: `negative_result`
- `paper_facing`: `false`
- `benchmark_promotion`: `false`

## Main Finding

The run demonstrates that the schema/config path can train a non-degenerate `predictive_ego_v1`
model from the crossing-conflict weighted dataset. It does not support promoting the trained
planner: final evaluation success was `0.08696`, below the configured `0.3` success gate.

## Key Metrics

- Mixed training samples: `40446`
- Feature schema: `predictive_ego_v1`
- Best validation ADE: `0.04837`
- Best validation FDE: `0.09735`
- Final evaluation episodes: `69`
- Final success rate: `0.08696`
- Final collision rate: `0.24638`
- Final mean minimum distance: `2.28687`
- Failed episodes: `63`

## Files

- [summary.json](summary.json): compact provenance, metrics, quality-gate status, and checksum
  anchors for the local source artifacts.
- [SHA256SUMS](SHA256SUMS): checksums for this compact evidence bundle.

## Source Artifact Boundary

The source artifacts were preserved from the local Slurm run directory before this compact bundle
was created. That ignored run tree includes the checkpoint, raw JSONL rows, and full local
summaries. This tracked bundle intentionally keeps only the compact public evidence needed for
issue routing and future synthesis.

## Claim Boundary

This is useful evidence for stopping blind #3254 reruns and focusing the next step on planner
tuning or negative-result synthesis. It is not a benchmark-strength planner result, a safety claim,
a paper-facing result, or a model promotion.
