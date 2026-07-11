# Data-driven scenario generation (#4932)

[#4932](https://github.com/ll7/robot_sf_ll7/issues/4932) is the staged plan for discovering
critical simulated episodes, distilling their critical windows, and cataloging the resulting
scenarios for later review. Generated scenarios are hypotheses, not benchmark evidence, and are
never added to release matrices automatically.

## Claim boundary

- Generated scenarios are hypotheses requiring manual review.
- A source-template load must not be reported as successful standalone replay of a distilled
  mid-episode state.
- No benchmark, paper, or dissertation conclusion follows from generated output.

## Current status

- [PR #4962](https://github.com/ll7/robot_sf_ll7/pull/4962) landed the trace-to-catalog schema and
  deterministic critical-segment distillation.
- The stage 1–3 integration path covers seeded sampling, catalog aggregation and deduplication,
  and replay-status validation. It requires executable proof before generated output is treated as
  software smoke evidence.
- [Issue #5203](https://github.com/ll7/robot_sf_ll7/issues/5203) owns the narrow replay adapter
  from a `generated-scenario-catalog-entry.v1` entry to standalone generated-only scenario YAML.
  Unsupported state must remain explicitly `not_representable_yet`; a passed source load alone is
  insufficient.
- Stage 4 has a separate deterministic archive sampler. It selects existing generated records
  without replacement, biases probability toward lower minimum-clearance values, and records the
  source-archive checksum plus every weight and random draw. The tracked CPU demo uses
  `configs/benchmarks/scenario_generation_rare_event_sampler.yaml`.

## Intended output contract

- A seeded run manifest records source scenarios, episode seeds, sampling policies, and claim
  boundary.
- Each distilled candidate records its source episode and seed, criticality signal and metrics,
  time window, review status, and any replay-representation warning.
- Generated catalog entries remain separate from the hand-authored benchmark matrix and carry
  `required_manual_review: true` and `benchmark_evidence: false`.

## Remaining research work

Calibrated importance-sampling estimators, learned proposals, exact mid-episode scenario-YAML
reconstruction, criticality-reproduction campaigns, and certification into hand-authored benchmark
families require separate evidence and review. Archive selection changes which existing hypotheses
are inspected first; it does not estimate a population failure probability.
