# Issue #6951 progress-shaped behavioral cloning diagnostic

This is diagnostic-only smoke evidence for the issue #6951 objective ablation. It does not
promote a learned policy, alter the released residual-BC configuration, or support a benchmark
or paper-facing claim.

## Claim boundary

The authorized run tested whether a bounded progress-weighted expert-action negative
log-likelihood (NLL) objective changes the behavior-cloning control arm on the preregistered
`planner_sanity_simple` smoke scenario. The result is a promotion non-result: neither arm
produced a successful policy in the evaluation cells.

Fallback/degraded rows: none observed in the collection, training, or evaluation logs. The
execution mode was native local CPU simulation/training. Raw datasets, checkpoints, and logs
remain worktree-local ignored artifacts; the compact summaries in this directory are the only
tracked evidence.

## Protocol

| Mechanism | Source issue | Evidence tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveats |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| Uniform Arm A vs. route-progress-weighted Arm B | #6951 | diagnostic-only smoke evidence | `issue_6951_arm_{a,b}_progress_weighted_bc.yaml`; 8 epochs, batch 64, learning rate 0.0003, CPU | training 111/112/113; evaluation 111/112/113 | dataset digest `aa70b611…07a09`; six local checkpoints; compact result summary | 9 evaluation cells per arm; success, collision, route-length reduction | promotion condition not met; retain the research direction as unresolved | 3-episode smoke dataset, one scenario, no nominal/full campaign, and the legacy converter duplicated terminal observations |

The dataset contained three expert episodes with action lengths 61, 63, and 56. The
route-progress arrays contained 62, 64, and 57 values respectively: one active route-polyline
length at the initial boundary plus one after each action. Arm B used `lambda=0.5`, normalization
scale `1.0`, and weight bounds `[0.5, 2.0]`; Arm A used uniform `[1.0, 1.0]` weights. Both arms
used the same dataset digest and code path.

## Observed result

| Arm | Evaluation cells | Successes | Collisions | Mean route-length reduction | Mean final remaining route length |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 9 | 0/9 | 9/9 | 3.254 m | 12.995 m |
| B | 9 | 0/9 | 9/9 | 3.489 m | 12.760 m |

The paired Arm B minus Arm A route-length-reduction difference was `+0.235 m` over the nine
train-seed/evaluation-seed cells. A bootstrap interval over those paired cells was
`[+0.191, +0.283] m` (95%; cell-level uncertainty only). This small diagnostic shift is not a
useful-policy result: all 18 rollouts collided and none completed the route. The evidence
therefore does not establish that progress weighting fixes the parked residual-BC failure.

## Provenance and next decision

- Source dataset: `output/benchmarks/expert_trajectories/issue_1428_orca_residual_bc_progress_v1_smoke.npz`
- Dataset SHA-256: `aa70b611875be2b6b67001e063bcf558ff00eff4ef398a4da498f33df8a07a09`
- Enabling code head: `28b5e1939` (based on `origin/main` `ce91ad5bd`)
- Machine-readable synthesis: [`summary.json`](summary.json)
- Artifact classification: [`artifact_provenance.json`](artifact_provenance.json)

Keep #1475 parked and #6951 open. A follow-up should first correct or explicitly freeze the
observation-history contract, then run a properly sized paired campaign with maintainer-frozen
success/progress thresholds and a durable dataset/checkpoint retrieval pointer. No released
default should change from this smoke result.
