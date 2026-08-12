# Issue #6971 — Safety-wrapper paired-campaign preregistration

> Status: preregistration / proposal only. No episodes were run, no compute was submitted, and
> no safety, benchmark, paper, or dissertation result is claimed.

Issue [#6971](https://github.com/ll7/robot_sf_ll7/issues/6971) freezes the next research question
after the retained-row instrumentation gate in [#6970](https://github.com/ll7/robot_sf_ll7/issues/6970):
for each declared planner, does the fixed safety wrapper change exact collision probability when
the same scenario and seed are run with the wrapper off and on? The packet is a design and
analysis contract, not a campaign launcher.

The machine-readable packet is
[`issue_6971_safety_wrapper_paired_preregistration.yaml`](../../configs/benchmarks/issue_6971_safety_wrapper_paired_preregistration.yaml).
Its checker is
[`check_issue_6971_safety_wrapper_preregistration.py`](../../scripts/validation/check_issue_6971_safety_wrapper_preregistration.py).

## Frozen design

The planned matrix is the complete resolved
[`classic_interactions_francis2023.yaml`](../../configs/scenarios/classic_interactions_francis2023.yaml)
suite (48 scenarios), the three existing planner keys (`orca`, `social_force`, and
`prediction_planner`), the named 20-seed schedule S20 (`111`–`130`), and two fixed wrapper arms:
`wrapper_off` and `wrapper_on`. The pairing key is `(planner, scenario_id, seed)`, so each
within-planner contrast has the same scenario and seed in both arms.

This is 48 × 3 × 20 × 2 = 5,760 planned episodes, or 1,920 episodes per planner. The source
factorial roster and the earlier three-seed design anchor remain pinned to the existing
[#4830](https://github.com/ll7/robot_sf_ll7) and
[#3501](https://github.com/ll7/robot_sf_ll7/issues/3501) contracts. The runner settings are fixed
to differential-drive kinematics, 100 steps at 0.1 seconds, two worker slots, subprocess arm
isolation, resumable output, force recording, and no videos.

## Outcomes and estimand

The one primary outcome is `exact_collision_probability`. For each planner, the primary estimand
is the mean `wrapper_on - wrapper_off` difference over the paired scenario-seed cells. A negative
effect is safety-improving only for that named planner and this fixed suite, subject to the
predeclared interval and practical-effect rule.

Secondary outcomes are near-miss probability, minimum predicted separation, completion
probability, false-positive stop rate, stop/yield latency, and wrapper intervention rate.
`progress_at_timeout` is the explicit task-performance cost outcome. Every outcome must use the
exact retained paths in the [#6970 metric contract](issue_6970_paired_effect_metric_contract.md);
legacy or proxy fields cannot substitute for a missing field.

The packet makes no universal claim across planners or environments, no transfer claim to unseen
maps, policies, hardware, or deployment conditions, and no safety certification claim.

## Analysis and decision rules

The paired estimator is computed separately by planner. Uncertainty uses a 1,000-replicate,
95 percent seed-block bootstrap with seed blocks containing the declared scenarios and both arms.
Zero differences remain data. Missing, non-finite, invalid, or unpaired retained fields fail
closed. The S20 schedule and metric roster cannot be changed after inspecting outcomes.

The practical primary threshold is an absolute difference of 0.05, while the precision target is a
95 percent interval width of 0.06 (half-width 0.03). This is an interval-width target, not a
significance promise or guarantee from a pilot variance estimate; the packet declares no pilot
variance basis. The width is evaluated after collection, not guaranteed in advance. For a named planner, the packet classifies a measured safety gain only when
the point estimate is at or below -0.05 and the interval is entirely below zero with complete
native, non-degraded paired data. A result that does not meet the gain rule is reported as no gain
when the precision target is met; incomplete data, wider intervals, or degraded execution are
inconclusive. All planners and all declared secondary outcomes must be reported.

## Exact retained-field manifest

The packet references
[`paired_effect_metric_contract_v1.yaml`](../../configs/benchmarks/paired_effect_metric_contract_v1.yaml)
and repeats its exact eight paths:

`metric_values.exact_collision_probability`, `metric_values.near_miss_probability`,
`metric_values.min_predicted_separation_m`, `metric_values.completion_probability`,
`metric_values.progress_at_timeout`, `metric_values.false_positive_stop_rate`,
`metric_values.stop_yield_latency_s`, and `metric_values.wrapper_intervention_rate`.

The preregistration validator checks source paths and SHA-256 digests for the #4830 campaign
config, #3501 design config, #6970 retained contract, scenario matrix, S20 seed source, runtime
validator, paired report builder, and the historical timing reference.

## Cost estimate

The estimate is planning evidence, not runtime evidence. It uses the durable historical
all-planners campaign summary as a transparent rate reference; that summary is not safety-wrapper
paired evidence.

| quantity | preregistered value |
| --- | ---: |
| planned episodes | 5,760 |
| sequential estimate | 4.70 wall-hours |
| ideal two-slot parallel estimate | 3.73 wall-hours |
| 25% headroom plus setup | 5.16 wall-hours |
| reserved wall clock | 6.0 hours |
| reserved worker-hours | 12.0 |
| modeled raw/report storage | 488 MiB |
| reserved storage | 1 GiB |

The rate reference omits future wrapper overhead, queue time, model loading, and storage variance.
A separately approved canary would have to measure those costs before any full submission.

## Readiness boundary

The packet is `blocked_pending_maintainer_go_no_go`. It does not authorize a Slurm, GPU, or local
campaign submission. Before execution, maintainers must confirm planner/model provenance, the
[#4826](https://github.com/ll7/robot_sf_ll7/issues/4826) per-arm subprocess-isolation gate, a
bounded canary, the retained-row/report validators, and the budget. Any future result remains a
diagnostic fixed-suite result unless a separate evidence review establishes a stronger claim.

## Validation

```text
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/check_issue_6971_safety_wrapper_preregistration.py --json
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/check_preregistration_inference_contract.py --json configs/benchmarks/issue_6971_safety_wrapper_paired_preregistration.yaml
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest -q tests/validation/test_issue_6971_safety_wrapper_preregistration.py
```

Passing these checks proves only that the preregistered contract is internally consistent and
source-pinned. It is not benchmark evidence.
