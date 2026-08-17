# Issue #7313: MPC solver canary diagnostic

> Claim boundary: this is diagnostic-only smoke evidence, not benchmark or paper-facing evidence.
> The canary validates solver and adapter wiring; it does not establish planner quality, safety,
> route-completion performance, or a comparison between the two arms. Confidence in the
> operational observations below is approximately 95%; the scientific interpretation is not
> applicable because every episode timed out at the deliberately short horizon.

## Question and provenance

The canary asked whether the two Issue #5579 incumbent prediction-MPC arms could execute their
native solver path on the declared six-cell tuning scope without solver failures or fallback
execution before spending compute on the full sensitivity campaign.

| Field | Value |
| --- | --- |
| Queue campaign | `issue-7313-mpc-solver-canary-20260817` |
| Slurm job | `14526` |
| Public commit | `462032df2abc3e086655935288c806b9df8bda2b` |
| Config | `configs/analysis/issue_5579_mpc_tuning_sensitivity_v2.yaml` |
| Config SHA-256 | `dfbb3f1b71c53c3c1018ca688d9efbaa2d7681dfe9e2f10e6894520011814033` |
| Packet SHA-256 | `a8f63791a5de3583c3bc128cf842954837233b514888018cbe857183b47832b6` |
| Seed | `101` |
| Horizon | `100` steps |
| Preserved artifact | `wandb://ll7/robot_sf/campaign-issue7313_mpc_solver_canary_20260817:v0` |
| Preserved manifest digest | `sha256:a5c16092c2579715a2578a2c35a3ceb5dcec84d5b838b577d9a7917861df5705` |

The private operations receipt records terminal scheduler state `COMPLETED`, batch exit `0:0`,
and derived exit `0:0`. The tracked summary below is intentionally compact; raw JSONL and cluster
paths remain in the durable private retrieval/preservation system.

## Observed result

The six eligible episodes were three paired scenarios for each of two arms:
`prediction_mpc` and `prediction_mpc_cbf`: six total episodes, three per arm. Every episode
reached the 100-step limit without a route-completion event. No episode
reported a collision event, solver failure, or fallback stop.

| Arm | Episodes | Route complete | Timeout | Collisions | Solver successes | Solver failures | Fallback stops | Minimum clearance (m) | Wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prediction_mpc` | 3 | 0 | 3 | 0 | 300 | 0 | 0 | 1.0242 | 47.6405 |
| `prediction_mpc_cbf` | 3 | 0 | 3 | 0 | 300 | 0 | 0 | 1.0242 | 29.5136 |

The paired scenario-level summary metrics matched across the two arms for all three scenarios.
The CBF wrapper recorded 100/100 interventions in every episode, with zero overrides and zero
hard-constraint violations. In this output contract, that intervention counter therefore does
not by itself show that the action changed: the recorded filtered action matched the proposed
action in the inspected terminal rows. The counter's semantics should be clarified before it is
used as a behavioral comparison metric.

The machine-readable compact summary is [`summary.json`](summary.json).

## Interpretation and next proof

What this canary supports:

- the declared six-cell admission and eligibility contract passed;
- the native solver path was exercised for 600 control steps per arm without solver failure;
- no fallback or degraded execution appeared in the six-row output;
- the CBF wrapper's runtime path was active and produced no hard-constraint violation in this
  short diagnostic.

What it does not support:

- a success-rate, safety, smoothness, speed, or planner-superiority claim;
- a conclusion that the two arms behave identically beyond the recorded summary fields;
- a conclusion that the CBF wrapper changes behavior in either direction;
- a conclusion about the full h600 tuning-budget sensitivity question.

The main limiting observation is that all six cells timed out at 100 steps, so route completion
was not observed. The next smallest proof step is a declared longer-horizon paired run that keeps
the same seed/scenario identities, verifies the CBF intervention semantics, and records route
completion as the primary outcome before any ranking or publication decision. This recommendation
has approximately 85% confidence and would change if the intended canary question is only solver
startup health rather than behavior under the full horizon.

## Reproduction boundary

The run used the public commit and config listed above and was submitted through the private
operations overlay. Reproduction requires that overlay's private queue/worktree contract and the
durable artifact receipt; this public note does not encode cluster names, partitions, credentials,
or local paths.
