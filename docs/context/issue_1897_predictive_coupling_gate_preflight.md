# Issue #1897 Predictive Coupling Gate Preflight

Related issues:

- Issue #1897: <https://github.com/ll7/robot_sf_ll7/issues/1897>
- Parent issue #1490: <https://github.com/ll7/robot_sf_ll7/issues/1490>
- Setup issue #1856: <https://github.com/ll7/robot_sf_ll7/issues/1856>
- Blocked expansion issues: #1505, #1506, #1507

## Decision

Keep the predictive-v2 expansion blocked. The local #1897 closed-loop gate failed because the
revised `phase_coupled_sequence_gate` row did not improve closed-loop success over
`baseline_like`.

The row produced a tiny clearance gain, but both rows had `0.0000` hard success and `0.0000`
global success. This is exactly the signal the #1856 gate was designed to reject: clearance-only
movement is not a reason to spend on the old four-way predictive-v2 training matrix.

## Evidence Boundary

Tracked compact evidence:
`docs/context/evidence/issue_1897_predictive_coupling_gate_2026-05-31/README.md`

Local raw campaign output remains disposable worktree-local state and is not referenced as a
durable dependency.

The checkpoint came from a local machine cache in the main checkout's ignored model-cache tree. It
is not a durable artifact pointer. The SHA-256 was:

```text
a28aed6d6ad7e1ebf597277ade1cf908efa6da038d0a9fcfdf80c7c31d8d1be1
```

This preflight is therefore useful routing evidence, not benchmark-strength or paper-facing
evidence.

## Result

| Variant | Hard success | Global success | Global mean min-distance |
|---|---:|---:|---:|
| `baseline_like` | 0.0000 | 0.0000 | 3.5460 |
| `phase_coupled_sequence_gate` | 0.0000 | 0.0000 | 3.5568 |

Closed-loop gate:

- status: `failed`
- reason: `global_success_delta_below_gate`
- global success delta: `0.0000`
- hard success delta: `0.0000`
- global mean-min-distance delta: `0.0108`

## Setup Fix

The first #1897 run failed before evaluation because
`configs/benchmarks/predictive_hard_seeds_v1.yaml` included deprecated `classic_crossing_*`
aliases while `configs/scenarios/classic_interactions.yaml` now exposes the corresponding
`classic_cross_trap_*` scenarios. This branch removes the stale alias rows and adds a regression
test so the default predictive success campaign can build its hard-case subset again.

## Routing

- Keep #1505, #1506, and #1507 blocked.
- Update #1490 with `revise_coupling_gate` or `stop_predictive_v2_expansion`; do not select the old
  four-way matrix from this result.
- If predictive-v2 work continues, the next child should revise the planner objective or checkpoint
  provenance boundary before running another gate.
