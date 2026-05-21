# Issue #1361 Command-Lattice Corridor-Deadlock Assessment (2026-05-20)

Date: 2026-05-20

Related issue:

- [robot_sf_ll7#1361](https://github.com/ll7/robot_sf_ll7/issues/1361): research: evaluate state-lattice baseline for corridor deadlocks

Related context:

- [issue_1318_teb_corridor_deadlock_eval.md](issue_1318_teb_corridor_deadlock_eval.md)
- [summary.json](evidence/issue_1318_teb_corridor_deadlock_2026-05-20/summary.json)
- [issue_1318_teb_corridor_deadlock_slice.yaml](../../configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml)
- [hybrid_rule_v3_fast_progress_static_escape.yaml](../../configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml)
- [hybrid_rule_v3_teb_like_rollout.yaml](../../configs/algos/hybrid_rule_v3_teb_like_rollout.yaml)

## Goal

Evaluate whether the smallest available lattice-like sampled local-planner baseline already gives
useful evidence on the #1318 classic-merging corridor-deadlock slice, and separate that evidence
from the stronger claim that a true state-lattice planner has been evaluated.

## Selected Proxy Baseline

Selected baseline: `hybrid_rule_v3_fast_progress_static_escape`

Why it satisfies a first-pass proxy boundary:

- It is a native Robot SF planner candidate, so no external framework or ROS bridge is required.
- It uses a finite sampled unicycle command lattice and rollout scorer rather than a pure reactive
  controller.
- Its base config, `configs/algos/hybrid_rule_v3_teb_like_rollout.yaml`, explicitly enables a
  topology-aware route-guide candidate surface with sampled linear/angular commands and lookahead
  distances.
- Its candidate config adds static-clearance escape, static recentering, and static corridor-transit
  behavior, which directly targets the static-obstacle local-minimum mechanism exposed by #1318.

Boundary: this is not a full state-lattice planner, not a full motion-primitive planner, and not a
timed-elastic-band optimizer. It lacks precomputed kinodynamic primitives, graph search over
discrete states, persistent multi-step primitive sequencing, and learned value scoring over
primitives. It is the smallest already-runnable finite command-lattice / sampled-rollout baseline
that satisfies the Robot SF planner contract and has same-subset evidence on the #1318 slice.

## Same-Subset Evidence

Issue #1318 already ran this selected baseline through the same map-runner episode-record contract
used for TEB and ORCA:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run robot_sf_bench run \
  --matrix configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml \
  --algo hybrid_rule_local_planner \
  --algo-config configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml \
  --benchmark-profile experimental \
  --horizon 600 \
  --out output/benchmarks/issue_1318_teb_corridor_deadlock_hybrid.jsonl \
  --workers 1 \
  --no-resume \
  --no-video \
  --structured-output json
```

Tracked compact evidence:
`docs/context/evidence/issue_1318_teb_corridor_deadlock_2026-05-20/summary.json`

| Algo | Scenario | Episodes | Successes | Collisions | Timeouts | Avg steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `teb` | `classic_merging_low` | 2 | 0 | 2 | 0 | 247.5 |
| `teb` | `classic_merging_medium` | 3 | 0 | 3 | 0 | 249.0 |
| `orca` | `classic_merging_low` | 2 | 1 | 1 | 0 | 280.5 |
| `orca` | `classic_merging_medium` | 3 | 1 | 2 | 0 | 307.0 |
| `hybrid_rule_local_planner` | `classic_merging_low` | 2 | 2 | 0 | 0 | 427.0 |
| `hybrid_rule_local_planner` | `classic_merging_medium` | 3 | 2 | 1 | 0 | 487.7 |

Seed-level interpretation from #1318:

- TEB collided with static geometry on all five selected seeds.
- ORCA produced two successes and three pedestrian collisions.
- The selected hybrid-rule command-lattice baseline solved four of five selected seeds, with one
  pedestrian collision on `classic_merging_medium` seed `112`.

## Verdict

Current tier: `sampled-rollout proxy evaluated / true state lattice still separate`.

The available same-subset evidence supports a narrower conclusion than "state lattice evaluated": a
native finite command-lattice / sampled-rollout baseline materially changes the corridor-deadlock
outcome compared with the in-repo TEB approximation. It does not fully solve the slice, but it is
clearly stronger on this narrow surface: four successes out of five versus TEB's zero successes and
ORCA's two.

This note therefore does not fully close the stronger acceptance path of evaluating a true
state-lattice planner. It narrows the decision: before adding a new planner family, first decide
whether the already-runnable sampled-rollout proxy is sufficient for this follow-up thread. If not,
the true state-lattice work should be split into a new implementation issue with a distinct
primitive library, collision checker, action extraction, and runtime budget.

## Limitations

- This is a narrow five-episode corridor-deadlock slice, not paper-facing benchmark evidence.
- The selected baseline is a sampled command-lattice rollout planner, not a full kinodynamic
  state-lattice or motion-primitive planner.
- The result does not prove general corridor robustness, because the remaining collision is a
  pedestrian collision and the slice is intentionally small.
- The tracked compact evidence is enough for issue sequencing, but any promotion or paper-facing
  claim would require a broader fixed protocol.

## Follow-Up Boundary

Recommended follow-ups, if maintainers want to continue this thread:

- Diagnose `classic_merging_medium` seed `112` for hybrid-rule pedestrian-collision mechanics.
- If a true state-lattice planner is still desired, open a separate implementation issue that
  specifies primitive library, collision checker, action extraction, and runtime budget.
- Keep external/reference TEB separate under Issue #1360; it requires a ROS bridge or clean-room
  implementation path, not a small adapter.

## Validation

This note reuses #1318 tracked evidence and adds no new benchmark output.

Validation commands for this PR:

```bash
rg -n "Issue #1361|hybrid_rule_v3_fast_progress_static_escape|command-lattice" docs/context docs/README.md
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
