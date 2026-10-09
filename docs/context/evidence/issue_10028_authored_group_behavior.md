# Authored pedestrian-group diagnostic for issue #10028

AI-GENERATED NEEDS-REVIEW. Independent pedestrian-behaviour and benchmark review is pending.

Runtime source: `d24f7908a03bb83feb32f8ecdf42878c14235e23`. Fresh-main comparator: `f42f769e08a102f44bbf2edce85746d3f8f2dc60`.
This note and its JSON companion preserve diagnostic observations, not release admission or a populated-scene planner-quality claim.

## Defect and regression proof

The scenario loader retained social-group definitions, but simulator population construction discarded them and registered every named pedestrian as a singleton. The two scenario configs also needed their intended memberships declared. The existing runtime fix instantiates authored groups before the role controller and restores initial memberships on reset. Join uses the anchors' centroid with its existing 0.2 m threshold. Unassigned pedestrians retain their singleton groups.

All four native regression cases fail against fresh main, then pass on the runtime source. The focused group, single-pedestrian and metadata checks pass 68 cases. The reset and one-step cases are the cheapest checks of population/reset wiring; the bounded 400-step native join is necessary because a synthetic near-target case cannot expose social repulsion preventing arrival. These tests use real authored configuration and physics, with no test-only production seam.

## Group census

The 48-scenario main roster was reset on dev seeds 1001–1003: 144 resets per revision. Scenarios with a multi-person group in this sample increase **18 to 20**. Only join and leave change; the other 46 scenarios retain identical reset group sizes and associated counts. This is a sampled runtime census, not a repository-wide count or navigation score.

Join changes from `[1,1,1]` to `[1,2]` at reset and reaches `[3]` at step 143 on each sampled seed. Mean size changes from 1 to 1.5 at reset and 3 after joining. Leave changes from `[1,1,1]` to `[3]` at reset, then `[1,2]` after its first active step; mean size changes from 1 to 3 and then 1.5. Three classic group-crossing scenarios already explicitly request positive sampled-group fractions; the fix declares two additional authored Francis groups. The JSON companion includes every changed census row and the bounded native probe.

**Running main changes the pedestrian behaviour of the released join_group and leave_group scenarios.** It also activates group forces for authored memberships. Frozen artifacts, prior result tables and release branches remain untouched; prior trajectories cannot be treated as the same interaction after this fix.

## Empty-world sweep

Main roster, all 14 arms, 48 scenarios, dev seeds 1001/1002, four workers and step traces: 1344 expected slots, 1344 written rows. Execution complete: **True**. Missing/failed/duplicate/unexpected slots and incomplete traces: 0/0/0/0/0. Every trace is actor-free (maximum pedestrian count 0) and every raw row has the runtime source SHA. Runtime: 3475.5 seconds.

Classification totals: **authored_budget_timeout: 73; success: 1131; wall_contact: 140**. Execution completion does not imply a green behaviour result. Every non-success slot is listed in the JSON companion, with observed contact steps, termination/budget, trace completeness, fallback flags and a raw-row digest covering the trace. Explicit fallback/degraded evidence from row metadata and arm summaries is excluded before success. Observed wall contact takes precedence over timeout or route completion. Remaining terminal categories are kept explicit. Budget timeouts use the authored horizon, not a shortened runner cap.

| Arm | Success | Wall contact | Budget timeout | Other/excluded |
| --- | ---: | ---: | ---: | ---: |
| goal | 63 | 25 | 8 | 0 |
| guarded_ppo | 87 | 0 | 9 | 0 |
| hybrid_rule_v4_fast_progress_static_escape | 87 | 0 | 9 | 0 |
| hybrid_rule_v4_fast_progress_static_escape_continuous | 84 | 0 | 12 | 0 |
| orca | 94 | 0 | 2 | 0 |
| ppo | 62 | 34 | 0 | 0 |
| prediction_planner | 79 | 17 | 0 | 0 |
| predictive_mppi | 92 | 0 | 4 | 0 |
| risk_dwa | 92 | 0 | 4 | 0 |
| sacadrl | 29 | 64 | 3 | 0 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4 | 91 | 0 | 5 | 0 |
| scenario_adaptive_hybrid_orca_v2_collision_guard_v4 | 91 | 0 | 5 | 0 |
| social_force | 86 | 0 | 10 | 0 |
| socnav_sampling | 94 | 0 | 2 | 0 |

No paired base empty-world sweep or width sweep was run. These are observed contact/terminal categories, not evidence that failures are new or pre-existing. No Slurm job was submitted for this source. Empty-world diagnostics cannot establish pedestrian-interaction quality or navigation admission; independent approval remains pending.

Replay from the recorded runtime source with the project environment and at most four workers:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_NO_SYNC=1 uv run python scripts/validation/run_empty_world_sweep.py --head-sha d24f7908a03bb83feb32f8ecdf42878c14235e23 --suite main --seeds 1001 1002 --workers 4 --output-dir output/validation/issue_10028_authored_groups --arm-isolation subprocess
```

## Artifact boundary

The JSON companion durably records all failure classifications, per-arm counts, census changes, native probes and hashes of the flattened rows and test logs. Full raw campaigns/traces are retained locally and have not been durably published; a checksum is integrity information, not archival custody. Ignored downloaded model caches are disposable dependencies, not newly trained artifacts. The later evidence commit changes documentation and inventory only; the tested runtime bytes are unchanged. Do not use this diagnostic note as release evidence.
