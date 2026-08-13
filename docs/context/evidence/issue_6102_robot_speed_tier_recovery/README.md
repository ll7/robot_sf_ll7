# Issue #6102 robot speed-tier campaign recovery

> **RECOVERED LOCALLY — NOT DURABLE BENCHMARK EVIDENCE AND NOT PAPER-FACING**

This packet records the recovery of the existing job `13828` result tree for the
native issue #5578 robot speed-tier campaign. The local copy is structurally
complete and passes the campaign synthesizer and its recorded file checksums after
rebasing the recorded paths to the local `output/` tree. The independent result
root named by the execution receipt is absent on this machine, however. The packet
therefore preserves a compact, reviewable recovery record without promoting the
local raw output to benchmark or dissertation evidence.

## What was verified

| Gate | Result |
| --- | --- |
| Private-ops job | `13828`, scheduler state `COMPLETED`, exit `0:0` |
| Producing commit | `481164b08d861e4af9777fe35734f88bda2754e9` with a clean worktree recorded |
| Frozen grid | 6 scenarios × 3 speed tiers × 4 planners × 30 seeds = 2,160 cells |
| Recorded execution status | `complete_native`; 2,160 native, 0 excluded |
| Local raw batches | 12 batches × 180 rows; 2,160 cell-summary rows |
| Campaign synthesis re-check | `grid_complete=true`, `all_native=true`, 2,160 cells |
| Local checksum re-check | 74/74 recorded files verified after path rebasing |
| Independent source copy | **missing**; the recorded external root does not exist locally |
| Admission | **not admitted**; `decision-required` remains open |

The descriptive synthesis reports the same planner ordering at all three speed
tiers (`scenario_adaptive_hybrid_orca_v2_collision_guard`, `orca`, `ppo`,
`prediction_planner`), with zero rank flips. This is descriptive only and is not a
planner-ranking claim. Of the 24 non-nominal planner-by-tier-by-metric rows, 10 are
classified `no_material_shift`, 8 `inconclusive`, and 6
`intervention_not_activated`; the latter are all the prediction-planner rows and
cannot answer a speed-effect question for that planner. These classifications are
retained as a synthesis diagnostic, not as an inferential result.

## Provenance boundary

The complete raw tree remains at the ignored local path recorded in
`recovery_manifest.json`. It is approximately 1.85 GB and is intentionally not
copied into Git. The compact manifest records the source commit, frozen command,
configuration, seed surface, manifest identity hash, local artifact hashes, and
the exact missing-copy condition. The tracked packet is metadata and a bounded
recovery summary; it is not a substitute for an independently recoverable raw
artifact or a reviewed result-interpretation packet.

## Re-check commands

From a checkout containing the local recovery output:

```bash
uv run python -c 'from scripts.benchmark.run_issue_5578_speed_tier_campaign import synthesize_from_cell_summaries; import json; r=synthesize_from_cell_summaries("output/issue5578-native-speed-tier/13828/cell_summaries.jsonl"); print(json.dumps({k:r[k] for k in ("evidence_status","per_cell_count","native_cell_count","excluded_cell_count","all_native","grid_complete")}, sort_keys=True))'
sed 's#/home/luttkule/external_data_hub/benchmark-results/robot_sf_ll7/issue5578/native-speed-tier/job-13828#output/issue5578-native-speed-tier/13828#' output/issue5578-native-speed-tier/13828/sha256sums.txt | sha256sum -c -
```

Both commands are validation of the local recovery only. They do not establish an
independent copy, public availability, or paper-facing eligibility.

## Next decision

`decision-required`: choose an independently recoverable artifact location and
owner, then re-run the promotion and interpretation gates against that copy. The
recommended action is **do not rerun the 2,160-cell campaign yet**: recover or
restore the missing result root first, because the local result already contains a
complete native grid and a rerun would create a second lineage without resolving
the custody gap. Any later admission must remain separate from this recovery
packet and must explicitly address the six non-activated prediction-planner rows.
