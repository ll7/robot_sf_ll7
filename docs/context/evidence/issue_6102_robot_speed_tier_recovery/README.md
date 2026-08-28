# Issue #6102 robot speed-tier campaign recovery

> **DURABLY PRESERVED — INTERPRETATION NOT YET ADMITTED OR PAPER-FACING**

This packet records the recovery of job `13828` for the native issue #5578
robot-speed-tier campaign. The recovered copy is structurally complete and passes
the campaign synthesizer and recorded file checksums. The same 76 source files are
now preserved in the immutable W&B artifact
`ll7/robot_sf/campaign-issue5578-native-speed-tier-job-13828:v0`. The artifact
manifest and every stored and decompressed source object were checked against
SHA-256 and byte-size records. This closes the custody gap, but does not by itself
admit a benchmark, planner-ranking, causal, dissertation, or paper claim.

## What was verified

| Gate | Result |
| --- | --- |
| Private-ops job | `13828`, scheduler state `COMPLETED`, exit `0:0` |
| Producing commit | `481164b08d861e4af9777fe35734f88bda2754e9` with a clean worktree recorded |
| Frozen grid | 6 scenarios × 3 speed tiers × 4 planners × 30 seeds = 2,160 cells |
| Recorded execution status | `complete_native`; 2,160 native, 0 excluded |
| Raw batches | 12 batches × 180 rows; 2,160 cell-summary rows |
| Campaign synthesis re-check | `grid_complete=true`, `all_native=true`, 2,160 cells |
| Independent source copy | W&B artifact `ll7/robot_sf/campaign-issue5578-native-speed-tier-job-13828:v0` |
| Artifact verification | 76/76 stored objects and 76/76 decompressed sources match manifest digests and sizes |
| Canonical synthesis parity | Stored and current outputs match after normalizing only `source_path` |
| Source-complete contrasts | 24/24 rows bound to the immutable synthesis member by issue #7980 |
| Admission | **not admitted**; a separate domain-aware admission decision is required |

The descriptive synthesis reports the same planner ordering at all three speed
tiers (`scenario_adaptive_hybrid_orca_v2_collision_guard`, `orca`, `ppo`,
`prediction_planner`), with zero rank flips. This is descriptive only and is not a
planner-ranking claim. Of the 24 registered non-nominal contrasts, 10 are
classified `no_material_shift`, 8 `inconclusive`, and 6
`intervention_not_activated`. The latter are the prediction-planner contrasts at
the two non-nominal tiers for collision, near-miss, and success rate. They cannot
answer a speed-effect question for that planner.

## Provenance boundary

The task-local hydration remains under ignored `output/` and is intentionally not
copied into Git. The durable dependency is the immutable W&B artifact above. The
compact manifest records the source commit, frozen command, configuration, seed
surface, manifest identity hash, artifact receipt, full verification result, and
canonical synthesis parity. This tracked projection is review metadata, not a
replacement for the raw artifact and not an admission receipt.

The result packet is `result_interpretation_packet.v1.json`. Its deterministic
caption, exact-digest review report, and source/output checksums are recorded in
`result_interpretation_caption.txt`, `result_interpretation_review.v1.json`, and
`SHA256SUMS`. No figure is emitted because this diagnostic packet has no controlled
visual assertion.

## Source-complete successor packet

Issue #7980 adds
`result_interpretation_packet.issue_7980.v1.json` without replacing or silently
mutating the reviewed packet above. The successor has one metric and one fail-closed
decision for each of the 24 registered planner-by-speed-tier-by-metric contrasts.
Each metric carries one versioned source-binding payload containing the complete
canonical synthesis row: pooled effect and standard error, 180-pair denominator,
both directional raw and Holm-adjusted tests and bounds, preregistered harm threshold,
activation diagnostics, classification, and the immutable `synthesis.json` SHA-256.

The successor still uses `smoke_diagnostic` / `diagnostic_only`. Its 18 activated
contrasts remain `inconclusive` pending a separate domain-aware admission decision;
the six inactive prediction-planner contrasts remain `invalid`. The tracked packet
therefore improves auditability without admitting a null effect, ranking, causal,
safety, dissertation, release, or paper-facing claim. Its deterministic caption and
checksums are `result_interpretation_caption.issue_7980.txt` and
`SHA256SUMS.issue_7980`. The packet, caption, and checksum manifest each have an
exact-hash `*.review.json` sidecar from the shared evidence-writer convention; the
caption and checksum manifest also carry inline `NEEDS-REVIEW` markers.

## Re-check commands

From a checkout containing a verified artifact hydration:

```bash
uv run python scripts/benchmark/run_issue_5578_speed_tier_campaign.py --synthesize output/issue_7792/job13828_wandb_v0/cell_summaries.jsonl --synthesis-out output/issue_7792/current_synthesis.json --json
uv run python scripts/analysis/build_result_interpretation_packet.py --input docs/context/evidence/issue_6102_robot_speed_tier_recovery/result_interpretation_packet.v1.json --validate-only
uv run python scripts/analysis/build_issue_7980_speed_tier_packet.py --synthesis output/issue_7980/source/synthesis.json --check
uv run python scripts/analysis/build_result_interpretation_packet.py --input docs/context/evidence/issue_6102_robot_speed_tier_recovery/result_interpretation_packet.issue_7980.v1.json --validate-only
(cd docs/context/evidence/issue_6102_robot_speed_tier_recovery && sha256sum -c SHA256SUMS.issue_7980)
```

The first command reproduces the compact synthesis. The later commands regenerate
and validate the tracked source-complete projection. None grants benchmark or
paper-facing eligibility.

## Next decision

Do not rerun the 2,160-cell campaign: the native grid and durable raw lineage are
complete. The source-complete successor resolves the missing-statistic and source-
binding gap, but it does not make the admission decision. Any later admission must
explicitly address the six non-activated prediction-planner contrasts and must not
promote the descriptive planner ordering into a ranking claim.
