# SLURM Issue Batch Status 2026-05-21

Related issues: #1167, #1344, #1354, #1358, #1391, #1392, #1398.

## Current Status

The 2026-05-20 batch was split after PR #1400 was closed without merge. The durable workflow
pieces landed through smaller PRs, while the remaining benchmark interpretation blocker is handled
by the #1398 analyzer fix.

| Issue | Current state | Result |
| --- | --- | --- |
| Issue #1391 | Closed | PR #1405 merged the feature-extractor SLURM output layout cleanup. |
| Issue #1392 | Closed | PR #1403 merged the generic camera-ready benchmark SLURM launcher. |
| Issue #1344 | Closed | PR #1399 merged the paired AMV primary-row protocol and compact evidence. |
| Issue #1354 | Open | Bounded cross-kinematics execution proof exists; paper-facing interpretation remains open. |
| Issue #1398 | In progress | This branch fixes false SNQI row mismatches for repeated planner keys across kinematics. |
| Issue #1167 | Closed | PR #1412 merged the obstacle-feature pipeline path; no SLURM comparison job was submitted in the 2026-05-20 batch. |
| Issue #1358 | Open | PR #1409 merged a residual guarded-PPO surface; a true learned residual training run still needs training lineage. |

No user SLURM jobs were queued when this note was written.

## Benchmark Evidence Carried Forward

Issue #1344 post-#1384 reruns completed cleanly and are recorded in
`docs/context/issue_1344_paired_amv_protocol_report.md`:

- nominal primary-row job `12572`: `COMPLETED 0:0`, `benchmark_success=true`
- stress primary-row job `12574`: `COMPLETED 0:0`, `benchmark_success=true`

Issue #1354 has bounded compatibility proof for `goal`, `orca`, and `social_force` across
`differential_drive`, `bicycle_drive`, and `holonomic`:

- initial job `12571`: `COMPLETED 0:0`, but analyzer reported row-level SNQI mismatches
- post-#1384 job `12575`: `COMPLETED 0:0`, no campaign warnings, but the old analyzer still
  reported row-level SNQI mismatches

The #1398 fix reinterprets the #1354 post-#1384 artifact by matching summary rows on
`(planner_key, kinematics)` instead of only `planner_key`. On the local job `12575` artifact, the
patched analyzer reports no automated inconsistencies.

Validation command for that reinterpretation:

```bash
RAW_ISSUE_1354_ROOT=output/benchmarks/issue_1354_post1384/cross_kinematics_v1_issue1354-cross-post1384_20260520_170814
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root "$RAW_ISSUE_1354_ROOT" \
  --output-json output/validation/issue_1398_cross_kinematics_analysis.json \
  --output-md output/validation/issue_1398_cross_kinematics_analysis.md
```

The raw campaign output remains local and ignored; set `RAW_ISSUE_1354_ROOT` to the checkout that
owns the generated `output/` tree before rerunning the command. Promote only compact summaries or
manifests if the #1354 report is updated for paper-facing use.

## Remaining SLURM Decisions

Issue #1354 should not launch a larger fixed-spec AMV campaign until the #1398 analyzer fix is merged and
the existing bounded proof is reinterpreted in the issue thread. After that, the next decision is
whether the bounded three-planner cross-kinematics surface is enough for an outlook note, or whether
to launch a broader #1353-dependent matrix.

Issue #1358 should not submit a generic PPO or policy-search smoke as the training run. PR #1409 gives the
bounded residual adapter and diagnostics surface, but the issue still asks for a learned residual
policy with explicit training config, checkpoint lineage, residual contribution diagnostics, and
nominal gate evidence.

Issue #1167 is closed after PR #1412 made the config-first obstacle-feature pipeline runnable. If the
actual same-seed training/evaluation comparison is still desired, open or reuse a follow-up issue
that submits the canonical config from `docs/context/issue_1167_predictive_obstacle_pipeline.md`
and reports ADE/FDE plus downstream planner metrics.
