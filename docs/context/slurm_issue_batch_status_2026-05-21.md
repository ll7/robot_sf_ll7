# SLURM Issue Batch Status 2026-05-21

Related issues: Issue #1167, Issue #1344, Issue #1354, Issue #1358, Issue #1391,
Issue #1392, Issue #1398.

## Canonical Source-Of-Truth Policy

This file remains the canonical SLURM issue-status ledger for the current open training and
artifact-rescue issue batch. Do not create a successor ledger until there is a concrete consumer
that needs a different path or machine-readable registry. Use
`docs/context/issue_1544_slurm_experiment_state_ledger.md` for the shared state vocabulary and
stale-trail closure protocol; use this file for the issue-by-issue source-of-truth rows.

Individual issue bodies and comments should carry only a short pointer or summary. When an issue
body, issue comment, PR text, and this ledger disagree, trust the newest source that includes both:

- a `source_of_truth` value pointing at this ledger or an explicitly named successor; and
- a `last_verified` date tied to a read-only issue/artifact inspection.

If no newer source names a replacement ledger, agents should use this file's row and fail closed on
missing durable artifacts. Local `output/...` paths, raw SLURM logs, and worktree-local checkpoints
remain triage clues until promoted through a durable artifact pointer or compact tracked evidence.

Use this short pointer in issue bodies or comments when reconciling duplicated status:

```yaml
slurm_issue_pointer:
  issue_number: 0000
  slurm_state: "not_submitted | submitted_running | completed_pending_artifact_promotion | artifact_rescue | rerun_required | failed_closed | inconclusive_close | completed_with_durable_evidence | parent_blocked | insufficient_data"
  source_of_truth: docs/context/slurm_issue_batch_status_2026-05-21.md
  last_verified: YYYY-MM-DD
  artifact_status: "missing | partial | durable | not_applicable"
  next_action: one short operational step
```

## Current Status

The 2026-05-20 batch was split after PR #1400 was closed without merge. The durable workflow
pieces landed through smaller PRs, while the remaining benchmark interpretation blocker is handled
by the Issue #1398 analyzer fix.

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

## Reusable SLURM Issue Status Block

Use this compact block in issue bodies, issue comments, and context notes when a training or
benchmark issue is waiting on SLURM/Auxme execution. It is human-readable status, not a new
machine-readable registry. The stricter stale-trail closure states remain documented in
`docs/context/issue_1544_slurm_experiment_state_ledger.md`. When a full block would duplicate this
ledger, prefer the short `slurm_issue_pointer` above.

```yaml
slurm_issue_status:
  issue_number: 0000
  state: not_submitted | submitted_running | completed_pending_artifact_promotion | artifact_rescue | rerun_required | failed_closed | inconclusive_close | completed_with_durable_evidence | parent_blocked | insufficient_data
  source_of_truth: docs/context/slurm_issue_batch_status_2026-05-21.md
  slurm_job_id: not_submitted
  branch: unknown
  commit: unknown
  config_path:
    - missing
  output_root: missing
  stdout_stderr_path: missing
  durable_artifact_pointer_status: missing
  next_action: record the single next operational step
  closure_condition: state the condition needed before this issue can close or transition
  last_update: YYYY-MM-DD
  comparison_status: pending
  status_basis: issue body | issue comment | context note | local inspection
  notes_non_evidence: local output paths are scratch until promoted to durable evidence
```

Field rules:

- `issue_number`, `state`, `source_of_truth`, `slurm_job_id`, `commit`, `config_path`, `output_root`,
  `durable_artifact_pointer_status`, and `next_action` should always be present.
- Use `not_submitted`, `missing`, or `unknown` rather than omitting fields.
- `parent_blocked` is for umbrella issues that should not be executed directly.
- `insufficient_data` is for issues where the current issue/context trail does not identify enough
  config, commit, or artifact information to submit or rescue a run safely.
- `output_root` and `stdout_stderr_path` may name local `output/...` locations for triage, but they
  are not durable evidence by themselves. Use the artifact vocabulary in
  `docs/context/artifact_evidence_vocabulary.md` before closing an issue as complete.

## Current SLURM-Needed Training Issue Snapshot 2026-05-30

This docs-only pass did not submit SLURM jobs, move artifacts, upload checkpoints, or write Project
metadata. It reviewed open issues carrying the `slurm` label and records conservative status blocks
only where the issue trail already named enough context. Missing `output/...` paths stay
non-evidence.

### Issue #1470 Oracle Imitation Dataset Collection

```yaml
slurm_issue_status:
  issue_number: 1470
  state: not_submitted
  source_of_truth: docs/context/slurm_issue_batch_status_2026-05-21.md
  slurm_job_id: not_submitted
  branch: issue-1397-oracle-imitation-launch-packet
  commit: unknown
  config_path:
    - configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml
  output_root: missing
  stdout_stderr_path: missing
  durable_artifact_pointer_status: missing dataset manifest, checksum, split validation, and artifact pointer
  next_action: validate the launch packet at the exact collection commit and select durable dataset storage before submission
  closure_condition: dataset manifest, checksums, split/leakage validation, and durable artifact pointer exist, or the trail is closed failed/inconclusive/rerun-required
  last_update: 2026-05-30
  comparison_status: pending split/leakage validation
  status_basis: issue body and docs/context/policy_search/SLURM/003_imitation_oracle_dataset_campaign.md
  notes_non_evidence: issue body names candidate launch commit 034ac79e, but the exact collection commit is not recorded as the execution commit
```

### Issue #1472 Learned Risk Model v1 SLURM Campaign

```yaml
slurm_issue_status:
  issue_number: 1472
  state: not_submitted
  source_of_truth: docs/context/slurm_issue_batch_status_2026-05-21.md
  slurm_job_id: not_submitted
  branch: issue-1395-learned-risk-launch-packet
  commit: unknown
  config_path:
    - configs/training/learned_risk_model_issue_1395_launch_packet.yaml
  output_root: missing
  stdout_stderr_path: missing
  durable_artifact_pointer_status: missing trace manifest, learned-risk checkpoint pointer, diagnostics report, and downstream evaluation summary
  next_action: validate the launch packet and replace pending trace/baseline artifact aliases with concrete durable pointers before training
  closure_condition: trace inputs, checkpoint, diagnostics, and stress/full-matrix comparison are durable, or the campaign is closed failed/inconclusive/rerun-required
  last_update: 2026-05-30
  comparison_status: pending stress/full-matrix comparison after training
  status_basis: issue body and docs/context/policy_search/SLURM/001_learned_risk_model_v1.md
  notes_non_evidence: launch-packet validity is not learned-risk training or benchmark evidence
```

### Issue #1474 Shielded PPO Repair Campaign

```yaml
slurm_issue_status:
  issue_number: 1474
  state: completed_with_durable_evidence
  source_of_truth: docs/context/slurm_issue_batch_status_2026-05-21.md
  slurm_job_id: 12674
  branch: issue-1474-shielded-ppo-repair
  commit: cc3e8552b0fa1ae47ddb3f42cd74443576c6e9c0
  config_path:
    - configs/training/ppo/ablations/expert_ppo_issue_1474_shielded_repair_collision20_5m.yaml
    - configs/training/shielded_ppo_issue_1396_launch_packet.yaml
    - SLURM/Auxme/issue_1474_shielded_ppo_repair.sl
    - scripts/dev/sbatch_shielded_ppo_repair_issue1474.sh
  output_root: output/slurm/issue1474-shielded-ppo-repair-job-12674/
  stdout_stderr_path:
    - output/slurm/12674-issue1474-shielded-ppo-repair.out
    - output/slurm/12674-issue1474-shielded-ppo-repair.err
  durable_artifact_pointer_status: durable trained checkpoint plus compact evidence; baseline and downstream benchmark artifacts still not promoted
  next_action: repair the guarded-PPO zero-motion smoke failure before nominal-sanity or stress escalation
  closure_condition: guarded smoke and nominal gates pass with runtime guard diagnostics, or the campaign is revised/rejected/closed as an unsuccessful repair candidate
  last_update: 2026-06-01
  comparison_status: guarded smoke failed launch-packet success gate; nominal-sanity blocked
  status_basis: issue comments, local worktree inspection, squeue/sacct, Slurm stdout/stderr, W&B artifact metadata, and guarded smoke report
  notes_non_evidence: >
    Failed warm-start probe 12673 showed the durable BR06 v3 checkpoint observation space no longer
    matches the current training env. Active retrain job 12674 intentionally starts from the current
    env with exactly the collision penalty repair delta. Interim health checks at
    2026-06-01T07:33+02:00 and 2026-06-01T08:35+02:00 showed RUNNING on a30, W&B run
    d8w8uykh, CUDA selected, num_envs=10, and progress from 819200 to 1536000/5000000
    timesteps reached. The first 500k evaluation gate emitted success_rate=0.36,
    collision_rate=0.61, snqi=-1.58, path_efficiency=0.782, and eval_episode_return=-0.437;
    the 1m evaluation gate emitted success_rate=0.66, collision_rate=0.33, snqi=-0.657,
    path_efficiency=0.817, and eval_episode_return=16.2. The 1.5m evaluation gate emitted
    success_rate=0.64, collision_rate=0.37, snqi=-0.708, path_efficiency=0.828, and
    eval_episode_return=21.0, suggesting the 1m safety improvement had not monotonically improved
    by 1.5m. Several PPO update blocks after 800k
    reported early stopping from max KL, so the training dynamics should be reviewed after the full
    run rather than accepted from interim improvement alone.
    Job 12674 later completed successfully and selected the 5m checkpoint with success_rate=0.83,
    collision_rate=0.16, snqi=-0.1023, path_efficiency=0.8145, and eval_episode_return=27.93.
    The W&B artifact
    ll7/robot_sf/ppo_expert_issue_1474_shielded_repair_collision20_5m-best-success:v5 is durable
    training evidence, but not benchmark or guarded-policy evidence. Guarded smoke job 12685_0
    completed with collision_rate=0.0 and near_miss_rate=0.0 but failed the launch-packet smoke
    gate because success_rate=0.0; diagnostics show zero-motion overconservative_stop behavior
    with decision_counts.goal_reached=80. Guard-saturated gains must not be reported as raw PPO
    improvement.
```

### Issue #1475 ORCA-Residual BC Smoke And Nominal Lineage

```yaml
slurm_issue_status:
  issue_number: 1475
  state: failed_closed
  source_of_truth: docs/context/slurm_issue_batch_status_2026-05-21.md
  slurm_job_id: 12749
  branch: gse-1475-smoke-20260605
  commit: 5faaa318d609f87730757d7fbda65b799178b5c5
  config_path:
    - configs/training/orca_residual/orca_residual_bc_issue_1428.yaml
    - docs/context/policy_search/SLURM/005_orca_residual_bc_lineage.md
  output_root: output/slurm/issue1475-orca-residual-bc-job-12749
  stdout_stderr_path:
    - output/slurm/12749-issue1475-orca-residual-bc.out
    - output/slurm/12749-issue1475-orca-residual-bc.err
  durable_artifact_pointer_status: compact summary and smoke report tracked; raw output paths remain local and non-durable
  next_action: revise ORCA-residual BC candidate or smoke target for low-progress timeout behavior before rerun
  closure_condition: revised bounded smoke records success before nominal escalation; fallback/degraded rows remain excluded
  last_update: 2026-06-05
  comparison_status: smoke failed closed; nominal escalation blocked
  status_basis: docs/context/evidence/issue_1475_orca_residual_bc_smoke_12749_summary.json and docs/context/policy_search/reports/2026-06-05_orca_residual_guarded_ppo_v0_smoke.md
  notes_non_evidence: diagnostic smoke status only; success_rate=0.0 with timeout_low_progress is revise evidence, not learned-residual success evidence
```

### Issue #1108 BC Warm-Start PPO Artifact Rescue

```yaml
slurm_issue_status:
  issue_number: 1108
  state: artifact_rescue
  source_of_truth: docs/context/slurm_issue_batch_status_2026-05-21.md
  slurm_job_id: 12472
  branch: issue-1108-bc-warm-start-ppo
  commit: unknown
  config_path:
    - configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml
    - configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml
    - SLURM/Auxme/issue_1108_bc_warm_start.sl
  output_root: output/slurm/issue1108-bcppo-job-12472/
  stdout_stderr_path: output/slurm/12472-issue1108-bc-warm-start.out
  durable_artifact_pointer_status: missing or not yet proven durable for dataset, BC checkpoint, PPO checkpoint, and policy-analysis comparison
  next_action: recover/promote the historical job trail or split a clean rerun follow-up
  closure_condition: classify as artifact_rescue_success, rerun_required, or inconclusive_close with durable evidence or explicit missing-evidence rationale
  last_update: 2026-05-30
  comparison_status: pending against 27dbe5xu and b60iopxt
  status_basis: issue body and docs/context/issue_1544_slurm_experiment_state_ledger.md
  notes_non_evidence: local output paths and historical logs are not durable evidence unless recovered and promoted
```

### Insufficient-Data Or Parent-Blocked SLURM Issues

Rows in this table are not execution blocks. They mark why a full SLURM status block would be
misleading until the parent issue is converted into a concrete execution child or the missing
status basis is added.

| Issue | Status | Status basis | Why this pass does not create a full execution block | Next action |
| --- | --- | --- | --- | --- |
| Issue #1358 | `parent_blocked` | Issue body | Parent issue; current body says execute through Issue #1475 only. | Wait for Issue #1475 to classify the ORCA-residual lane as continue/revise/stop. |
| Issue #1490 | `parent_blocked` | Issue body and `docs/context/issue_1543_predictive_v2_negative_audit.md` | Predictive-v2 umbrella; current body says do not execute directly after Issue #1543. | Maintainer selects a revised predictive-v2 hypothesis or child sequence. |
| Issue #1496 | `parent_blocked` | Issue body | Blocked by Issue #1470 durable oracle-imitation dataset. | Wait for Issue #1470 dataset manifest/checksums and split validation. |

## Benchmark Evidence Carried Forward

Issue #1344 reruns after Issue #1384 completed cleanly and are recorded in
`docs/context/issue_1344_paired_amv_protocol_report.md`:

- nominal primary-row job `12572`: `COMPLETED 0:0`, `benchmark_success=true`
- stress primary-row job `12574`: `COMPLETED 0:0`, `benchmark_success=true`

Issue #1354 has bounded compatibility proof for `goal`, `orca`, and `social_force` across
`differential_drive`, `bicycle_drive`, and `holonomic`:

- initial job `12571`: `COMPLETED 0:0`, but analyzer reported row-level SNQI mismatches
- job `12575` after Issue #1384: `COMPLETED 0:0`, no campaign warnings, but the old analyzer still
  reported row-level SNQI mismatches

The Issue #1398 fix reinterprets the Issue #1354 artifact from after Issue #1384 by matching summary rows on
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
manifests if the Issue #1354 report is updated for paper-facing use.

## Remaining SLURM Decisions

Issue #1354 should not launch a larger fixed-spec AMV campaign until the Issue #1398 analyzer fix is merged and
the existing bounded proof is reinterpreted in the issue thread. After that, the next decision is
whether the bounded three-planner cross-kinematics surface is enough for an outlook note, or whether
to launch a broader Issue #1353-dependent matrix.

Issue #1358 should not submit a generic PPO or policy-search smoke as the training run. PR #1409 gives the
bounded residual adapter and diagnostics surface, but the issue still asks for a learned residual
policy with explicit training config, checkpoint lineage, residual contribution diagnostics, and
nominal gate evidence.

Issue #1167 is closed after PR #1412 made the config-first obstacle-feature pipeline runnable. If the
actual same-seed training/evaluation comparison is still desired, open or reuse a follow-up issue
that submits the canonical config from `docs/context/issue_1167_predictive_obstacle_pipeline.md`
and reports ADE/FDE plus downstream planner metrics.
