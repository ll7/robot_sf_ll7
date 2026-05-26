# Issue #1544 Slurm Experiment State Ledger

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1544>

## Goal

Keep SLURM execution issues from staying open behind stale local trails. Each issue should carry one
explicit state block that says whether the run is not started, live, waiting on artifact promotion,
needs rescue, needs a rerun, closed as failed/inconclusive, or is complete with durable evidence.

## Canonical Surface

Keep the ledger human-readable in issue bodies/comments and `docs/context/` notes for now. The
repository already has machine-readable registries for assets and catalogs such as
`model/registry.yaml`, `maps/registry.yaml`, and `docs/context/policy_search/candidate_registry.yaml`,
but none is a generic SLURM execution-state ledger. Do not invent a new registry schema inside this
workflow note.

## Required Fields

Record these fields in every state block:

| Field | Requirement |
| --- | --- |
| `state` | One of the exact states below. |
| `issue_number` | GitHub issue number. |
| `slurm_job_id` | Real job id, or explicit `not_submitted`. |
| `branch` | Branch used for the last meaningful execution attempt. |
| `commit` | Commit SHA if known; use `unknown` when the issue trail does not record it. |
| `config_path` | Config or launcher path(s) that define the run. |
| `output_root` | Repository-root-relative `output/...` path, or explicit `missing`. |
| `stdout_stderr_path` | Repository-root-relative log path, or explicit `missing`. |
| `durable_artifact_pointer_status` | `missing`, `partial`, or a concrete durable pointer summary. |
| `next_action` | Single next operational step. |
| `closure_condition` | Exact condition that allows the issue to close or transition. |

Useful extras when known: `last_update`, `scheduler_status`, `comparison_status`, and
`notes_non_evidence`.

## States

| State | Use when | Closure / transition rule |
| --- | --- | --- |
| `not_submitted` | No real SLURM job has been launched yet. | Submit a real run, or close only if the issue is intentionally cancelled/superseded with an explicit non-claim. |
| `submitted_running` | A real job is queued/running and the referenced trail is still inspectable. | Do not close directly from a vague "still running" note; reclassify after the job finishes, fails, or the trail goes stale. |
| `completed_pending_artifact_promotion` | The run produced the needed local outputs, but durable pointers and/or required comparison notes are still missing. | Promote durable artifacts and comparison evidence, then move to `completed_with_durable_evidence`; if the local trail decays first, move to `artifact_rescue`. |
| `artifact_rescue` | The issue records a meaningful job/result root, but the current checkout no longer has the referenced local trail or only non-durable placeholders survive. | Rescue/promote the missing evidence and reclassify, or close the stale trail as `rerun_required` or `inconclusive_close`. |
| `rerun_required` | The prior run is not sufficient or not recoverable, but the experiment still matters. | Close the stale trail only after linking the fresh rerun issue/plan with current commit/config ownership. |
| `failed_closed` | The run failed with enough diagnostics to treat the issue as a closed failure rather than a pending execution trail. | Close with the failure reason, exact non-claim, and why no rerun is being taken now. |
| `inconclusive_close` | The issue is ending without proof-grade success or failure evidence. | Close with the missing evidence called out explicitly and no benchmark/paper-strength claim. |
| `completed_with_durable_evidence` | Required artifacts, pointers, and follow-up comparison/proof are durable and linked. | Close the issue with the durable evidence links and final outcome statement. |

## Stale-Trail Closure Protocol

1. Refresh the latest state block from the issue body/comments before touching the issue.
2. Treat `output/...` paths as local scratch or exploratory support only unless a durable pointer,
   tracked evidence copy, or release artifact is linked.
3. If the issue says a job finished but the named `output_root` or `stdout_stderr_path` is missing
   in the active checkout/worktree, reclassify the issue to `artifact_rescue` immediately instead of
   leaving it under `submitted_running` or `completed_pending_artifact_promotion`.
4. In `artifact_rescue`, do one bounded follow-up: either recover/promote the real dataset,
   checkpoint, manifest, or report trail, or conclude that the trail cannot support the issue.
5. If rescue fails but the experiment still matters, close the stale trail as `rerun_required` and
   point to the successor issue/config. If rescue fails and no fresh run is justified now, close as
   `inconclusive_close`.
6. Only use `completed_with_durable_evidence` when the issue's acceptance surface is durable, not
   merely present under `output/`.

## Local Output vs Durable Evidence

- Local `output/...` paths are not enough to close a SLURM execution issue by themselves.
- Dry-run manifests, placeholder checkpoints, or files with obviously non-real metadata such as
  `dry_run=true`, `episode_count=1` for a 141-episode target, or `wall_clock_hours=0.0` remain
  non-evidence until replaced or backed by a durable pointer.
- Acceptable durable surfaces include a tracked evidence bundle under `docs/context/evidence/`, a
  reviewed registry entry that already points to an external durable artifact, or a release/W&B
  artifact pointer recorded in the issue or note.
- Required downstream analysis remains part of closure. A finished training job without the promised
  policy-analysis comparison stays incomplete.

See also `docs/context/artifact_evidence_vocabulary.md` and
`docs/context/slurm_multi_worktree_branch_workflow.md`.

## Copy/Paste Block

```yaml
slurm_experiment_state:
  state: not_submitted
  issue_number: 0000
  slurm_job_id: not_submitted
  branch: issue-0000-short-slug
  commit: unknown
  config_path:
    - configs/path/to/config.yaml
  output_root: missing
  stdout_stderr_path: missing
  durable_artifact_pointer_status: missing
  next_action: submit or document why submission is blocked
  closure_condition: close only after durable evidence exists or the trail is explicitly closed as failed/inconclusive/rerun-required
  last_update: YYYY-MM-DD
  comparison_status: pending
  notes_non_evidence: local output paths stay non-durable until promoted
```

## Conservative Example: Issue #1108

Based on the recorded body/comments for [#1108](https://github.com/ll7/robot_sf_ll7/issues/1108),
the conservative state is `artifact_rescue`, not `completed_pending_artifact_promotion`.

```yaml
slurm_experiment_state:
  state: artifact_rescue
  issue_number: 1108
  slurm_job_id: 12472
  branch: issue-1108-bc-warm-start-ppo
  commit: unknown
  config_path:
    - configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml
    - configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml
  output_root: output/slurm/issue1108-bcppo-job-12472/
  stdout_stderr_path: output/slurm/12472-issue1108-bc-warm-start.out
  durable_artifact_pointer_status: missing
  next_action: recover the real job-12472 dataset/checkpoint trail or split a clean rerun follow-up
  closure_condition: close only after durable dataset/BC/PPO evidence and comparison exist, or after the stale trail is explicitly closed as rerun_required/inconclusive_close
  last_update: 2026-05-24
  comparison_status: pending against 27dbe5xu and b60iopxt
  notes_non_evidence: >
    Current checkout lacks the recorded log/output root; surviving local dataset/checkpoint files are
    dry-run placeholders or non-durable local files and do not complete the issue.
```

Why `artifact_rescue`:

- the issue trail records a real SLURM job (`12472`) and named local paths,
- the 2026-05-24 refresh says the current checkout no longer has that log or output root,
- local files under `output/benchmarks/...` are explicitly called out as dry-run placeholders or
  non-durable local copies,
- the required policy-analysis comparison is still pending.

If the real `12472` outputs are later rescued and promoted, move forward from this state. If they
cannot be recovered, close the stale trail as `rerun_required` or `inconclusive_close` instead of
pretending the issue is merely "still in progress."

## Related Surfaces

- `docs/context/issue_1108_bc_warm_start_execution.md`
- `docs/context/artifact_evidence_vocabulary.md`
- `docs/context/slurm_multi_worktree_branch_workflow.md`
- <https://github.com/ll7/robot_sf_ll7/issues/1108>
