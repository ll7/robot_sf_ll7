---
name: goal-slurm-experiment
description: Keep one skill-owned Robot SF learning or training SLURM job active by selecting the best
  current experiment candidate, closing implementation gaps through an issue-to-PR workflow, and
  submitting the validated job from its owning worktree.
category: research-iteration
kind: orchestrator
phase: implementation
requires_write: true
requires_slurm: true
requires_benchmark_artifacts: false
delegates_to:
- experiment-context
- goal-issue-implementation
- slurm-campaign-submit
- artifact-provenance
- context-note-maintainer
output_schema: skill_run_summary.v1
---

# Goal SLURM Experiment

Use this skill for the always-on background training lane: keep at most one skill-owned learning or
training SLURM job pending/running, and when none exists, find and prepare the best next experiment.

It orchestrates:

- `experiment-context` for canonical config, command, artifact, and validation routing.
- `goal-issue-implementation` for issue-to-PR work when a candidate has a bounded implementation gap.
- `slurm-campaign-submit` for preflight, suitability, submission, and immediate health checks.
- `artifact-provenance` and `context-note-maintainer` for durable handoff and evidence boundaries.

## When to use

Use this skill when the user wants a research-engineer loop that keeps generating learning data
points through SLURM training runs while preserving the main research lanes.

Typical prompts include:

- "keep one training job running"
- "find the next best SLURM experiment"
- "run the best current training candidate"
- "use lucky shots to generate more learning evidence"

Do not use it for:

- one-off local analysis with no training or SLURM candidate;
- benchmark interpretation after a job already completed, unless that analysis is needed before
  selecting the next training job;
- PR review cleanup with no training follow-up;
- CARLA or external-runtime jobs when the current machine cannot satisfy their runtime contract.

## Read First

- `AGENTS.md`
- `docs/maintainer_values.md`
- `local.machine.md` when present
- `SLURM/AGENTS.md`
- `docs/dev/slurm_submission.md`
- `docs/context/issue_1544_slurm_experiment_state_ledger.md`
- `.agents/skills/experiment-context/SKILL.md`
- `.agents/skills/goal-issue-implementation/SKILL.md`
- `.agents/skills/slurm-campaign-submit/SKILL.md`

## Workflow

1. Run the occupancy gate.
   - Refresh live jobs with:
     `squeue --me --format='%i %j %T %P %Q %y %b %M %l %S %R'`
   - Refresh recent outcomes with:
     `sacct -u "$USER" --starttime now-3days --format=JobID,JobName%32,State,ExitCode,Partition,Elapsed,Start,End -P`
   - Treat only jobs named or labeled with the skill prefix `gse-` as the active training lane.
   - If a `gse-` training job is pending or running, monitor it and stop without submitting another
     training job.
   - Do not count validation-only PR gate jobs as the active training job.

2. Discover experiment candidates.
   - Query open issues with training, SLURM, `resource:slurm`, learned-policy, PPO, BC, imitation,
     predictive, Dreamer, ORCA-residual, or launch-packet signals.
   - Inspect the issue body and relevant comments before trusting labels.
   - Check recent context notes and launch packets so completed, blocked, and artifact-rescue lanes
     are not rerun.
   - Prefer issues in `state:ready` with `resource:slurm` or a clear training follow-up over
     blocked parents or analysis-only issues.

3. Classify every plausible candidate.
   - `ready_to_prepare`: bounded implementation or launch-packet gap remains, but the issue is worth
     preparing now.
   - `ready_to_submit`: config, launcher, artifact contract, and validation route are already clear.
   - `already_running`: the same experiment already has an active job.
   - `completed_needs_analysis`: do analysis or artifact promotion instead of rerunning.
   - `blocked_dependency`: missing durable artifacts, exact command surface, required commit, runtime,
     credentials, or maintainer decision.
   - `analysis_only`: useful research work, but not a training submission candidate.

4. Select one candidate.
   - Rank by expected research value, uncertainty reduction, readiness, no duplicate completed run,
     smallest implementation gap, and fit for one low-interference background job.
   - Prefer a single conservative GPU job over an array. Use an array only when the issue explicitly
     requires one submission with throttling.
   - If multiple candidates tie, choose the one with the clearest acceptance contract and cheapest
     validation path.

5. Prepare through the owning worktree.
   - Use a sibling worktree under `../robot_sf_ll7.worktrees/issue-<number>-<slug>` when code or
     config changes are needed.
   - Sync with `origin/main` before implementation work.
   - Use `goal-issue-implementation` for full issue-to-PR work when the candidate needs code,
     launcher, config, docs, or test changes before submission.
   - Keep the candidate config-first. Avoid ad-hoc CLI overrides except short-lived diagnostics.

6. Validate by risk tier.
   - `local_preflight`: cheap login-node checks such as path checks, skill registry checks, shell
     syntax, `--help`, validators, and `--dry-run`.
   - `slurm_pr_gate`: full PR readiness or all-tests verification. Submit this as a SLURM job; do
     not run it locally on the login node.
   - `training_submission`: the actual research/training job. Submit only after required preflight
     passes and any necessary `slurm_pr_gate` has passed or is explicitly deferred because the issue
     is exploratory.

Before submission, keep hypothesis tracking close to the experiment by default. The queue entry,
launch packet, issue comment, private ops ledger, or issue-specific context note should capture the
lightweight hypothesis fields: hypothesis, variant/config, expected signal, result classification,
artifact pointer or snapshot, and next decision. Do not require a central hypothesis ledger for an
ordinary exploratory training run.

Create or update a central hypothesis ledger only when the research family needs cross-run belief
management: many related runs, confusing or contradictory outcomes, repeated negative results,
duplicate variant risk, claim-boundary movement, dissertation or paper synthesis, or a decision
about what the repository believes now rather than what to submit next.

7. Submit exactly one training job.
   - When the selected candidate has or should have a queue entry, update
     `experiments/submission_queue.yaml` and run
     `uv run python scripts/dev/submit_training_jobs.py --dry-run` before submission.
   - Use `uv run python scripts/dev/submit_training_jobs.py --submit` for auto-submit only after
     the entry is `ready_to_submit`, `auto_submit: true`, local machine policy allows SLURM, and
     duplicate checks pass.
   - Submit through existing wrappers such as `scripts/dev/sbatch_use_max_time.sh` or a
     candidate-specific helper under `scripts/dev/`.
   - Use a short skill-owned job name: `gse-<issue>-<slug>`.
   - Capture job id, branch, commit, config path, SLURM script, output root, and stdout/stderr path.
   - For long-running training, capture the predeclared early-stop criteria from the experiment card
     or launch packet: metric, threshold, check cadence, minimum runtime/timesteps, cancel
     condition, and diagnostic-preservation action.
   - Immediately check `squeue`, `sacct`, and early stderr when the job starts.

8. Monitor with long, low-churn intervals.
   - Estimate remaining wall time from observed timesteps/throughput, configured horizon, and eval
     overhead.
   - For healthy multi-hour training jobs, prefer a small number of long sleeps over frequent polls:
     usually two to four checks until the expected finish window, with shorter checks only near
     completion, suspected early failure, or a scheduler state change.
   - Do not open a PR or commit for every scheduled eval gate. Preserve interim eval metrics in an
     issue comment or local note only when they change the operational decision, expose a new
     failure mode, or the user explicitly wants live reporting.
   - If a predeclared low-signal cancel condition is met, complete the preservation action before
     cancelling or immediately after cancellation: record branch, commit, config, logs, manifest,
     output root, and durable artifact status. Label the result diagnostic or failed as appropriate,
     not benchmark proof.
   - Make the durable context-note update and PR after the job finishes, fails, or reaches a
     campaign decision point that changes what should run next.
   - Keep interim metrics labeled as live training state, not benchmark or guarded-policy evidence.

9. Record the handoff.
   - Comment on the issue or update the relevant context note with the job id, branch, commit,
     config, logs, output root, durable artifact status, validation status, and next action.
   - Keep local `output/` paths labeled as non-durable until `artifact-provenance` classifies or
     promotes them.

## Candidate Policy

Ready launch packets are not automatically runnable. Do not submit when the latest issue trail or
context note still requires durable artifact aliases, exact execution commits, a concrete launcher,
base checkpoint provenance, or a maintainer decision.

If the best candidate has completed results that need analysis, route to the relevant analysis skill
or `context-note-maintainer`. Rerunning is wasteful unless the prior run is proven insufficient and a
fresh rerun is the stated next action.

## Guardrails

- Never submit more than one `gse-` training job at a time.
- Never bypass `scripts/dev/submit_training_jobs.py` for a queue-backed training candidate unless
  the queue entry is blocked or the helper cannot represent the existing launcher; document that
  exception in the handoff.
- Never count `slurm_pr_gate` validation jobs as the active training job.
- Do not run full PR readiness or all-tests gates locally; submit them to SLURM.
- Do not submit from an unintended branch, dirty worktree, or stale `origin/main` base without
  calling out the risk.
- Do not let a `slurm` or `resource:slurm` label override issue comments, context notes, or the
  submission suitability gate.
- Do not create PR churn for routine in-progress eval gates; wait for completion/failure or a real
  decision-changing signal before opening durable follow-up PRs.
- Do not let central hypothesis ledgers become a required gate for every exploratory training run;
  use per-experiment notes until public decisions, claims, or cross-run synthesis need a shared
  belief surface.
- Do not treat queued, running, fallback, degraded, or local-only `output/` artifacts as completed
  evidence.
- Do not treat a cancelled low-signal run as useful diagnostic evidence unless the early-stop
  criteria were declared before submission and the preservation action was completed.
- Do not spend a training allocation on a bug that can be fixed from logs, config inspection, a
  validator, or a cheap local preflight.
- Keep benchmark and paper-facing claims conditional until durable artifacts and analysis exist.

## Output

Use `skill_run_summary.v1` or a compact equivalent. Include:

- `occupancy_state`: `idle`, `skill_job_active`, `validation_job_active`, or `blocked_no_slurm`.
- `active_job`: job id, job name, state, partition, and reason when present.
- `candidate_query`: issue filters, context notes, and repo surfaces inspected.
- `selected_candidate`: issue, branch/worktree, config, launcher, and ranking basis.
- `candidate_state`: `ready_to_prepare`, `ready_to_submit`, `already_running`,
  `completed_needs_analysis`, `blocked_dependency`, or `analysis_only`.
- `validation_route`: `local_preflight`, `slurm_pr_gate`, `training_submission`, plus what passed,
  failed, or was deferred.
- `submission`: job id, command surface, output root, log paths, and immediate health check result
  when a job was submitted.
- `next_action`: the one concrete follow-up for the user or next agent.
