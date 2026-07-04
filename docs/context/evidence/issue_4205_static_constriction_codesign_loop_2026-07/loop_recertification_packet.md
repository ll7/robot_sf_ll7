<!-- AI-GENERATED (robot_sf#4205 / job 13294, 2026-07-04) — NEEDS-REVIEW -->
# Co-Design Loop Re-Certification Packet (job 13294)

**Claim boundary:** single pre-registered worked example (issue #4205,
contract `configs/research/issue_4205_static_constriction_codesign_loop_v1.yaml`,
PR #4218; hydration preflight passed 2026-07-04). Diagnostic tier. Feeds diss
F-C4(iii). **No mitigation-superiority, planner-superiority, paper, or
dissertation claim is made here** — and see the headline finding below: this
run does NOT support a "generalizing safety gain" wording.

## Provenance

- Slurm job **13294** (imech192 l40s), COMPLETED 0:0, elapsed 1:20 (27
  episodes; the pre-registered "low"-density constriction scenarios terminate
  quickly), campaign `issue_4205_codesign_loop_run_20260704`.
- Frozen PPO checkpoint `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`
  (job-13175 failure-row lineage), hydration manifest verified fail-closed
  (`--require-hydrated-checkpoint` exit 0; algo-config sha byte-identical to
  the pre-registration).
- Grid exactly as pre-registered: 3 arms × scenarios
  [classic_bottleneck_low, classic_head_on_corridor_low, narrow_passage] ×
  seeds [111, 112, 113]; trace fields present on 27/27 rows.
- Artifacts: ledger row 13294; metrics + per-episode rows mirrored at
  `imech156-u:~/git/robot_sf_ll7/output/issue4205-codesign-loop-run/13294/`.
- Attempt lineage (transparency): jobs 13286–13293 were infrastructure/runner
  defects, each root-caused and fixed publicly (#4378 profile plumbing, #4394
  map resolution, #4406 dict-valued trace fields; #4403 parallel-path fix
  pending; plus private sbatch worktree/uv fixes). Attempt 9 is the first
  clean execution.

## Per-arm results (9 episodes/arm)

| arm | success | collisions | near-miss | deadlocks | note |
| --- | --- | --- | --- | --- | --- |
| ppo_frozen (discovered-failure baseline) | 7/9 (0.778) | 0 | 1 | **0** | the targeted failure family did not occur |
| ppo_frozen_wrapper_on (hardening arm) | 6/9 (0.667) | 0 | **0** | 0 | removes the near-miss; costs one success (conservatism) |
| ppo_frozen_cbf_on (hardening arm) | 7/9 (0.778) | 0 | 1 | 0 | metric-identical to baseline |

## Headline finding (honest null on the gain leg)

1. **The loop MECHANICS are demonstrated end-to-end** — pre-registration →
   frozen-checkpoint hydration with fail-closed identity checks → hardened
   arms (wrapper / CBF) → re-certification run with native trace capture.
   This is the process evidence F-C4(iii) needs for its
   discover→attribute→harden→re-certify narrative.
2. **The targeted failure mode (static deadlock) did not reproduce in the
   baseline on the pre-registered surface** (0 deadlocks in 9 baseline
   episodes). Consequently **no hardening gain is measurable from this run**:
   there was nothing for the hardened arms to prevent. A "measured,
   generalizing safety gain" wording is NOT supported by job 13294.
3. Secondary observations at n=9/arm (all differences are single episodes —
   treat as anecdotes, not effects): the wrapper eliminated the baseline's
   one near-miss at the cost of one success (the expected conservatism
   trade); CBF changed nothing on this surface.

## Interpretation of the null + next-step options (decision for the author/diss side)

The job-13175 PPO failure rows that motivated the loop came from the full
h500 benchmark surface; the pre-registered re-certification family used the
*low-density* variants of the constriction scenarios, where the frozen policy
simply does not fail. Options, in increasing cost:

- **(a) Word F-C4(iii) as process contribution:** the adversarial-assurance
  loop is demonstrated as machinery with a bounded worked example; the gain
  leg remains explicitly open. Zero additional compute.
- **(b) Second pre-registered pass on a failure-expressing surface:** same
  arms and discipline, scenario variants where the baseline demonstrably
  deadlocks (e.g. the high-density constriction family, selected from the
  13175/13268 failure rows *before* running). One more small l40s job.
- **(c) Fold into the #4206 trace-capable h600 re-run** (11-planner roster,
  seeds 20–24): its ppo rows will carry trace-verified mechanism labels on
  the full surface and can serve as the "discover" leg at scale, with a
  targeted re-certification to follow.

The pre-registration's own fail-closed doctrine applies: this packet reports
the contract outcome as-is; no post-hoc surface substitution is performed.

<!-- /AI-GENERATED -->
