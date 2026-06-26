# Issue #2557 Recovered Reward-Curriculum Seed Runs (updated 2026-06-26)

Related issues: [#2557](https://github.com/ll7/robot_sf_ll7/issues/2557),
[#2919](https://github.com/ll7/robot_sf_ll7/issues/2919),
[#3203](https://github.com/ll7/robot_sf_ll7/issues/3203),
[#3266](https://github.com/ll7/robot_sf_ll7/issues/3266). Recovery tooling:
[#3590](https://github.com/ll7/robot_sf_ll7/pull/3590) (merged).

## Status

Diagnostic as of 2026-06-26. Three completed `expert_ppo_issue_2557_reward_curriculum_promotion_10m`
seed runs trained and evaluated successfully but lost their training-run manifests to the
cross-worktree serializer failure (the manifest write raised on an `evaluation_scenario_config`
path materialised under a sibling worktree). The training compute was intact on the cluster; only
the `benchmarks/ppo_imitation/runs/<run_id>.json` manifest was missing. All three manifests were
**backfilled from the retained artefacts** with
`scripts/tools/backfill_training_run_manifest.py` (added in the now-merged #3590), and the full
~2.1 GB evidence per run (checkpoints, W&B, episode logs, per-scenario eval) was retrieved locally.

These runs are recorded here as **diagnostic-tier seed evidence only**. They are **not**
benchmark-success or paper-grade results (see caveats below).

## Recovered Runs

| Job | Issue | Seed | Success (mean, CI95) | Collision | SNQI | Path eff. | Episode return | Wall | Scenarios |
|---|---|---|---|---|---|---|---|---|---|
| 13024 | #3266 | 509 | 0.831 [0.810, 0.851] | 0.167 | −0.0692 | 0.832 | 18.64 | 16.23 h | 70 |
| 12949 | #2919 | 506 | 0.851 [0.831, 0.870] | 0.144 | +0.0169 | 0.826 | 18.49 | 13.76 h | 70 |
| 12950 | #3203 | 508 | 0.810 [0.790, 0.831] | 0.187 | −0.1115 | 0.830 | 17.33 | 16.07 h | 70 |

Metrics are aggregate means across the 70-scenario evaluation profile, taken from each run's
backfilled `expert_policies/<policy>.json` manifest. All three carry `validation_state: draft`.

## Provenance & Recovery

- **Failure mode**: cross-worktree manifest loss — `serialize_training_run` raised on a path
  outside both the artefact root and the repo root, aborting the post-training manifest write and
  discarding evidence. Fixed forward in #3590 (lenient basename fallback for optional eval/config
  path fields; `episode_log_path` stays strict).
- **Recovery**: `backfill_training_run_manifest.py` reconstructs the training-run manifest from
  the artefacts the pipeline already wrote (expert-policy manifest → metrics/seeds, perf json →
  run_id/wall-clock, eval-by-scenario → scenario coverage, episode/eval/perf files → path fields).
  No training or evaluation was re-run. Each recovered manifest is tagged as backfilled in its
  `notes`.
- **Wall-clock cross-checks**: backfilled wall-clock (16.23 / 13.76 / 16.07 h) matches the
  private ledger's recorded elapsed for each job, confirming the perf-summary provenance.

## Diagnostic Classification & Caveats

These runs **must not** be cited as benchmark-success, ranking, or Results-chapter evidence:

- **SNQI is at or below zero** (−0.069 / +0.017 / −0.111). #3266 is specifically the
  "resolve PPO SNQI validity blockers" campaign; these values indicate the social-navigation
  quality index is marginal/unestablished, not a positive result.
- **Collision rate is high** (14–19%). The issue_2557 campaign success criterion is "final success
  improves over baseline **without increased collision rate**"; these runs do not clear it.
- **Single-seed manifests**: each manifest covers one seed; cross-seed variance statistics are not
  yet established (see next section).

Allowed wording: *"Recovered reward-curriculum expert-policy seed runs exist with payload-complete,
provenance-tracked manifests for seed-variance bookkeeping and pipeline diagnostics only; their
marginal SNQI and elevated collision rate mean they do not establish benchmark-success or
Results-chapter evidence."*

## In-flight Sweep & Next Steps

A bounded seed-variance fill of the contiguous 501–511 block is in flight on the a30 partition
(jobs 13153/13154/13155, seeds 507/510/511), submitted for variance statistics only under an
explicit override of the campaign `do_not_rerun_without_new_hypothesis` policy. Once those finish,
retrieve, and analyse, a **consolidated 501–511 seed-variance evidence note** should supersede this
stub with mean ± CI across the full block, still classified diagnostic until the SNQI/collision
caveats are repaired under a new hypothesis.
