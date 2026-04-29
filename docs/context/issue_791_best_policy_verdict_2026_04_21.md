# Issue-791 — best-performing-policy verdict (2026-04-21)

Context snapshot at the end of Wave-7 training (jobs 11872, 11873, 11885 all
COMPLETED; benchmark 11886 PENDING).

## Current evidence

| Source | Eval set | best success | best collision |
|---|---|---|---|
| Leader 11724 (WandB `ibo3aqus`) | eval superset (in-distribution) | 0.929 | — |
| Seed replica 11872 (seed 231) | eval superset | 0.900 @ 6.82M | 0.100 |
| Seed replica 11873 (seed 1337) | eval superset | 0.914 @ 6.29M | 0.086 |
| Broad-training 11885 | all-scenarios manifest (harder) | 0.886 @ 4.19M | 0.114 |
| Benchmark 11871 (partial) | camera-ready matrix, `horizon: 100` | 0.121 | 0.567 |

Seed-variance across 11724/11872/11873 at best checkpoint ≈ **±0.014**, inside
the ±0.02 promotion band from `issue_791_wave6_results_and_benchmark_orca_block.md`.

## Relevant prior memory

- `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`:
  eval-aligned training explains ~97% of the 0.586 → 0.929 PPO lift; curriculum
  / capacity / foresight together contribute the remaining ~3%.
- `memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md`: paper
  framing is "strong policy on a broad scenario matrix", not OOD. Seed variance
  on the benchmark matrix is the primary evidence vehicle.
- `issue_791_wave6_results_and_benchmark_orca_block.md` (horizon-gap section):
  benchmark `horizon: 100` (10s) vs training `max_episode_steps: 400–600`
  (40–60s) is the root-cause hypothesis for the 0.929 → 0.121 collapse.

## Queue anything more right now?

**No.** Benchmark 11886 is still pending and will either confirm or refute the
horizon-mismatch hypothesis. Queuing further training before those numbers land
risks spending GPU hours on the wrong lever.

## Best-performing-policy recipe

Given the evidence, the dominant remaining lever is **horizon alignment**. Every
other lever we have already pulled is either saturated (distribution alignment)
or low-yield (more seeds, more capacity, more scenarios):

- **Base recipe (keep as-is):** leader hyperparameters — LR `7.5e-5`, `clip_range`
  `0.1`, `target_kl` `0.02`, `n_epochs` `4`, `batch_size` `256`; `grid_socnav`
  feature extractor with grid channels `[64, 128, 128]` and socnav hidden
  `[256, 256]`; reward curriculum; predictive foresight enabled; env22 subproc.
- **Training scenarios (keep):** eval-aligned (`ppo_full_maintained_eval_v1.yaml`).
  Distribution alignment is the single biggest contributor to the 0.929 number;
  do not dilute it.
- **Change:** set training `max_episode_steps` to match benchmark `horizon: 100`
  (or align the benchmark upward — see tradeoff). This is the only lever with
  measurable remaining upside on benchmark numbers.
- **Seeds:** 3 replicas (leader + 2 more) is sufficient given the observed
  ±0.014 variance. No additional seed sweep needed for the narrow claim.

## Tradeoff to flag before committing

Retraining with a 10 s horizon will change what the policy optimizes — it may
learn a more aggressive pace that trades collision rate for success under time
pressure, which is a different policy character than the current 0.929 leader.

If benchmark 11886 shows the leader dominated by **collision** failures (not
**timeout** failures), then horizon is not the bottleneck and 11724 ships as-is.
If 11886 shows **timeouts** dominating, the horizon-matched retrain is the right
next step.

## Decision sequence

1. Wait for 11886 to complete.
2. Inspect per-episode termination reasons in the benchmark output (timeout vs
   collision vs success).
3. Branch:
   - **Timeouts dominate** → queue one horizon-matched retrain of the leader
     recipe (`max_episode_steps: 100`, same hyperparams, same scenarios). This
     is the single-shot attempt to close the gap; no need for a second sweep.
   - **Collisions dominate** → 11724 is the final candidate. Promote it, update
     the registry, and move to manuscript writeup.
   - **Mixed with neither dominant** → treat as distribution/scenario sensitivity
     (a different problem); revisit the benchmark config rather than retraining.

## Non-actions (explicitly deprioritized)

- More seeds of the current recipe — variance is already tight.
- More capacity / wider networks — Wave-5/6 attribution shows near-zero residual.
- Exploration boost, reward rescaling — recorded dead ends in
  `issue_791_dead_ends_and_deprioritized.md`.
- OOD held-out suite — deprioritized by maintainer decision (2026-04-20).
- Further broad-training runs beyond 11885 — 11885 is kept as a robustness probe,
  not a promotion candidate.

## Cross-references

- `docs/context/issue_791_promotion_campaign_128k_256k.md`
- `docs/context/issue_791_wave6_results_and_benchmark_orca_block.md`
- `docs/context/issue_791_dead_ends_and_deprioritized.md`
- `memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md`
- `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
