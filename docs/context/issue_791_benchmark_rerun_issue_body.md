# Draft GitHub issue body - rerun camera-ready benchmark with issue-791 Wave-5 leader

`gh` CLI is not installed on the workstation that triaged this campaign, so this draft is
checked in for someone with `gh` (or a GitHub MCP-equipped agent) to file. SLURM job 11798
already exists as a fast in-cluster sanity benchmark; this issue tracks the publication-grade
rerun and follow-up release work.

---

## Suggested title

`benchmark: rerun paper_experiment_matrix_v1 with issue-791 Wave-5 PPO leader (eval-aligned + large capacity)`

## Suggested labels

`benchmark`, `issue-791`, `policy-promotion`

## Body

### Summary

The issue-791 promotion campaign produced a new PPO leader on the camera-ready benchmark
adapter contract. The full-suite camera-ready benchmark must be rerun with the new candidate
swapped into the `ppo` planner key so the publication numbers and SNQI evidence reflect the
current leader.

### Why this is needed

- Wave-4 / Wave-5 ablations on the `ppo_full_maintained_eval_v1.yaml` scenario superset lifted
  every PPO architecture by 0.25-0.50 absolute success - a direct OOD-gap closure that the
  earlier 11566 plateau (0.586) could not reach with any architectural lever alone.
- New leader: SLURM job 11724, WandB run `ll7/robot_sf/ibo3aqus`.
  - Best eval (70 episodes, eval-aligned): success_rate=**0.929**, collision_rate=**0.071**,
    SNQI=**0.353** at step 9,961,472 / 10,000,000.
  - Final eval matched best (success=0.929, collision=0.071) - clean convergence.
- Existing benchmark configs still pin the `ppo` planner key to the BR-06 v3 baseline
  (`configs/baselines/ppo_15m_grid_socnav.yaml`). They will not pick up the new leader without
  the configs added in this campaign.

### What's already done in this repository (PR-ready, no benchmark rerun yet)

- Leader artifact saved to a stable model_cache location:
  `output/model_cache/ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417/model.zip`
- Benchmark adapter config: `configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml`
- Camera-ready benchmark config swapping the `ppo` key to the new leader:
  `configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml`
- Model registry entries added in `model/registry.yaml` for the leader and two runner-up
  candidates (jobs 11723, 11660).
- New SLURM benchmark wrapper: `SLURM/Auxme/issue_791_benchmark.sl`
- In-cluster sanity benchmark already submitted on l40s as **SLURM job 11798**
  (`output/slurm/11798-issue791-benchmark.out` once started).

### What this issue tracks

1. **Publication rerun.** Run the camera-ready campaign again on the canonical comparison
   matrix using the new leader, with full bootstrap (300 samples, 95% CI) and AMV coverage,
   producing a new publication bundle.
2. **Decide promotion.** With the rerun bootstraps in hand, decide whether to update
   `configs/baselines/ppo_15m_grid_socnav.yaml` (the canonical promoted-PPO baseline) to point
   at the new leader, or to keep the leader as a parallel `ppo_eval_aligned_large_capacity`
   entry pending OOD validation.
3. **Held-out OOD eval.** Build a held-out scenario family (excluding the `issue_596_*`
   atomic archetypes already inside `ppo_full_maintained_eval_v1.yaml`) and re-run the leader
   on that family. Without it, the 0.929 figure is in-distribution after expanded coverage,
   not a generalization claim - the publication wording must reflect this.

### Hard caveats that must appear in any external write-up

- The Wave-5 leader was trained on the eval superset
  (`configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`). Eval numbers above are
  in-distribution by construction. Report as benchmark-set performance, not OOD generalization.
- We do not yet have a clean "vanilla baseline + eval-aligned (no curriculum, default capacity)"
  control run. Wave-6 control L (job 11799 RUNNING) closes that gap; without it the +0.929 lift
  cannot be cleanly attributed between distribution alignment, curriculum, and capacity.

### Suggested commands

```bash
# Camera-ready rerun (publication path) - submit on l40s
ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
ISSUE791_BENCHMARK_LABEL=issue791-eval-aligned-leader-11724-publication \
sbatch SLURM/Auxme/issue_791_benchmark.sl

# Local preflight to inspect coverage and matrix before consuming a slot
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_791/preflight
```

### Validation gates

- AMV coverage warn-only stays clean (current campaign baseline).
- SNQI contract: rank alignment, outcome separation, max component dominance all within
  the camera-ready bounds defined in the config.
- All planners listed in `paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml`
  finish in `native` or `adapter` mode (no `fallback` / `degraded` outcomes per the
  fail-closed benchmark fallback policy in
  `docs/context/issue_691_benchmark_fallback_policy.md`).

### References

- Promotion campaign log: `docs/context/issue_791_promotion_campaign_128k_256k.md`
- Benchmark fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`
- Issue 791: original quality-gate ablation track.
