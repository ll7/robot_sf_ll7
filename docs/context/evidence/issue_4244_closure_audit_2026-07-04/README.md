# Issue 4244 Closure Audit

This audit maps issue [#4244](https://github.com/ll7/robot_sf_ll7/issues/4244)
acceptance criteria to merged pull requests and the current local contract check.
Issue #4244 asks for a pre-declared seven-arm learned-policy comparison matrix,
not training execution or a benchmark result.

Evidence status: contract complete for closure. This is preregistration and
smoke/dry-run evidence only; it is not benchmark evidence, does not submit
Slurm or GPU work, and does not edit paper or dissertation claims.

## Acceptance Evidence

| Criterion | Evidence | Status |
| --- | --- | --- |
| One config with identical environment, scenario, seed, step, evaluation, and per-arm budget across seven arms. | PR [#4251](https://github.com/ll7/robot_sf_ll7/pull/4251) added `configs/training/comparison_matrix/issue_4244_seven_arm_preregistration.yaml` with the exact seven-arm roster and shared-budget contract. PR [#4475](https://github.com/ll7/robot_sf_ll7/pull/4475) replaced the PPO-Mamba placeholder with `configs/training/ppo/issue_4014_ppo_mamba_smoke.yaml`. | Met. |
| Matrix-runner entry point with queue-plan shape and no training execution in this slice. | PR [#4251](https://github.com/ll7/robot_sf_ll7/pull/4251) added `scripts/training/run_comparison_matrix_preregistration.py`; the runner emits `training_executed=false`, `slurm_or_gpu_submitted=false`, and planned-not-submitted queue entries in dry-run output. | Met. |
| Explicit inclusion gates in config: smoke contracts fail closed, identical-budget gate, and offline-online SAC enters only after standalone offline-pretraining gate. | PR [#4251](https://github.com/ll7/robot_sf_ll7/pull/4251) added per-arm smoke and identical-budget gates. PR [#4257](https://github.com/ll7/robot_sf_ll7/pull/4257) produced the issue #4245 offline-pretraining evidence summaries. PR [#4347](https://github.com/ll7/robot_sf_ll7/pull/4347) wired those summaries into the SAC arm gate. | Met. |
| Config-validation tests and CPU dry-run matrix runner over two arms. | PR [#4251](https://github.com/ll7/robot_sf_ll7/pull/4251) added `tests/training/test_comparison_matrix_preregistration_issue_4244.py` and the two-arm dry-run CLI test. PR [#4347](https://github.com/ll7/robot_sf_ll7/pull/4347) extended SAC gate tests. This PR keeps the focused test green and adds a regression check for the landed mechanism-schema pointer. | Met. |
| Pre-registered analysis plan: rank table, confidence intervals, and per-mechanism breakdown schema instrumentation landed. | PR [#4251](https://github.com/ll7/robot_sf_ll7/pull/4251) preregistered rank-table and bootstrap confidence-interval fields. PR [#4255](https://github.com/ll7/robot_sf_ll7/pull/4255), PR [#4301](https://github.com/ll7/robot_sf_ll7/pull/4301), PR [#4305](https://github.com/ll7/robot_sf_ll7/pull/4305), and PR [#4373](https://github.com/ll7/robot_sf_ll7/pull/4373) landed and closed issue #4242 mechanism/exposure schema instrumentation. This PR updates the #4244 analysis plan to point at issue #4242 instead of the SAC gate issue #4245. | Met after this PR. |

## Closure Boundary

This audit supports closing issue #4244 after this PR merges. The remaining work
is operational, not an acceptance gap for the preregistration slice: private queue
owners still need passing smoke manifests and actual campaign execution before
any benchmark, paper, or dissertation claim can be made.

No full benchmark campaign, Slurm or GPU submission, paper claim, dissertation
claim, release, merge, or deletion was performed by this audit.
