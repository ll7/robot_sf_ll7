# Issue #3214 Closure Audit

Issue: [#3214](https://github.com/ll7/robot_sf_ll7/issues/3214)

Status on 2026-07-05: `blocked_not_closable`.

This audit maps the issue acceptance criteria to merged evidence and explains why the issue should
not be closed yet. It is a closure-audit artifact only: no Slurm scheduler or GPU job was
submitted, no full benchmark campaign was run, no model checkpoint was promoted, and no paper or
dissertation claim was edited.

## Live-Thread Inputs

- Issue body acceptance criteria, read from the live issue on 2026-07-05.
- Issue comments through 2026-07-04 19:38:33 UTC, including the latest blocked decision.
- Merged PRs referenced by the issue thread: #3255, #3515, #3869, and #4467.
- Follow-up issue [#3254](https://github.com/ll7/robot_sf_ll7/issues/3254), closed as a verified
  negative result.

## Criteria To Evidence

| Acceptance criterion | Status | Evidence |
| --- | --- | --- |
| Hard-case-weighted training config and explicit reproducible data-weighting spec. | Met for the local config/tooling slice. | PR [#3255](https://github.com/ll7/robot_sf_ll7/pull/3255) merged `configs/training/predictive/predictive_crossing_conflict_hardcase_mixing_issue_3214.yaml`, `build_predictive_mixed_dataset.py` weighting-spec support, pipeline plumbing, and focused tests. The tracked spec is launch/config/tooling-only and does not claim a retrained checkpoint. |
| At least one retrained checkpoint evaluated against baseline on `predictive_hard_seeds_v1`, with navigation gate separate from average displacement error/final displacement error (ADE/FDE). | Partially met as diagnostic negative evidence, not promotable benchmark evidence. | Follow-up issue #3254 ran Slurm job 13042 and PR [#3515](https://github.com/ll7/robot_sf_ll7/pull/3515) preserved the compact negative-result bundle at `docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23/`. The bundle records best validation ADE 0.04837 and FDE 0.09735, but final closed-loop success 0.08696 below the 0.30 gate. The #3254 closing comment additionally records baseline checks: `predictive_proxy_selected_v2_full` at 0.101 and documented camera-ready `prediction_planner` at 0.069. |
| Decision per stop rule and uncertainty; checkpoint promoted durably through registry or Weights & Biases (W&B), not local-only. | Blocked for checkpoint promotion; decision is recorded as negative/blocked. | PR #3515 records W&B run id `3tu3tmee` and compact evidence, but explicitly keeps large checkpoint/raw output out of git and does not promote a model registry entry. The #3254 closing comment classifies the result as a legitimate negative/limitation result and says not to rerun without a public control-law-side change. |
| Result classified on the evidence ladder; no fallback or degraded promotion. | Met for classification; issue remains open because the classification is negative/blocked, not success evidence. | PR #3515 and `docs/context/issue_3254_predictive_crossing_conflict_negative_result.md` classify the evidence as `negative_training_result_not_benchmark_promotion`, not paper-facing and not benchmark promotion. The tracked readiness packet `configs/training/predictive/predictive_retraining_readiness_issue_3214.yaml` keeps launch state blocked. PR [#3869](https://github.com/ll7/robot_sf_ll7/pull/3869) added the static readiness packet/checker, and PR [#4467](https://github.com/ll7/robot_sf_ll7/pull/4467) added packet-to-pipeline hard-seed summary consistency validation. |

## Remaining Blockers

The latest live issue comment says no non-duplicative local code slice remains. Current tracked
state agrees:

- `configs/training/predictive/predictive_retraining_readiness_issue_3214.yaml` sets the launch
  decision to blocked after the verified negative retraining result.
- Required inputs before rerun are `control_law_change_config`, `updated_public_config_or_pr`,
  `checkpoint_provenance_plan`, and `hard_seed_evaluation_plan`.
- A control-law-side change is not satisfied by the existing #3214 packet; the latest issue comment
  names it as the binding blocker.

## Closure Decision

Do not close #3214 from this audit. The local config/tooling criterion is satisfied and the
evidence/result-classification criteria are documented, but the checkpoint-promotion and rerun
criteria remain blocked on missing public inputs. The smallest correct next action is not another
guard/checker PR; it is a public control-law-side change plus checkpoint and hard-seed evaluation
provenance plan, after which the retraining packet can be revisited.
