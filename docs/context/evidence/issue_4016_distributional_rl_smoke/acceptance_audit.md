# Issue #4016 Acceptance Audit

This audit maps issue #4016 closure criteria to merged implementation evidence. It is conservative: diagnostic smoke evidence is not treated as benchmark-strength safety evidence.

- Closure status: `complete`.
- Claim boundary: closure audit of merged implementation evidence; diagnostic-only smoke evidence is not benchmark-strength safety evidence.
- No full benchmark campaign, Slurm/GPU submission, or paper/dissertation claim edit was performed.

## Merged PR Evidence

- PR #4157: distributional RL primitives: lattice, quantile critic, risk objectives.
- PR #4215: QR-DQN smoke trainer and runtime adapter.
- PR #4283: trainer/adapter handoff proof.
- PR #4476: diagnostic mean-vs-risk comparison report contract.
- PR #4672: real smoke checkpoint materialized through mean and CVaR manifests.

## Criteria

| Criterion | Status | Evidence | Remaining work |
| --- | --- | --- | --- |
| QR-DQN-style distributional critic trains on a smoke scenario. | `met` | PR #4215 added the QR-DQN smoke trainer path.<br>PR #4672 records output/models/distributional_rl/issue_4016/training_manifest.json as the source for checked-in smoke manifests.<br>summary.json records fallback_or_degraded=False. | None |
| Risk-sensitive selection runs in map_runner/runtime adapter. | `met` | PR #4215 added robot_sf/baselines/distributional_rl.py and robot_sf/benchmark/map_runner_policies/distributional_rl.py.<br>PR #4672 materialized cvar_lower runtime diagnostics from the merged adapter. | None |
| Same checkpoint can be evaluated in mean and cvar_lower selection modes. | `met` | mean checkpoint: output/models/distributional_rl/issue_4016/qr_dqn_issue_4016_smoke.pt<br>cvar checkpoint: output/models/distributional_rl/issue_4016/qr_dqn_issue_4016_smoke.pt<br>matched seed=True, matched total_timesteps=True. | None |
| Mean-value comparator on the same discrete action lattice is available. | `met` | PR #4672 added configs/baselines/distributional_rl_issue_4016_mean.yaml.<br>qr_dqn_mean_manifest.json risk_objective='mean'. | None |
| Matched-seed diagnostic comparison is recorded. | `met` | PR #4476 added scripts/analysis/compare_distributional_rl_issue_4016.py.<br>PR #4672 checked in distributional_rl_risk_comparison.json and .md.<br>comparison_status='valid_diagnostic'. | None |
| Reports include collision, near-miss, min-clearance, success/progress, and path-efficiency tradeoffs. | `met` | Required metric keys present in smoke manifests: True.<br>benchmark_runner_measured=True.<br>Measured benchmark-runner manifests remain diagnostic-only; no paper-facing safety claim is promoted. | None |
| Fallback/degraded rows are explicitly excluded or marked non-evidence. | `met` | summary fallback_or_degraded=False.<br>comparison fallback_degraded_rows={'excluded': 0, 'included_as_non_evidence': 0}. | None |
| Claim boundary is explicit and does not promote paper-facing safety claims. | `met` | summary evidence_tier='diagnostic-only'.<br>comparison benchmark_safety_claim=False. | None |

## Closure Decision

All listed criteria are met; #4016 can close when this audit is accepted.
