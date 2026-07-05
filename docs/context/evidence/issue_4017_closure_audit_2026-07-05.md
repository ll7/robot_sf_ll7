# Issue #4017 Closure Audit

Plain-language summary: issue #4017 now has the CPU-side constrained reinforcement
learning lane implemented and checked; the only named remainder is the broader
empirical benchmark campaign, which is outside this worker's authorized scope.

Claim boundary: closure audit only. This file maps the issue acceptance criteria
to merged pull requests and live issue comments. It does not claim benchmark-strength,
paper-grade, or dissertation safety evidence.

Live issue reviewed: <https://github.com/ll7/robot_sf_ll7/issues/4017>

Audit timestamp: 2026-07-05

## Evidence Table

| Criterion | Evidence | Status |
| --- | --- | --- |
| Constrained PPO training runs through a smoke scenario without crashing | PR #4214 added `scripts/training/train_constrained_rl.py`, paired smoke configs, and dry-run/config tests for the constrained training entrypoint. | Met for CPU smoke path |
| Constraint costs and multipliers are logged | PR #4155 added safety-cost extraction, `LagrangeMultiplierState`, `ConstrainedRewardWrapper`, and diagnostics including constraint costs, multipliers, raw task reward, constrained reward, and terminal episode summaries. PR #4214 integrated those diagnostics into the training path and manifest/trace outputs. | Met |
| At least one constraint budget violation causes multiplier update | PR #4259 added the diagnostic comparison/report path and tests requiring budget-violation constraints and multiplier-changed constraints in the constrained trace. PR #4477 added a fail-closed readiness checker that blocks if no positive budget violation or no Lagrange multiplier update is observed. | Met for diagnostic readiness contract |
| Unconstrained and constrained policies are evaluated on the same scenario set and paired seeds | PR #4214 added paired constrained/unconstrained smoke configs. PR #4259 added `configs/benchmarks/issue_4017_constrained_rl_diagnostic.yaml` and comparison logic that enforces matched seed and matched timestep checks. | Met for diagnostic comparison contract |
| Comparison report is diagnostic-only and includes caveats | PR #4259 added `docs/context/evidence/issue_4017_constrained_rl/README.md` and `scripts/analysis/compare_constrained_rl_issue_4017.py` with `evidence_tier: diagnostic-only`, claim-boundary text, fallback/degraded handling, and non-evidence blockers. | Met |
| Existing metric semantics are not redefined | PRs #4155, #4214, #4259, and #4477 add training, wrapper, report, and readiness surfaces without changing benchmark metric definitions. | Met by diff inspection |
| Safety-wrapper and uncertainty-buffer functionality are not mixed into the training claim | The merged PR sequence stays scoped to constrained-RL reward wrapping, training, diagnostic comparison, and readiness checking. Issue comments explicitly keep #3501 safety-wrapper work and #3974 uncertainty-buffer work out of scope. | Met |
| Broader empirical campaign or benchmark-strength safety claim | Live issue comments state this remains open and requires GPU training plus benchmark evaluation under Slurm. This worker is not authorized to submit compute, and the ready-queue directive says an issue whose only remainder is a campaign run counts as complete for this CPU-only closure task. | Out of scope here |

## Merged Pull Requests

- PR #4155: <https://github.com/ll7/robot_sf_ll7/pull/4155>
- PR #4214: <https://github.com/ll7/robot_sf_ll7/pull/4214>
- PR #4259: <https://github.com/ll7/robot_sf_ll7/pull/4259>
- PR #4477: <https://github.com/ll7/robot_sf_ll7/pull/4477>

## Closure Call

Close #4017 through the closure-audit pull request because the repository now has
the complete CPU-side diagnostic constrained-RL lane requested by the issue
implementation plan, and the only remaining item is a broader empirical campaign
that this worker is explicitly not authorized to run. Any later benchmark-strength
or paper-facing safety claim should be tracked as a separate compute-authorized
campaign with its own provenance and validation boundary.

Forbidden actions confirmed: no Slurm submission, no GPU/full benchmark campaign,
no merge, no release, no deletion, and no paper/dissertation claim edit.
