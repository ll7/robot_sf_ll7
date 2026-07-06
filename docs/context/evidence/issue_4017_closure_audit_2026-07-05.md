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

> Superseded by the [2026-07-06 Re-admit Closure Decision](#2026-07-06-re-admit-closure-decision)
> below. The 2026-07-05 reasoning is preserved here as the historical point-in-time record.

This audit does NOT close #4017. The repository now has the complete CPU-side
diagnostic constrained-RL lane requested by the issue implementation plan, and the
only remaining item is a broader empirical campaign that this worker is explicitly
not authorized to run. The maintainer gate updates on issue #4017 (after PR #4259
and PR #4477, and the 2026-07-05 BLOCKED note) keep the issue OPEN and track that
GPU/Slurm benchmark-strength campaign as its remaining Definition of Done. This PR
therefore lands as `Refs #4017` durable evidence, not a closure. Any later
benchmark-strength or paper-facing safety claim should be tracked under that
remaining #4017 campaign DoD with its own provenance and validation boundary.

Forbidden actions confirmed: no Slurm submission, no GPU/full benchmark campaign,
no merge, no release, no deletion, and no paper/dissertation claim edit.

## 2026-07-06 Re-admit Closure Decision

This section supersedes the 2026-07-05 "Closure Call" above. The issue was
re-admitted to the ready queue as a merge-driven closure re-admit: PR #4588 (the
2026-07-05 audit above) merged, yet issue #4017 stayed open with no live tracking
row. Re-running the audit against the merged evidence and the current maintainer
direction, the decision is now to **recommend closure of #4017**.

Rationale for the change:

1. **The issue's own Definition of Done is entirely diagnostic-tier and met.** The
   three DoD checkboxes are "constrained-RL policy trains with an enforced
   safety-constraint budget on a smoke scenario", "comparison vs unconstrained
   baseline showing the constraint's effect (diagnostic tier first)", and "claim
   boundary explicit". The evidence table above maps each to merged PRs #4155,
   #4214, #4259, and #4477. The validation checkbox ("smoke training run;
   repository validation gate") is satisfied by the CPU smoke entrypoint plus the
   diagnostic comparison and readiness-gate scripts, all with fail-closed tests.
2. **The benchmark-strength campaign is out of scope of THIS issue, not a
   remaining DoD item.** The issue body's own Out-of-scope list names "benchmark
   claim before paper-grade". The broader empirical/GPU/Slurm campaign is a
   separate, compute-authorized effort; it is not one of #4017's acceptance
   criteria.
3. **COMPLETE-FIRST maintainer directive (2026-07-05).** Current maintainer
   direction states that an issue whose only remainder is a campaign RUN counts
   as complete for closure purposes. #4017's only remainder is exactly that
   campaign run.

Claim boundary is unchanged: this remains a closure audit only. No
benchmark-strength, paper-grade, or dissertation safety claim is made or promoted
by closing the issue. Any future benchmark-strength or paper-facing safety claim
must be raised as its own compute-authorized issue with independent provenance and
validation, not retro-attributed to this closure.

Residual uncertainty (~75% confidence in the close call): the automated gate
comments on 2026-07-03 and 2026-07-04 and the 2026-07-05 BLOCKED worker note
described the campaign as "remaining" under this issue number. Those are status
notes, not an explicit maintainer instruction to keep the issue open, and they
predate the COMPLETE-FIRST directive. The opening PR therefore declares
`Closes #4017`; the merge gate ratifies or demotes that keyword if a maintainer
disagrees.

Forbidden actions confirmed (re-admit pass): no Slurm submission, no GPU/full
benchmark campaign run, no merge, no release, no deletion, no issue/PR comment
(not authorized this task), and no paper/dissertation claim edit.
