# Issue #3300 Closure Audit

This audit maps issue #3300 acceptance criteria to merged evidence and records the closure
boundary after PR #4601 produced the first observed false-positive actor-injection replay result.

## Claim Boundary

- Evidence status: CPU-local smoke/diagnostic evidence.
- Replay mode: executable planner/environment replay, not trace-derived diagnostics.
- Scope: false-positive actor injection in observation-quality replay, reported separately from other
  observation-noise effects.
- Closure call: original #3300 acceptance criteria are met by merged PRs. The broader full replay
  campaign remains outside this CPU-only closure-audit slice and is not promoted here.
- Out of scope: full benchmark campaign, Slurm/GPU submission, hardware-calibrated sensor realism,
  robustness promotion, paper/dissertation claim edits.

## Acceptance Mapping

| Criterion from #3300 | Status | Evidence |
| --- | --- | --- |
| Reproducible command runs planner/environment false-positive actor injection, or fails closed with an actionable blocker. | Met | PR #4390 added the first executable replay evidence path and classified incompatible execution as `blocked_unavailable`; PR #4413 added fail-closed view readiness checks; PR #4431 and PR #4486 ran structured-pedestrian paired smoke replays; PR #4601 ran ORCA nominal/perturbed executable replay commands. |
| Report distinguishes live executable replay evidence from trace-derived diagnostics. | Met | PR #4390 introduced executable replay reporting; PR #4486 and PR #4601 evidence bundles record `replay_mode: executable` and caveat CPU-local smoke/diagnostic scope. |
| Report records false-positive safety effects separately from false-negative/noise effects. | Met | PR #4431 and PR #4601 use issue-specific false-positive actor-injection profiles and reports; `docs/context/evidence/issue_3300_orca_reactive_false_positive_2026-07-05/summary.json` records only false-positive actor-injection deltas. |
| Report includes scenario, seed, planner mode, perturbation family, execution mode, fallback/degraded status, and caveats. | Met | PR #4439 pre-registered the stronger matrix and readiness checker; PR #4486 records three scenarios, seeds `0` and `3300`, planner `goal`, executable mode, no fallback/degraded rows, and `scenario_too_weak`; PR #4601 records planner `orca`, six paired rows, profile hash `1c77dd478a6b`, no unmatched rows, and CPU-smoke caveats. |
| If actor injection remains unavailable, record exact missing prerequisite and next smallest proof step. | Superseded by met replay | PR #4390 and PR #4413 recorded the blocker when structured pedestrian observations were unavailable. PR #4431 removed that blocker by injecting structured pedestrian slots, and PR #4601 observed a reactive planner response. |
| Produce a bounded result for the missing #2927 false-positive acceptance dimension: observed, diagnostic-only, blocked, or scenario-too-weak. | Met | PR #4486 produced `scenario_too_weak` on the stronger goal-planner matrix. PR #4601 produced `observed` on ORCA: 101 injected pedestrians across six paired rows changed predeclared replay outcomes including speed, curvature, jerk, clearance, and near-miss metrics. |
| Keep hardware sensor realism and paper-facing claims out of scope. | Met | All merged evidence bundles caveat CPU smoke/diagnostic status. This closure audit does not add claim-map, benchmark-report, paper, dissertation, or hardware-calibration edits. |

## Evidence Trail

| PR | Merged | Contribution | Closure relevance |
| --- | --- | --- | --- |
| PR 4390 | 2026-07-04 | Added CPU-local executable replay evidence bundle and classified no-injection execution as `blocked_unavailable`. | Proved the replay/report path and preserved a fail-closed blocker instead of treating no injection as success. |
| PR 4413 | 2026-07-04 | Added readiness checker for incompatible or underspecified observation views. | Made requested actor injection fail closed when planner observations cannot carry structured pedestrians. |
| PR 4431 | 2026-07-04 | Added structured SocNav pedestrian-slot false-positive injection support, paired smoke configs, and evidence. | Removed the structured-observation blocker; result remained `scenario_too_weak` for the goal planner. |
| PR 4439 | 2026-07-04 | Pre-registered a stronger structured false-positive matrix and readiness checker. | Bounded the next empirical replay matrix before running it. |
| PR 4486 | 2026-07-04 | Added stronger-matrix closure-audit evidence bundle. | Demonstrated executable injection across three scenarios and two seeds, classified `scenario_too_weak`. |
| PR 4601 | 2026-07-06 | Added ORCA reactive false-positive replay evidence, metric-name correction, configs, tests, and evidence bundle. | Supplies the missing `observed` replay: injected false-positive pedestrians changed predeclared outcomes. |

## Closure Decision

Issue #3300 can close on the merged PR set above. The remaining "full replay campaign" noted after
PR #4601 is intentionally outside this closure audit's CPU-only authorization and is not required to
close the smoke/diagnostic acceptance dimension. Any future full campaign should be tracked as a
separate claim-promotion task with predeclared campaign scope, artifact provenance, and benchmark
claim boundaries.

## Artifacts Consulted

- `docs/context/evidence/issue_3300_false_positive_actor_injection_2026-07-04/`
- `docs/context/evidence/issue_3300_stronger_matrix_closure_audit_2026-07-04/`
- `docs/context/evidence/issue_3300_orca_reactive_false_positive_2026-07-05/`
- `configs/benchmarks/issue_3300_orca_fp_nominal_smoke.yaml`
- `configs/benchmarks/issue_3300_orca_fp_perturbed_smoke.yaml`
- `configs/benchmarks/observation_noise/issue_3300_false_positive_orca_close_v1.yaml`
- `robot_sf/benchmark/false_positive_replay_report.py`
- `tests/benchmark/test_false_positive_actor_injection_replay.py`
