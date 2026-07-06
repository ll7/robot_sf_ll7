# Issue #3501 Closure Audit

Plain-language summary: issue #3501 is not ready to close. Merged pull requests have delivered
the planner-agnostic safety wrapper, benchmark runtime wiring, preregistration/checker support,
false-stop diagnostic, deadlock-recovery runtime integration, and paired effect-size report
builder. The remaining acceptance criteria require the actual paired ablation run and report on
executed rows, which are compute-gated and outside the CPU-only closure-audit lane.

**Reconciled 2026-07-06 after PR #4646:** the prior CPU-only follow-up to wire
`DeadlockRecoveryMonitor` into `map_runner_episode` is complete. The only remaining issue-level
criteria are the paired campaign run and effect-size report on durable executed rows.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/3501>
- Wrapper core: <https://github.com/ll7/robot_sf_ll7/pull/3591>
- Runtime integration: <https://github.com/ll7/robot_sf_ll7/pull/4137>
- Manifest/checker hardening: <https://github.com/ll7/robot_sf_ll7/pull/4108>,
  <https://github.com/ll7/robot_sf_ll7/pull/4112>,
  <https://github.com/ll7/robot_sf_ll7/pull/4116>,
  <https://github.com/ll7/robot_sf_ll7/pull/4128>,
  <https://github.com/ll7/robot_sf_ll7/pull/4136>
- Preregistration support: <https://github.com/ll7/robot_sf_ll7/pull/4223>
- False-stop diagnostic: <https://github.com/ll7/robot_sf_ll7/pull/4298>
- Paired report builder: <https://github.com/ll7/robot_sf_ll7/pull/4598>
- Deadlock-recovery stage: <https://github.com/ll7/robot_sf_ll7/pull/4636>
- Deadlock-recovery runtime wiring: <https://github.com/ll7/robot_sf_ll7/pull/4646>

## Claim Boundary

This is a closure-audit integration report. It does not run a benchmark campaign, submit Slurm or
GPU compute, edit paper/dissertation claims, or promote a mitigation-effect claim. The delivered
capability is opt-in diagnostic infrastructure: fixed-threshold safety wrapper stages, benchmark
runtime binding, planned-row checks, and a paired report builder that can consume executed rows once
the campaign has run.

## Acceptance Mapping

| Acceptance criterion from #3501 | Evidence | Status |
| --- | --- | --- |
| Composable `SafetyWrapper` around the action interface, off by default and opt-in per run. | PR #3591 added the pure planner-agnostic wrapper in `robot_sf/robot/safety_wrapper.py`; PR #4636 added `DeadlockRecoveryMonitor` as the fourth predeclared stage. | Delivered. |
| Wrapper includes clearance/TTC monitoring, speed cap near pedestrians, hard stop/yield veto, and deadlock recovery. | PR #3591 delivered clearance/TTC, speed-cap, and hard-stop/yield stages; PR #4636 delivered stateful deadlock recovery; PR #4646 wires that monitor into the benchmark episode loop. | Delivered. |
| Thresholds are predeclared, fixed, and planner-agnostic, with no post-hoc per-planner tuning. | `SafetyWrapperConfig`, `DeadlockRecoveryConfig`, and `SafetyWrapperRuntimeConfig` enforce default thresholds; PRs #4128, #4136, #4137, and #4646 add fail-closed validation for threshold/config drift. | Delivered. |
| Safety context is constructed from simulator state and requested command before execution. | PR #4137 added `compute_safety_context_from_env(...)` and runtime parser hardening in `robot_sf/benchmark/safety_wrapper_runtime.py`. | Delivered. |
| Runtime wrapper binding sits on the benchmark action path. | PR #4137 wired `apply_runtime_safety_wrapper(...)` into `run_map_episode`; PR #4646 added per-episode `DeadlockRecoveryMonitor` construction and step-loop integration. | Delivered. |
| Factorial `planner x {wrapper_off, wrapper_on}` ablation over the fixed scenario set and paired seeds is run. | PR #4136 and `configs/research/safety_wrapper_ablation_v1.yaml` define the planned paired design; PR #4223 preregisters the CPU-smoke plan. No executed paired campaign rows are present in the issue thread. | Open, compute-gated. |
| Ablation results emit into the safety-event ledger / episode metadata path (#3482). | PR #4137 attaches safety-wrapper episode summaries through benchmark metadata/ledger validation for executed episodes. This needs the actual paired run to produce rows. | Open, depends on compute-gated run. |
| Primary outcomes are reported: exact-collision probability, near-miss probability, minimum predicted separation, completion probability, progress at timeout, false-positive stop rate, stop/yield latency, and intervention rate, including positive and negative effects. | PR #4598 added `safety_wrapper_factorial_report.py`, a paired effect-size report builder that fails closed on incomplete pairs or missing primary metrics; PR #4298 added false-stop diagnostic support. No real paired rows have been reported yet. | Open, compute-gated. |
| Results remain diagnostic until durable; no paper-facing deployment-safety claim is made. | Manifest/report paths label dry-run or builder output as not benchmark evidence unless backed by executed rows; this audit makes no mitigation-effect claim. | Delivered. |

## Closure Decision

Do not close #3501 yet. The CPU-only implementation lane is complete after PR #4646, but the issue's
research acceptance claim still requires:

1. Run the paired `planner x {wrapper_off, wrapper_on}` ablation over the fixed scenario set and
   paired seeds, emitting executed rows through the safety-event ledger path.
2. Run the paired effect-size report builder on those executed rows and keep the interpretation
   diagnostic-tier unless durable evidence supports a stronger claim.

This audit should be referenced with `Refs #3501`, not with a closing keyword.

## Local Verification

Audit-time validation for this docs-only reconciliation:

```bash
git diff --check
rg -n "4646|DeadlockRecoveryMonitor|compute-gated|closing keyword" docs/context/evidence/issue_3501_closure_audit_2026-07-04.md
```
