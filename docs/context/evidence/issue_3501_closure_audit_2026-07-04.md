# Issue #3501 Closure Audit

Plain-language summary: the merged PRs below delivered the planner-agnostic safety wrapper and its
factorial-ablation harness, preregistration, false-stop diagnostic, and paired effect-size report
builder. This audit maps each acceptance criterion to the evidence and states honestly what remains.
**Reconciled 2026-07-06:** the deadlock-recovery stage (a named criterion, previously deferred) is
now implemented, and the two remaining criteria are compute-gated — an actual paired campaign RUN
and its ledger/report emission on executed rows. The earlier "all criteria satisfied / closable"
wording is superseded by this reconciliation.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/3501>
- Merged Implementation PRs:
  - Wrapper Core (3 stateless stages): <https://github.com/ll7/robot_sf_ll7/pull/3591>
  - Runtime Integration: <https://github.com/ll7/robot_sf_ll7/pull/4137>
  - Manifest Builder: <https://github.com/ll7/robot_sf_ll7/pull/4136>
  - Preregistration Config: <https://github.com/ll7/robot_sf_ll7/pull/4223>
  - False-Stop Diagnostic: <https://github.com/ll7/robot_sf_ll7/pull/4298>
  - Paired Effect-Size Report Builder: <https://github.com/ll7/robot_sf_ll7/pull/4598>
  - Deadlock-Recovery Stage: this PR
- Audit dates: 2026-07-04 (initial), 2026-07-06 (reconciliation)

## Claim Boundary

This is a closure-audit integration report only. It does not run a full benchmark campaign, submit
Slurm or GPU compute, or edit paper or dissertation claims. The delivered capability is an opt-in
runtime safety wrapper (four predeclared stages), its dry-run manifest/preregistration checking
harness, and a paired effect-size report builder that consumes completed ablation rows. No paired
ablation has been executed, so no mitigation-effect size exists yet.

## Acceptance Mapping

| Acceptance criterion from #3501 | Delivered evidence | Status |
| --- | --- | --- |
| Composable `SafetyWrapper` implemented around the action interface (clearance/TTC prediction, speed cap near pedestrians, hard stop/yield veto, **deadlock recovery**), off by default and opt-in per run. | PR #3591 added `robot_sf/robot/safety_wrapper.py` with the three stateless stages (`apply_safety_wrapper`). The stateful **deadlock-recovery** stage (`DeadlockRecoveryMonitor`) is added by this PR — opt-in, disabled by default, rotates in place to break a freeze and never adds forward speed (veto preserved). Tested in `tests/test_safety_wrapper.py` and `tests/test_safety_wrapper_deadlock.py`. | Stages implemented (this PR completes deadlock recovery); runtime wiring of the monitor pending. |
| Wrapper thresholds predeclared (planner-agnostic, fixed; no post-hoc per-planner tuning). | Enforced via `SafetyWrapperConfig` in `robot_sf/robot/safety_wrapper.py` (defensive caution radius = 2.0m, capped speed = 0.5 m/s, TTC veto = 1.0s, clearance veto = 0.3m). Checked by manifest validation. | Delivered by PR #3591. |
| Safety-context construction from simulator state and command immediately before step. | PR #4137 added `compute_safety_context_from_env` in `robot_sf/benchmark/safety_wrapper_runtime.py` to retrieve pre-step robot/pedestrian position, heading, and velocity. | Delivered by PR #4137. |
| Opt-in runtime config wrapper binding in benchmark runner. | PR #4137 wired `apply_runtime_safety_wrapper` and `ineligible_safety_wrapper_step_record` into `run_map_episode` under `robot_sf/benchmark/map_runner_episode.py`. | Delivered by PR #4137. |
| Factorial `planner × {wrapper off, wrapper on}` ablation **run** over a fixed scenario set with paired seeds. | Harness exists: PR #4136 `safety_wrapper_ablation_manifest.py` (off/on contrast, paired seeds, planner IDs); `configs/research/safety_wrapper_ablation_v1.yaml`; PR #4223 preregistered the CPU-smoke plan. **No paired ablation has been executed** — the preregistration harness builds *planned* rows only. | **Open — compute-gated** (needs a campaign RUN, out of the audit's scope). |
| Ablation results emitted into the safety-event ledger (#3482) and episode metadata. | Emission path exists: PR #4137 `run_map_episode` attaches validated `safety_wrapper` summary metadata for serialization. Requires *executed* episode rows to emit. | **Open — compute-gated** (depends on the ablation run above). |
| Primary outcomes reported (collision/near-miss probability, min separation, completion, progress-at-timeout, false-stop rate, latency, intervention rate), positive AND negative effects. | Reporting contract exists: PR #4598 `safety_wrapper_factorial_report.py` builds per-pair and per-planner mean on-minus-off effect sizes and fails closed on incomplete pairs / missing metrics; PR #4298 added the false-stop diagnostic. **No real rows to report yet.** | **Open — compute-gated** (report builder ready; awaits executed rows). |
| Preregistration CPU-smoke execution verification. | PR #4223 added `configs/benchmarks/issue_3501_safety_wrapper_factorial_preregistration_cpu_smoke.yaml` and `robot_sf/benchmark/safety_wrapper_factorial_preregistration.py` to build and verify a CPU-smoke preregistration plan. | Delivered by PR #4223. |
| Result kept diagnostic-tier until durable; no paper-facing deployment-safety claim. | The manifest builder tags dry-run outputs as `not_benchmark_evidence`; the report builder tags `benchmark_evidence: False` and a `claim_boundary`. | Enforced by design. |

## Closure Decision

**Not fully closable yet.** After this PR's deadlock-recovery stage, every *implementable* (CPU-only,
no-compute) criterion is delivered: all four wrapper stages, runtime binding, ablation manifest +
checker, preregistration, false-stop diagnostic, and the paired effect-size report builder. Two
criteria remain and both are **compute-gated**, not implementable in this lane:

1. run the paired `planner × {wrapper_off, wrapper_on}` ablation over the fixed scenario set + paired
   seeds (the excluded campaign RUN), emitting executed rows into the #3482-style ledger;
2. build the paired effect-size report on those executed rows (the report builder is ready) and keep
   it diagnostic-tier.

One additional implementable follow-up remains for the ablation to actually exercise the new stage:
**wire `DeadlockRecoveryMonitor` into the benchmark runtime step loop** (the runtime path is
currently stateless). This is a small, CPU-only slice but is scoped out of this closure-audit PR to
keep the blast radius zero for default runs.

Per the maintainer COMPLETE-FIRST rule, an issue whose only remainder is a campaign RUN counts as
complete for the implementation lane; the honest research status, however, is that **no
mitigation-effect size has been measured yet**, so this PR uses `Refs #3501` (not `Closes`) with the
remaining checklist above. An authorized actor may close #3501 once the paired campaign has run and
the diagnostic report is emitted, or may split the campaign RUN into a dedicated gated-run issue.

## Local Verification

Audit-time validation for this docs-only slice:

```bash
./.venv/bin/pytest tests/test_safety_wrapper.py tests/test_safety_wrapper_deadlock.py \
  tests/benchmark/test_safety_wrapper_runtime.py \
  tests/benchmark/test_safety_wrapper_ablation_manifest.py \
  tests/benchmark/test_safety_wrapper_factorial_preregistration.py \
  tests/benchmark/test_safety_wrapper_factorial_report.py
git diff --check
```
