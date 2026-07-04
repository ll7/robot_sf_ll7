# Issue #3501 Closure Audit

Plain-language summary: merged PRs #3591, #4137, #4136, #4223, and #4298 delivered the planner-agnostic safety wrapper and factorial-ablation execution/preregistration harness requested by issue #3501. This audit maps each acceptance criterion to the merged evidence so the issue can be closed by an authorized follow-up.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/3501>
- Merged Implementation PRs:
  - Wrapper Core: <https://github.com/ll7/robot_sf_ll7/pull/3591>
  - Runtime Integration: <https://github.com/ll7/robot_sf_ll7/pull/4137>
  - Manifest Builder: <https://github.com/ll7/robot_sf_ll7/pull/4136>
  - Preregistration Config: <https://github.com/ll7/robot_sf_ll7/pull/4223>
  - False-Stop Diagnostic: <https://github.com/ll7/robot_sf_ll7/pull/4298>
- Audit date: 2026-07-04

## Claim Boundary

This is a closure-audit integration report only. It does not run a full benchmark campaign, submit Slurm or GPU compute, or edit paper or dissertation claims. The delivered capability is an opt-in runtime safety wrapper and its dry-run manifest/preregistration checking harness.

## Acceptance Mapping

| Acceptance criterion from #3501 | Delivered evidence | Status |
| --- | --- | --- |
| Composable `SafetyWrapper` implemented around the action interface (clearance/TTC prediction, speed cap near pedestrians, hard stop/yield veto), off by default and opt-in per run. | PR #3591 added `robot_sf/robot/safety_wrapper.py` implementing `apply_safety_wrapper` with hard-stop, speed cap, and pass-through stages. Tested in `tests/test_safety_wrapper.py`. | Delivered by PR #3591. |
| Wrapper thresholds predeclared (planner-agnostic, fixed; no post-hoc per-planner tuning). | Enforced via `SafetyWrapperConfig` in `robot_sf/robot/safety_wrapper.py` (defensive caution radius = 2.0m, capped speed = 0.5 m/s, TTC veto = 1.0s, clearance veto = 0.3m). Checked by manifest validation. | Delivered by PR #3591. |
| Safety-context construction from simulator state and command immediately before step. | PR #4137 added `compute_safety_context_from_env` in `robot_sf/benchmark/safety_wrapper_runtime.py` to retrieve pre-step robot/pedestrian position, heading, and velocity. | Delivered by PR #4137. |
| Opt-in runtime config wrapper binding in benchmark runner. | PR #4137 wired `apply_runtime_safety_wrapper` and `ineligible_safety_wrapper_step_record` into `run_map_episode` under `robot_sf/benchmark/map_runner_episode.py`. | Delivered by PR #4137. |
| Factorial `planner × {wrapper off, wrapper on}` ablation manifest and validation. | PR #4136 added `robot_sf/benchmark/safety_wrapper_ablation_manifest.py` verifying off/on contrast, paired seeds, and planner IDs. Configured in `configs/research/safety_wrapper_ablation_v1.yaml`. | Delivered by PR #4136. |
| Ablation results emitted into the safety-event ledger (#3482) and episode metadata. | PR #4137 updated `run_map_episode` to construct and attach `safety_wrapper` summary metadata, which is validated and serialized into the ledger via `build_event_ledger(record)`. | Delivered by PR #4137. |
| Primary outcomes reported (probabilities, min separation, false stop, latency, etc.). | PR #4137 and PR #4298 added `analyze_false_stop_diagnostic` to categorize hard-stops using a forward-window clearance persistence check, and summarized clearances, TTCs, and intervention rates. | Delivered by PRs #4137, #4298. |
| Preregistration CPU-smoke execution verification. | PR #4223 added `configs/benchmarks/issue_3501_safety_wrapper_factorial_preregistration_cpu_smoke.yaml` and `robot_sf/benchmark/safety_wrapper_factorial_preregistration.py` to build and verify a CPU-smoke preregistration plan. | Delivered by PR #4223. |
| Result kept diagnostic-tier until durable; no paper-facing deployment-safety claim. | The manifest builder explicitly tags dry-run outputs as `not_benchmark_evidence` and defines the `claim_boundary` to prevent premature paper or dissertation claims. | Enforced by design. |

## Closure Decision

All acceptance criteria in the live issue thread are satisfied by merged PRs. Issue #3501 is closable once an authorized actor can post this criterion-to-evidence closure comment and close the issue.

## Local Verification

Audit-time validation for this docs-only slice:

```bash
./.venv/bin/pytest tests/test_safety_wrapper.py tests/benchmark/test_safety_wrapper_runtime.py tests/benchmark/test_safety_wrapper_ablation_manifest.py tests/benchmark/test_safety_wrapper_factorial_preregistration.py
git diff --check
```
