# Issue #4448 Closure Integration Audit

Plain-language summary: merged PRs now cover the public code path needed for issue
[#4448](https://github.com/ll7/robot_sf_ll7/issues/4448): the guarded PPO arm remains
rostered, is marked accepted-unavailable when its checkpoint observation contract is missing,
and the camera-ready campaign runner records dependency-gated planners as `not_available`
instead of attempting them. The remaining open item is empirical h600 campaign rerun/resume
validation, which requires compute authorization and is not benchmark evidence until it runs.

## Scope

- Issue: [#4448](https://github.com/ll7/robot_sf_ll7/issues/4448)
- Implementation PR: [#4449](https://github.com/ll7/robot_sf_ll7/pull/4449)
- Prior audit PR: [#4504](https://github.com/ll7/robot_sf_ll7/pull/4504)
- Campaign-path follow-up PR: [#4550](https://github.com/ll7/robot_sf_ll7/pull/4550)
- Audit refresh date: 2026-07-05
- Claim boundary: public config, preflight, and CPU-only campaign skip behavior. This audit did
  not run the h600 benchmark campaign, submit Slurm or GPU work, change paper claims, close the
  guarded PPO checkpoint promotion dependency, or claim empirical F-C4(ii) evidence.

## Acceptance Evidence

| Criterion from #4448 | Evidence | Status |
| --- | --- | --- |
| `guarded_ppo` stays rostered and documented. | PR #4449 keeps `guarded_ppo` in `configs/benchmarks/paper_experiment_matrix_v1_h600_trace_capable_rerun.yaml` with `availability_gate: dependency_gated` and `fail_closed_reason: guarded_ppo_checkpoint_observation_contract_missing`. The issue #4206/#4448 checker reports a 12-arm roster and includes `guarded_ppo` in the runnable config planner keys. | Met for public config and preflight scope. |
| Missing checkpoint observation contract is accepted-unavailable, not an unexpected preflight failure. | PR #4449 preserves `availability_gate` and `fail_closed_reason` on camera-ready planner specs and emits preflight planner rows with `status: not_available` for dependency-gated planners. `tests/validation/test_issue_4206_trace_capable_h600_rerun_preregistration.py::test_runnable_h600_preflight_reports_guarded_ppo_accepted_unavailable` covers this path. | Met for public preflight scope. |
| Campaign success and evidence accounting exclude unavailable arms instead of silently dropping them. | PR #4550 updates `robot_sf/benchmark/camera_ready/campaign.py` so `availability_gate: dependency_gated` planners synthesize a fail-closed `not_available` summary without calling `run_batch`. `tests/benchmark/test_camera_ready_campaign.py` covers accepted-unavailable campaign rows and the `accepted_unavailable_only` execution status. | Met for CPU-only campaign skip/accounting behavior; empirical h600 output still pending. |
| Public preflight validator is green. | PR #4504 recorded local validation: `scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/validation/test_issue_4206_trace_capable_h600_rerun_preregistration.py -q` passed with 19 tests, and the checker passed with a 12-arm roster, 11 required-available planners, and `guarded_ppo` accepted-unavailable. | Met for public validator. |
| Empirical rerun/resume preserves completed episodes and confirms available-arm campaign completion. | The issue thread comment on 2026-07-04 and the later #4525/#4550 path fix show this still needs a compute-authorized h600 resume/rerun and downstream cross-cut consumption check. This audit has `compute_submit: false` authorization and does not mutate private queue state. | Residual, intentionally out of scope for this PR. |
| No silent planner drop, no claim change, and no dissertation or paper edit. | PRs #4449, #4504, and #4550 keep `guarded_ppo` visible as accepted-unavailable and add only config, validation, campaign fail-closed behavior, tests, and evidence documentation. This note does not encode target-host or packet-lineage queue state. | Met for this PR. |

## Integration Decision

Issue #4448 should remain open only if maintainers want the empirical h600 rerun/resume and
downstream issue #4206 cross-cut proof tracked in this same issue. The smallest remaining public
repository slice is now documentation/state integration, because the code path that blocked the
campaign runner was delivered by #4550 after the original closure audit.

Next empirical action when compute is authorized:

1. Resume or rerun the h600 trace-capable campaign over required-available arms.
2. Confirm `guarded_ppo` appears as accepted-unavailable with
   `guarded_ppo_checkpoint_observation_contract_missing`, not as an unexpected failure.
3. Retrieve the output on the analysis host and rerun the issue #4206 cross-cut builder.
4. Treat resulting F-C4(ii) conclusions as evidence only over available arms, with the gated arm
   recorded as an exclusion.

## Local Validation From Prior Merged Audit

```text
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/validation/test_issue_4206_trace_capable_h600_rerun_preregistration.py -q
PASS: 19 tests passed.

scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/check_issue_4206_trace_capable_h600_rerun_preregistration.py \
  --config configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml \
  --run-config configs/benchmarks/paper_experiment_matrix_v1_h600_trace_capable_rerun.yaml \
  --manifest-out output/issue_4448_closure_audit/preflight_manifest.json
PASS: trace-capable h600 re-run pre-registration valid
PASS: runnable trace-capable h600 config valid
```
