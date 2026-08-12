# Issue #6934 — BRNE Action-Weight Contract

This note records the narrow contract repair that prevents mean-normalized BRNE weights from
scaling a native unicycle command by the number of control samples. It is diagnostic evidence
only: it does not promote BRNE to a benchmark arm or establish a planner, safety, realism, or
paper claim.

## Contract decision

BRNE (Bayesian Recursive Nash Equilibrium) at pinned upstream commit
`633a5cdcb39ab27f18b596cb8cb1968644f82391` uses a nonnegative sample-weight row with
`mean(weights) == 1`:

- `weights_update_nb` divides each updated row by `np.mean(weights[i])`.
- `brne_nav` applies the corridor mask and mean-normalizes the robot row again.
- The pinned Python node computes `np.mean(ulist_essemble[:,:,0] * weights[0], axis=1)` and
  `np.mean(ulist_essemble[:,:,1] * weights[0], axis=1)`.
- The pinned C++ node and sampling test use the equivalent `arma::mean(control * weights)`.

The upstream dynamics consume native unicycle controls directly: `v` is linear speed in m/s and
`omega` is yaw rate in rad/s. Therefore the Robot SF adapter must use a weighted mean over the
effective sample axis. A weighted sum is not equivalent: with 42 effective samples and unit-mean
weights it scales the command by roughly 42 before the separate safety clamp.

The adapter now implements `weighted_mean_plan_step_first_over_samples` for the pinned
`(plan_steps, samples, command)` layout and retains the legacy samples-first normalization path.
It fails closed for malformed/non-finite/negative or unnormalized sample weights. The configured
`v_max=2.0` m/s and `omega_max=1.0` rad/s safety limits are unchanged; the repair does not hide
or relax saturation by changing those limits.

## Hand-checkable fixture

For the first plan step, use controls
`[[0.2, -0.2], [0.4, 0.0], [0.6, 0.2]]` and weights `[0.5, 1.0, 1.5]`. The weights have unit
mean, so the accepted upstream contract gives:

```text
v     = (0.2*0.5 + 0.4*1.0 + 0.6*1.5) / 3 = 7/15 m/s
omega = (-0.2*0.5 + 0.0*1.0 + 0.2*1.5) / 3 = 1/15 rad/s
```

The corresponding weighted sum would be `[1.4, 0.2]`, demonstrating the sample-count scale
error. The focused planner tests also cover the plan-step-first adapter path, samples-first
normalization, malformed tensors, non-finite controls, negative weights, and unnormalized rows.

## Frozen intervention

The required same-matrix intervention used the pinned source, fallback disabled, and single-thread
isolation:

```bash
NUMBA_NUM_THREADS=1 LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml \
  --output-dir output/benchmarks/issue_6934_brne_action_scale_20260812T073000Z
```

The exact final output directory was
`output/benchmarks/issue_6934_brne_action_scale_20260812T073000Z`. Its compact report hashes
were JSON `645d494b20deac358521abd4cb08057cca7e9f4389136118a9f4b0e28c6eb2df` and Markdown
`dd0ce23e45018633fef9dc2896b93a61954ca488b84e621cfe0af64838f59a82`.

The run was exact and native for BRNE (`3/3` pairs, `3/3` runtime-eligible, `3/3`
non-degenerate, effective samples `42`, fallback/degraded `0`, corridor violations `0`). Every
BRNE trace recorded `weighted_mean_plan_step_first_over_samples`; no step exceeded the 100 ms
budget. BRNE reached the goal in `0/3` rows, collided in `0/3`, and timed out with the per-row
deadlock diagnostic active in `3/3` rows. The observed final action values were `v=0.4` on the
first step and `v≈0.04`, `omega=0` thereafter in the recorded seed-111 trace; all applied values
were within the existing safety limits. ORCA and Social Force remained paired diagnostic
comparators (`3/3` and `2/3` eligible goal-reaching respectively), not a ranking claim.

The intervention supports the narrow conclusion that the former weighted-sum construction was a
real action-scale contract defect, with approximately 99% confidence from direct pinned-source
agreement and the hand fixture. It does not establish that correcting the defect is sufficient for
goal-reaching; the three-seed result remains diagnostic-only, with approximately 90% confidence in
that negative conclusion until a separately authorized progress/control hypothesis is tested.

Raw episode JSONL and the staged GPL source remain ignored and worktree-local. The compact report
hashes and exact output directory belong in the issue/PR handoff for the final exact-head run.
