# Issue #6934 — BRNE Action-Weight Scaling

Status: **bounded source-aligned fix validated; diagnostic-only planner result**.

Issue: [#6934](https://github.com/ll7/robot_sf_ll7/issues/6934)

## Finding

The pinned upstream Bayesian Recursive Nash Equilibrium (BRNE) core normalizes
each agent's weights by their sample mean. In the staged source, `brne.py` uses `agent_weights /=
np.mean(agent_weights)`, so the effective sample weights have mean `1` and sum
to the effective sample count (`42` for the frozen Robot SF diagnostic, from a
requested `49`). The upstream Robot Operating System (ROS) controller then
computes each robot command with a sample mean in `brne_nav.py`:

```python
np.mean(ulist_essemble[:, :, 0] * weights[0], axis=1)
np.mean(ulist_essemble[:, :, 1] * weights[0], axis=1)
```

Robot SF previously used `np.sum` at this adapter boundary. That multiplied
the source-aligned command by the effective sample count before the unchanged
safety clamp. The bounded fix changes both supported tensor layouts to
`np.mean` and renames mechanism telemetry from `sum_*` to `mean_*`. It does
not change the velocity or angular limits, fallback policy, scenario scope, or
upstream source.

## Validation

Focused source and planner tests passed:

```bash
NUMBA_NUM_THREADS=1 uv run pytest -q \
  tests/baselines/test_brne_source_smoke.py tests/baselines/test_brne_planner.py
```

Result: **47 passed**.

The frozen corridor diagnostic was rerun with the exact three seeds, single
Numba threading, and fallback disabled:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 NUMBA_NUM_THREADS=1 \
uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml \
  --output-dir output/benchmarks/issue_6934_brne_action_scaling_final_20260812T050300Z
```

The report completed with exact pair coverage and no fallback/degraded rows.
All three BRNE rows were native, mechanism-trace-valid, runtime-eligible, and
non-degenerate; corridor violations were `0/3`. Selected linear actions were
`0.04`–`0.40 m/s` on every seed, angular action was `0 rad/s`, and the
aggregation formula was `mean_plan_step_first_over_samples`. Goal-reaching
remained `0/3`, so the correction removes the sample-count scale error but does
not establish that BRNE solves the corridor task.

| Seed | Native/eligible | Goal reached | Non-degenerate | First → last `v` (m/s) | Max `|ω|` (rad/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 111 | yes | no | yes | 0.40 → 0.04 | 0 |
| 112 | yes | no | yes | 0.40 → 0.04 | 0 |
| 113 | yes | no | yes | 0.40 → 0.04 | 0 |

Compact machine-readable evidence is in
[`issue_6934_brne_action_scaling_summary.json`](evidence/issue_6934_brne_action_scaling_summary.json).
The raw episode files and staged GPL source remain ignored, worktree-local
artifacts; the report hashes and provenance are retained in that summary.

## Decision and boundary

Accept the bounded aggregation correction as the completion of #6934. The
result is still smoke/diagnostic evidence only: it supports source-contract
alignment and action-scale diagnosis, not planner ranking, safety, realism,
matched-compute, objective, or paper claims. The remaining `0/3` goal-reaching
result requires the separately scoped [#6938](https://github.com/ll7/robot_sf_ll7/issues/6938)
progress/mechanism experiment before any broader BRNE campaign.
