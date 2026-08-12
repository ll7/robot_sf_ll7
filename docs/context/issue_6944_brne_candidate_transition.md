# Issue #6944 — BRNE Candidate Transition

Status: **controller-parity mismatch observed; diagnostic-only**.

Bayesian Recursive Nash Equilibrium (BRNE) emits a low-speed command after the
first planner-observation transition in the frozen corridor diagnostic. This
slice adds bounded candidate-control, weight, upstream-activation,
nominal-command, and applied-environment-action summaries so the transition can
be separated from candidate construction, pedestrian selection, mean
aggregation, and the safety clamp. It does not establish planner quality,
ranking, safety, realism, matched-compute parity, or paper evidence.

## Frozen diagnostic

- Scenario: `classic_head_on_corridor_low` from
  `configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml`.
- Seeds: `111`, `112`, and `113`; horizon `500`; timestep `0.1` seconds.
- Native BRNE only for the mechanism interpretation; fallback is disabled.
- Upstream source: `MurpheyLab/brne` at
  `633a5cdcb39ab27f18b596cb8cb1968644f82391`, GPL-3.0, staged locally and not
  vendored or redistributed.
- Integration base after refresh: `a25045b4ae02802c7ff58747924c898fbc841d0f`.
- Implementation commit: `8f9438632e794f084db72bb016a14b539bbca648`.

Reproduce with:

```bash
NUMBA_NUM_THREADS=1 LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
  uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml \
  --output-dir output/benchmarks/issue_6944_candidate_trace_<timestamp>
```

The run completed with exact `3/3` pair coverage, `3/3` native and eligible
BRNE rows, no fallback/degraded rows, no corridor violations, and zero clipped
steps. The tracked compact handoff is
[issue_6944_brne_candidate_transition_summary.json](evidence/issue_6944_brne_candidate_transition_summary.json).

## Observed transition

The adapter records `brne-candidate-distribution.v1` summaries for finite
candidate controls and mean-normalized robot weights. The first-to-second
planner-observation transition is identical across the three seeds:

| Seed | Candidate mean `v` (m/s) | Weighted mean `v` (m/s) | Weight mean | Weight std | Clipped steps | Goal |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 111 | 0.400 → 0.040 | 0.400 → 0.040 | 1.0 → 1.0 | 0.0 → 0.0 | 0 | no |
| 112 | 0.400 → 0.040 | 0.400 → 0.040 | 1.0 → 1.0 | 0.0 → 0.0 | 0 | no |
| 113 | 0.400 → 0.040 | 0.400 → 0.040 | 1.0 → 1.0 | 0.0 → 0.0 | 0 | no |

The candidate distribution itself shifts: its velocity range changes from
`0.20–0.60 m/s` to `0.00–0.08 m/s`, while the robot weights remain uniform.
The weighted command therefore follows the candidate shift rather than being
suppressed by weight normalization or aggregation. This is a falsifiable null
result for the weighting/normalization explanation in this frozen slice, not a
causal explanation of the upstream behavior.

## Controller-parity fields

The expanded trace now exposes the candidate-generation inputs and the command
boundary for each native seed:

| Seed | Nominal command | Observed / within `<3.5 m` / passed | Pre-clamp → selected | Applied environment action | Clipped |
| ---: | --- | --- | --- | --- | ---: |
| 111 | `0.40/0.00 → 0.04/0.00`, `straight_constant` | `2/0/2` | `0.40/0.00 → 0.04/0.00` | `0.40/0.00 → ~0.00/0.00` | 0 |
| 112 | `0.40/0.00 → 0.04/0.00`, `straight_constant` | `2/0/2` | `0.40/0.00 → 0.04/0.00` | `0.40/0.00 → ~0.00/0.00` | 0 |
| 113 | `0.40/0.00 → 0.04/0.00`, `straight_constant` | `2/0/2` | `0.40/0.00 → 0.04/0.00` | `0.40/0.00 → ~0.00/0.00` | 0 |

This confirms a bounded upstream-controller-parity mismatch in the diagnostic:
the adapter uses a constant straight nominal command and passes nearest agents
without applying the upstream `<3.5 m` activation gate. The trace does not
change either behavior, and it does not establish whether aligning either one
would improve progress.

All three rows retain positive signed goal-distance progress in early, middle,
and late phases, but none reaches the goal. The remaining explanations include
upstream candidate generation, planner state, and control-horizon behavior. No
intervention was run here, and no separate implementation issue is created by
this diagnostic slice.

## Decision and boundary

Keep the result diagnostic-only and do not change the source pin, action limits,
fallback/degraded policy, benchmark or ranking scope, safety/realism
interpretation, or paper-facing surfaces. Raw episode JSONL and the staged GPL
source remain ignored, worktree-local artifacts; the tracked summary is a
compact provenance handoff rather than a raw episode archive.

If further work is authorized, test one upstream-controller-parity,
planner-state, or control-horizon intervention at a time on the same frozen
matrix. The parity mismatch is now a confirmed diagnostic lead, not a validated
fix or benchmark result.

## Validation

- `uv run pytest -q tests/baselines/test_brne_planner.py tests/benchmark/test_issue_6464_brne_corridor_diagnostic.py` — `61 passed`
- `NUMBA_NUM_THREADS=1 uv run pytest -q tests/baselines/test_brne_source_smoke.py` — `6 passed`
- `uv run pytest -q tests/benchmark/test_map_runner_utils.py -k run_map_episode_records_synthetic_actuation_metrics` — `1 passed`
- focused Ruff check and `git diff --check` — passed

## Report integrity

The exact local report directory is
`output/benchmarks/issue_6944_controller_parity_20260812T101747Z/`.

- `diagnostic_report.json` SHA-256:
  `4438f5df1e6d3fe8eeffa85fa433c24388b4f862accc4a8b77c5adec5e8f3e0a`
- `diagnostic_report.md` SHA-256:
  `8abc23a4e0449168c481d10aae2501ebc586f9b8eba9b336cc21fad0349ffbb8`
