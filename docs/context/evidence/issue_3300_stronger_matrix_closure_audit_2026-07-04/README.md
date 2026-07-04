# Issue #3300 Stronger Matrix Closure Audit

This bundle records a closure audit for issue #3300 after the merged false-positive
actor-injection replay slices. It promotes the smallest missing evidence item from the
2026-07-04 issue thread: a paired nominal-vs-perturbed executable replay report over the
pre-registered stronger structured-pedestrian matrix.

## Claim Boundary

- Evidence status: smoke/diagnostic evidence only.
- Replay mode: executable camera-ready benchmark smoke, not trace-derived diagnostics.
- Scope: three pedestrian-bearing scenarios, two fixed seeds, one CPU-local `goal` planner
  config using `socnav_state`.
- Result classification: `scenario_too_weak`.
- Out of scope: full benchmark campaign, Slurm/GPU submission, hardware sensor-realism model,
  paper/dissertation claim edits, or robustness promotion.

## Inputs

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/3300>
- Branch head used for this audit: `9b34cf32f`
- Scenario matrix: `configs/scenarios/sets/issue_3300_false_positive_stronger_structured_matrix.yaml`
- Scenarios: `single_ped_crossing_orthogonal`, `issue_3233_near_field_observation_noise`,
  `francis2023_intersection_wait`
- Planner: `goal`
- Observation mode: `socnav_state`
- Seeds: `0`, `3300`
- Nominal config: `configs/benchmarks/issue_3300_false_positive_stronger_nominal_smoke.yaml`
- Perturbed config: `configs/benchmarks/issue_3300_false_positive_stronger_perturbed_smoke.yaml`
- Perturbation profile:
  `configs/benchmarks/observation_noise/issue_3300_false_positive_actor_injection_v1.yaml`

## Commands

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/benchmark/check_false_positive_matrix_readiness_issue_3300.py \
  --nominal-config configs/benchmarks/issue_3300_false_positive_stronger_nominal_smoke.yaml \
  --perturbed-config configs/benchmarks/issue_3300_false_positive_stronger_perturbed_smoke.yaml \
  --min-scenarios 3 \
  --min-seeds 2

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3300_false_positive_stronger_nominal_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_3300_stronger_matrix_closure \
  --campaign-id issue_3300_stronger_nominal_closure_audit \
  --skip-publication-bundle

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3300_false_positive_stronger_perturbed_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_3300_stronger_matrix_closure \
  --campaign-id issue_3300_stronger_perturbed_closure_audit \
  --skip-publication-bundle

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/benchmark/build_false_positive_replay_report_issue_3300.py \
  --nominal-jsonl output/benchmarks/issue_3300_stronger_matrix_closure/issue_3300_stronger_nominal_closure_audit/runs/goal__differential_drive/episodes.jsonl \
  --perturbed-jsonl output/benchmarks/issue_3300_stronger_matrix_closure/issue_3300_stronger_perturbed_closure_audit/runs/goal__differential_drive/episodes.jsonl \
  --output-json docs/context/evidence/issue_3300_stronger_matrix_closure_audit_2026-07-04/summary.json \
  --output-csv docs/context/evidence/issue_3300_stronger_matrix_closure_audit_2026-07-04/robustness_delta.csv \
  --output-md docs/context/evidence/issue_3300_stronger_matrix_closure_audit_2026-07-04/false_positive_replay_report.md \
  --replay-mode executable
```

## Result

- Matrix readiness: `status=ready`; three scenarios, two seeds, `socnav_state`; probe added
  one synthetic pedestrian.
- Nominal executable smoke: six episodes, `benchmark_success`, no fallback/degraded rows.
- Perturbed executable smoke: six episodes, `benchmark_success`, no fallback/degraded rows.
- Paired report: six matched nominal-vs-perturbed rows; no unmatched rows.
- Injection summary: `pedestrians_added=102`, `steps_with_noise=102`.
- Classification: `scenario_too_weak` because false-positive injection occurred but no
  predeclared route-completion or selected-action outcome changed under this pinned smoke.

## Acceptance Evidence

| Criterion from #3300 | Evidence |
| --- | --- |
| Reproducible command runs planner/environment false-positive actor injection or fails closed with an actionable blocker. | PR #4431 made the structured-pedestrian replay injectable; PR #4439 pre-registered the stronger matrix and readiness checker; this bundle ran the paired nominal/perturbed stronger matrix commands successfully. |
| Report distinguishes live replay evidence from trace-derived diagnostics. | `summary.json` records `replay_mode: executable`; this README and `false_positive_replay_report.md` state the trace-derived boundary explicitly. |
| Report records false-positive safety effects separately from false-negative effects. | `summary.json` and `robustness_delta.csv` are issue #3300 false-positive-only artifacts. The perturbation profile is `issue_3300_false_positive_actor_injection_v1`; no false-negative slice is mixed into this bundle. |
| Report includes scenario, seed, planner mode, perturbation family, execution mode, fallback/degraded status, and caveats. | `summary.json`, `robustness_delta.csv`, this README, and the camera-ready command summaries record three scenarios, seeds `0` and `3300`, planner `goal`, `socnav_state`, false-positive profile hash, executable mode, no fallback/degraded rows, and diagnostic-only caveats. |
| If unavailable, record exact missing prerequisite and next smallest proof step. | Earlier merged PRs recorded `blocked_unavailable` for incompatible observation views (#4390, #4413). This audit is no longer unavailable; it classifies the stronger executable smoke as `scenario_too_weak`. |
| No hardware-calibrated sensor or paper-facing claim is made without durable runtime evidence. | This bundle states diagnostic-only scope and explicitly excludes full benchmark campaigns, Slurm/GPU submission, hardware sensor realism, and paper/dissertation claims. |

## Closure Recommendation

The issue acceptance criteria are met at the issue's requested smoke/diagnostic tier. The correct
closure state is not "observed robustness"; it is a completed executable false-positive replay
dimension with stronger-matrix classification `scenario_too_weak`. Close issue #3300 after this
bundle lands, unless the maintainer wants a broader empirical follow-up beyond the original child
issue scope.

## Artifacts

- `summary.json`: structured false-positive actor-injection replay report.
- `robustness_delta.csv`: six-row paired nominal-vs-perturbed delta table.
- `false_positive_replay_report.md`: generated compact Markdown report.
- Raw benchmark outputs remain under ignored `output/benchmarks/issue_3300_stronger_matrix_closure/`.
