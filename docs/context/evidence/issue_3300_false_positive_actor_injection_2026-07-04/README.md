# Issue #3300 False-Positive Actor-Injection Replay Evidence

This diagnostic replay attempted the smallest local nominal-vs-perturbed false-positive
actor-injection comparison for issue #3300 and classified the result as `blocked_unavailable`.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- Replay mode: executable camera-ready benchmark smoke, not trace-derived diagnostics.
- Scope: one local CPU scenario, one planner, one seed.
- Out of scope: hardware sensor realism, broad benchmark campaign, Slurm/GPU submission, and
  paper-facing claims.

## Inputs

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/3300>
- Commit: `d90c9cda73f073f3b08201a366c18044c85dc552`
- Scenario matrix:
  `configs/scenarios/sets/issue_3300_false_positive_single_ped_smoke.yaml`
- Scenario: `single_ped_crossing_orthogonal`
- Planner: `goal`
- Seed: `0`
- Nominal config: `configs/benchmarks/issue_3300_false_positive_nominal_smoke.yaml`
- Perturbed config: `configs/benchmarks/issue_3300_false_positive_perturbed_smoke.yaml`
- Perturbation profile:
  `configs/benchmarks/observation_noise/issue_3300_false_positive_actor_injection_v1.yaml`
- Perturbation profile hash: `f3149ea5da06`

## Commands

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3300_false_positive_nominal_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_3300 \
  --campaign-id issue_3300_nominal_single_ped_smoke \
  --skip-publication-bundle

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3300_false_positive_perturbed_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_3300 \
  --campaign-id issue_3300_false_positive_single_ped_smoke \
  --skip-publication-bundle

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/benchmark/build_false_positive_replay_report_issue_3300.py \
  --nominal-jsonl output/benchmarks/issue_3300/issue_3300_nominal_single_ped_smoke/runs/goal__differential_drive/episodes.jsonl \
  --perturbed-jsonl output/benchmarks/issue_3300/issue_3300_false_positive_single_ped_smoke/runs/goal__differential_drive/episodes.jsonl \
  --output-json docs/context/evidence/issue_3300_false_positive_actor_injection_2026-07-04/summary.json \
  --output-csv docs/context/evidence/issue_3300_false_positive_actor_injection_2026-07-04/robustness_delta.csv \
  --output-md docs/context/evidence/issue_3300_false_positive_actor_injection_2026-07-04/false_positive_replay_report.md \
  --replay-mode executable
```

## Result

- Classification: `blocked_unavailable`.
- Reason: the perturbed executable episode recorded `pedestrians_added: 0` and
  `steps_with_noise: 0`.
- Pairing: one nominal row matched one perturbed row with no unmatched rows.
- Fallback/degraded status: campaign rows completed in native execution mode with no fallback or
  degraded rows, but they are not successful false-positive actor-injection evidence because the
  actor injection did not materialize in planner-facing observations.

## Artifacts

- `summary.json`: structured issue #3300 report.
- `robustness_delta.csv`: per-episode nominal-vs-perturbed delta.
- `false_positive_replay_report.md`: human-readable compact report.

Raw benchmark outputs remain under ignored `output/benchmarks/issue_3300/`.

## Residual Blocker

The `goal` planner ran with `observation_mode: goal_state`; even on the single-pedestrian crossing
scenario, the planner-facing observation did not expose a compatible `pedestrians` block for the
false-positive actor-injection hook. The next smallest empirical step is to run the same paired
profile with a CPU-local planner/config whose effective observation view exposes structured
pedestrian observations, or to add a fail-closed config preflight that rejects #3300 replay configs
when the selected planner observation mode cannot carry injected pedestrians.
