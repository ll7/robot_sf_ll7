# Issue #3300 Structured False-Positive Actor-Injection Replay Evidence

This bundle records a bounded CPU-local executable replay for issue #3300 using a
planner observation view that exposes structured pedestrian slots.

## Claim Boundary

- Evidence status: smoke/diagnostic evidence only.
- Replay mode: executable camera-ready benchmark smoke, not trace-derived diagnostics.
- Scope: one local CPU scenario, one planner, one seed.
- Out of scope: full benchmark campaign, Slurm/GPU submission, hardware sensor realism, and
  paper-facing robustness claims.

## Inputs

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/3300>
- Replay source: this PR branch head, based on
  `25495f55b`.
- Scenario matrix: `configs/scenarios/sets/issue_3300_false_positive_single_ped_smoke.yaml`
- Scenario: `single_ped_crossing_orthogonal`
- Planner: `goal`
- Observation mode: `socnav_state`
- Observation level: `tracked_agents_no_noise`
- Seed: `0`
- Nominal config: `configs/benchmarks/issue_3300_false_positive_nominal_smoke.yaml`
- Perturbed config: `configs/benchmarks/issue_3300_false_positive_perturbed_smoke.yaml`
- Perturbation profile:
  `configs/benchmarks/observation_noise/issue_3300_false_positive_actor_injection_v1.yaml`

## Commands

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3300_false_positive_nominal_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_3300_structured \
  --campaign-id issue_3300_nominal_structured_ped_smoke_v5 \
  --skip-publication-bundle

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3300_false_positive_perturbed_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_3300_structured \
  --campaign-id issue_3300_false_positive_structured_ped_smoke_v5 \
  --skip-publication-bundle

scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/benchmark/build_false_positive_replay_report_issue_3300.py \
  --nominal-jsonl output/benchmarks/issue_3300_structured/issue_3300_nominal_structured_ped_smoke_v5/runs/goal__differential_drive/episodes.jsonl \
  --perturbed-jsonl output/benchmarks/issue_3300_structured/issue_3300_false_positive_structured_ped_smoke_v5/runs/goal__differential_drive/episodes.jsonl \
  --output-json docs/context/evidence/issue_3300_structured_false_positive_actor_injection_2026-07-04/summary.json \
  --output-csv docs/context/evidence/issue_3300_structured_false_positive_actor_injection_2026-07-04/robustness_delta.csv \
  --output-md docs/context/evidence/issue_3300_structured_false_positive_actor_injection_2026-07-04/false_positive_replay_report.md \
  --replay-mode executable
```

## Result

- Classification: `scenario_too_weak`.
- Injection occurred: `pedestrians_added: 5`, `steps_with_noise: 5`.
- Pairing: one nominal row matched one perturbed row; no unmatched rows.
- Outcome delta: no selected-action or route-completion change under this one pinned smoke.
- Fallback/degraded status: both campaign rows completed natively; no fallback/degraded rows were
  counted as success evidence.

## Artifacts

- `summary.json`: structured issue #3300 replay report.
- `robustness_delta.csv`: one-row nominal-vs-perturbed delta.
- `false_positive_replay_report.md`: generated compact Markdown report.
- Raw benchmark outputs remain under ignored `output/benchmarks/issue_3300_structured/`.

## Residual Risk

This closes the compatible structured-pedestrian replay gap, but it does not show a behavioral
effect. A broader or stronger scenario/seed matrix is required before any robustness or perception
claim can move beyond this smoke/diagnostic boundary.
