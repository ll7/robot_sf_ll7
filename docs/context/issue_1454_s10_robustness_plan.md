# Issue #1454 S10 Robustness Campaign Plan

# Goal

Issue #1454 adds a staged S10 robustness surface for the broader planner rows from #1353. Stage A
is the required fixed `horizon: 100` stress-matrix reference; Stage B is the gated
scenario-specific h500 horizon comparison, run only after Stage A is usable.

# Boundaries

- In scope: versioned sibling benchmark configs, contract tests for seed and row compatibility, and
  a durable execution plan.
- Out of scope: replacing frozen paper-facing release configs, changing metric semantics, retuning
  planners, adding scenarios, running S20, or treating fallback/degraded rows as benchmark success.

# Evidence

- GitHub issue: <https://github.com/ll7/robot_sf_ll7/issues/1454>
- Stage A row lineage: `configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml`
- Seed schedule: `configs/benchmarks/seed_sets_v1.yaml` (`paper_eval_s10`, preserving
  `[111, 112, 113]` first)
- Scenario-horizon lineage: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
  and `configs/policy_search/scenario_horizons_h500.yaml`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`
- Prior h500 interpretation: `docs/context/issue_1023_scenario_horizon_benchmark.md`
- Preflight evidence: `docs/context/evidence/issue_1454_s10_preflight_2026-05-22/README.md`

# Steps

1. Run Stage A preflight:
   `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/issue_1454_s10_fixed_h100_broader_baselines.yaml --mode preflight --campaign-id issue1454-s10-fixed-h100-preflight`
2. If preflight is structurally valid, run Stage A in a resumable local or tmux path:
   `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/issue_1454_s10_fixed_h100_broader_baselines.yaml --campaign-id issue1454-s10-fixed-h100`
3. Analyze Stage A and write a short go/no-go note before Stage B. Stage B should proceed only when
   Stage A has a comparable planner/scenario/seed table and no unresolved campaign-level failure.
4. If gated in, run Stage B preflight and campaign with
   `configs/benchmarks/issue_1454_s10_scenario_horizons_h500_broader_baselines.yaml`.
5. Compare fixed h100 against scenario horizons with matched planner/scenario/seed rows. Keep any
   future h500-only candidate rows out of the primary verdict.
6. Preserve compact evidence under `docs/context/evidence/`; keep raw `output/` campaign data local
   unless promoted to a durable artifact store.

# Validation

- Config contract:
  `uv run pytest tests/benchmark/test_issue_1454_s10_robustness_configs.py -q`
- Runner/config regression slice:
  `uv run pytest tests/benchmark/test_issue_1454_s10_robustness_configs.py tests/benchmark/test_camera_ready_campaign.py -q`
- Required campaign proof before interpreting results: Stage A preflight, Stage A campaign summary,
  analyzer output, and Stage A go/no-go note. Stage B proof is required only after the gate passes.
  The 2026-05-22 preflight bundle proves only config expansion and row/seed/horizon wiring.

# Risks / Follow-ups

- Runtime risk remains medium to high because S10 multiplies the broader row count and includes PPO.
- `socnav_bench`, `socnav_sampling`, and fallback-capable rows must stay visible as unavailable,
  degraded, or non-success when dependencies do not meet the benchmark contract.
- Longer horizons change exposure time. Improvements in success must be interpreted beside
  collision, near-miss, unfinished, runtime, and SNQI deltas.
- The final report should recommend whether a separate S20 issue is warranted; this issue does not
  execute S20 by default.
