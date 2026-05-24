# Issue #1454 S10 Preflight Evidence

This bundle preserves compact preflight evidence for the two staged issue #1454 configs.

- Fixed h100 Stage A config:
  `configs/benchmarks/issue_1454_s10_fixed_h100_broader_baselines.yaml`
- Scenario-horizon Stage B config:
  `configs/benchmarks/issue_1454_s10_scenario_horizons_h500_broader_baselines.yaml`
- Raw local output roots:
  `output/benchmarks/camera_ready/issue1454-s10-fixed-h100-preflight`
  and `output/benchmarks/camera_ready/issue1454-s10-scenario-horizons-h500-preflight`

Both preflights resolved 8 planner rows, 48 scenarios, the `paper_eval_s10` seed set
`[111, 112, 113, 114, 115, 116, 117, 118, 119, 120]`, and `paper-matrix-v1`.
The fixed-h100 surface reports `horizon_mode: fixed` and `horizon: 100`; the Stage B surface
reports `horizon_mode: scenario_horizons` with
`configs/policy_search/scenario_horizons_h500.yaml`.

This is preflight evidence only. It proves config expansion, AMV/comparability artifact generation,
and row/seed/horizon wiring. It does not claim Stage A or Stage B campaign outcome quality.
