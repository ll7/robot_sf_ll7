# Issue #4445 Chapter 8 Statistics Reproducibility Packet

- **Overall status**: `blocked`
- **Evidence tier**: `diagnostic-only`
- **Source status**: blocked: diss#337 now provides audited expected values and frozen-release provenance, but this Robot SF packet still lacks repo-local source rows for recomputation.
- **Manifest**: `docs/context/evidence/issue_4445_ch8_statistics_reproducibility/source_manifest.json`
- **Manifest SHA-256**: `849c76b9af0018c1d9951866d4ffcccd5cd85d1ce3f25581629fbd9797687f30`

## Statistics

### ch8_eta_squared_success_mean

- **Kind**: `partial_eta_squared`
- **Status**: `blocked_missing_source_data`
- **Computed**: `{}`
- **Expected**: `{"interaction_eta_squared": 0.462, "planner_eta_squared": 0.15, "scenario_family_eta_squared": 0.388, "tolerance": 0.001}`
- **Blockers**: data block is missing or empty

### ch8_eta_squared_near_misses_mean

- **Kind**: `partial_eta_squared`
- **Status**: `blocked_missing_source_data`
- **Computed**: `{}`
- **Expected**: `{"interaction_eta_squared": 0.456, "planner_eta_squared": 0.09, "scenario_family_eta_squared": 0.454, "tolerance": 0.001}`
- **Blockers**: data block is missing or empty

### ch8_eta_squared_time_to_goal_norm_mean

- **Kind**: `partial_eta_squared`
- **Status**: `blocked_missing_source_data`
- **Computed**: `{}`
- **Expected**: `{"interaction_eta_squared": 0.512, "planner_eta_squared": 0.113, "scenario_family_eta_squared": 0.375, "tolerance": 0.001}`
- **Blockers**: data block is missing or empty

### ch8_spearman_success_time_to_goal_minus_0_998

- **Kind**: `spearman_rho`
- **Status**: `blocked_missing_source_data`
- **Computed**: `{}`
- **Expected**: `{"n": 238, "tolerance": 0.0005, "value": -0.998}`
- **Blockers**: data block is missing or empty

### ch8_spearman_success_near_misses_plus_0_024

- **Kind**: `spearman_rho`
- **Status**: `blocked_missing_source_data`
- **Computed**: `{}`
- **Expected**: `{"n": 238, "tolerance": 0.0005, "value": 0.024}`
- **Blockers**: data block is missing or empty

### ch8_bootstrap_ppo_rank_1_ci

- **Kind**: `bootstrap_mean_ci`
- **Status**: `blocked_missing_source_data`
- **Computed**: `{}`
- **Expected**: `{"rank_ci": [1, 1], "samples": 10000, "tolerance": 0}`
- **Blockers**: data block is missing or empty

### ch8_bootstrap_rank_swap_bounds

- **Kind**: `bootstrap_mean_ci`
- **Status**: `blocked_missing_source_data`
- **Computed**: `{}`
- **Expected**: `{"rank_ci_by_planner": {"goal": [5, 5], "orca": [2, 3], "prediction_planner": [4, 4], "sacadrl": [6, 6], "social_force": [7, 7], "socnav_sampling": [2, 3]}, "samples": 10000, "tolerance": 0}`
- **Blockers**: data block is missing or empty

## Claim Boundaries

- diagnostic-only reproducibility packet; not benchmark evidence
- no paper or dissertation claim is established by this script alone
- missing source data or expected targets produce blocked status
