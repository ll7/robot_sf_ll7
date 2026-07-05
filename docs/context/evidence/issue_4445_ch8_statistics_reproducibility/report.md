# Issue #4445 Chapter 8 Statistics Reproducibility Packet

- **Overall status**: `reproducible`
- **Evidence tier**: `diagnostic-only`
- **Source status**: reproducible: Zenodo DOI 10.5281/zenodo.19563812 release 0.0.2 table bundle source rows registered repo-locally
- **Manifest**: `docs/context/evidence/issue_4445_ch8_statistics_reproducibility/source_manifest.json`
- **Manifest SHA-256**: `eae05e943f7789e50cd372b191414ea99a612d8f873299877f1b1a6980945052`

## Statistics

### ch8_eta_squared_success_mean

- **Kind**: `partial_eta_squared`
- **Status**: `matches_expected`
- **Computed**: `{"interaction_eta_squared": 0.46203158470290495, "planner_eta_squared": 0.15007190373169396, "scenario_family_eta_squared": 0.38789651156540106}`
- **Expected**: `{"interaction_eta_squared": 0.462, "planner_eta_squared": 0.15, "scenario_family_eta_squared": 0.388, "tolerance": 0.001}`

### ch8_eta_squared_near_misses_mean

- **Kind**: `partial_eta_squared`
- **Status**: `matches_expected`
- **Computed**: `{"interaction_eta_squared": 0.45580654590187186, "planner_eta_squared": 0.089945369502653, "scenario_family_eta_squared": 0.4542480845954751}`
- **Expected**: `{"interaction_eta_squared": 0.456, "planner_eta_squared": 0.09, "scenario_family_eta_squared": 0.454, "tolerance": 0.001}`

### ch8_eta_squared_time_to_goal_norm_mean

- **Kind**: `partial_eta_squared`
- **Status**: `matches_expected`
- **Computed**: `{"interaction_eta_squared": 0.5115646587748238, "planner_eta_squared": 0.11337201627804212, "scenario_family_eta_squared": 0.37506332494713407}`
- **Expected**: `{"interaction_eta_squared": 0.512, "planner_eta_squared": 0.113, "scenario_family_eta_squared": 0.375, "tolerance": 0.001}`

### ch8_spearman_success_time_to_goal_minus_0_998

- **Kind**: `spearman_rho`
- **Status**: `matches_expected`
- **Computed**: `{"value": -0.998398476564385}`
- **Expected**: `{"n": 238, "tolerance": 0.0005, "value": -0.998}`

### ch8_spearman_success_near_misses_plus_0_024

- **Kind**: `spearman_rho`
- **Status**: `matches_expected`
- **Computed**: `{"value": 0.02377355264928945}`
- **Expected**: `{"n": 238, "tolerance": 0.0005, "value": 0.024}`

### ch8_bootstrap_ppo_rank_1_ci

- **Kind**: `bootstrap_mean_ci`
- **Status**: `matches_expected`
- **Computed**: `{"observed_rank": 1, "rank_ci": [1, 1], "samples": 10000, "seed": 42}`
- **Expected**: `{"rank_ci": [1, 1], "samples": 10000, "tolerance": 0}`

### ch8_bootstrap_rank_swap_bounds

- **Kind**: `bootstrap_mean_ci`
- **Status**: `matches_expected`
- **Computed**: `{"rank_ci_by_planner": {"goal": [5, 5], "orca": [2, 3], "prediction_planner": [4, 4], "sacadrl": [6, 6], "social_force": [7, 7], "socnav_sampling": [2, 3]}, "samples": 10000, "seed": 42}`
- **Expected**: `{"rank_ci_by_planner": {"goal": [5, 5], "orca": [2, 3], "prediction_planner": [4, 4], "sacadrl": [6, 6], "social_force": [7, 7], "socnav_sampling": [2, 3]}, "samples": 10000, "tolerance": 0}`

## Claim Boundaries

- diagnostic-only reproducibility packet; not benchmark evidence
- no paper or dissertation claim is established by this script alone
- missing source data or expected targets produce blocked status
