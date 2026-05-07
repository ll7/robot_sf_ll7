# Issue 1038 H500 SNQI Contract Decision

Date: 2026-05-06

Related issues and notes:

- `ll7/robot_sf_ll7#1038` Calibrate SNQI contract for h500 scenario-horizon benchmark.
- `ll7/robot_sf_ll7#1023` Scenario-horizon benchmark surface.
- `docs/context/issue_1023_scenario_horizon_benchmark.md`
- `docs/context/issue_635_snqi_v3_paper_contract.md`
- `docs/context/issue_838_snqi_calibration_analysis.md`
- `docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/`

## Goal

Resolve the h500 scenario-horizon SNQI failure without overwriting the camera-ready v3 metric
semantics. The question is whether the h500 surface should get separate SNQI assets, stay
experimental, or change its release claim boundary.

## Evidence Surface

The decision uses the tracked compact local evidence bundle:

- campaign ID: `issue1023_scenario_horizons_candidates_local_2026-05-06`
- config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
- scenarios: `48`
- planners: `9`
- episodes: `1296`
- runtime: `1966.588` seconds
- execution result: all 9 planner rows completed
- fallback/degraded planner rows: none
- SNQI assets:
  - `configs/benchmarks/snqi_weights_camera_ready_v3.json`
  - `configs/benchmarks/snqi_baseline_camera_ready_v3.json`

The raw episode JSONL files were intentionally omitted from the tracked evidence bundle, so this
note does not claim a fresh full SNQI recomputation. It records the decision from the preserved
`reports/snqi_diagnostics.json`, `reports/snqi_diagnostics.md`, campaign tables, and analyzer
summary.

## Failure Decomposition

The h500 candidate-augmented run fails the SNQI contract because rank alignment falls below the
configured fail threshold:

| diagnostic | value | threshold | interpretation |
|---|---:|---:|---|
| rank alignment Spearman | `0.1833` | fail below `0.3` | fails |
| outcome separation | `0.2035` | fail below `0.0`, warn below `0.05` | passes |
| dominant component mean abs | `0.0936` | warn above `0.24`, fail above `0.27` | passes |
| dominant component | `success_reward` | n/a | success drives the largest share, but not enough to breach dominance |

The component-direction check is not the problem. All variable component correlations align with
the expected direction:

| metric | expected direction | Spearman vs SNQI |
|---|---|---:|
| success | positive | `0.556244` |
| time_to_goal_norm | negative | `-0.535121` |
| near_misses | negative | `-0.736912` |
| comfort_exposure | negative | `-0.325752` |
| force_exceed_events | negative | `-0.397857` |
| jerk_mean | negative | `-0.077849` |

`collisions` is degenerate in the SNQI diagnostic slice and cannot independently validate ordering
there, even though the campaign table records high collision-rate differences between planners.

## Component And Ablation Findings

The highest mean absolute SNQI contributions are:

| component | mean absolute contribution |
|---|---:|
| `success_reward` | `0.093613` |
| `near_penalty` | `0.084540` |
| `time_penalty` | `0.077256` |

One-at-a-time ablations show that the h500 planner ordering is sensitive to the success and
near-miss terms:

| ablated weight | planner-rank correlation | order changed |
|---|---:|---|
| `w_success` | `0.300000` | yes |
| `w_near` | `0.683333` | yes |
| `w_jerk` | `0.950000` | yes |
| `w_force_exceed` | `0.983333` | yes |
| `w_time` | `1.000000` | no |
| `w_comfort` | `1.000000` | no |
| `w_collisions` | `1.000000` | no |

The diagnostic optimizer found h500-specific weights with better rank alignment:

| metric | v3 configured | h500 local calibrated |
|---|---:|---:|
| `w_success` | `0.190458` | `0.050170` |
| `w_time` | `0.094911` | `0.052625` |
| `w_collisions` | `0.104835` | `0.134012` |
| `w_near` | `0.308258` | `0.338514` |
| `w_comfort` | `0.179831` | `0.233160` |
| `w_force_exceed` | `0.069211` | `0.111190` |
| `w_jerk` | `0.052495` | `0.080329` |

Those weights are diagnostic only. They reduce the success term sharply while increasing comfort,
force, jerk, collision, and near-miss weights. That is exactly the kind of surface-specific
retuning that should not silently replace the fixed camera-ready v3 contract.

## Recommendation

Decision: keep h500 scenario horizons as an experimental sensitivity surface for now.

Do not generate or promote new h500 SNQI assets in this issue. If maintainers later want a separate
h500 SNQI version, it should be a deliberately versioned asset set, for example v4 or
`h500-experimental`, produced from a fresh campaign with raw episodes retained and with paper claim
wording reviewed separately.

Do not publish a release tag from the h500 candidate-augmented local evidence. The run is useful
because it proves the 9-planner long-horizon campaign executes end to end without fallback or
degraded planner rows, but it is not clean release evidence because:

- the current v3 SNQI contract status is `fail`,
- rank alignment is the failing dimension,
- planner ordering is sensitive to success and near-miss weights,
- the experimental candidates improve completion while still showing high near-miss exposure,
- the evidence is local non-Slurm provenance and the raw episode JSONL is not tracked.

The h500 surface can be cited as a sensitivity/confounding analysis beside the fixed-horizon paper
matrix, with explicit caveats. It should not replace the fixed-horizon camera-ready benchmark or
the checked-in v3 SNQI assets.

## Validation And Artifact Notes

Commands used for this decision pass:

```bash
gh issue view 1038 --json number,title,state,body,labels,milestone,url,assignees,projectItems
```

```bash
jq '{contract_status, rank_alignment_spearman, outcome_separation, objective_score, dominant_component, component_dominance, weight_sensitivity, thresholds, calibrated_weights, configured_weights, positioning}' \
  docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/reports/snqi_diagnostics.json
```

```bash
jq '.weight_sensitivity[] | {weight_name, metric_name, configured_weight_share, mean_abs_contribution, planner_rank_correlation_if_ablated, planner_order_changed_if_ablated, sensitivity_rank}' \
  docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/reports/snqi_diagnostics.json
```

The compact evidence bundle now includes
`docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/reports/snqi_sensitivity.csv`,
reconstructed directly from the tracked `snqi_diagnostics.json` so the bundle matches the documented
SNQI analyzer artifact shape.

## Follow-Up Boundary

Open a follow-up only if maintainers decide h500 needs a separate calibrated SNQI asset set. That
follow-up should require:

- a fresh h500 campaign or mounted raw episode JSONL,
- explicit version names and SHA-256 provenance for new weights and baselines,
- analyzer output with no unexplained inconsistencies,
- paper wording that separates fixed-horizon v3 SNQI from any h500-specific metric contract.
