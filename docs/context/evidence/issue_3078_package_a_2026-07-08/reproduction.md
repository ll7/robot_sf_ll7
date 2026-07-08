# Package A Pipeline Output

- Issue: `#3078`
- Classification: `diagnostic`
- Generated: `2026-07-08T06:59:01.184728+00:00`

## Artifacts

- `seed_sufficiency_analysis.json`
- `baseline_table.csv`
- `transfer_delta.csv`
- `fig_transfer_delta.png`
- `claim_card.json`

## Seed Sufficiency

- advisory_campaigns: `['issue_1484_broader_cross_kinematics_2026-05-28']`
- campaign_count: `2`
- ranking_instability_rows: `7`
- scenario_family_winner_changes: `1`
- underpowered_or_unstable: `True`

## Transfer Delta

| Planner | Transfer Delta (SNQI) | Direction |
|---|---|---|
| goal | -0.7883 | negative_transfer |
| hybrid_rule_v3_fast_progress | N/A | incomplete |
| hybrid_rule_v3_fast_progress_static_escape | N/A | incomplete |
| hybrid_rule_v3_fast_progress_static_escape_continuous | N/A | incomplete |
| orca | -0.9554 | negative_transfer |
| ppo | N/A | incomplete |
| prediction_planner | N/A | incomplete |
| sacadrl | N/A | incomplete |
| scenario_adaptive_hybrid_orca_v1 | N/A | incomplete |
| scenario_adaptive_hybrid_orca_v2_collision_guard | N/A | incomplete |
| social_force | -1.0121 | negative_transfer |
| socnav_sampling | N/A | incomplete |

## Partition Validation

- Held-out-family partition manifest: **valid**

## Reasons for Classification

- synthetic_fixture_heldout_used
- diagnostic: claim-card review required before promotion

---

This output is diagnostic evidence. Real campaign execution is required before benchmark-level classification.
