# diss#331 h600 SNQI Weight-Set Ranking Snippet

Diagnostic h600 per-weight-set analysis only: this closes the evidence-gap shape for Social Navigation Quality Index (SNQI) rank preservation on retained jobs 13268 and 13273, inherits the three-seed caveat, and does not choose canonical weights or edit claims.

## Top-3 Rows

|weight_set_id|rank|planner_key|snqi_score|source_run|
|---|---|---|---|---|
|default_uniform_1p0|1|ppo|-0.608015877087|confirm_13268|
|default_uniform_1p0|2|orca|-0.892846493797|confirm_13268|
|default_uniform_1p0|3|socnav_sampling|-1.56588207726|confirm_13268|
|camera_ready_v2|1|ppo|-0.0643671533613|confirm_13268|
|camera_ready_v2|2|orca|-0.124587742339|confirm_13268|
|camera_ready_v2|3|socnav_sampling|-0.234826744826|confirm_13268|
|camera_ready_v3|1|ppo|-0.0313572830381|confirm_13268|
|camera_ready_v3|2|orca|-0.1408991705|confirm_13268|
|camera_ready_v3|3|socnav_sampling|-0.144037848781|confirm_13268|
|model_canonical_v1|1|orca|-0.0892846493797|confirm_13268|
|model_canonical_v1|2|socnav_sampling|-0.156588207726|confirm_13268|
|model_canonical_v1|3|prediction_planner|-0.158372109152|confirm_13268|

## Pairwise Agreement

|left_weight_set|right_weight_set|spearman|kendall_tau|pairwise_disagreement_rate|top1_same|top3_jaccard|
|---|---|---|---|---|---|---|
|camera_ready_v2|camera_ready_v3|0.9|0.777777777778|0.111111111111|True|1|
|camera_ready_v2|default_uniform_1p0|0.983333333333|0.944444444444|0.0277777777778|True|1|
|camera_ready_v2|model_canonical_v1|0.516666666667|0.555555555556|0.222222222222|False|0.5|
|camera_ready_v3|default_uniform_1p0|0.883333333333|0.722222222222|0.138888888889|True|1|
|camera_ready_v3|model_canonical_v1|0.383333333333|0.333333333333|0.333333333333|False|0.5|
|default_uniform_1p0|model_canonical_v1|0.533333333333|0.611111111111|0.194444444444|False|0.5|

Artifacts: `docs/context/evidence/issue_3810_h600_interpretation_2026-07/snqi_weight_set_h600_rank_table.csv`, `docs/context/evidence/issue_3810_h600_interpretation_2026-07/snqi_weight_set_h600_pairwise_agreement.csv`, `docs/context/evidence/issue_3810_h600_interpretation_2026-07/snqi_weight_set_h600_report.json`, and `docs/context/evidence/issue_3810_h600_interpretation_2026-07/SHA256SUMS`.

Author's canonical-weight ruling remains unchanged until decided separately.
