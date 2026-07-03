# Issue #4207 Certification-Transfer Interaction-Validity Annotation

Post-hoc, diagnostic-only annotation. This does NOT run a new simulation; it classifies the interaction status of each cell from the aggregate metrics already recorded in `summary.json`.

- Near-field band: `5` m (`social_force_default` and `hsfm_total_force_v1` only diverge inside it).
- Interaction status counts: `{'non_interacting': 8}`
- Model sensitivity exercised: `False`

**Verdict:** Model sensitivity was NOT exercised: every recorded cell stayed outside the 5 m pedestrian near field, so the certification-transfer statuses are vacuous and do NOT demonstrate certification robustness.

## Per-cell interaction status

| planner_key | evaluation_model | gate_status | interaction_status | min_clearance_m | robot_ped_within_5m_frac | episodes |
| --- | --- | --- | --- | --- | --- | --- |
| goal | social_force_default | pass | non_interacting | 24.1214633051 | 0 | 3 |
| goal | hsfm_total_force_v1 | pass | non_interacting | 24.1214633051 | 0 | 3 |
| ppo | social_force_default | pass | non_interacting | 19.8766165182 | 0 | 3 |
| ppo | hsfm_total_force_v1 | pass | non_interacting | 19.8766165182 | 0 | 3 |
| prediction_planner | social_force_default | fail | non_interacting | 24.1765022681 | 0 | 3 |
| prediction_planner | hsfm_total_force_v1 | fail | non_interacting | 24.1765022681 | 0 | 3 |
| guarded_ppo | social_force_default | pass | non_interacting | 19.8853869032 | 0 | 3 |
| guarded_ppo | hsfm_total_force_v1 | pass | non_interacting | 19.8853869032 | 0 | 3 |

## Provenance

- Source summary: `summary.json` (sha256 `2ed383ee44f2746730c38dc1f32093e9d094c03292c4d634484a182c0b640295`)
- Companion CSV: `interaction_validity.csv`
- Claim boundary: diagnostic certification-transfer interaction validity; no deployment, safety, benchmark-strength, or paper/dissertation claim.
