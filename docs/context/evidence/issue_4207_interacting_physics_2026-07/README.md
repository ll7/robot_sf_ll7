# Issue #4207 Certification-Transfer Probe Evidence

This compact packet records a CPU diagnostic probe for certification-transfer between `social_force_default` and `hsfm_total_force_v1` pedestrian models.

The provisional release gates are not deployment approval. Missing gate metrics are `not_evaluable`, never `pass`. Transfer flips are reported as model-assumption fragility in the gate decision, not as a failed experiment.

## Counts

- Gate status counts: `{'fail': 4, 'pass': 4}`
- Transfer status counts: `{'stable_fail': 8, 'stable_pass': 8}`
- Interaction status counts: `{'interacting': 4, 'non_interacting': 12}`
- Physics near-field confirmed: `True`
- Max robot-pedestrian within-5m fraction: `0.10558655435990806`
- Minimum clearance: `-0.024278621748445417`
- Interacting gate cells: `2` of `8`
- Model sensitivity exercised: `True`
- Flip cases: `0`

If `Model sensitivity exercised` is `false`, every transfer cell is `non_interacting` or `unknown`: the robot never entered the pedestrian near field, so the stable statuses above are vacuous and do not demonstrate certification robustness.

## Files

- `summary.json`
- `metadata.json`
- `certification_gate_cells.csv`
- `certification_transfer_matrix.csv`
- `metric_deltas_by_model.csv`
- `flip_cases.csv`
- `claim_boundary.md`
- `SHA256SUMS`
