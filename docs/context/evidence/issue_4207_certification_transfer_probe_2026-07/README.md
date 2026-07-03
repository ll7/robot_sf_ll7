# Issue #4207 Certification-Transfer Probe Evidence

This compact packet records a CPU diagnostic probe for certification-transfer between `social_force_default` and `hsfm_total_force_v1` pedestrian models.

The provisional release gates are not deployment approval. Missing gate metrics are `not_evaluable`, never `pass`. Transfer flips are reported as model-assumption fragility in the gate decision, not as a failed experiment.

## Counts

- Gate status counts: `{'pass': 6, 'fail': 2}`
- Transfer status counts: `{'stable_pass': 12, 'stable_fail': 4}`
- Flip cases: `0`

## Files

- `summary.json`
- `metadata.json`
- `certification_gate_cells.csv`
- `certification_transfer_matrix.csv`
- `metric_deltas_by_model.csv`
- `flip_cases.csv`
- `claim_boundary.md`
- `SHA256SUMS`
