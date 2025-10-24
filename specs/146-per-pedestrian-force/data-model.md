# Data Model: Per-Pedestrian Force Quantiles

## Entities

- EpisodeData (existing)
  - robot_pos: (T,2) float
  - robot_vel: (T,2) float
  - robot_acc: (T,2) float
  - peds_pos: (T,K,2) float
  - ped_forces: (T,K,2) float (may contain NaN for absent ped/timestep)
  - goal: (2,) float
  - dt: float

- Metrics (output map)
  - ped_force_q50: float (NaN if K==0 or no finite samples)
  - ped_force_q90: float (same semantics)
  - ped_force_q95: float (same semantics)

## Validation Rules

- If K==0 → all ped_force_qxx keys must be present with NaN values.
- If at least one pedestrian has ≥1 finite sample, metrics must be finite numbers (unless all peds have no finite samples → NaN).
- Quantile values must be ≥ 0.0 (non-negative magnitudes).
