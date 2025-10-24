# Metrics Contract: Per-Pedestrian Force Quantiles

## New Metric Keys

- `ped_force_q50`: float
- `ped_force_q90`: float
- `ped_force_q95`: float

## Semantics

- For each pedestrian k, compute force magnitudes over time M_k = { ||F_{k,t}||_2 } over timesteps where present.
- Compute Q_k(q) = quantile(M_k, q) individually per pedestrian.
- Episode value = mean over pedestrians: (1/K) Σ_k Q_k(q).
- If K==0 (no pedestrians), all keys are present with NaN values.
- If a pedestrian has no finite samples, exclude from the mean; if all are excluded, return NaN.

## Compatibility

- Backwards compatible; new keys only. Existing keys `force_q50`, `force_q90`, `force_q95` remain unchanged as aggregated metrics.
- Episode schema supports additional metric keys via `additionalProperties`.
