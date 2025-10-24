# Research: Per-Pedestrian Force Quantiles

Date: 2025-10-24

## Decisions

- Quantile definition: compute per-pedestrian quantiles first, then average: for each pedestrian k, let M_k = {||F_{k,t}||_2} over timesteps where present; Q_k(q) = quantile(M_k, q); episode value = mean_k Q_k(q).
- Naming convention: new keys use `ped_force_q{50,90,95}` to distinguish from existing aggregated `force_q{50,90,95}`.
- Presence handling: use nan-aware operations. If a pedestrian is absent at some timesteps, those entries in `ped_forces` should be NaN; we will apply `np.nanquantile` along the time axis to ignore missing entries.
- Degenerate cases: if a pedestrian has a single finite sample, quantiles equal that value; if no finite samples for the entire episode, exclude from the mean (if all peds excluded, return NaN overall).
- Performance: vectorize with NumPy; avoid Python loops; complexity O(T×K). Use `np.linalg.norm(..., axis=2)` to compute magnitudes and `np.nanquantile(mags, q=[...], axis=0)` to compute per-ped quantiles.

## Rationale

- Aggregated quantiles (flattened over all (t,k)) can mask individual outliers; per-ped averages preserve per-agent experience before aggregation, aligning with social comfort interpretation.
- nan-aware ops allow handling variable ped presence without brittle masks or additional schema changes.
- Keeping new keys distinct avoids ambiguity and preserves backward compatibility (Constitution VII).

## Alternatives Considered

- Weighted average by ped presence duration (mean over Q_k(q) weighted by |T_k|): rejected for first version to keep semantics simple and comparable across episodes; can be added later as a separate metric if needed.
- Using median across pedestrians instead of mean: rejected per spec; mean is requested; median could be added later.
- Flattened quantile (current implementation): retained as the existing `force_q*` metrics; not a replacement.

## Open Questions (resolved)

- How to treat totally missing ped force series (all NaN)? → Exclude from mean; if all excluded, return NaN.
- Should we clip negative or infinite magnitudes? → Use `np.nan_to_num` prior to norm to map NaN/inf to finite values only if needed; prefer `nanquantile` and require inputs to be finite where present.
