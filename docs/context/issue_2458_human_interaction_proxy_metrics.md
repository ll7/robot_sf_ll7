# Issue 2458 Human-Interaction Proxy Metrics

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2458>

## What Was Implemented

`robot_sf/benchmark/metrics.py` now supports an optional diagnostic metric group behind:

```bash
compute_all_metrics(..., experimental_human_interaction_proxy=True)
```

The opt-in path emits flat `human_proxy_*` fields plus a structured
`metrics.human_interaction_proxy` block with
`schema_version: human-interaction-proxy.v1`.

Canonical reductions:

- `human_discomfort_exposure_m_s`
- `intrusion_duration_s`
- `time_to_yield_s`
- `robot_yield_distance_m`
- `pedestrian_path_deviation_proxy_m`
- `group_split_intrusion_available`

## Claim Boundary

These are simulation proxies for mechanism reports only. They are not validated human comfort,
human-subject, safety, or paper-grade social-compliance metrics.

`group_split_intrusion_available` is false in the current implementation because `EpisodeData` does
not carry group-membership labels. Group-split intrusion should remain excluded until a grouped
pedestrian contract exists.

## Proof Surface

Targeted tests cover:

- opt-in behavior and empty-crowd stability,
- a crafted trajectory with exact formula checks for discomfort exposure, intrusion duration,
  time-to-yield, and robot-yield distance,
- post-processing into the structured `human_interaction_proxy` block,
- episode schema validation,
- aggregate flattening from schema-backed records.

The formula and unit contract is documented in `docs/ped_metrics/metrics_spec.md`.
