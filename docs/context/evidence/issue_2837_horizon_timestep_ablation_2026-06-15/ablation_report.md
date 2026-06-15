# Horizon and Timestep Ablation Report

## Claim Boundary

**analysis_only_not_navigation_evidence: this report compares forecast horizon and output-timestep presets on open-loop trace fixtures. It does not prove navigation value, closed-loop benefit, safety improvement, or benchmark-strength predictor quality.**

## Reproducibility

- **Issue:** #2837
- **Generated at (UTC):** 2026-06-15T00:00:00+00:00
- **Command:** `uv run python scripts/benchmark/build_horizon_timestep_ablation_report.py --issue 2837 --generated-at-utc 2026-06-15T00:00:00+00:00`
- **Repo HEAD:** `d042684a`
- **Horizon ladder (s):** [0.5, 1.0, 1.6, 2.0, 3.0]
- **dt ladder (s):** [0.1, 0.2, 0.4, 0.5]

## Ablation Summary (horizon x dt_s)

| horizon_s | dt_s | evaluated/total | samples | mean miss rate | mean calib. error | mean coll. rel. error | runtime (s) | artifact size (B) | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.1 | 7/9 | 99 | 0.00% | 0.0500 | 0.5232 | 0.0036 | 88310 | evaluated |
| 0.5 | 0.2 | 7/9 | 54 | 0.00% | 0.0500 | 0.5238 | 0.0019 | 88310 | evaluated |
| 0.5 | 0.4 | 7/9 | 27 | 0.00% | 0.0500 | 0.5476 | 0.0011 | 88310 | evaluated |
| 0.5 | 0.5 | 7/9 | 23 | 0.00% | 0.0500 | 0.5476 | 0.0010 | 88310 | evaluated |
| 1 | 0.1 | 7/9 | 54 | 0.00% | 0.0500 | 0.5714 | 0.0019 | 88310 | evaluated |
| 1 | 0.2 | 7/9 | 27 | 0.00% | 0.0500 | 0.5714 | 0.0011 | 88310 | evaluated |
| 1 | 0.4 | 7/9 | 18 | 0.00% | 0.0500 | 0.5714 | 0.0009 | 88310 | evaluated |
| 1 | 0.5 | 7/9 | 14 | 0.00% | 0.0500 | 0.5714 | 0.0008 | 88310 | evaluated |
| 1.6 | 0.1 | 2/9 | 16 | 0.00% | 0.0500 | 0.0000 | 0.0006 | 88310 | evaluated |
| 1.6 | 0.2 | 2/9 | 8 | 0.00% | 0.0500 | 0.0000 | 0.0004 | 88310 | evaluated |
| 1.6 | 0.4 | 2/9 | 4 | 0.00% | 0.0500 | 0.0000 | 0.0002 | 88310 | evaluated |
| 1.6 | 0.5 | 3/9 | 5 | 0.00% | 0.0500 | 0.0000 | 0.0003 | 88310 | evaluated |
| 2 | 0.1 | 0/9 | 0 | NA | NA | NA | 0.0000 | 88310 | unavailable |
| 2 | 0.2 | 0/9 | 0 | NA | NA | NA | 0.0000 | 88310 | unavailable |
| 2 | 0.4 | 0/9 | 0 | NA | NA | NA | 0.0000 | 88310 | unavailable |
| 2 | 0.5 | 0/9 | 0 | NA | NA | NA | 0.0000 | 88310 | unavailable |
| 3 | 0.1 | 0/9 | 0 | NA | NA | NA | 0.0000 | 88310 | unavailable |
| 3 | 0.2 | 0/9 | 0 | NA | NA | NA | 0.0000 | 88310 | unavailable |
| 3 | 0.4 | 0/9 | 0 | NA | NA | NA | 0.0000 | 88310 | unavailable |
| 3 | 0.5 | 0/9 | 0 | NA | NA | NA | 0.0000 | 88310 | unavailable |

## Preset Recommendations

| preset | horizon_s | dt_s | status | samples | miss rate | runtime (s) | intended use |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| short | 0.5 | 0.1 | recommended | 30 | 0.00% | 0.0012 | near-term collision relevance; fine output granularity |
| medium | 1.6 | 0.2 | recommended | 4 | 0.00% | 0.0002 | intent/goal horizon; moderate granularity |
| long | - | - | unavailable | - | - | - | route-scale lookahead; coarse granularity where trace length permits |

## Per-Trace, Per-Cell Status

| family | label | horizon_s | requested_dt_s | actual_dt_s | status | samples | ADE (m) | miss rate | NLL | calib. error | coll. rel. error | runtime (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| corridor_interaction | default_social_force | 0.5 | 0.1 | 0.1 | evaluated | 30 | 0.0243 | 0.00% | 1.2677 | 0.0500 | 0.0000 | 0.0012 |
| corridor_interaction | default_social_force | 0.5 | 0.2 | 0.2 | evaluated | 16 | 0.0798 | 0.00% | 1.2712 | 0.0500 | 0.0000 | 0.0005 |
| corridor_interaction | default_social_force | 0.5 | 0.4 | 0.4 | evaluated | 8 | 0.1025 | 0.00% | 1.2769 | 0.0500 | 0.0000 | 0.0003 |
| corridor_interaction | default_social_force | 0.5 | 0.5 | 0.5 | evaluated | 6 | 0.0994 | 0.00% | 1.2881 | 0.0500 | 0.0000 | 0.0002 |
| corridor_interaction | default_social_force | 1 | 0.1 | 0.1 | evaluated | 20 | 0.0769 | 0.00% | 1.9522 | 0.0500 | 0.0000 | 0.0006 |
| corridor_interaction | default_social_force | 1 | 0.2 | 0.2 | evaluated | 10 | 0.1337 | 0.00% | 1.9685 | 0.0500 | 0.0000 | 0.0003 |
| corridor_interaction | default_social_force | 1 | 0.4 | 0.4 | evaluated | 6 | 0.2355 | 0.00% | 1.9741 | 0.0500 | 0.0000 | 0.0002 |
| corridor_interaction | default_social_force | 1 | 0.5 | 0.5 | evaluated | 4 | 0.3059 | 0.00% | 2.0177 | 0.0500 | 0.0000 | 0.0002 |
| corridor_interaction | default_social_force | 1.6 | 0.1 | 0.1 | evaluated | 8 | 0.2951 | 0.00% | 2.5859 | 0.0500 | 0.0000 | 0.0003 |
| corridor_interaction | default_social_force | 1.6 | 0.2 | 0.2 | evaluated | 4 | 0.5206 | 0.00% | 2.6453 | 0.0500 | 0.0000 | 0.0002 |
| corridor_interaction | default_social_force | 1.6 | 0.4 | 0.4 | evaluated | 2 | 0.9742 | 0.00% | 2.7644 | 0.0500 | 0.0000 | 0.0001 |
| corridor_interaction | default_social_force | 1.6 | 0.5 | 0.5 | evaluated | 2 | 0.9119 | 0.00% | 2.7348 | 0.0500 | 0.0000 | 0.0001 |
| corridor_interaction | default_social_force | 2 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | default_social_force | 2 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | default_social_force | 2 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | default_social_force | 2 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | default_social_force | 3 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | default_social_force | 3 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | default_social_force | 3 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | default_social_force | 3 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | ammv_social_force | 0.5 | 0.1 | 0.1 | evaluated | 30 | 0.0243 | 0.00% | 1.2677 | 0.0500 | 0.0000 | 0.0009 |
| corridor_interaction | ammv_social_force | 0.5 | 0.2 | 0.2 | evaluated | 16 | 0.0798 | 0.00% | 1.2712 | 0.0500 | 0.0000 | 0.0005 |
| corridor_interaction | ammv_social_force | 0.5 | 0.4 | 0.4 | evaluated | 8 | 0.1025 | 0.00% | 1.2769 | 0.0500 | 0.0000 | 0.0003 |
| corridor_interaction | ammv_social_force | 0.5 | 0.5 | 0.5 | evaluated | 6 | 0.0994 | 0.00% | 1.2881 | 0.0500 | 0.0000 | 0.0003 |
| corridor_interaction | ammv_social_force | 1 | 0.1 | 0.1 | evaluated | 20 | 0.0769 | 0.00% | 1.9522 | 0.0500 | 0.0000 | 0.0006 |
| corridor_interaction | ammv_social_force | 1 | 0.2 | 0.2 | evaluated | 10 | 0.1337 | 0.00% | 1.9685 | 0.0500 | 0.0000 | 0.0003 |
| corridor_interaction | ammv_social_force | 1 | 0.4 | 0.4 | evaluated | 6 | 0.2355 | 0.00% | 1.9741 | 0.0500 | 0.0000 | 0.0002 |
| corridor_interaction | ammv_social_force | 1 | 0.5 | 0.5 | evaluated | 4 | 0.3059 | 0.00% | 2.0177 | 0.0500 | 0.0000 | 0.0002 |
| corridor_interaction | ammv_social_force | 1.6 | 0.1 | 0.1 | evaluated | 8 | 0.2951 | 0.00% | 2.5859 | 0.0500 | 0.0000 | 0.0003 |
| corridor_interaction | ammv_social_force | 1.6 | 0.2 | 0.2 | evaluated | 4 | 0.5206 | 0.00% | 2.6453 | 0.0500 | 0.0000 | 0.0002 |
| corridor_interaction | ammv_social_force | 1.6 | 0.4 | 0.4 | evaluated | 2 | 0.9742 | 0.00% | 2.7644 | 0.0500 | 0.0000 | 0.0001 |
| corridor_interaction | ammv_social_force | 1.6 | 0.5 | 0.5 | evaluated | 2 | 0.9119 | 0.00% | 2.7348 | 0.0500 | 0.0000 | 0.0001 |
| corridor_interaction | ammv_social_force | 2 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | ammv_social_force | 2 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | ammv_social_force | 2 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | ammv_social_force | 2 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | ammv_social_force | 3 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | ammv_social_force | 3 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | ammv_social_force | 3 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| corridor_interaction | ammv_social_force | 3 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 0.5 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 0.5 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 0.5 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 0.5 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 1 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 1 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 1 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 1 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 1.6 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 1.6 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 1.6 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 1.6 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 2 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 2 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 2 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 2 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 3 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 3 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 3 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| crossing_proxy | synthetic_crossing_proxy_orca | 3 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 0.5 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 0.5 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 0.5 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 0.5 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 1 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 1 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 1 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 1 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 1.6 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 1.6 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 1.6 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 1.6 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 2 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 2 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 2 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 2 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 3 | 0.1 | 0.1 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 3 | 0.2 | 0.2 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 3 | 0.4 | 0.4 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| bottleneck | minimal_fixture | 3 | 0.5 | 0.5 | limited_no_pedestrian_motion | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 0.5 | 0.1 | 0.1 | evaluated | 11 | 0.0000 | 0.00% | 1.2625 | 0.0500 | 0.0909 | 0.0004 |
| occluded_emergence | deterministic_occluded_emergence | 0.5 | 0.2 | 0.2 | evaluated | 6 | 0.1000 | 0.00% | 1.2714 | 0.0500 | 0.1667 | 0.0002 |
| occluded_emergence | deterministic_occluded_emergence | 0.5 | 0.4 | 0.4 | evaluated | 3 | 0.1000 | 0.00% | 1.2714 | 0.0500 | 0.3333 | 0.0001 |
| occluded_emergence | deterministic_occluded_emergence | 0.5 | 0.5 | 0.5 | evaluated | 3 | 0.0000 | 0.00% | 1.2625 | 0.0500 | 0.3333 | 0.0001 |
| occluded_emergence | deterministic_occluded_emergence | 1 | 0.1 | 0.1 | evaluated | 6 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 0.0000 | 0.0002 |
| occluded_emergence | deterministic_occluded_emergence | 1 | 0.2 | 0.2 | evaluated | 3 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 0.0000 | 0.0001 |
| occluded_emergence | deterministic_occluded_emergence | 1 | 0.4 | 0.4 | evaluated | 2 | 0.2000 | 0.00% | 1.9536 | 0.0500 | 0.0000 | 0.0001 |
| occluded_emergence | deterministic_occluded_emergence | 1 | 0.5 | 0.5 | evaluated | 2 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 0.0000 | 0.0002 |
| occluded_emergence | deterministic_occluded_emergence | 1.6 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 1.6 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 1.6 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 1.6 | 0.5 | 0.5 | evaluated | 1 | 0.1000 | 0.00% | 2.5276 | 0.0500 | 0.0000 | 0.0001 |
| occluded_emergence | deterministic_occluded_emergence | 2 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 2 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 2 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 2 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 3 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 3 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 3 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| occluded_emergence | deterministic_occluded_emergence | 3 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 0.5 | 0.1 | 0.1 | evaluated | 7 | 0.0000 | 0.00% | 1.2625 | 0.0500 | 1.0000 | 0.0003 |
| signalized_crossing | signalized_crossing_semantic_metadata | 0.5 | 0.2 | 0.2 | evaluated | 4 | 0.0700 | 0.00% | 1.2669 | 0.0500 | 1.0000 | 0.0002 |
| signalized_crossing | signalized_crossing_semantic_metadata | 0.5 | 0.4 | 0.4 | evaluated | 2 | 0.0700 | 0.00% | 1.2669 | 0.0500 | 1.0000 | 0.0001 |
| signalized_crossing | signalized_crossing_semantic_metadata | 0.5 | 0.5 | 0.5 | evaluated | 2 | 0.0000 | 0.00% | 1.2625 | 0.0500 | 1.0000 | 0.0001 |
| signalized_crossing | signalized_crossing_semantic_metadata | 1 | 0.1 | 0.1 | evaluated | 2 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 1.0000 | 0.0001 |
| signalized_crossing | signalized_crossing_semantic_metadata | 1 | 0.2 | 0.2 | evaluated | 1 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 1.0000 | 0.0001 |
| signalized_crossing | signalized_crossing_semantic_metadata | 1 | 0.4 | 0.4 | evaluated | 1 | 0.1400 | 0.00% | 1.9443 | 0.0500 | 1.0000 | 0.0001 |
| signalized_crossing | signalized_crossing_semantic_metadata | 1 | 0.5 | 0.5 | evaluated | 1 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 1.0000 | 0.0001 |
| signalized_crossing | signalized_crossing_semantic_metadata | 1.6 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 1.6 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 1.6 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 1.6 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 2 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 2 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 2 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 2 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 3 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 3 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 3 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| signalized_crossing | signalized_crossing_semantic_metadata | 3 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 0.5 | 0.1 | 0.1 | evaluated | 7 | 0.0000 | 0.00% | 0.4516 | 0.0500 | 1.0000 | 0.0003 |
| goal_directed_crossing | goal_directed_crossing_fixture | 0.5 | 0.2 | 0.2 | evaluated | 4 | 0.0600 | 0.00% | 0.4588 | 0.0500 | 1.0000 | 0.0002 |
| goal_directed_crossing | goal_directed_crossing_fixture | 0.5 | 0.4 | 0.4 | evaluated | 2 | 0.0600 | 0.00% | 0.4588 | 0.0500 | 1.0000 | 0.0001 |
| goal_directed_crossing | goal_directed_crossing_fixture | 0.5 | 0.5 | 0.5 | evaluated | 2 | 0.0000 | 0.00% | 0.4516 | 0.0500 | 1.0000 | 0.0001 |
| goal_directed_crossing | goal_directed_crossing_fixture | 1 | 0.1 | 0.1 | evaluated | 2 | 0.0000 | 0.00% | 1.1245 | 0.0500 | 1.0000 | 0.0001 |
| goal_directed_crossing | goal_directed_crossing_fixture | 1 | 0.2 | 0.2 | evaluated | 1 | 0.0000 | 0.00% | 1.1245 | 0.0500 | 1.0000 | 0.0001 |
| goal_directed_crossing | goal_directed_crossing_fixture | 1 | 0.4 | 0.4 | evaluated | 1 | 0.1200 | 0.00% | 1.1392 | 0.0500 | 1.0000 | 0.0001 |
| goal_directed_crossing | goal_directed_crossing_fixture | 1 | 0.5 | 0.5 | evaluated | 1 | 0.0000 | 0.00% | 1.1245 | 0.0500 | 1.0000 | 0.0001 |
| goal_directed_crossing | goal_directed_crossing_fixture | 1.6 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 1.6 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 1.6 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 1.6 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 2 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 2 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 2 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 2 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 3 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 3 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 3 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| goal_directed_crossing | goal_directed_crossing_fixture | 3 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 0.5 | 0.1 | 0.1 | evaluated | 7 | 0.1200 | 0.00% | 0.5071 | 0.0500 | 0.5714 | 0.0003 |
| waiting_with_intent_change | waiting_intent_change_fixture | 0.5 | 0.2 | 0.2 | evaluated | 4 | 0.0900 | 0.00% | 0.4732 | 0.0500 | 0.5000 | 0.0002 |
| waiting_with_intent_change | waiting_intent_change_fixture | 0.5 | 0.4 | 0.4 | evaluated | 2 | 0.0600 | 0.00% | 0.4588 | 0.0500 | 0.5000 | 0.0001 |
| waiting_with_intent_change | waiting_intent_change_fixture | 0.5 | 0.5 | 0.5 | evaluated | 2 | 0.0600 | 0.00% | 0.4660 | 0.0500 | 0.5000 | 0.0001 |
| waiting_with_intent_change | waiting_intent_change_fixture | 1 | 0.1 | 0.1 | evaluated | 2 | 0.4500 | 0.00% | 1.3321 | 0.0500 | 1.0000 | 0.0001 |
| waiting_with_intent_change | waiting_intent_change_fixture | 1 | 0.2 | 0.2 | evaluated | 1 | 0.4200 | 0.00% | 1.3045 | 0.0500 | 1.0000 | 0.0001 |
| waiting_with_intent_change | waiting_intent_change_fixture | 1 | 0.4 | 0.4 | evaluated | 1 | 0.3000 | 0.00% | 1.2164 | 0.0500 | 1.0000 | 0.0001 |
| waiting_with_intent_change | waiting_intent_change_fixture | 1 | 0.5 | 0.5 | evaluated | 1 | 0.4200 | 0.00% | 1.3045 | 0.0500 | 1.0000 | 0.0001 |
| waiting_with_intent_change | waiting_intent_change_fixture | 1.6 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 1.6 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 1.6 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 1.6 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 2 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 2 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 2 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 2 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 3 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 3 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 3 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| waiting_with_intent_change | waiting_intent_change_fixture | 3 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 0.5 | 0.1 | 0.1 | evaluated | 7 | 0.0000 | 0.00% | 1.2625 | 0.0500 | 1.0000 | 0.0003 |
| route_conflict_goal | route_conflict_goal_fixture | 0.5 | 0.2 | 0.2 | evaluated | 4 | 0.0539 | 0.00% | 1.2651 | 0.0500 | 1.0000 | 0.0002 |
| route_conflict_goal | route_conflict_goal_fixture | 0.5 | 0.4 | 0.4 | evaluated | 2 | 0.0539 | 0.00% | 1.2651 | 0.0500 | 1.0000 | 0.0001 |
| route_conflict_goal | route_conflict_goal_fixture | 0.5 | 0.5 | 0.5 | evaluated | 2 | 0.0000 | 0.00% | 1.2625 | 0.0500 | 1.0000 | 0.0001 |
| route_conflict_goal | route_conflict_goal_fixture | 1 | 0.1 | 0.1 | evaluated | 2 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 1.0000 | 0.0001 |
| route_conflict_goal | route_conflict_goal_fixture | 1 | 0.2 | 0.2 | evaluated | 1 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 1.0000 | 0.0001 |
| route_conflict_goal | route_conflict_goal_fixture | 1 | 0.4 | 0.4 | evaluated | 1 | 0.1077 | 0.00% | 1.9407 | 0.0500 | 1.0000 | 0.0001 |
| route_conflict_goal | route_conflict_goal_fixture | 1 | 0.5 | 0.5 | evaluated | 1 | 0.0000 | 0.00% | 1.9355 | 0.0500 | 1.0000 | 0.0001 |
| route_conflict_goal | route_conflict_goal_fixture | 1.6 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 1.6 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 1.6 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 1.6 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 2 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 2 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 2 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 2 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 3 | 0.1 | 0.1 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 3 | 0.2 | 0.2 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 3 | 0.4 | 0.4 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |
| route_conflict_goal | route_conflict_goal_fixture | 3 | 0.5 | 0.5 | horizon_longer_than_trace | 0 | None | None | None | None | None | 0.0000 |

## Missing Trace Families

These scenario families have no durable trace fixtures and were not evaluated:

- **dense_pedestrian_interaction**: no durable trace fixture available
- **bottleneck_with_motion**: existing bottleneck fixture has zero pedestrian velocity

## Interpretation

This ablation varies only the forecast output horizon and output timestep on a bounded set of repository trace fixtures.  It does not change simulator physics step semantics.  Long-horizon and coarse-dt rows are frequently unavailable because the durable fixtures are short (1-2 s).  Preset recommendations are diagnostic suggestions for forecast-output configuration, not evidence that any preset improves navigation, safety, or closed-loop planner performance.