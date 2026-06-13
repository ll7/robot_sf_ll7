# Issue #2749 - Observation Noise Diagnostics: Paired Step-Diagnostics Comparison (stress_slice rerun)

**Result classification:** `diagnostic_only`
**Not benchmark-strength evidence.**

## Configuration

| Field | Value |
|---|---|
| Candidate | `hybrid_rule_v0_minimal` |
| Stage | `stress_slice` |
| Scenario | `classic_bottleneck_medium` |
| Scenario index | 0 |
| Seed | 111 |
| Horizon | 40 steps |
| Algorithm | `hybrid_rule_local_planner` |
| Pedestrians in scenario | 1 (map single_ped marker) |

## Perturbation Config

| Parameter | Baseline | Perception-limited |
|---|---|---|
| position_noise_std_m | 0.0 | 0.1 |
| position_noise_bound_m | 0.0 | 0.2 |
| missed_detection_probability | 0.0 | 0.25 |
| occlusion_distance_m | null | 3.0 |
| delay_steps | 0 | 1 |
| seed | null | 2749 |
| Evidence class | `ideal_state` | `perception_limited` |
| Noise profile | `none` | `bounded_gaussian` |

## Progress Comparison

| Metric | Baseline | Perception-limited | Delta |
|---|---|---|---|
| Net goal progress | -10.646 | -10.646 | 0.0 |
| Best goal progress | 0.0 | 0.0 | 0.0 |
| Progress steps | 38 | 38 | 0 |
| Regression steps | 2 | 2 | 0 |
| Stagnant steps | 0 | 0 | 0 |
| Closest robot-ped distance | 14.913m | 14.913m | 0.0 |
| Pedestrian collisions | 0 | 0 | 0 |
| Obstacle collisions | 0 | 0 | 0 |
| Robot collisions | 0 | 0 | 0 |

## Planner Comparison

| Metric | Baseline | Perception-limited | Delta |
|---|---|---|---|
| Selected sources | dw:36, pf:4 | dw:36, pf:4 | identical |
| Fallback count | 0 | 0 | 0 |
| Rejection counts | sc:778 | sc:778 | identical |
| Unavailable counts | cs:40 | cs:40 | identical |

## Observation Perturbation Evidence

| Metric | Baseline | Perception-limited |
|---|---|---|
| Actor count (per step) | 1 | 1 |
| Observed actor count range | [1, 1] | [0, 1] |
| Missed actor count range | [0, 0] | [0, 1] |
| Occluded actor count range | [0, 0] | [0, 1] |
| Evidence class | ideal_state | perception_limited |
| Noise profile | none | bounded_gaussian |

The perception-limited variant shows active occlusion masking on most steps (observed=0 at steps 5 and 39) and 1 missed detection at step 15. Despite this, planner decisions and trajectory outcomes are identical to baseline.

## Key Finding

The scenario contains 1 pedestrian (position [20.0, 6.0] at step 0, closest approach ~14.9m). The perturbation infrastructure is working correctly: occlusion masking and missed detections are active. However, because the pedestrian is far from the robot and the planner's decisions are dominated by **static obstacle clearance** (778 static_clearance rejections, nearest obstacle ~1.84m), the observation noise has **no measurable effect** on planner behavior, progress, or risk outcomes.

This confirms the perturbation plumbing works on a pedestrian-present scenario but does **not** measure behavioral degradation, because the scenario's single distant pedestrian does not influence the planner's decisions.

## Commands

```bash
# Baseline
uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v0_minimal --stage stress_slice \
  --scenario-index 0 --horizon 40 \
  --output-dir output/validation/issue-2749-observation-noise-diagnostics/baseline

# Perception-limited
uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v0_minimal --stage stress_slice \
  --scenario-index 0 --horizon 40 \
  --observation-noise-std-m 0.1 --observation-noise-bound-m 0.2 \
  --missed-detection-probability 0.25 --occlusion-distance-m 3.0 \
  --observation-delay-steps 1 --observation-perturbation-seed 2749 \
  --output-dir output/validation/issue-2749-observation-noise-diagnostics/perception_limited
```

## Trace/Report Paths

- `output/validation/issue-2749-observation-noise-diagnostics/baseline/trace.json`
- `output/validation/issue-2749-observation-noise-diagnostics/baseline/report.md`
- `output/validation/issue-2749-observation-noise-diagnostics/perception_limited/trace.json`
- `output/validation/issue-2749-observation-noise-diagnostics/perception_limited/report.md`

## Limitations

1. **Single scenario, single seed** - not statistically meaningful; only classic_bottleneck_medium at seed 111
2. **Pedestrian too far to influence planner** - closest approach ~14.9m; planner decisions dominated by static obstacle clearance (~1.84m)
3. **Short horizon (40 steps)** - limited trajectory diversity; robot regressed (net -10.646) in both variants
4. **Only 1 pedestrian** - limited actor diversity for testing multi-agent perception degradation
5. **diagnostic_only** - confirms infrastructure works with a pedestrian-present scenario; does not measure degradation

## Summary JSON

See `summary.json` in the same directory.
