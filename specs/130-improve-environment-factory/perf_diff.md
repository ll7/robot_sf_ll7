---
title: Factory Creation Performance Diff (T031)
purpose: Documents tightening of performance guard from +10% to +5% and current vs baseline metrics.
updated: 2025-09-23
---

## Overview
Task T031 tightens the factory creation performance regression guard (FR-017) from a +10% mean allowance to +5%. This file captures the current measured metrics vs the original baseline stored in `results/factory_perf_baseline.json` and justifies the threshold reduction.

## Baseline (from results/factory_perf_baseline.json)
- Timestamp: 2025-09-23T07:11:50Z
- Iterations: 5

| Factory | Baseline Mean (ms) | Baseline p95 (ms) | Baseline Raw Samples (ms) |
|---------|--------------------|-------------------|---------------------------|
| make_robot_env | 218.33 | 760.27 | 940.75, 37.95, 37.31, 38.36, 37.30 |
| make_image_robot_env | 656.24 | 2397.37 | 2970.01, 106.78, 55.44, 93.77, 55.19 |

Baseline exhibits pronounced cold‑start outliers (first invocation cost) dominating p95 and inflating the mean.

## Current Measurement (post T029/T030, pre T031 commit)
Collected immediately before tightening threshold (5 iterations, identical methodology; fast demo disabled):

| Factory | Current Mean (ms) | Current p95 (ms) | Current Raw Samples (ms) |
|---------|-------------------|------------------|---------------------------|
| make_robot_env | 180.93 | 820.45 | 820.45, 20.72, 20.84, 21.98, 20.65 |
| make_image_robot_env | 169.58 | 611.37 | 611.37, 86.32, 37.87, 74.49, 37.85 |

## Delta Summary

| Factory | Mean Δ (ms) | Mean Δ % | p95 Δ (ms) | p95 Δ % |
|---------|-------------|----------|-----------|---------|
| make_robot_env | -37.41 | -17.13% | +60.18 | +7.92% |
| make_image_robot_env | -486.66 | -74.16% | -1785.99 | -74.50% |

Notes:
1. Robot env mean improved ~17%; single cold start still observed (820ms) slightly higher than baseline p95 (760ms) but irrelevant for mean guard.
2. Image env shows substantial improvement (>70% faster mean) and reduced tail latency.
3. Variability dominated by first initialization (import + JIT / asset load). Subsequent creations stable (<90ms).

## Guard Adjustment Rationale
- Previous hard threshold: +10% (1.10 multiplier) left unnecessary headroom given improvements.
- New hard threshold: +5% (1.05 multiplier) aligns with spec requirement and still tolerates minor environmental noise.
- Introduced narrower soft warning band: >+5% and <=+8% to surface creeping regressions early without hard fail.

## Test Update
`tests/perf/test_factory_creation_perf.py` constants changed:
```
THRESHOLD = 1.05
SOFT_THRESHOLD = 1.08
```
Error messages now explicitly state +5% ceiling.

## Action Items / Follow Ups
- (Optional) Add cold-start exclusion logic (discard first timing) if p95 volatility continues to distract diagnostics.
- Consider extending iterations (e.g., 10) in local developer baseline script while keeping CI iterations at 5 for speed.
- Rebaseline only if systematic improvements (e.g., asset caching) make current numbers persistently lower over multiple days.

## Definition of Done Check (T031)
- [x] Threshold tightened to +5%.
- [x] Diff documented with baseline vs current metrics and rationale.
- [x] Soft warning band adjusted.
- [x] Performance test expected to pass under new threshold (current means well below +5%).

---
End of perf diff.
