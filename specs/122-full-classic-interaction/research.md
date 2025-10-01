# Phase 0 Research: Full Classic Interaction Benchmark

Purpose: Resolve open clarifications, define statistical and operational methodology, and lock decisions feeding Phase 1 design.

## 1. Open Clarifications & Resolutions

| Topic | Question | Decision | Rationale | Alternatives Considered |
|-------|----------|----------|-----------|-------------------------|
| Reference Hardware | What is the baseline machine for runtime targets? | 8‑core (performance) Apple M1/M2/M3 class or 8 vCPU Linux VM; 16GB RAM. | Representative of developer laptops & moderate CI runners; aligns with current perf notes. | High-end workstation (overestimates perf); 2‑core CI runners (too restrictive). |
| Minimum Episodes / Scenario | How many episodes needed to reach CI half‑width thresholds? | Start with 150 episodes per (archetype,density). Adaptive: after every +30 episodes recompute CI; stop early if all primary metrics meet thresholds or cap at 250. | Empirical rule-of-thumb for binomial (collision/success) with p in [0.05,0.3] gives half‑width ≈ 1.96*sqrt(p(1-p)/n) < 0.02 when n≥~230; early stop saves time. | Fixed n=200 (less adaptive); sequential SPRT (more complex). |
| Effect Size Definitions | Which effect sizes to report? | For rates: absolute difference (p_high - p_low) + Cohen's h; for means: difference of means Δμ and standardized (Glass Δ using low density SD). | Provides intuitive absolute diff plus a standardized form; avoids overcomplicating with Hedge's g for small n. | Only raw differences (less interpretability); purely standardized (less tangible). |
| Video Annotation Elements | What overlays are required? | Robot path (trail), current positions (robot + pedestrians), collision events (red flash marker), success/goal (green marker at end), stalled/timeouts (yellow border), timestep counter, scenario id text. | Minimal yet informative; keeps rendering cost low. | Adding force vectors (clutter); per‑pedestrian velocity arrows (visual overload). |
| Scaling Acceptance (NFR-003) | How to quantify near-linear scaling? | Define efficiency E(k)=T(1)/(k*T(k)). Accept >=0.8 for k ≤ 8. Report in sufficiency report. | Simple metric widely used; easy to compute from wall-clock durations. | Speedup only (less normalized). |
| Bootstrap Parameters | How many samples for CIs? | Default 1000 bootstrap samples; seed settable; can downgrade to 300 in smoke mode. | Balance accuracy vs runtime. | 10k samples (too slow); 100 (unstable tails). |
| CI Computation for Zero Events | How to handle zero collision counts? | Use Wilson score interval for binomial metrics instead of normal approximation. | Provides non-zero upper bound; standard for low counts. | Jeffreys interval (similar results); Clopper-Pearson (conservative). |
| Early Stop Criteria | When to stop adaptive sampling? | If all tracked metrics meet thresholds for two consecutive evaluation windows (hysteresis) OR cap reached. | Prevents stopping on transient lucky batch. | Single check (risk of premature stop). |

## 2. Statistical Methodology

### 2.1 Metrics Categories
- Binary rates: success_rate, collision_rate.
- Continuous: time_to_goal, path_efficiency, average_speed, snqi.

### 2.2 Confidence Intervals
- Binary: Wilson interval (p ± range) at 95%.
- Continuous: Nonparametric bootstrap of mean, median, p95 (1000 resamples; smoke=300). Store: mean, median, p95, and CI for mean & median.
- SNQI: treat like continuous.

### 2.3 Effect Sizes
- Rate difference: Δp = p_high - p_low; Cohen's h = 2*arcsin(√p_high) - 2*arcsin(√p_low).
- Means: Δμ = μ_high - μ_low; Glass Δ = (μ_high - μ_low)/σ_low.

### 2.4 Precision Thresholds (Default)
| Metric | Half‑Width Target |
|--------|------------------|
| collision_rate | ≤ 0.02 |
| success_rate | ≤ 0.03 |
| time_to_goal_mean | ≤ 5% of mean |
| path_efficiency_mean | ≤ 5% of mean |
| snqi_mean | ≤ 0.05 absolute |

Time-to-goal/path efficiency relative width: (CI_high - CI_low)/(2*mean).

### 2.5 Adaptive Sampling Loop
1. Start n0=150.
2. Run episodes in batches of 30 (per scenario) until thresholds met or n>=250.
3. After each batch: recompute metrics & CIs; if all pass for two successive checks → early stop.
4. Record per-scenario sampling log for transparency.

### 2.6 Resume Semantics
- Episode id = hash(archetype,density,seed,horizon,algo,config_version).
- Manifest (JSON) caches ids and counts; mismatch in scenario matrix hash triggers WARNING and requires explicit --force-continue flag to use old episodes.

### 2.7 Randomness Control
- Master seed controls seed sequence generation per scenario; deterministic ordering to support resume.

## 3. Operational Methodology

### 3.1 Directory Layout
```
results/classic_full_<timestamp>/
  manifest.json
  config.json
  episodes/episodes.jsonl
  aggregates/summary.json
  reports/statistical_sufficiency.json
  reports/effect_sizes.json
  plots/*.pdf (and .png)
  videos/*.mp4
  logs/benchmark.log
```

### 3.2 Plot Set
- distributions_<metric>.pdf (per group)
- trajectories_<archetype>.pdf
- kde_positions_<archetype>_<density>.pdf
- pareto_snqi_time_<archetype>.pdf (example trade-off)
- force_heatmap_<archetype>_<density>.pdf (if force data available; else skipped with note)

### 3.3 Video Generation
- Select representative episode per archetype: highest SNQI among successful episodes (fallback: median time_to_goal).
- Render to MP4 30fps; embed overlay annotations.

### 3.4 Smoke Mode Adjustments
| Aspect | Full | Smoke |
|--------|------|-------|
| episodes/start | 150 | 5 |
| batch size | 30 | 5 |
| max episodes | 250 | 10 |
| bootstrap samples | 1000 | 300 |
| plots | full | minimal (only distributions + one trajectory) |
| videos | full set | skipped (flagged) |

## 4. Risk Assessment & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Runtime over 4h | Delays nightly pipeline | Adaptive early stop + parallel workers tuned; scaling report to adjust workers. |
| Zero collision scenarios produce misleading 0 CI | False safety impression | Wilson interval; explicit note in report. |
| Video dependency (ffmpeg) missing | Partial artifact failure | Catch exception; mark video status skipped in report; continue. |
| Large JSONL size | Storage strain | Compressed optional copy (future); current size acceptable (<200MB). |
| Bootstrap runtime heavy | Slow aggregation | Vectorized numpy operations; allow lower sample override. |
| Resume after matrix change | Corrupted stats | Hash comparison + warning + require --force-continue. |

## 5. Decisions Summary
- Adaptive sampling with Wilson + bootstrap chosen for balance of rigor & runtime.
- Dual effect size approach (absolute + standardized) for interpretability.
- Representative video selection based on SNQI success episodes.
- Directory layout standardized with subfolders for clarity & CI checks.
- Smoke mode drastically reduced but structural mirrors full.

## 6. Deferred / Future Considerations
- Bayesian credible intervals (optional enhancement) – deferred.
- Parallel plot generation multiprocessing – if plotting becomes bottleneck.
- Compressed episodes sidecar (.zst) – future storage optimization.

## 7. All Unknowns Resolved
No remaining NEEDS CLARIFICATION markers after this document.
