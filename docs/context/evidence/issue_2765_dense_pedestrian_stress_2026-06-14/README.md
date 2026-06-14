# Issue #2765: Dense-Pedestrian-Stress Observation-Noise Envelope - 2026-06-14

## Claim Boundary

**Diagnostic trace-derived evidence only. Not paper-facing benchmark proof.** This evaluates near-field observation-noise robustness on a multi-pedestrian dense-stress trace fixture with 3 converging actors.

## Reproducibility

- **Issue:** #2765
- **Generated at (UTC):** 2026-06-14T07:12:43.816481+00:00
- **Command:** `uv run python scripts/benchmark/run_dense_stress_observation_envelope.py --output-dir docs/context/evidence/issue_2765_dense_pedestrian_stress_2026-06-14`
- **Repo HEAD:** `779a294e`
- **Fixture:** `tests/fixtures/analysis_workbench/simulation_trace_export_v1/dense_pedestrian_stress_episode_0000.json`
- **Scenario:** dense_pedestrian_stress
- **Pedestrian count:** 3

## Conditions

### noop

- **Description:** No perturbation applied (baseline).
- **Noise profile:** none
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `diagnostic_only`
  - No-perturbation baseline. Provides reference multi-pedestrian trajectory and action selection without observation noise.

### low_noise

- **Description:** Bounded Gaussian position noise std=0.10 m, bound=0.20 m on all actors.
- **Noise profile:** bounded_gaussian
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `forecast_ambiguity_detected`
  - Multiple constant-velocity forecast ellipses overlap (120 overlap events). Dense scene creates forecast ambiguity under observation perturbation.

### medium_noise

- **Description:** Bounded Gaussian position noise std=0.30 m, bound=0.60 m on all actors.
- **Noise profile:** bounded_gaussian
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `forecast_ambiguity_detected`
  - Multiple constant-velocity forecast ellipses overlap (120 overlap events). Dense scene creates forecast ambiguity under observation perturbation.

### high_noise

- **Description:** Bounded Gaussian position noise std=0.50 m, bound=1.00 m on all actors.
- **Noise profile:** bounded_gaussian
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `forecast_ambiguity_detected`
  - Multiple constant-velocity forecast ellipses overlap (120 overlap events). Dense scene creates forecast ambiguity under observation perturbation.

### partial_missed_detection

- **Description:** 50% missed detection probability on each actor independently.
- **Noise profile:** missed_detection
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `forecast_ambiguity_detected`
  - Multiple constant-velocity forecast ellipses overlap (120 overlap events). Dense scene creates forecast ambiguity under observation perturbation.

### full_missed_detection

- **Description:** 100% missed detection probability (all actors dropped).
- **Noise profile:** missed_detection
- **Pedestrians:** 3
- **First observed step:** None
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** None
- **Yield feasible (first observed):** None
- **Classification:** `scenario_too_weak`
  - All pedestrians suppressed by missed detection or occlusion mask; no observation signal reaches the policy.

### single_actor_occlusion

- **Description:** Occlude only the first actor (ped_a), others visible.
- **Noise profile:** occlusion_mask
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `forecast_ambiguity_detected`
  - Multiple constant-velocity forecast ellipses overlap (120 overlap events). Dense scene creates forecast ambiguity under observation perturbation.

### two_actor_occlusion

- **Description:** Occlude first two actors (ped_a, ped_b), third visible.
- **Noise profile:** occlusion_mask
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `forecast_ambiguity_detected`
  - Multiple constant-velocity forecast ellipses overlap (120 overlap events). Dense scene creates forecast ambiguity under observation perturbation.

### delay_2_steps

- **Description:** 2-step delayed observation for all actors.
- **Noise profile:** delayed_observation
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `forecast_ambiguity_detected`
  - Multiple constant-velocity forecast ellipses overlap (120 overlap events). Dense scene creates forecast ambiguity under observation perturbation.

### medium_noise_with_occlusion

- **Description:** Medium Gaussian noise (std=0.30 m) + occlusion of the first actor.
- **Noise profile:** bounded_gaussian
- **Pedestrians:** 3
- **First observed step:** 0
- **Closest distance:** 0.1562
- **Forecast overlap events:** 120
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `forecast_ambiguity_detected`
  - Multiple constant-velocity forecast ellipses overlap (120 overlap events). Dense scene creates forecast ambiguity under observation perturbation.

## Classification Legend

- **forecast_ambiguity_detected**: overlapping forecast ellipses from multiple actors.
- **robustness_evidence**: perturbation affected observation and may impact policy.
- **scenario_too_weak**: scenario too weak (all pedestrians suppressed or too distant).
- **policy_insensitive**: perturbation did not change policy action sequence.
- **diagnostic_only**: mixed effects; needs broader evidence.
- **inconclusive**: insufficient data for mechanism classification.

## Caveats

- Deterministic single-seed fixture (seed=2765), single scenario family.
- No live planner replay; action proxies are from the stored trace.
- Forecast ambiguity is computed from constant-velocity Gaussian baselines.
- Not paper-facing benchmark evidence.