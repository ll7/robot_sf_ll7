# Observation-Noise Robustness Envelope

## Claim Boundary

**Diagnostic trace-derived evidence only. Not paper-facing benchmark proof.**
This evaluates near-field observation-noise robustness on a single occluded-emergence trace fixture. Results are diagnostic, not statistically powered population claims.

## Reproducibility

- **Issue:** #2755
- **Generated at (UTC):** 2026-06-13T11:14:14.038693+00:00
- **Command:** `uv run python scripts/benchmark/run_observation_noise_envelope.py --output-dir docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13`
- **Repo HEAD:** `2aa3c468`
- **Fixture:** `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_episode_0000.json`
- **Scenario:** issue_2756_occluded_emergence
- **First visible step:** 5

## Conditions

### noop

- **Description:** No perturbation applied (baseline).
- **Noise profile:** none
- **First observed step:** 5
- **Response delay:** 0 steps from first-visible
- **Closest distance:** 0.5
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `diagnostic_only`
  - No-perturbation baseline. Provides reference robot-pedestrian trajectory and action selection without observation noise.

### low_noise

- **Description:** Bounded Gaussian position noise with std=0.10 m, bound=0.20 m.
- **Noise profile:** bounded_gaussian
- **First observed step:** 5
- **Response delay:** 0 steps from first-visible
- **Closest distance:** 0.5
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `diagnostic_only`
  - Perturbation produced mixed effects. Classified as diagnostic-only pending broader seed/scenario evidence.

### medium_noise

- **Description:** Bounded Gaussian position noise with std=0.30 m, bound=0.60 m.
- **Noise profile:** bounded_gaussian
- **First observed step:** 5
- **Response delay:** 0 steps from first-visible
- **Closest distance:** 0.5
- **Stop feasible (first observed):** True
- **Yield feasible (first observed):** True
- **Classification:** `diagnostic_only`
  - Perturbation produced mixed effects. Classified as diagnostic-only pending broader seed/scenario evidence.

### missed_detection_only

- **Description:** Single pedestrian fully missed (probability=1.0).
- **Noise profile:** missed_detection
- **First observed step:** None
- **Closest distance:** 0.5
- **Stop feasible (first observed):** None
- **Yield feasible (first observed):** None
- **Classification:** `scenario_too_weak`
  - Pedestrian fully missed after fixture visibility begins; no observation signal reaches the policy. Cannot test policy robustness.

### occlusion_only

- **Description:** Single pedestrian position/velocity zeroed by occlusion mask.
- **Noise profile:** occlusion_mask
- **First observed step:** None
- **Closest distance:** 0.5
- **Stop feasible (first observed):** None
- **Yield feasible (first observed):** None
- **Classification:** `scenario_too_weak`
  - Pedestrian position/velocity zeroed by occlusion after fixture visibility begins; no observation signal reaches the policy.

### delay_only

- **Description:** 2-step delayed observation for the single pedestrian.
- **Noise profile:** delayed_observation
- **First observed step:** 7
- **Response delay:** 2 steps from first-visible
- **Closest distance:** 0.5
- **Stop feasible (first observed):** False
- **Yield feasible (first observed):** True
- **Classification:** `robustness_evidence`
  - Pedestrian observed at step 7 (2 steps after first-visible). Perturbation delayed/masked the observation and may have affected policy response timing.

### combined

- **Description:** Medium Gaussian noise + occlusion mask on the pedestrian.
- **Noise profile:** bounded_gaussian
- **First observed step:** None
- **Closest distance:** 0.5
- **Stop feasible (first observed):** None
- **Yield feasible (first observed):** None
- **Classification:** `scenario_too_weak`
  - Pedestrian position/velocity zeroed by occlusion after fixture visibility begins; no observation signal reaches the policy.

## Classification Legend

- **robustness_evidence**: perturbation affected observation and may impact policy.
- **scenario_too_weak**: scenario too weak (pedestrian never observed or too distant).
- **policy_insensitive**: perturbation did not change policy action sequence.
- **diagnostic_only**: mixed effects; needs broader evidence.
- **blocked**: not applicable for this trace-derived evaluation.

## Caveats

- Single deterministic fixture (seed=111), single scenario family.
- No live planner replay; action proxies are from the stored trace, not re-executed.
- Stop/yield feasibility is from fixture metadata, not re-derived.
- Not paper-facing benchmark evidence.