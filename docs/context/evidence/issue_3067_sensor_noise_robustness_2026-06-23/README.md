# Issue #3067: Sensor-Noise Robustness Slice (clean / noisy / partial)

**Claim boundary:** `diagnostic_only`. **Evidence tier:** `smoke`. **Paper-grade:** `false`.

This is a bounded, trace-derived observation-robustness slice. It is **NOT** a
real-sensor certification and **NOT** a sim-to-real transfer claim. Perturbations
are non-calibrated benchmark robustness noise applied to observed actor state, not
a hardware sensor model. Robustness is interpreted **separately** from nominal
(stored-trace) policy performance.

## Reproducibility

- **Command:**
  `uv run python scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py --output-dir output/issue_3067_sensor_noise/run`
- **Seed:** 3067 (same seed across all rows)
- **Repo HEAD:** `cbec866b` (worktree branch `issue-3067-sensor-noise-robustness`)
- **Fixture:** `tests/fixtures/analysis_workbench/simulation_trace_export_v1/dense_pedestrian_stress_episode_0000.json`
  (pedestrian-dominated, 3 converging actors, 20 frames, scenario `dense_pedestrian_stress`)
- **Generated:** 2026-06-23

## Matrix (clean / noisy / partial)

| row | family | observation level | status |
| --- | --- | --- | --- |
| clean | clean | oracle_full_state | ok |
| noisy_low | noisy | tracked_agents_with_noise (std 0.10 m / bound 0.20 m) | ok |
| noisy_medium | noisy | tracked_agents_with_noise (std 0.30 m / bound 0.60 m) | ok |
| partial_occlusion | partial | occluded_partial_state (occlude nearest actor) | ok |
| partial_missed_detection | partial | occluded_partial_state (50% missed detection) | ok |

## Headline clean-vs-perturbed deltas

Clean reference: `min_observed_distance_m=0.1562`, `observation_continuity=1.0`,
`near_field_exposure_frames=20`, `total_observed_actor_observations=60`.

| row | d(min_obs_dist_m) | d(continuity) | d(near_field_frames) | d(obs_count) |
| --- | --- | --- | --- | --- |
| noisy_low | +0.0055 | 0.0 | 0 | 0 |
| noisy_medium | +0.0492 | 0.0 | 0 | 0 |
| partial_occlusion | +0.2212 | 0.0 | 0 | -20 |
| partial_missed_detection | +0.1465 | -0.15 | -5 | -36 |

**Non-null result:** observation perturbations measurably change the observed state
the policy would consume. Bounded Gaussian noise shifts the nearest-observed
distance; partial observation (occlusion / missed detection) drops actor
observations, reduces observation continuity, and removes near-field exposure
frames.

## Overall classification: `diagnostic`

Perturbations ran on a same-seed pedestrian-dominated fixture and moved
observed-state metrics beyond the clean reference (non-null deltas). This is
diagnostic-only: single fixture, single seed, trace-derived (no live planner
replay). Not a benchmark, not a certification, not a sim-to-real claim.

## Fail-closed behavior

- Incomplete per-row perturbation metadata -> overall `blocked`.
- All perturbed rows degraded (all actors suppressed) -> overall `non-claim`.
- Degraded / invalid rows are never counted as success.
- Vocabulary is stable: overall ∈ {benchmark, diagnostic, blocked, non-claim};
  row status ∈ {ok, degraded, invalid, not-available}.

## Caveats

- Single deterministic pedestrian-dominated fixture, single seed.
- Metrics are observed-state proxies, not re-executed planner outcomes.
- Nominal performance (stored-trace behavior on the clean observation) is kept
  distinct from robustness (how the observed state changes under perturbation).
