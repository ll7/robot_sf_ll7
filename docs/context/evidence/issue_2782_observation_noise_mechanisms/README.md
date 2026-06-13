# Observation-Noise Mechanism Classification

## Claim Boundary

**Diagnostic mechanism-layer classification only. Not paper-facing benchmark proof.**
This classifies each perturbation condition from the observation-noise envelope by the pipeline layer at which it had (or did not have) an effect: observation source, command, trajectory, or timing.
The label table is the valid vocabulary; the distribution section shows which labels were actually assigned for this evidence bundle.

## Reproducibility

- **Issue:** #2782
- **Generated at (UTC):** 2026-06-13T15:23:51.094473+00:00
- **Command:** `uv run python scripts/tools/classify_observation_noise_mechanisms.py --evidence docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13/summary.json --output-dir docs/context/evidence/issue_2782_observation_noise_mechanisms`
- **Repo HEAD:** `165444d1`
- **Source evidence:** `docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13/summary.json`
- **Fixture:** `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_episode_0000.json`
- **Scenario:** issue_2756_occluded_emergence
- **First visible step:** 5

## Mechanism Label Vocabulary

| Label | Pipeline layer | Meaning |
|---|---|---|
| `noise_stayed_below_decision_threshold` | Decision | Noise present but did not cross the decision boundary |
| `observation_affected_source_but_not_command` | Observation -> Command | Observation perturbed but command unchanged |
| `observation_did_not_affect_selected_source` | Observation | Perturbation did not reach the policy input |
| `command_changed_but_trajectory_did_not` | Command -> Trajectory | Command changed but trajectory unchanged |
| `delay_shifted_stop_timing` | Timing | Delay shifted stop/yield decision timing |
| `occlusion_changed_first_actionable_frame` | Observation timing | Occlusion changed when policy first saw the pedestrian |
| `scenario_had_no_actionable_conflict` | Scenario | No actionable conflict for noise testing |
| `stored_action_proxy_prevents_live_conclusion` | Proxy boundary | Action proxies from stored trace; live replay required |
| `diagnostic_only` | Reference | Baseline or mixed-effects reference |
| `inconclusive` | Fallback | Insufficient data for classification |

## Conditions

### noop

- **Mechanism label:** `diagnostic_only`
  - No-perturbation baseline; mechanism classification is not applicable to the reference trajectory.
- **Prior classification:** `diagnostic_only`
- **Noise profile:** none
- **Response delay:** 0 steps
- **Closest distance:** 0.5 m

### low_noise

- **Mechanism label:** `noise_stayed_below_decision_threshold`
  - Gaussian noise (std=0.10 m) applied but policy command sequence matches baseline. Noise stayed below the decision threshold for this planner/scenario combination.
- **Prior classification:** `diagnostic_only`
- **Noise profile:** bounded_gaussian
- **Response delay:** 0 steps
- **Closest distance:** 0.5 m

### medium_noise

- **Mechanism label:** `observation_affected_source_but_not_command`
  - Gaussian noise (std=0.30 m) perturbed the observation source but the policy command sequence matches baseline. The perturbation did not propagate to the command layer.
- **Prior classification:** `diagnostic_only`
- **Noise profile:** bounded_gaussian
- **Response delay:** 0 steps
- **Closest distance:** 0.5 m

### missed_detection_only

- **Mechanism label:** `occlusion_changed_first_actionable_frame`
  - Perturbation completely suppressed the pedestrian observation (missed detections or occlusion mask). The first actionable frame was changed or never reached.
- **Prior classification:** `scenario_too_weak`
- **Noise profile:** missed_detection
- **Closest distance:** 0.5 m

### occlusion_only

- **Mechanism label:** `occlusion_changed_first_actionable_frame`
  - Perturbation completely suppressed the pedestrian observation (missed detections or occlusion mask). The first actionable frame was changed or never reached.
- **Prior classification:** `scenario_too_weak`
- **Noise profile:** occlusion_mask
- **Closest distance:** 0.5 m

### delay_only

- **Mechanism label:** `delay_shifted_stop_timing`
  - Observation arrived 2 step(s) after first-visible (step 7). Stop/yield feasibility at first observation differs from baseline, indicating the delay shifted decision timing.
- **Prior classification:** `robustness_evidence`
- **Noise profile:** delayed_observation
- **Response delay:** 2 steps
- **Closest distance:** 0.5 m

### combined

- **Mechanism label:** `occlusion_changed_first_actionable_frame`
  - Perturbation completely suppressed the pedestrian observation (missed detections or occlusion mask). The first actionable frame was changed or never reached.
- **Prior classification:** `scenario_too_weak`
- **Noise profile:** bounded_gaussian
- **Closest distance:** 0.5 m

## Mechanism Distribution

| Mechanism label | Conditions | Count |
|---|---|---|
| `delay_shifted_stop_timing` | delay_only | 1 |
| `diagnostic_only` | noop | 1 |
| `noise_stayed_below_decision_threshold` | low_noise | 1 |
| `observation_affected_source_but_not_command` | medium_noise | 1 |
| `occlusion_changed_first_actionable_frame` | missed_detection_only, occlusion_only, combined | 3 |

## Caveats

- Single deterministic fixture (seed=111), single scenario family.
- Action proxies are from the stored trace, not re-executed; command -> trajectory conclusions require live replay.
- Mechanism labels are rule-based heuristics, not causal proof.
- Not paper-facing benchmark evidence.