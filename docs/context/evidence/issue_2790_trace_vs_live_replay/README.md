# Issue #2790 Trace vs Live Replay Comparison Report

- **Status**: completed
- **Evidence Grade**: `analysis_only`
- **Prefilter Trustworthy**: `False`
- **Recommended Action**: `demote_to_debugging_only`
- **Decision Verdict**: Trace-derived diagnostics failed to predict live replay outcomes correctly (delay_only was a false positive).

## Decision Rule Context

- *Decision Rule*: If trace-derived diagnostics predict live replay ranking correctly, keep them as a cheap prefilter. If they do not, demote trace-derived artifacts to debugging-only evidence.
- *Action Outcome*: Since trace-derived delay sensitivity did not reproduce in the live replay DWA wrapper run (resulting in a `false_positive` for `delay_only`), the trace-derived envelope artifacts are demoted to debugging-only evidence and must not be used as a cheap prefilter.

## Comparison Table

| Condition | Trace Label | Live Cmd Changed | Live Prog Changed | Comparison Label | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `noop` | `diagnostic_only` | `False` | `False` | **`confirmed`** | Trace and live replay agree on no actionable sensitivity / no-effect. |
| `low_noise` | `diagnostic_only` | `False` | `False` | **`confirmed`** | Trace and live replay agree on no actionable sensitivity / no-effect. |
| `medium_noise` | `diagnostic_only` | `False` | `False` | **`confirmed`** | Trace and live replay agree on no actionable sensitivity / no-effect. |
| `missed_detection_only` | `scenario_too_weak` | `False` | `False` | **`confirmed`** | Trace and live replay agree on no actionable sensitivity / no-effect. |
| `occlusion_only` | `scenario_too_weak` | `False` | `False` | **`confirmed`** | Trace and live replay agree on no actionable sensitivity / no-effect. |
| `delay_only` | `robustness_evidence` | `False` | `False` | **`false_positive`** | Trace predicted delay sensitivity but live replay showed no command/progress changes. |
| `combined` | `scenario_too_weak` | `False` | `False` | **`confirmed`** | Trace and live replay agree on no actionable sensitivity / no-effect. |

## Trace Condition Details

### `noop`
- **Trace Prediction**: `diagnostic_only`
  - *Rationale*: No-perturbation baseline. Provides reference robot-pedestrian trajectory and action selection without observation noise.
- **Live Replay Outcome**: `unknown`
  - *Rationale*: 

### `low_noise`
- **Trace Prediction**: `diagnostic_only`
  - *Rationale*: Perturbation produced mixed effects. Classified as diagnostic-only pending broader seed/scenario evidence.
- **Live Replay Outcome**: `policy_insensitive`
  - *Rationale*: Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical.

### `medium_noise`
- **Trace Prediction**: `diagnostic_only`
  - *Rationale*: Perturbation produced mixed effects. Classified as diagnostic-only pending broader seed/scenario evidence.
- **Live Replay Outcome**: `policy_insensitive`
  - *Rationale*: Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical.

### `missed_detection_only`
- **Trace Prediction**: `scenario_too_weak`
  - *Rationale*: Pedestrian fully missed after fixture visibility begins; no observation signal reaches the policy. Cannot test policy robustness.
- **Live Replay Outcome**: `policy_insensitive`
  - *Rationale*: Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical.

### `occlusion_only`
- **Trace Prediction**: `scenario_too_weak`
  - *Rationale*: Pedestrian position/velocity zeroed by occlusion after fixture visibility begins; no observation signal reaches the policy.
- **Live Replay Outcome**: `policy_insensitive`
  - *Rationale*: Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical.

### `delay_only`
- **Trace Prediction**: `robustness_evidence`
  - *Rationale*: Pedestrian observed at step 7 (2 steps after first-visible). Perturbation delayed/masked the observation and may have affected policy response timing.
- **Live Replay Outcome**: `policy_insensitive`
  - *Rationale*: Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical.

### `combined`
- **Trace Prediction**: `scenario_too_weak`
  - *Rationale*: Pedestrian position/velocity zeroed by occlusion after fixture visibility begins; no observation signal reaches the policy.
- **Live Replay Outcome**: `policy_insensitive`
  - *Rationale*: Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical.

## Claim Boundary & Preservation

This comparison preserves the repository's fail-closed evidence discipline. Agreement or disagreement on a single scenario/seed must not be treated as broad experimental validity. Trace-derived agreement does not replace live replay for benchmark claims.