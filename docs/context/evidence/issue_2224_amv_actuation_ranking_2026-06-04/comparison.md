# Policy Search AMV Actuation Comparison

| Candidate | Stage | Success | Collision | Near Miss | Command Clip | Yaw Saturation | Signed Braking Peak | Failure Mode |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `hybrid_rule_v3_fast_progress` | `amv_actuation_smoke` | 0.0000 | 0.0000 | 0.0000 | 0.2750 | 0.0000 | -2.5000 | `timeout_low_progress: 1` |
| `actuation_aware_hybrid_rule_v0` | `amv_actuation_smoke` | 0.0000 | 0.0000 | 0.0000 | 0.1875 | 0.0000 | -2.5000 | `timeout_low_progress: 1` |

Classification: `diagnostic_direction_only`.

Interpretation: actuation-aware scoring reduced command clipping on this one-episode smoke slice,
but both candidates still timed out. Do not use this as planner-ranking or paper-facing AMV
evidence.

