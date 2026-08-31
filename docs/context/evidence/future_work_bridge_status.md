<!-- AI-GENERATED (#8048) - NEEDS-REVIEW -->
# Future-Work Bridge Status Summary

<!-- schema: future_work_bridge_summary.v1 -->

This summary documents the current engineering capability and empirical evidence distance for four future-work bridge directions in Robot SF. Implementation progress reduces engineering distance but does not itself close the dissertation empirical evidence gap.

## Bridge Status Matrix

| Bridge | Implemented Surface | Evidence Status | Missing Decisive Proof | Strongest Safe Interpretation |
| --- | --- | --- | --- | --- |
| [`carla_cross_simulator_bridge`](future_work_cards/carla_cross_simulator_bridge.v1.json) | • Pinned CARLA client/server connector with versioned packaging.<br>• Standalone headless CARLA runtime checks and replay harness entry points. | `diagnostic_only` (diagnostic_only) | • Matched actor-complete cross-simulator scenario replay between Robot SF and CARLA.<br>• Coordinate, temporal, and action mapping formal equivalence proof. | A pinned CARLA live-replay prototype exists and has demonstrated client/server connection plus bounded replay handling; matched actor-complete replay, metric parity, and cross-simulator validation remain unestablished. |
| [`route_choice_homotopy_observability`](future_work_cards/route_choice_homotopy_observability.v1.json) | • Deterministic classification of route side (left/right passing) and topological homotopy consistency.<br>• Static topological feature extractors for planned trajectories on SVG and grid maps. | `synthetic_fixture` (diagnostic_only) | • Human behavioral ground-truth or preference datasets validating route observability.<br>• Empirical proof that visible topological features improve human trajectory prediction or perceived social comfort. | The repository can deterministically classify route side and homotopy consistency on synthetic fixtures; whether those observables improve human predictability or social acceptance remains unevaluated. |
| [`incident_to_scenario_provenance`](future_work_cards/incident_to_scenario_provenance.v1.json) | • Fail-closed schema distinguishing source facts, extracted hypotheses, simulator assumptions, and replay identity.<br>• Deterministic checksum-covered scenario generation from structured incident descriptors. | `synthetic_fixture` (diagnostic_only) | • Ingestion and validation of real-world public transportation or robot collision incident reports.<br>• Human-audited extraction accuracy and representativeness bounds. | A fail-closed provenance contract can distinguish source facts, extracted hypotheses, simulator assumptions, and replay identity for a synthetic incident fixture; real-report validity and representativeness remain future work. |
| [`amv_actuation_realism_bridge`](future_work_cards/amv_actuation_realism_bridge.v1.json) | • Bounded 2D unicycle and differential-drive kinematic models with acceleration, velocity, and jerk limits.<br>• Literature-backed longitudinal e-scooter proxy acceleration and deceleration profiles. | `unsupported_proxy` (diagnostic_only) | • Physical vehicle platform system identification (measured command-to-motion latency, motor response curves).<br>• Rotational dynamics, tire slip, terrain-dependent friction, and non-holonomic yaw inertia. | Public longitudinal e-scooter evidence provides a bounded proxy-source basis, while platform-specific yaw, latency, dynamics, and physical calibration remain absent. |

## Claim Boundary & Caution

> [!IMPORTANT]
> None of the future-work bridges documented here have established physical transfer, human preference 
> validation, legal fault attribution, or unconstrained benchmark admission. 
> All evidence is currently diagnostic-only, proxy-based, or synthetic-fixture-only.

## Versioned Card Manifest

- **CARLA Cross-Simulator Bridge** (`carla_cross_simulator_bridge`): card digest `84d9f66c6abbb54a...`, relationship: `introduced_after_anchor`
- **Route Choice and Homotopy Observability** (`route_choice_homotopy_observability`): card digest `df156582547b0b39...`, relationship: `introduced_after_anchor`
- **Incident-to-Scenario Provenance** (`incident_to_scenario_provenance`): card digest `8f5eb53c7a0994e6...`, relationship: `introduced_after_anchor`
- **AMV Actuation Realism Bridge** (`amv_actuation_realism_bridge`): card digest `27a398b3f4158993...`, relationship: `present_at_anchor`
