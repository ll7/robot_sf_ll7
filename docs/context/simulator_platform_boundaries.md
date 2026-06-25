# Simulator-platform boundaries and adapter evidence needs (#3579)

**Status:** repository-facing research-planning note. **Purpose:** make explicit what
`robot_sf_ll7` is designed to validate **directly**, and what would require adapters,
external simulator support, or separate evidence before a benchmark claim is defensible.
This note exists to prevent overclaiming — it does **not** assert that any external-platform
support exists before an adapter or evidence artifact is available.

## What `robot_sf_ll7` validates natively

`robot_sf_ll7` is a social-force / AMV social-navigation benchmark stack. Native, directly
producible evidence:

- scenario configurations and pedestrian/robot trajectories (social-force pedestrians, with
  reactive and — where wired — replay/open-loop modes);
- per-episode safety/quality metrics: collision, near-miss, minimum clearance/separation,
  TTC-style breaches, progress/efficiency, comfort, and oscillation;
- planner comparisons over scenario families, with the event ledger
  (`EpisodeEventLedger.v1`) as the per-episode source of truth and fail-closed handling of
  fallback/degraded/not-available rows;
- internal-proxy kinematic/perception diagnostics (e.g. rollover proxy, MOTP/speed contract,
  trace-level safety predicates) that are explicitly **diagnostic / non-hardware** until
  durable evidence.

These are simulation-only by construction: they are not real-world pedestrian-realism or
hardware-safety claims.

## Platform-family boundary table

For each family: the evidence it would add, the claims it does **not** license on its own,
the likely adapter/interface need, and the minimum artifact required before a benchmark claim
strengthens.

| Platform family | Supported evidence (role) | Unsupported on its own | Likely adapter / interface need | Minimum artifact before stronger claims |
| --- | --- | --- | --- | --- |
| **`robot_sf_ll7`** native social-force / AMV stack | Social-navigation planner comparison, safety/clearance/progress metrics, scenario-family coverage, diagnostic proxies | Real-world pedestrian realism; hardware tip-over / actuation limits; sensor-realism | — (native) | Versioned configs + seeds + event-ledger provenance; predeclared metric definitions |
| **CARLA-style** sensor/urban-scene sim | Sensor pipeline, perception under realistic rendering, urban scene geometry | Social-force pedestrian-interaction claims; native social metrics without re-derivation | A scene/agent bridge mapping CARLA actors ↔ robot_sf scenarios + metric re-derivation (a `robot_sf_carla_bridge` exists as availability-gated client glue, not a validated parity path) | Documented parity / claim-boundary report for any cross-sim metric; provenance for the rendered conditions |
| **SUMO-style** traffic-flow sim | Macroscopic vehicle/traffic flow, network-level routing | Pedestrian-rich close-interaction safety; per-episode clearance/near-miss | A flow→agent adapter; mismatch in interaction granularity must be stated | A defined mapping from flow outputs to per-episode safety metrics, with caveats |
| **Vadere / crowd-dynamics** sim | Calibrated pedestrian-dynamics models, crowd flow/fundamental-diagram behavior | Robot-in-the-loop planner evaluation; reactive robot-pedestrian coupling | A pedestrian-model import or parameter-transfer adapter; identifiability caveats (social-force params are coupled/weakly identifiable) | A documented parameter-provenance + realized-distribution audit before treating priors as validated |
| **Navground / SocNavBench / HuNavSim** social-nav benchmarks | Cross-benchmark interoperability, grounded-replay (recorded) trajectories, shared protocols | Reactive-pedestrian claims from replay benchmarks (replay pedestrians do not respond to the robot); ranking transfer without a reactivity analysis | Scenario/metric converters (Robot SF ↔ external format) + a reactivity validity boundary | A scenario-format converter + a quantified reactive-vs-replay validity-boundary report |

## How to use this note

- Before promoting any cross-platform result, confirm the row's "minimum artifact" exists and
  is tracked; absent that, label the result diagnostic-only or out-of-scope.
- Keep external-platform support claims gated on a concrete adapter or evidence artifact —
  never assume support exists because a family appears in this table.

## Related issues

- Simulation-fidelity sensitivity / simulator-dependence validity boundary: #3207.
- Cross-benchmark policy comparison across Robot SF and external social-nav suites: #3287.
- Robot SF ↔ SocNavBench / HuNavSim scenario converters (interop): #3285.
- Sim↔real evidence integration: #3293.
- Reactive-vs-replay pedestrian-reactivity validity boundary: #3573.
- Real-trajectory ingestion / dataset staging contract: #3065.
- Benchmark-governance / fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`.
