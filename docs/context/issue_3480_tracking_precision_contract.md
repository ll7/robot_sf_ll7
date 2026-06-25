# Issue #3480 — Tracking-precision drift mask + speed contract (diagnostic-only)

**Status:** diagnostic / internal-proxy. **Evidence grade:** idea-level proxy, not a
measured sensor-noise model and not paper-facing. Reuse durable runtime evidence and
the observation-noise portfolio (#2777 / #2927) before any perception-robustness claim.

## What this is

`robot_sf/benchmark/tracking_precision_contract.py` implements the tracking-precision
lever from issue #3480 so the "perfect sensing" assumption (uncorrupted raycast truth,
zero tracking decay) can be stressed:

- a **tracking-drift mask** that perturbs tracked actor coordinates by a target
  MOTP-like precision, and
- a precision→speed **operational contract** that drops the planner max-speed cap to a
  defensive ceiling once tracking precision degrades past a threshold `T_u`.

## Model (`tracking_precision_contract.v1`)

- MOTP is the mean Euclidean tracking error. For an isotropic 2D Gaussian with per-axis
  std `σ`, the mean Euclidean error is `σ·√(π/2)` (Rayleigh mean), so a target
  `MOTP = m` uses `σ = m / √(π/2)`. **MOTP = 0 ⇒ exact pass-through.**
- Minimum-separation safety is computed on the **corrupted** observation vector, not on
  ground truth.
- Operational contract: `speed_cap = defensive_speed_cap` when `MOTP ≥ T_u`, else
  `default_speed_cap`. `is_contract_honored` checks an applied cap respects the ceiling.

### Versioned contract defaults (non-hardware proxy)

| field | default | meaning |
| --- | --- | --- |
| `precision_threshold_m` (`T_u`) | 2.5 m | MOTP at/above which the defensive cap applies |
| `default_speed_cap` | 2.0 m/s | cap under good tracking precision |
| `defensive_speed_cap` | 0.5 m/s | cap under degraded precision |

## Scope boundary

**Diagnostic-only and side-effect free.** It does not alter the live perception, action,
or benchmark loop. Wiring the drift mask and speed-cap into the stepping loop (and any
benchmark safety-vs-precision sweep) is a deliberate opt-in follow-up so existing runs
are not silently altered. The Gaussian drift is a **proxy**, not a hardware sensor model.

## Tests

`tests/benchmark/test_tracking_precision_contract.py` proves MOTP=0 pass-through, that a
large drifted sample's mean error matches the target MOTP (Rayleigh mapping), seed
reproducibility, minimum-separation on the observed vector, the speed-cap threshold
behavior, honored/violated contract checks, parameter validation, and the telemetry
schema.

## Related

- #3300 — false-positive actor-injection replay (related, distinct).
- #2777 / #2927 — observation-noise live replay / portfolio.
- #3293 — sim↔real evidence integration.
