# Issue #3479 — Internal-proxy rollover stability margin (diagnostic-only)

**Status:** diagnostic / internal-proxy. **Evidence grade:** idea-level proxy, not
hardware-calibrated and not paper-facing. **Governance:** honors the #2416 / #2417
gate — blocked from hardware-calibrated AMV profiles or AMV safety claims until
real-source provenance (#1585 / #2000) is accepted.

## What this is

`robot_sf/robot/rollover_proxy.py` implements the closed-form lateral-stability
("dynamic rollover") proxy proposed in issue #3479, so a narrow-track three-wheeled
platform's tip-over risk — invisible to planar bicycle / differential-drive /
holonomic models — can be surfaced as a per-step signal.

## Model (`rollover_proxy.v1`)

- lateral acceleration `a_y ≈ v · ω`
- critical lateral acceleration `a_y,crit = g · (t_w / (2 · h_c)) · (a / L)`
- stability margin `= clamp(1 − |a_y| / a_y,crit, 0, 1)` (1 = stable, 0 = at/over the
  proxy tip-over threshold)
- `ROLLOVER_CRITICAL` when the margin reaches 0.

### Versioned proxy geometry (`RolloverProxyParams`, non-hardware)

| symbol | field | default (m) | meaning |
| --- | --- | --- | --- |
| `t_w` | `track_width_m` | 0.80 | lateral wheel track |
| `h_c` | `cog_height_m` | 0.60 | centre-of-gravity height |
| `a` | `front_axle_to_cog_m` | 0.50 | front axle → CoG |
| `L` | `wheelbase_m` | 1.20 | wheelbase |
| `g` | `gravity_m_s2` | 9.81 | gravity |

These defaults are **aligned with the benchmark-surface source of truth**
`robot_sf.benchmark.metrics.evaluate_stability_margin` (the reviewer-supplied TWV proxy
`t_w=0.8`, `L=1.2`, `h_c=0.6`, `a=0.5`) so the runtime diagnostic and the benchmark column
`rollover_min_stability_margin` cannot diverge (issue #3587); a parametrized test cross-checks
that the two produce identical margins. They carry **no hardware authority**. For the defaults,
`a_y,crit ≈ 2.73 m/s²`.

## Scope boundary

This increment is **diagnostic-only and side-effect free**: it computes the margin,
the `ROLLOVER_CRITICAL` classifier, and a schema-tagged telemetry record, and changes
**no** planner, training, or benchmark behavior. Wiring the terminal flag + an opt-in
reward penalty into the stepping loop (and any benchmark comparison) is a deliberate
follow-up so existing runs are not silently altered.

## Tests

`tests/test_rollover_proxy.py` proves a feasible command stays stable, an over-yaw
command trips `ROLLOVER_CRITICAL`, the margin is clamped/monotonic/sign-independent,
non-physical geometry fails closed, and telemetry is schema-tagged and labelled
`internal_non_hardware`.

## Related

- #3466 — three-wheeled tip-over feasibility on the *benchmark-artifact* surface.
- #2416 / #2417 — AMV governance gates (blocking hardware-calibrated profiles).
- #1585 / #2000 — real-source provenance that would unblock hardware-facing claims.
