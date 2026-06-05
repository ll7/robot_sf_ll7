# Issue #2224 Synthetic AMV Actuation Ranking Evidence

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2224>

This compact evidence bundle preserves the reviewable outputs from a bounded diagnostic comparison
between `hybrid_rule_v3_fast_progress` and `actuation_aware_hybrid_rule_v0` on
`amv_actuation_smoke`.

## Files

- [comparison.json](comparison.json): machine-readable two-candidate result plus baseline context.
- [comparison.md](comparison.md): rendered comparison table with synthetic actuation columns.
- [manifest.json](manifest.json): commands, local output paths, and claim boundary.

## Result

The actuation-aware candidate reduced command clipping from 0.2750 to 0.1875 on this one-episode
smoke slice, with zero collisions and zero yaw saturation for both candidates. Both candidates
still timed out with `timeout_low_progress`, so this is diagnostic direction only, not planner
ranking evidence.

