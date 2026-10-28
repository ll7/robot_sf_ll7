# Post-hoc sensitivity: crashed robot removed instead of stopped

Not preregistered. Written after the main results were seen. The 400 traverse
episodes of `G0E1` and `orca` were rerun with `--failure-parking remove`: after
a collision, the robot leaves the scene, as a successful robot does. All other
rows are the main-run rows.

- **Measured robot-field agreement.** All 400 rerun episodes match their main-run
  counterparts on steps, termination, success, pedestrian and wall contacts, timeout,
  and closest robot-pedestrian distance. Full rows differ in commit and source hashes.
- **The guard's pedestrian-cost contrast changes sign.** G1E1 − G0E1, pooled,
  traverse, averaging cell/seed means with equal cell and seed weights:
  - Censored delay: +2.3 s [1.5, 3.1]. The main run gave −2.2 s.
  - Blocked share: +12.5 pp [9.8, 15.1]. The main run gave −18.3 pp.
  - Under the same rule text this would trigger
    `O2_success_improves_but_pedestrian_cost_rises`.
- **What this means.** The design cannot identify the guard's own pedestrian
  effect independent of parking. Guard-off robots usually fail by crashing early,
  so their effect on pedestrians depends on what a crashed robot does next. Both
  policy-specific effects are estimable; neither parking rule is a neutral counterfactual.
- **Robust to the rule.** The guarded arm never fails, so its absolute
  pedestrian cost against no robot is the same under both rules: 7.6 s in the
  cell/seed mean censored delay and 18.1% in the cell/seed mean blocked share,
  with equal cell and seed weight.
- **Descriptive check on both-success pairs** (`../posthoc_both_success.json`):
  on (cell, seed) pairs where both hybrid arms complete, both robots leave
  after completion, so the parking rule does not matter there. In the
  bottleneck cells (43 and 38 pairs), the guarded robot delays pedestrians
  slightly less in the cell/seed mean: −0.8 s and −1.9 s. The doorway cells have only 6 and 3 such
  pairs, too few to read. This subset is selected on outcome.
