# Goal

Record measured robot speed and spawn context for 0.0.8 collision events so contact
classes can be audited without changing the historical collision count or event ledger.

# Scope

- Opt in with `collision_attribution_version: v1` in the scenario identity. Default
  scenarios retain their prior output bytes.
- Preserve reset overlap as a separate setup verdict. A first-step pedestrian
  collision with the same reset-overlapping row is `reset_overlap_contact`.
  Other pedestrian step collisions distinguish a matched respawn contact,
  stationary robot contact, moving robot contact, and unresolved speed.
  Preserve exact collision counts.
- No pedestrian force-law change or planner-caused collision rate in this packet.

# Evidence sources

- Issue #9729 and the #9668 0.0.8 correction ruling.
- `map_runner_episode.py` step collisions, `spawn_validity.py` matched respawn
  collisions, `event_ledger.py` canonical ledger and reconciliation.

# Steps

1. Add opt-in speed capture at the runtime step and a versioned ledger provenance
   block with explicit speed threshold and measurement source.
2. Keep unknown/non-finite speed unresolved; require internally consistent
   attribution fields in ledger reconciliation.
3. Test default output, stationary/moving/respawn/reset/missing-speed cases, then
   inspect native traces where feasible.

# Decisions and risks

- `0.05 m/s` is a declared diagnostic stationary threshold, not a causal rule.
  Measured speed is displacement over the collision simulation step, not an
  instantaneous velocity at the exact collision instant. The saved collision
  event carries the source speed and step index; the provenance block must
  agree with it. A release gate must remeasure speed from the retained trace
  and bind the row to source/config and artifact checksums, because jointly
  altered copies within a row cannot be detected by ledger reconciliation.
- Respawn matching inherits the existing timed pedestrian-row match; a match
  remains diagnostic until simulator-level reinjection and release comparison pass.
- A grouped event with multiple pedestrian partners stays
  `unresolved_multi_pedestrian_attribution` if any partner matches reset overlap
  or respawn evidence. A single matched partner cannot classify the whole event.
- An observed collision is never relabeled as planner-caused by this block.

# Validation route

- Focused event-ledger and map-runner tests; Ruff check and format; exact-head
  self-review and independent review. Native rows, if run, remain diagnostic.
- No nominal rate claim until source/config hashes and a candidate-wide
  classification gate pass.

# Recovery / handoff

Work is isolated on `issue-9729-contact-attribution-20260928`, stacked on
`#9873` head `6332c5c0`. Preserve the draft PR and its exact head for review.

# Quarantined diagnostics and replacement proof

- Before the #9668 seed rule was merged, a three-step
  `classic_cross_trap_low` seed 111 goal episode and a 100-step zero-action
  `francis2023_robot_crowding` seed 113 trace were run as diagnostics. Both
  had no contact. Quarantine both episodes: they are excluded from all release
  evidence and must not inform v4 tuning, the freeze, Social Navigation Quality
  Index (SNQI) calibration, or the campaign. The seed 113 no-contact trace does
  not establish absence of the historical stationary-contact defect.
- The saved-row regression uses development seed 1001 with a deterministic
  contact environment to prove the scenario marker enables measured speed,
  step index, and class in the saved `contact_provenance.v1` row. It is a code
  diagnostic, not native benchmark or release evidence. The quarantined native
  runs are not needed for this PR's implementation check and are not reused.
  Any further stepped diagnostic must use development seeds 1001-1030 or
  diagnostic seeds 103-105; no planner step is allowed on seeds 111-140.
