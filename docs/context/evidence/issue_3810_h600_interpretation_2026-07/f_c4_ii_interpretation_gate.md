<!-- AI-GENERATED (robot_sf#4195, 2026-07-03) — NEEDS-REVIEW -->
# F-C4(ii) Interpretation Gate — h600 Hybrid-Roster Integration

**Evidence status:** `diagnostic-only`. This note integrates the pre-registered
h600 hybrid-roster packet (job 13282) with the retained h600 confirm/extended
bundle (jobs 13268 / 13273) so the issue #4195 F-C4(ii) checklist can be read
from committed artifacts rather than only from issue comments. The evidence in
this directory remains diagnostic-tier; the spine claim tier is set by the
maintainer sign-off recorded below, not by this note.

**Sign-off status:** `author_signoff: RECORDED` (2026-07-03 evening,
issue #4195 maintainer comment). F-C4 was promoted draft→supported in the
dissertation spine (diss commit `0d853df`) for pillars (i)+(ii) *at the guarded
wording*; pillar (iii) remains draft pending the falsification→hardening loop
documentation (outside #4195). **This note records that outcome and the guarded
boundary as durable evidence; it does not itself promote a claim or edit the
dissertation.**

F-C4(ii): *hybrid (control-law) arms retain their advantage over
prediction-equipped arms at the long horizon (h600).*

## Integrated inputs (all in this directory unless noted)

| leg | job | role | provenance |
| --- | --- | --- | --- |
| confirm | 13268 | probe roster, 7/7 rows valid | git `1cb7dc31`, matrix hash `c10df617a87c` (`source_manifest.json`) |
| extended roster | 13273 | probe roster + prediction MPC / MPC+CBF arms | matrix hash `c10df617a87c` (shared-arm comparability `pass`) |
| hybrid roster | 13282 | 4 pre-registered hybrid/scenario-adaptive arms | git `b4800dc828e6`, matrix hash `c10df617a87c` (`hybrid_roster_h600_transfer_packet.md`) |

Comparability precondition for the gate: the scenario-matrix hash is identical
(`c10df617a87c`) across all three legs, so the hybrid arms and the
confirm/extended arms were evaluated on the same 48-scenario surface. Seed set
`[111, 112, 113]` (3 seeds, `eval`) is shared; horizon fixed at 600.
`comparability_check.md` records the confirm-vs-extended shared-arm check
(`pass`, 7 shared arms). The hybrid leg's comparability rests on the identical
matrix hash and pre-registration (`docs/context/issue_4230_h600_hybrid_roster_preregistration.md`),
not on a re-run of the confirm/extended shared-arm mapping.

## Gate reading — F-C4(ii) diagnostic boundary (the promoted guarded wording)

### SUPPORTED (guarded wording promoted to F-C4 supported, diss `0d853df`)

1. **Hybrid arms dominate every prediction-equipped arm at h600, with disjoint
   CIs.** Worst hybrid success 0.771 vs best prediction arm 0.569 → Δ ≥ 0.20,
   no 95% CI overlap (`hybrid_roster_h600_transfer_packet.md`). This extends the
   control-law-bound reading from the 13268/13273 chain to the pre-registered
   hybrid roster at long horizon: arms that change the **control law** clear arms
   that add **prediction** to a fixed control law [pillar (i)+(ii)].
2. **h500 → h600 point-estimate rank transfer holds.** The four h500 hybrid
   leaders are the top four point-estimate arms at h600 (0.771–0.799), above the
   entire extended roster — the transfer question #4230 pre-registered.
3. **CBF filtering does not rescue prediction:** adding CBF to prediction_mpc
   moves success/collision by <0.01 (bundle rows); the prediction deficit is
   structural, not a missing safety wrapper.

### DIAGNOSTIC-ONLY (explicitly excluded from the promoted claim)

- **Hybrids vs ORCA:** point-estimate lead (+0.03 to +0.06) but ORCA's 0.743
  lies inside every hybrid 95% episode-level CI at the S3/144-episode budget →
  **no separated success-vs-ORCA claim is licensed** (this exclusion is exactly
  the guarded-wording caveat the maintainer promoted under). Per-seed direction
  and the SNQI lead (hybrids −0.09..−0.12 vs ORCA −0.198) favor the hybrids but
  stay diagnostic. Escalation path if a separated claim is wanted: pre-declared
  S30 schedule (#4304, deferred by ruling 2026-07-03).
- **Hybrid-internal ordering:** the four hybrid arms (0.771–0.799) are a
  statistical tie among themselves; no within-hybrid ranking claim.
- **Extended-roster prediction arms** (`prediction_mpc`, `prediction_mpc_cbf`)
  remain extended-only; they are diagnostic relative to the confirm probe roster.

### NOT SUPPORTED (must not be claimed from this bundle)

- No horizon-only causality: h500-vs-h600 rank deltas are roster artifacts
  (different rosters/seed budgets); see `horizon_sensitivity_report.md`.
- No S20-grade seed claims (h600 ran 3 seeds; S20 is h500-only).
- No exposure-normalized statements: interaction-exposure diagnostics are
  `blocked_missing_required_fields` (`interaction_exposure_diagnostics.md`).
- No simulator-realism, real-world-safety, or generalized planner-superiority
  claims. Pillar (iii) stays draft (outside #4195).

## Integration report

- **Blockers closed:** the missing h600 hybrid-roster leg (follow-up (a) of the
  2026-07-03 claim-boundary proposal) now exists as a committed, checksummed,
  provenance-pinned artifact and is read into the gate; author sign-off is
  recorded; the #4195 interpretation chain is closed.
- **Blockers remaining (intentional, outside #4195):** exposure leg (item 5 /
  #3977/#3978) stays fail-closed until native exposure fields land; pillar (iii)
  falsification→hardening documentation is separate diss work.
- **New blockers:** none.
- **Next empirical action (only if a separated hybrid-vs-ORCA success claim is
  later wanted):** the pre-declared S30 seed top-up (#4304), currently deferred.
  Not required for the SUPPORTED items above or for the promoted claim.

## Author sign-off

`author_signoff: RECORDED` — 2026-07-03 evening (issue #4195). F-C4
draft→supported (diss commit `0d853df`), pillars (i)+(ii) at the guarded wording
above; pillar (iii) draft. This evidence note is the durable citation surface for
that guarded boundary and promotes nothing on its own.

<!-- /AI-GENERATED -->
