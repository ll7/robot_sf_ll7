# Scenario-window research dossier — 2026-09-08

Status: preparation-only terminal packet. The complete machine-readable record is
[scenario_window_research_dossier_2026-09-08.json](scenario_window_research_dossier_2026-09-08.json).
This work follows the live #7381 programme and the preparation rulings in #7383
and #7384; the attached packet is guidance/evidence, not a replacement authority.

Refresh provenance for this repair: current `origin/main` is
`ade95312d31462c8a66208948f9c8528910de0c7`; the live #8612 and #8620 tips are
`db1215d6e5d1a19598d4f6a4ba10b119229f66b7` and
`4ef0858d33c91291129098c08975525a54fb0121`; and the tested #8622 code head is
`75e69876a3a141d8e3f231110a1d557f65794177`. The stack is refreshed for inspection,
but live domain authority and native campaign admission remain blocked.

## Outcome

The bounded result is an auditable implementation seam, not a shortened benchmark:

- RW-01/RW-02 source and native-contract repairs are in [PR #8612](https://github.com/ll7/robot_sf_ll7/pull/8612).
- RW-03/RW-04 typed state inventory, JSON+NPZ snapshot prototype, and no-op
  comparator are in [PR #8620](https://github.com/ll7/robot_sf_ll7/pull/8620).
- RW-05 deterministic relevance selection and the RW-06/RW-07 gated plan are in
  [PR #8622](https://github.com/ll7/robot_sf_ll7/pull/8622).
- No campaign, planner ranking, manuscript, release artifact, or new mainline
  evidence was produced. RW-06 and RW-07 remain `BLOCKED_ADMISSION`.

The supported snapshot subset is one-robot native `SimulatorCounterfactualModel`
state at a pre-step/action boundary. It preserves absolute world time and remaining
budget, uses stable actor/behavior identities, validates map/config/code/planner/
checkpoint/platform/timestep compatibility before mutation, and loads numeric NPZ
payloads with pickle disabled. Planner/controller memory, sensor history, and
metric/event accumulators remain explicit unsupported modes; this is not a fresh
environment checkpoint for those modes.

## Three preparation timelines

All rows below are synthetic/native preparation fixtures, not production findings.
The complete parent remains the authority and all actors/static geometry remain
retained.

| timeline | candidate signals | proposed interval | evidence grade |
| --- | --- | --- | --- |
| no event | clearance 3 m, closing speed 0 m/s, TTC 10 s, no conflict/fallback/contact | none | synthetic preparation |
| doorway interaction | precursor/visibility at step 2; clearance 0.8/0.7 m and closing speed 0.3 m/s at steps 4–5; one hysteresis hold | steps 2–8 (pre 2, post 2) | synthetic preparation |
| multi-stage interaction | precursor at 1; clearance at 3; path conflict at 5; contact at 8 | steps 1–11 after merge | synthetic preparation |

The selector records clearance, closing velocity, TTC, closest approach, braking
margin, visibility latency, path conflict, fallback/saturation, progress, stall,
discomfort, and collision. Every signal carries units, provenance, availability
timing, prediction assumptions, missingness, and an explicit hindsight flag.
Missing values are unknown, never safe zeroes. The current Option-A rules are
proposed only: clearance 1 m, closing speed 0.2 m/s, TTC 3 s, closest approach
1 m, braking margin 0 m, visibility latency 0.5 s, stall 2 s, discomfort 0.5,
pre/post-roll 2 steps, merge gap 1 step, and one hysteresis step. No rule is marked
approved.

## Unsafe crop and refusal boundary

For the multi-stage timeline, a deliberately late crop containing steps 6–11 drops
the precursor at step 1 and earlier decisions at steps 3 and 5. The immutable
manifest validator rejects it with `unsafe crop: selected excerpt omits required
precursor steps`. The full parent digest, actor IDs, original step indices, and
complete rows remain linked in the proposal. This is a fidelity refusal, not a
claim that the physical event is impossible.

## Proposed fidelity vector

These are candidate rules for a later author decision, not frozen acceptance
thresholds.

| property | unit | proposed direction | approval |
| --- | --- | --- | --- |
| state/observation/action equivalence | native values per step | exact equality | proposed |
| verdict agreement | categorical | exact category | proposed |
| failure-mechanism agreement | categorical trace label | exact when labels exist; otherwise not evaluable | proposed |
| primary metric error | metric-specific units | bound absolute or relative error before validation | not frozen |
| planner ranking | ordering/rank correlation | only with full metric support and independent parents | blocked until transfer |
| replay determinism | state/RNG stream | exact same-platform continuation for the typed snapshot/no-op subset; the last-avoidable engine remains contact-presence/step only | subset with explicit limitation |
| precursor/avoidability event | step/event identity | zero missing declared precursor; finite search failure is unknown | proposed |

## Dependency and gated experiment plan

The state inventory is [simulator_state_inventory.v1.json](simulator_state_inventory.v1.json).
RW-01/RW-02 establish route, horizon, action, collision, and response contracts;
RW-03/RW-04 establish the supported snapshot/no-op seam; RW-05 consumes retained
parent rows. RW-06 (capability-conditioned avoidability witnesses) and RW-07
(paired transfer/cost break-even) require stage-specific live authority and remain
blocked. The checked-in plan keeps the four required modes together:

1. full parent;
2. same-policy exact restart;
3. pose/velocity-only reset as a negative control;
4. deliberately late/no-preroll crop as a negative control.

It does not launch any of them. Cross-planner prefixes, recurrent hidden state,
spatial actor removal, and full-episode metric recomposition are not silently
assumed valid.

## Cost and sample-size boundary

The native doorway fixture measured 19,657 bytes of JSON metadata plus 3,893 bytes
of compressed numeric payload (23,550 bytes total; seven pedestrians). This is an
engineering receipt only. The accounting used by the plan is:

```text
C_plain = K * C(T)
C_window = C_parent + C_selection + C_validation + K * (C_load + C(W))
```

No positive break-even was measured because no admitted pilot ran. A proposed
validation design uses 30 independent parents to expose feasibility bugs and 300
independent parents for a roughly 1% one-sided 95% upper bound after zero observed
failures; both are held out by parent/scenario/seed lineage and are not executed in
this lane.

## Terminal hypothesis ledger

The JSON dossier gives every hypothesis an allowed terminal status. In brief:

- `supported`: route restoration, horizon refusal, the native typed subset's
  same-object round trip, and the deterministic preparation selector.
- `rejected`: a late crop that omits a declared precursor is safe.
- `not_evaluable`: shortened continuation's compute break-even.
- `blocked`: fresh-process/recurrent coverage, planner transfer/ranking, and
  PELT/learned selector comparison.

The remaining action is review of this preparation packet and, only if authority
changes, a separate admission check. There is no pending autonomous loop and no
manuscript-submission change in this branch.
