# Trace-example screening (exsel protocol), robot_sf side

Status at this commit: preregistration only. The two claim contracts and the screening
orders below were committed and pushed before the first screening episode ran. Results,
ledgers, traces and provenance are added in later commits on this branch.

- `exsel-6.4-measure-walkthrough/`: EXSEL-6.4-MEASURE-WALKTHROUGH (social_force,
  classic_doorway_high).
- `exsel-7.2-narrow-doorway/`: EXSEL-7.2-NARROW-DOORWAY (orca and goal,
  francis2023_narrow_doorway).
- `tools/`: config derivation, single-run wrapper, screening driver, acceptance logic and
  its tests.

The protocol is the dissertation's frozen trace-example selection protocol
(ll7/diss `docs/context/trace_example_selection_protocol.yaml`, decision
EXSEL-TRACE-SELECTION-PROTOCOL-2026-10-28). Each `contract.yaml` copies the claim block,
the rules and the acquisition spec verbatim and binds the execution identity.
Development-seed diagnostic on the unmodified 0.0.8 release code; never release evidence.
