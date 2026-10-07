# Behaviour receipt gate integration

This gate is not ready to enable until the release integration owner lands
`configs/benchmarks/releases/behaviour_gate_0_1_0.json` on the target branch.
Merge that dependency before the gate PR. Missing scope remains a blocker;
there is no default roster, inferred roster, or tooling exemption for a missing inventory.

The owner reviews the complete release arms, maps, vehicle identity and vehicle-specific
exceptions. The inventory includes `arms`, `maps`, `vehicle_id` and `exceptions`.
For named treatment arms, `arm_algorithms` binds each arm key to its canonical
registry algorithm. Without a mapping the arm key itself must identify the registry
algorithm. Receipts bind the canonical JSON SHA-256 of this reviewed inventory.
No inventory is authored by the receipt validator or its synthetic test fixtures.

The workflow checks out the PR source SHA with complete history and tags. The
baseline resolves to the latest published software release tag, excluding model
and other artifact releases. Source and synthetic merge identities must never be
interchanged.

All benchmark paths are in scope, including the map and non-map row producers,
compatibility wrappers, episode types, schemas and campaign accounting. In each
receipt row `algorithm` identifies the bound controller; `execution_mode` records
its command space. Native-capable algorithms require native commands; adapter-only
algorithms require registry-supported adapter commands. `controller_executed`
attests that the intended controller (including its solver, when applicable) really
produced commands. `fallback` and `degraded` must both be false. Independent review
must confirm these fields against the source-bound raw execution artifacts;
adaptation alone does not prove solver execution.

Inventory installation, source-bound execution, durable classification evidence
and independent review remain integration prerequisites. Unit fixtures prove
validator behavior, never scientific execution or inventory authority.
