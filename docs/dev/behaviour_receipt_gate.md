# Behaviour receipt gate integration

The release integration inventory is installed at
`configs/benchmarks/releases/behaviour_gate_0_1_0.json`. It binds the 14 named
arms and 51 scenario/map slots from the empty-world sweep's `main` and `width`
inputs, with explicit arm-to-registry algorithm mappings. Map keys are scenario
names, matching sweep row identities; shared SVGs retain separate slots when
scenario settings differ. The inventory records repository-relative map paths,
map digests and campaign/matrix source digests. Receipts bind its canonical JSON
SHA-256, including those provenance fields.

The vehicle is `differential_drive_r1m`: the current differential-drive body with
1.0 m radius. The existing 2.0 m narrow doorway remains an infeasible-by-design
exception, linked to the source declaration at the integration base commit.
This does not certify a sweep or admit scientific conclusions. The release
integration owner maintains this inventory when arms, maps or body change;
missing inventory still fails closed for behaviour receipts.

The inventory itself is gate tooling, so an inventory-only PR is exempt, just
like validator/test changes. A PR also touching a planner or campaign remains
in scope. Docs-only PRs do not read inventory or query release metadata.

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
