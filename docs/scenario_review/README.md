# Scenario review

Executable contracts for reviewing retained traces and scenarios
([review contracts](./review_contracts.md)): versioned review bundles,
visualization specs, component request/result envelopes with capability
descriptors, and experiment recipes. Diagnostic tooling only — no
scientific admission, benchmark results, or simulator/planner changes.

The contracts are consumed by offline components; the first consumer is the
[SREV-15 review workbench](./review_contracts.md#review-workbench-srev-15)
(`python -m robot_sf.render.review_workbench`), which renders a local,
network-free artifact/provenance view plus a presentation plan.
