# Issue #8570: Bounded adversarial falsification answerability

The checked-in packet is a source-bound, diagnostic-only preparation contract. It records a
reproducible design and explicitly authorizes no simulator, planner, optimizer, campaign,
benchmark, safety, robustness, or scientific claim.

## Contract and source custody

- Packet: [`configs/adversarial/issue_8570_bounded_falsification_answerability_v1.yaml`](../../configs/adversarial/issue_8570_bounded_falsification_answerability_v1.yaml)
- Validator, loader, and canonical self-digest: `robot_sf/benchmark/research_answerability.py`
- Schema: `robot_sf/benchmark/schemas/adversarial_falsification_answerability.v1.json`
- Frozen base: `c61b0f93683e1f9d2c83125f1d830b3d0162f3d1`; the packet stores raw SHA-256 hashes for
  the #7340 search-space config, station-platform template, and referenced SVG map.
- The packet's explicit variable/actor/unit/bound map is checked against the parsed #7340
  search-space config, not only against copied packet values.

In #7340 template mode, `pedestrian_delay_s` is provenance-only: the existing loader permits
`wait_at` only for explicit trajectories. Its binding is therefore marked non-runtime-effective,
and the packet's `research_answerability.v1` gate remains blocked until a future adapter proves a
runtime-effective binding and supplies interpretable native/replay outputs.

## Declared diagnostic design

The lexicographic objective is feasibility, kinematic criticality, controllability risk,
diversity, normalized perturbation cost, then candidate digest, with deterministic tie breaks and
no scalarization. Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is the primary declared
proposal arm; seeded random and Halton controls each receive the same 64-candidate budget for each
of three search seeds. Five distinct held-out confirmation seeds are recorded separately. These
are design allocations, not executed rows or powered evidence.

The packet reuses `research_answerability.v1` and
`robot_sf.adversarial.feasibility_first.build_scenario_feasibility_ledger`; invalid or unavailable
predicates remain visible and outside safety denominators. Its outcome vocabulary distinguishes
`result`, `null`, `inconclusive`, `invalid`, `unavailable`, and `blocked`.

Focused tests cover self-digest and source/bound tampering, seed/budget accounting, complete
outcomes, the template-delay gate, and deterministic feasibility/rejection accounting. No test
launches a simulator or other compute campaign.
