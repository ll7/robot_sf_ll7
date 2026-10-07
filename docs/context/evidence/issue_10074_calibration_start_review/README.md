<!-- AI-GENERATED (robot_sf#10094) - NEEDS-REVIEW -->
# Pedestrian development review: PR #10094

Diagnostic-only; model acceptance remains blocked. Defaults stay OFF.

[Populated summary](populated_summary.json) records ON versus OFF with all three
pedestrians intact, dev seeds 1001-1005, 300 production substrate steps, and both
1.2 m doorway and 0.8 m bottleneck openings. [Small sample](sample.json) retains
initial, first-step and final positions for seed 1001. Raw trajectories stay external.

[Custody manifest](custody.json) preserves original archive and member checksums.
Historical bulk evidence is removed from feature history. External availability
is recorded separately from preservation; an unverified pointer is no proof.
The old registry had 492 findings before and after: no locator repair is claimed.
Only compact artifacts in this PR receive catalog/registry bookkeeping.

The earlier 1440-row actor-free sweep validates default-OFF preservation only.
Its ON equality is tautological for pedestrian changes, and supplies no evidence
of pedestrian interaction correctness. The complete behaviour gate remains open:
other planner arms unrun, width goal metric-reference rejection and width MPPI
predictor-contract rejection (issue #10180). Populated probes are domain review
evidence and do not close that gate.
