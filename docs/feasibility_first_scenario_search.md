# Feasibility-first scenario search diagnostic

This fixture diagnostic rejects contradictory scenario candidates before risk-search ranking, so invalid candidates do not silently enter a safety denominator. It implements the bounded contract from [issue #7315](https://github.com/ll7/robot_sf_ll7/issues/7315).

## Claim boundary

- Evidence status: `diagnostic-only` fixture protocol.
- The command does not execute the simulator, planner, or a risk campaign.
- It does not establish safety, planner weakness, discovery superiority, or transfer from the source paper.
- The source method is adapted to Robot SF's adversarial candidate vocabulary; no autonomous-driving transfer claim is made.
- A larger campaign requires separate approval and a real scenario/seed manifest.

## Contract

Every candidate carries four separate checks:

1. `kinematic_reachability`: the route fits the declared kinematic envelope.
2. `behavioral_consistency`: robot and virtual-road-user timing assumptions agree.
3. `geometry_traffic`: geometry and traffic constraints are admissible.
4. `simulator_validity`: a simulator or validator accepted the input.

Each check is `pass`, `fail`, or `unavailable`. Missing or contradictory fields fail closed: only four explicit `pass` values make a candidate eligible. Rejection reasons and candidate identities remain in the report, while rejected candidates are excluded from safety denominators.

Eligible candidates are ordered by a deterministic hierarchy:

1. kinematic criticality,
2. controllability/risk,
3. diversity,
4. stable candidate identifier.

This is intentionally not a weighted scalar. The fixture compares that risk-feature ordering with a seeded uniform draw from the complete candidate pool. Safety-event severity is explicitly unavailable because no simulator runs.

The report also names the existing adversarial random sampler as the intended baseline, but marks it `not_executed` in this fixture because that path needs a real scenario/search-space input. The status is a provenance guard, not a result.

## Reproduce the fixture

```bash
uv run python scripts/validation/run_feasibility_first_scenario_search.py \
  --config configs/benchmarks/issue_7315_feasibility_first_smoke.yaml \
  --output output/issue_7315_feasibility_first/report.json
```

The JSON report records the configuration digest, candidate and sampling seeds, rejection ledger, selected IDs, method summaries, and governance flags. `output/` is temporary worktree-local output; the report is not a durable benchmark artifact.

## Next proof step

To move beyond the fixture, provide a versioned scenario/seed manifest, attach real geometry/traffic and simulator-validation evidence, and pre-register a compute-bounded comparison against the existing adversarial baseline. Until then, fixture differences are research direction evidence only, with uncertainty dominated by the unexecuted simulator and the small candidate pool.
