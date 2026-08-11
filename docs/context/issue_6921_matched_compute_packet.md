# Issue #6921 — Matched-budget reactive versus open-loop comparison packet

Status: diagnostic-only comparison packet.
Evidence grade: not benchmark evidence. No paper, metric, or safety claim.

## Plain-language summary

This slice freezes a versioned, machine-checkable comparison packet that
names the two arms for a future reactive-versus-open-loop comparison under
identical compute budgets.  The nominal Social Force / open-loop arm
evaluates candidate residual proposals through a one-shot bounded controller.
The reactive residual-search arm evaluates candidates through the full
bounded controller with iterative jerk rate-limiting, geometry projection,
and separation enforcement carried across macro-action steps.

The packet is a specification, not an execution path.  It freezes scenario
IDs, seeds, timestep, physics steps, macro-action cadence, candidate
evaluations, total budget, bounds, objective proxy, provenance, and
fail-closed exclusions.  No campaign is run, no SLURM job is submitted, and
no benchmark or paper-facing claim is made.

## What ships (the contract)

- `configs/adversarial/issue_6921_matched_compute_packet.yaml`:
  - `matched_compute_packet.v1` schema with frozen arms, scenario, budget,
    seeds, bounds, provenance, exclusions, and domain-approval gate.
  - Two named arms: `social_force_open_loop` and `residual_search_reactive`.
  - Matched budget: 9 candidate evaluations per arm per macro-action boundary.
  - Frozen scenario IDs: `crossing_ttc_low`, `crossing_ttc_medium`.
  - Frozen seeds: 1101, 2202, 3303 for scenarios; 42 for search and adversary.
- `tests/adversarial/test_matched_compute_packet.py`:
  - Config loading and required-field validation.
  - Budget mapping non-vacuity check (budget > 0).
  - Bounds identity check (residual bounds not relaxed).
  - Exclusion completeness check (forbidden exclusions present).
  - Domain-approval gate assertion.
- `docs/context/issue_6921_matched_compute_packet.md`:
  - This note.

## Design decisions

### Packet as frozen specification

The packet is a versioned YAML file, not a runnable script.  This keeps the
comparison definition stable while deferring execution to a future slice that
requires domain-approval.  The schema version (`matched_compute_packet.v1`)
allows future packets to extend or revise the comparison without breaking
backward compatibility.

### Matched budget

Both arms evaluate exactly 9 candidates per macro-action boundary.  This
equals `grid_points_per_dim ** 2` for the 2-D residual action space (3 x 3).
The budget is not vacuous: it exercises the full grid enumeration path.

### Identical bounds and objective

Hard residual bounds (acceleration, jerk, speed, heading, route deviation,
separation) are frozen and identical across arms.  The objective proxy
(`maximize_residual_magnitude`) is the same diagnostic proxy used by the
residual-search slice (#6911).  No bound or objective is relaxed between arms.

### Provenance chain

The packet references the residual-search config (#6911), the residual-adversary
config (#4360), and the dispatchable inventory (#4360) without introducing new
runtime modules.  All bounds and parameters are inherited from these existing
configs.

## Canonical validation command

```bash
uv run pytest tests/adversarial/test_matched_compute_packet.py -v
```

This runs the focused contract tests that prove the packet is well-formed,
the budget is non-vacuous, bounds are not relaxed, and the domain-approval
gate is present.  It does not execute any campaign or simulation.

## Claim boundary (what this slice does NOT do)

This is a diagnostic-only specification slice.  It makes **no** benchmark,
planner-ranking, safety, or paper-facing claim.  It runs **no** campaign,
**no** SLURM job, and **no** simulation.  It adds **no** new stress-case
metric.  It does **not** implement the comparison runner, the reactive
policy, or the open-loop evaluation path.  It does **not** enable or
authorize any future execution; that requires explicit domain-approval.

## Domain-approval gate

Any future campaign execution, benchmark run, or stronger claim from this
packet requires maintainer domain-aware approval.  The packet is frozen as
a diagnostic-only specification; no execution path is enabled by this config
alone.  The gate is asserted in the contract tests and documented in the
YAML config.

## Deferred slices

- Comparison runner implementation (executes both arms and produces a report).
- Reactive residual-search policy integration (the actual reactive arm).
- Open-loop candidate evaluation path (the actual open-loop arm).
- Campaign execution with domain-approval gate.
- Stress-case validity/strength metrics (requires Domain-Aware Approval).
- CMA-ES or MCTS search-baseline adversary (sequenced before PPO).
- PPO / learned adversary (only after the search baseline is measured).
