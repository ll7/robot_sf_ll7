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
identity, seeds, timestep, physics steps, macro-action cadence, candidate
evaluations, total budget, bounds, objective proxy, provenance, and
fail-closed exclusions.  No campaign is run, no SLURM job is submitted, and
no benchmark or paper-facing claim is made.

## What ships (the contract)

- `configs/adversarial/issue_6921_matched_compute_packet.yaml`:
  - `matched_compute_packet.v1` schema with frozen arms, scenario, budget,
    seeds, bounds, provenance, exclusions, and domain-approval gate.
  - Two named arms: `social_force_open_loop` and `residual_search_reactive`.
  - Matched budget derived from simulation geometry: 10 macro-actions per
    episode, 9 candidates per macro-action per arm, 90 per-arm episode total,
    180 all-arms episode total.
  - Scenario identity grounded in the checked-in `crossing_ttc_template`
    (seed 123), not invented IDs.
  - Explicit `seed` fields on both arms' `residual_search` and
    `residual_adversary`, matching the packet-level seed values (42).
  - `target_ped_idx: [0]` documented as a deliberate single-target
    matched-cost choice, diverging from the upstream `-1` all-target example.
- `tests/adversarial/test_matched_compute_packet.py`:
  - Config loading and required-field validation.
  - Budget arithmetic consistency tests (macro-actions, per-macro-action,
    per-arm episode, all-arms episode).
  - Arms max-candidates match budget and arm budgets are equal.
  - Seed field presence and value-matching tests.
  - Provenance and template/search-space path resolution tests.
  - Scenario template identity and seed validation.
  - target_ped_idx single-target assertion and upstream divergence check.
  - Bounds identity check (residual bounds not relaxed).
  - Exclusion completeness check (forbidden exclusions present).
  - Domain-approval gate assertion.
  - Config round-trip tests for ResidualSearchConfig and ResidualAdversaryConfig.
- `docs/context/issue_6921_matched_compute_packet.md`:
  - This note.

## Design decisions

### Packet as frozen specification

The packet is a versioned YAML file, not a runnable script.  This keeps the
comparison definition stable while deferring execution to a future slice that
requires domain-approval.  The schema version (`matched_compute_packet.v1`)
allows future packets to extend or revise the comparison without breaking
backward compatibility.

### Budget derived from simulation geometry

Budget fields are explicitly named and derived from `total_sim_steps` (50)
divided by `physics_steps_per_macro_action` (5) = 10 macro-actions per
episode.  Each arm evaluates 9 candidates per macro-action boundary
(`grid_points_per_dim ** 2` = 3 ** 2 for the 2-D residual action space).
Per-arm episode total is 10 * 9 = 90.  All-arms episode total is 90 * 2 = 180.
No field named `total_candidate_evaluations` with ambiguous scope exists.

### Scenario identity grounded in checked-in template

Scenario identity is `crossing_ttc_template` defined by the checked-in
template at `configs/scenarios/templates/crossing_ttc.yaml` with template
seed 123.  No invented `crossing_ttc_low` / `crossing_ttc_medium` IDs are
used.  No generated manifest is claimed.

### Identical bounds and objective

Hard residual bounds (acceleration, jerk, speed, heading, route deviation,
separation) are frozen and identical across arms.  The objective proxy
(`maximize_residual_magnitude`) is the same diagnostic proxy used by the
residual-search slice (#6911).  No bound or objective is relaxed between arms.

### Explicit seed fields in both arms

Both arms carry explicit `seed: 42` in `residual_search` and
`residual_adversary` sections, matching the packet-level `search_seed` and
`residual_adversary_seed` values.  This makes the seed contract machine-checkable.

### target_ped_idx: [0] as deliberate choice

Both arms set `target_ped_idx: [0]`, a deliberate single-target matched-cost
choice.  This diverges from the upstream `issue_4360_residual_adversary.yaml`
which uses `-1` (all-target).  The single-target setting ensures both arms
perturb exactly one pedestrian, making the comparison cost matched per
targeted agent.

### Provenance chain

The packet references the scenario template, the residual-search config
(#6911), the residual-adversary config (#4360), the search space, and the
dispatchable inventory without introducing new runtime modules.  All provenance
paths are tested to resolve on disk.

## Canonical validation command

```bash
uv run pytest tests/adversarial/test_matched_compute_packet.py -v
```

This runs the focused contract tests that prove the packet is well-formed,
the budget arithmetic is consistent, provenance paths resolve, both arms'
max-candidates match the budget, arm budgets are equal, seeds match, bounds
are not relaxed, and the domain-approval gate is present.  It does not execute
any campaign or simulation.

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
