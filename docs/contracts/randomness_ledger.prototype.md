# Randomness-Ledger Episode Sidecar — Prototype Contract

> Status: **non-normative prototype**. Seed-sensitivity provenance only. Not emitted by any
> simulator, planner, pedestrian, or benchmark execution. Do not treat as benchmark,
> paper-grade, or causal-seed evidence.

Parent: design-gap issue [#5617](https://github.com/ll7/robot_sf_ll7/issues/5617) —
named stochastic-factor provenance for seed-effect explanation (documented, not scheduled).
Prototype slice: issue
[#6466](https://github.com/ll7/robot_sf_ll7/issues/6466).

## What this is

A single per-episode JSON sidecar that names each stochastic factor an episode consumed,
identifies the random stream it drew from, records how many draws it took, and links the
entry to episode and seed provenance.

- Schema: [`randomness_ledger.prototype.schema.json`](randomness_ledger.prototype.schema.json)
  (JSON Schema draft-07), stable prototype identifier
  `https://robot-sf.dev/contracts/randomness_ledger.prototype.v1.json`, version string
  `randomness_ledger.prototype.v1`.
- Valid fixture: `tests/data/randomness_ledger/episode_seed_23.prototype.ledger.json`.
- Contract tests: `tests/tooling/test_randomness_ledger_contract.py`.

## Claim boundary (important)

This prototype records **seed-sensitivity** provenance only. A seed is an index into
pseudorandom streams, so a statement like "seed 23 produced the collision" is not
explanatory. Naming which factor drew from which stream lets reports phrase outcome
differences across seeds as **sensitivity patterns**, never as causal attribution.

The defensible chain that *would* support causal explanation — seed → recorded vector of
named random choices → altered trace feature → changed event sequence → confirmed by
independent controlled replay — requires independent per-factor replay, which this
prototype explicitly does **not** provide (see parent #5617). State checkpoints in the
schema are opaque tokens with no defined replay semantics.

## Out of scope

- No runtime producer, recorder, or replay under `robot_sf/` or `scripts/`.
- No integration with environment, planner, pedestrian, or benchmark execution.
- No migration of historical episode artifacts.
- No change to existing contracts (`training_summary.schema.json`, `summary_markdown.md`)
  or shared doc indexes (`docs/README.md`, `docs/context/INDEX.md`).

## Validation

```bash
uv run pytest tests/tooling/test_randomness_ledger_contract.py -q
uv run ruff check tests/tooling/test_randomness_ledger_contract.py
uv run ruff format --check tests/tooling/test_randomness_ledger_contract.py
```
