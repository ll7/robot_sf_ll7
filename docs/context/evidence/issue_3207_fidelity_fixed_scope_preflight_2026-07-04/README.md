# Issue #3207 — Full fixed-scope fidelity-sensitivity preflight (2026-07-04)

First-use evidence for the full fixed-scope **preflight** added by the #3207
pre-registration slice. This is a **launch/readiness artifact only** — it is not
benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, and
not paper-facing evidence. No benchmark episodes were run to produce it.

## What this records

`fidelity_fixed_scope_preflight.json` is the deterministic preflight packet built
from the shipped study config `configs/research/fidelity_sensitivity_v1.yaml` at
`origin/main` base commit `b619d1be4`.

- `decision`: `preflight_ready` — the contract-level launch checks pass.
- `materialized_scope.run_cells_per_scenario`: `108` (3 planner groups × 12 axis
  variants × 3 seeds), enumerated per scenario in the fixed scenario set.
- `planner_resolution`: every planner group resolves to a non-placeholder
  algorithm-readiness catalog entry (`orca` → `orca`, `default_social_force` →
  `social_force`, `hybrid_rule_v0_minimal` → `hybrid_rule_local_planner`).
- `primary_metric`: `snqi` is identifiable by contract.
- `launch_prerequisites`: `preflight_ready` is **not** execution-ready. The
  campaign runner is still a bounded two-planner slice, ORCA needs rvo2, the
  hybrid planner needs explicit opt-in, and runtime rank-identifiability must be
  re-checked on measured results.

## Regenerate

```bash
uv run python scripts/benchmark/preflight_fidelity_fixed_scope.py \
  --config configs/research/fidelity_sensitivity_v1.yaml \
  --out output/fidelity_sensitivity/preflight --require-ready
```

The `git_head` field differs by checkout; all other fields are deterministic for
a given config.
