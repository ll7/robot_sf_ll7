# Issue #6911 — Deterministic finite-budget grid search over residual adversary proposals

Status: diagnostic-only capability slice.
Evidence grade: not benchmark evidence. No paper, metric, or safety claim.

## Plain-language summary

This slice adds a diagnostic-only search baseline that evaluates a finite grid
of candidate residual accelerations through the existing bound pipeline.  The
search is a brute-force enumeration over an explicitly discretised action space;
no optimizer dependency (CMA-ES, MCTS, PPO) is introduced.  The algorithm and
diagnostic objective proxy are named in the checked-in config and emitted in
every diagnostic record.

The search plugs into the existing `ResidualAdversaryPolicy` protocol and is
wrapped by `BoundedResidualAdversary`, which enforces every hard bound
(acceleration, jerk, speed, heading, route, walkable-space, inter-agent
separation).  No existing bound is bypassed or weakened; the nominal Social
Force wiring is unchanged.

## What ships (the contract)

- `robot_sf/ped_npc/residual_search.py`:
  - `ResidualSearchConfig` — validated, frozen config with algorithm name,
    objective proxy, grid resolution, budget, seed, and action bounds.
  - `FiniteGridSearchPolicy` — deterministic grid-search policy implementing
    `ResidualAdversaryPolicy`. Every candidate is evaluated by an isolated
    `BoundedResidualAdversary`, so candidate accounting uses the same
    acceleration, jerk, speed, heading, route, walkable-space, and separation
    contract as the runtime controller.
  - `SearchDiagnosticRecord` — compact, deterministic JSON record with schema
    version, config digest, seed, source revision, candidate/action ordering,
    candidate accounting, and all residual bound settings.
  - `compute_config_digest` — SHA-256 digest of the canonical config
    serialisation for reproducibility tracking.
- `configs/adversarial/issue_6911_residual_search.yaml` — versioned config
  with fixed seed, tiny budget, explicit action bounds, and representative
  CPU-only input.
- `tests/adversarial/test_residual_search.py` — focused tests for config
  validation, deterministic output, invalid-candidate accounting, and
  bound-preserving integration.
- `docs/context/issue_6911_residual_search.md` — this note.

## Design decisions

### Action grid

The search enumerates a Cartesian product of `grid_points_per_dim` evenly-spaced
values per action dimension, yielding `grid_points_per_dim ** 2` candidates.  The
grid is deterministic and reproducible.

### Objective proxy

The default proxy is `maximize_residual_magnitude`: for each targeted pedestrian
the candidate whose bounded residual has the largest Euclidean norm is selected.
This is a simple diagnostic proxy; it does not measure adversarial strength,
planner vulnerability, or safety.

### Per-pedestrian independent search

Each targeted pedestrian is searched independently. This keeps the budget
predictable while avoiding joint-space combinatorial explosion. The configured
budget is a total cap for the proposal, and candidate/action order is retained
in the diagnostic record. The live controller enforces pairwise separation
again after the search returns.

### Candidate bound evaluation

Each grid candidate is evaluated by a fresh `BoundedResidualAdversary` with a
fixed one-candidate policy. This applies the stateful jerk, geometry, and
separation pipeline before the diagnostic objective is computed. The selected
candidate is then evaluated once more by the live controller, which carries the
actual prior residual state.

## Diagnostic record schema

The candidate/action arrays below are abbreviated to keep the schema example
compact; the emitted record contains the complete evaluated order.

```json
{
  "accepted": 3,
  "action_bounds": {"max_mps2": 1.5, "min_mps2": -1.5},
  "algorithm_name": "finite_grid_search_v1",
  "bound_settings": {"max_jerk_mps3": 7.5, "target_ped_idx": [0]},
  "budget": 9,
  "candidate_actions_mps2": [[-1.5, -1.5], [0.0, -1.5]],
  "candidate_order": ["ped_0:grid_000", "ped_0:grid_001"],
  "config_digest": "abcdef0123456789",
  "grid_points_per_dim": 3,
  "invalid": 0,
  "num_targeted_peds": 1,
  "objective_proxy": "maximize_residual_magnitude",
  "rejected": 6,
  "schema_version": "residual_search_diagnostic.v1",
  "seed": 42,
  "source_revision": "<git HEAD SHA>",
  "total_evaluated": 9
}
```

All keys are alphabetically sorted. No timestamps or absolute paths appear in
the canonical record. Candidate/action order is explicit, and repeated runs
from the same config and seed produce byte-equivalent JSON.

## Canonical smoke command

```bash
uv run python -c "
from pathlib import Path
import yaml, numpy as np
from robot_sf.ped_npc.residual_adversary import (
    BoundedResidualAdversary, ResidualAdversaryConfig,
)
from robot_sf.ped_npc.residual_search import (
    FiniteGridSearchPolicy, ResidualSearchConfig,
)

cfg_path = Path('configs/adversarial/issue_6911_residual_search.yaml')
payload = yaml.safe_load(cfg_path.read_text())
algo = payload['algorithm']
action = payload['action_bounds']
search_cfg = ResidualSearchConfig(
    algorithm_name=algo['name'],
    objective_proxy=algo['objective_proxy'],
    grid_points_per_dim=algo['grid_points_per_dim'],
    max_candidates=algo['max_candidates'],
    seed=algo['seed'],
    action_min_mps2=action['min_mps2'],
    action_max_mps2=action['max_mps2'],
)
residual_cfg = ResidualAdversaryConfig(**dict(payload['residual_adversary']))
policy = FiniteGridSearchPolicy(search_cfg, residual_cfg, dt_s=0.1, num_peds=2)
adversary = BoundedResidualAdversary(
    config=residual_cfg, policy=policy, dt_s=0.1, num_peds=2,
)
pos = np.array([[3.0, 1.0], [2.0, 4.0]])
vel = np.array([[0.5, 0.0], [0.0, 0.3]])
spd = np.array([1.5, 1.2])
robot = ((0.0, 0.0), 0.0)
for _ in range(10):
    adversary.step_residual(pos.copy(), vel.copy(), spd.copy(), robot)
print(policy.last_record.to_json(indent=2))
"
```

Run twice and compare output for byte-equivalence.  This proves deterministic
search bookkeeping, not planner quality or safety.

**Output / claim-status**: capability-only smoke evidence — deterministic
search plumbing, config digest, and diagnostic accounting.  This is **not**
benchmark, safety, stress-strength, or paper-facing evidence.

## Claim boundary (what this slice does NOT do)

This is a capability-only slice.  It makes **no** benchmark, planner-ranking,
safety, or paper-facing claim.  It adds **no** new stress-case metric.  The
grid search is a diagnostic baseline, not a statement about adversarial
strength or planner vulnerability.  It does **not** replace, compare against,
or imply anything about CMA-ES, MCTS, PPO, or matched-compute campaigns.

## Deferred slices

- CMA-ES or MCTS search-baseline adversary (sequenced before PPO).
- PPO / learned adversary (only after the search baseline is measured).
- Matched-compute comparison vs open-loop scenario optimisation.
- Stress-case validity/strength metrics (requires Domain-Aware Approval).
