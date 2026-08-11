# Issue #6917 — Alternative diagnostic objective proxy for residual grid search

Status: diagnostic-only capability slice.
Evidence grade: not benchmark evidence. No paper, metric, or safety claim.

## Plain-language summary

This slice adds one alternative diagnostic objective proxy,
`minimize_predicted_robot_distance`, to the existing deterministic finite-budget
grid search baseline (#6911). The proxy ranks candidates by how close the
one-step predicted pedestrian position places the targeted pedestrian to the
robot, using positions, velocities, and the robot pose already available at the
search seam. No planner or episode-state coupling is introduced.

## What ships (the contract)

- `robot_sf/ped_npc/residual_search.py`:
  - `SUPPORTED_OBJECTIVE_PROXIES` — frozen set of recognised proxy names
    (`maximize_residual_magnitude`, `minimize_predicted_robot_distance`).
  - `ResidualSearchConfig` validates that `objective_proxy` is in the
    supported set; rejects unsupported or malformed values.
  - `_evaluate_candidate` accepts an `objective_proxy` keyword. For
    `minimize_predicted_robot_distance` it computes the negative Euclidean
    distance from the one-step predicted position (nominal velocity displacement
    plus bounded residual displacement) to the robot position, preserving the
    existing maximise-convention ranking.
  - `FiniteGridSearchPolicy.propose_residual` passes the configured proxy
    through to `_evaluate_candidate`.
  - `SearchDiagnosticRecord` and `to_json` emit the objective proxy name so
    every diagnostic record explains the candidate ranking.
- `configs/adversarial/issue_6917_objective_proxy.yaml` — versioned config
  using the new proxy with the same tiny fixed-seed grid as #6911.
- `tests/adversarial/test_residual_search.py` — focused tests for the new
  proxy: config validation, two distinct finite scores, deterministic repeated
  records, malformed input rejection, budget accounting, full
  candidate-through-controller coverage, bound preservation, and cross-proxy
  differentiation.
- `docs/context/issue_6917_objective_proxy.md` — this note.

## Design decisions

### Proxy selection: predicted robot proximity

The chosen proxy uses one-step predicted pedestrian proximity to the robot:
`predicted_position = position + velocity * dt_s + bounded_residual * dt_s^2`,
then `score = -||predicted_position - robot_position||`. This is a one-step
observable interaction signal from inputs already present at the search seam
(`positions`, `velocities`, `robot_pose`, `dt_s`, `bounded_residual`). It does
not require planner access, episode-state coupling, or a scientific stress
metric.

### Negative distance convention

The search loop ranks candidates by `score > best_score + EPSILON`, i.e. it
maximises. By returning negative distance, candidates that place the pedestrian
closest to the robot are selected without modifying the accounting logic.

### Fail-closed on malformed input

When `objective_proxy` is `minimize_predicted_robot_distance` but `robot_pose`
is missing or non-finite, the evaluation catches the `ValueError` and returns
`(0.0, False)`, preserving the existing invalid-candidate accounting path.

## Diagnostic record schema (new fields)

The `objective_proxy` field in the diagnostic record is the only schema
addition. The existing schema version `residual_search_diagnostic.v1` is
unchanged because the record already carries `objective_proxy`:

```json
{
  "objective_proxy": "minimize_predicted_robot_distance",
  "schema_version": "residual_search_diagnostic.v1"
}
```

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

cfg_path = Path('configs/adversarial/issue_6917_objective_proxy.yaml')
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

Run twice and compare output for byte-equivalence. This proves deterministic
search bookkeeping, not planner quality or safety.

**Output / claim-status**: capability-only smoke evidence — deterministic
search plumbing, config digest, and diagnostic accounting with the alternative
proxy. This is **not** benchmark, safety, stress-strength, or paper-facing
evidence.

## Claim boundary (what this slice does NOT do)

This is a capability-only slice. It makes **no** benchmark, planner-ranking,
safety, or paper-facing claim. It adds **no** new stress-case metric. The
alternative proxy is a diagnostic ranking signal, not a statement about
adversarial strength or planner vulnerability. It does **not** replace, compare
against, or imply anything about CMA-ES, MCTS, PPO, or matched-compute
campaigns. The proxy is selected from signals already available at the search
 seam; it does not invent a scientific stress metric.

## Deferred slices

- Matched-compute comparison vs open-loop scenario optimisation (the claim to
  test: reactivity finds failures open-loop search cannot, at equal simulator
  budget).
- Stress-case validity/strength metrics (requires Domain-Aware Approval).
- Calibration of the proximity proxy against planner-observed near-miss rates
  (requires planner instrumentation not in this slice).
