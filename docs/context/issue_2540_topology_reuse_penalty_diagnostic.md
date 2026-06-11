# Issue #2540 Topology Primary-Route Reuse-Penalty Diagnostic

Issue: [#2540](https://github.com/ll7/robot_sf_ll7/issues/2540)
Status: current, diagnostic-only implementation.

## Claim Boundary

This note implements and records the primary-route reuse-penalty mechanism selected by
[#2600](issue_2600_topology_revision_decision.md) via [#2563](issue_2563_topology_corrective_revision.md)
narrowing. It is diagnostic-only: the mechanism must pass a controlled paired diagnostic before
any benchmark, planner promotion, or leaderboard claim.

## Implementation

Added an explicit primary-route reuse-penalty mechanism to
`robot_sf/planner/topology_guided_local_policy.py`. The penalty is off by default; it activates
only when the candidate config enables `primary_route_reuse_penalty_enabled: true`.

### Mechanism

When enabled, the planner tracks recent hypothesis selections in a bounded deque. If the primary
route has been selected at least `primary_route_reuse_penalty_min_prior_primary_selections` times
within the last `primary_route_reuse_penalty_cooldown_steps` steps, and at least one alternative
hypothesis has an `eligible_near_parity_alternative` gate reason, then the primary route's
`selection_score` is reduced by `primary_route_reuse_penalty_weight * recent_primary_count`.

This allows near-parity alternatives to win selection when the primary route has been
repeatedly reselected without sufficient route-progress justification.

### Config Fields

| Field | Default | Description |
|---|---|---|
| `primary_route_reuse_penalty_enabled` | `false` | Enable the reuse-penalty mechanism |
| `primary_route_reuse_penalty_weight` | `1.0` | Penalty weight multiplied by recent primary count |
| `primary_route_reuse_penalty_cooldown_steps` | `3` | Window of recent steps to track |
| `primary_route_reuse_penalty_min_prior_primary_selections` | `2` | Minimum recent primary selections to trigger penalty |

### Diagnostic Fields

The decision output and route-corridor diagnostic payload now include:

- `reuse_penalty_applied` (bool): whether the penalty was applied this step
- `reuse_penalty_reason` (str | None): human-readable reason when applied
- `recent_primary_selection_count` (int): count of primary-route selections in the cooldown window
- `eligible_near_parity_alternative_exists` (bool): whether any alternative passed the near-parity gate

`scripts/validation/run_topology_hypothesis_diagnostics.py` also aggregates those fields under
`summary.topology_reuse_penalty` as applied-step counts, eligible near-parity alternative steps,
maximum recent primary-route count, and reason counts.

### New Candidate Config

`configs/policy_search/candidates/topology_guided_hybrid_rule_v0_reuse_penalty.yaml` is the
diagnostic candidate with the reuse penalty enabled. The baseline
`topology_guided_hybrid_rule_v0.yaml` is preserved unchanged for paired comparison.

## Diagnostic Launch Packet

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0_reuse_penalty \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2540_primary_route_reuse_penalty
```

Paired baseline comparator:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2540_baseline_comparator
```

## Required Diagnostic Fields

The diagnostic trace must expose:

- `reuse_penalty_applied`, `reuse_penalty_reason`, `recent_primary_selection_count`, `eligible_near_parity_alternative_exists`
- `summary.topology_reuse_penalty` with applied-step counts, eligible near-parity alternative counts, max recent primary-route count, and reason counts
- All existing fields from [#2563](issue_2563_topology_corrective_revision.md): `diagnostic_status`, topology status counts, route-selector selected-hypothesis counts, near-parity gate reasons, local command-source counts, topology-command influence counts, route-progress deltas, hypothesis switch count, terminal outcome

## Pass / Revise / Stop Rule

- `continue`: diagnostic completes, non-primary topology-command influence is preserved or
  increased against #2530 baseline, route-progress evidence improves without worse terminal
  behavior, and the paired comparator does not explain the gain away.
- `revise`: the mechanism runs but progress remains weak, terminal behavior stays
  `horizon_exhausted`, or switching volatility increases without route-progress benefit.
- `stop`: non-primary topology-command influence collapses to zero, the diagnostic fails closed on
  required fields, or the mechanism only changes route labels without a plausible progress signal.

## Evidence Chain

- [#2258](issue_2258_topology_primary_route_audit.md): primary-route overselection
- [#2403](issue_2403_topology_selection_score_decision.md): `primary_route_overselected` classification
- [#2518](issue_2518_topology_near_parity_gate.md): near-parity gate can produce non-primary selections
- [#2530](issue_2530_topology_near_parity_corrective_smoke.md): `revise` classification, weak route progress
- [#2563](issue_2563_topology_corrective_revision.md): selected reuse-penalty mechanism
- [#2570](issue_2570_topology_revise_status_propagation.md): diagnostic-only status propagation
- [#2600](issue_2600_topology_revision_decision.md): narrowed #2540 to this mechanism

## Validation

```bash
PYTHONPATH=. scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/planner/test_topology_guided_local_policy.py -q
PYTHONPATH=. scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/validation/test_run_topology_hypothesis_diagnostics.py -q
PYTHONPATH=. scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check robot_sf/planner/topology_guided_local_policy.py scripts/validation/run_topology_hypothesis_diagnostics.py tests/planner/test_topology_guided_local_policy.py tests/validation/test_run_topology_hypothesis_diagnostics.py
PYTHONPATH=. scripts/dev/run_worktree_shared_venv.sh -- uv run ruff format --check robot_sf/planner/topology_guided_local_policy.py scripts/validation/run_topology_hypothesis_diagnostics.py tests/planner/test_topology_guided_local_policy.py tests/validation/test_run_topology_hypothesis_diagnostics.py
PYTHONPATH=. scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2540_topology_reuse_penalty_diagnostic.md --path docs/context/README.md --path docs/context/INDEX.md --path docs/context/catalog.yaml --path docs/context/policy_search/candidate_registry.yaml --path docs/context/policy_search/candidate_registry_summary.md
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

## No Full Diagnostic Run

No full paired diagnostic has been run yet. The implementation, candidate registry entry, and
diagnostic summary plumbing are in place and targeted tests pass. The next step is to run the
canonical paired diagnostic from the launch packet above against the #2530 baseline before treating
any outcome as planner improvement.
