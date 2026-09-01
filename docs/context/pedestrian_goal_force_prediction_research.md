# Public pedestrian goal and force prediction research

This note maps the current research direction for observation-only pedestrian goal candidates and
force-aware prediction; the delivered implementation is a bounded interface, not evidence that
the inferred goals match human intent.

## Current state

Issue [#8068](https://github.com/ll7/robot_sf_ll7/issues/8068) supplies the one-frame heading
posterior baseline. Issue [#8073](https://github.com/ll7/robot_sf_ll7/issues/8073) adds the public
candidate-generation boundary used before inference. Both paths are actor-safe: map annotations,
public route geometry, and causal tracked state are allowed; assigned routes, true goals, active
waypoint indices, and future trajectories are not.

The candidate provider keeps final destinations separate from active waypoints, preserves distinct
route signatures up to a configured homotopy cap, records path tangents and fallback mode, and
emits direction-only open rays plus an unconditional unknown hypothesis. An oracle-only coverage
evaluator is available after the candidate set is frozen. Its output is coverage diagnostics only.

## Research question and competing explanations

Can public map geometry, semantic destinations, route topology, and current tracked state provide
adequate goal coverage without forcing a wrong finite destination or leaking simulator truth?

The bounded implementation is compatible with these exploratory explanations:

1. Candidate-set coverage may be the first bottleneck for route-level inference.
2. A path tangent around geometry may explain immediate force direction better than a direct endpoint
   vector.
3. Open-ray and unknown hypotheses may preserve ambiguity when map topology is incomplete.
4. Increasing candidate density may improve coverage while harming runtime and calibration.

These are hypotheses, not results. Testing them requires a preregistered comparator, held-out
episodes, explicit route/goal provenance, and a separate evidence decision.

## Delivered contract and evidence boundary

The canonical owner is `robot_sf/prediction/goal_candidate_provider.py`. The config-first smoke is
[`run_goal_candidate_provider_smoke.py`](../../scripts/research/run_goal_candidate_provider_smoke.py),
and its compact receipt is
[`issue_8073_goal_candidate_provider_smoke_2026-09-01.json`](evidence/issue_8073_goal_candidate_provider_smoke_2026-09-01.json).

The receipt covers open room, straight corridor, doorway, crossing, multiple homotopies, blocked
destination, and an intentionally absent true goal. It records candidate-set/config/map digests,
stable candidate IDs, role/source/provenance summaries, rejection reasons, observed runtime, and
separate coverage diagnostics. The evidence status is `smoke evidence` with the claim boundary
`implementation_integrity_and_candidate_coverage_only`; it does not establish prediction accuracy,
calibration, planner improvement, human preference, benchmark ranking, or paper-facing conclusions.

Reproduce it with:

```bash
scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- uv run python \
  scripts/research/run_goal_candidate_provider_smoke.py \
  --output-json output/issue-8073-goal-candidate-smoke.json
```

## Next bounded research directions

- Compare direct endpoint, public route tangent, and force-coupled likelihoods on held-out tracks
  while keeping candidate generation frozen.
- Measure candidate coverage and unknown-needed rates by map family before interpreting posterior
  accuracy.
- Add a stateful route/waypoint posterior only after the candidate bytes and oracle boundary are
  independently audited.
- Study source ablations and candidate-cap/runtime trade-offs with public-prior provenance; do not
  fit priors from evaluation trajectories.

The condition that would change the priority is a reproducible held-out result showing that public
candidate coverage is already adequate and that posterior error, rather than candidate-set error,
dominates. Until then, the candidate-generation and coverage boundary remains exploratory and
diagnostic rather than benchmark evidence.
