# Issue #1387 Tentabot-Style Value Scorer Spike

## Scope

This spike adds Robot SF-native `tentabot_value_scorer_*` candidates without importing upstream
Tentabot code, training data, ROS/Gazebo dependencies, OctoMap assets, or source checkpoints. They
are experimental candidates behind `allow_testing_algorithms: true`, not Tentabot parity claims and
not benchmark-ready evidence.

## Candidate Contract

- Candidate config: `configs/policy_search/candidates/tentabot_value_scorer_v0.yaml`
- Algo config: `configs/algos/tentabot_value_scorer_v0.yaml`
- Runtime algorithm: `hybrid_rule_local_planner`
- Planner variant: `tentabot_value_scorer_v0`
- Candidate lattice: existing hybrid-rule unicycle command lattice plus route-guide and
  route-corridor recovery candidates.
- Feature scope: route progress, static clearance, pedestrian distance/TTC, smoothness, and command
  bounds.
- V0 scorer: hand-scored linear teacher weights from Robot SF hybrid-rule diagnostics. This is a
  supervised-spike baseline in the sense that the weights imitate the repository's current safe
  hand-scored teacher, not a trained learned model.
- V1 static-gated scorer: same clean-room lattice and value terms with an added deterministic
  static-safety demotion tier. The retained v1 config demotes low-clearance candidates across all
  accepted sources unless they make positive progress without worsening static clearance.

## Diagnostics

The planner records `value_scorer` metadata in both aggregate diagnostics and the latest decision:

- `profile`
- `training_source`
- `candidate_lattice`
- `source_parity_claim: false`
- `upstream_code_used: false`
- `observation_scope`

Per-step diagnostics continue to expose selected command/source/score, score terms, top-k accepted
candidates, rejected examples, rejection counts by source, nearest pedestrian/static distances, TTC,
route-corridor diagnostics, and unavailable counts/examples for optional candidate sources such as
the route-corridor subgoal primitive. These fields are the audit surface for selected, rejected, and
unavailable candidates.

## Validation Plan

Required local checks:

```bash
uv run pytest tests/planner/test_hybrid_rule_local_planner.py -q
uv run python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v0 --stage smoke --workers 1
uv run python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v0 --stage nominal_sanity --workers 1
```

Compare the same-seed outputs against `risk_dwa` and the active `hybrid_rule_v3_*` baseline before
classifying the spike as continue, revise, or stop.

## 2026-05-20 Result

Validation commands run:

```bash
uv run pytest tests/planner/test_hybrid_rule_local_planner.py -q
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v0 --stage smoke --workers 1
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v0 --stage nominal_sanity --workers 1
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_fast_progress_static_escape --stage nominal_sanity --workers 1
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate risk_dwa_camera_ready --stage nominal_sanity --workers 1
```

Summary:

| Candidate | Stage | Episodes | Success | Collision | Near miss | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `tentabot_value_scorer_v0` | smoke | 1 | 1.0000 | 0.0000 | 0.0000 | pass |
| `tentabot_value_scorer_v0` | nominal_sanity | 18 | 0.2222 | 0.1111 | 0.2222 | revise |
| `hybrid_rule_v3_fast_progress_static_escape` | nominal_sanity | 18 | 0.2222 | 0.0000 | 0.2222 | revise |
| `risk_dwa_camera_ready` | nominal_sanity | 18 | 0.0000 | 0.3333 | 0.1667 | revise |

Classification: **revise**.

Interpretation: the clean-room Tentabot-style candidate path works and exposes the expected
diagnostics, but the v0 hand-scored value weights do not beat the relevant hybrid-rule baseline on
same-seed nominal sanity and introduce two static collisions. It is safer than the registered
`risk_dwa_camera_ready` comparison on collision rate, but it is not a promotion candidate. The next
iteration should tune or learn only after preserving the hybrid-rule static collision gate.

## 2026-05-31 Issue #1832 Progress-Recovery Probe

Issue #1832 revisited the low-progress timeout failure after the #1826 safety retune. Validation
used the same `tentabot_value_scorer_v0` smoke and nominal-sanity stages:

```bash
LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v0 --stage smoke --horizon 80 --workers 1 --output-dir output/policy_search/tentabot_value_scorer_v0/smoke/issue1832_final_h80
LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v0 --stage nominal_sanity --workers 2 --output-dir output/policy_search/tentabot_value_scorer_v0/nominal_sanity/issue1832_final
```

Summary:

| Candidate state | Stage | Episodes | Success | Collision | Near miss | Low-progress timeouts | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| retained regression-recovery gate | smoke | 1 | 1.0000 | 0.0000 | 0.0000 | 0 | pass |
| retained regression-recovery gate | nominal_sanity | 18 | 0.2222 | 0.0556 | 0.1667 | 11 | revise |
| progress-pressure config probe | nominal_sanity | 18 | 0.2222 | 0.1667 | 0.2222 | 8 | rejected |
| static-recovery config probe | nominal_sanity | 18 | 0.2222 | 0.1111 | 0.1667 | 10 | rejected |

Classification: **revise / stop this retune lane**.

Interpretation: treating negative goal-distance progress as stalled is the correct fail-closed
activation semantics for corridor-subgoal recovery, but it did not change the nominal-sanity
aggregate for the current candidate. The config retunes that reduced low-progress timeouts did so
by increasing static collisions, so they were not retained. Future work should add a stronger
route-aware progress objective or trace-level recovery policy before changing speed/clearance
weights again.

## 2026-05-31 Issue #1877 Static-Safety Gate Probe

Issue #1877 tested a bounded `tentabot_value_scorer_v1_static_gated` variant after #1832 showed
that lower low-progress timeouts were easy to buy by increasing static collisions. The new variant
keeps the v0 hand-scored value terms and records both `raw_value_score` and
`static_safety_gate` diagnostics, then subtracts a deterministic penalty from low-clearance
candidates that do not make safe positive progress.

Validation commands run:

```bash
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/planner/test_hybrid_rule_local_planner.py -q
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/validation/test_validate_policy_search_registry.py -q
scripts/dev/run_worktree_shared_venv.sh -- ruff check robot_sf/planner/hybrid_rule_local_planner.py tests/planner/test_hybrid_rule_local_planner.py
LOGURU_LEVEL=WARNING scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v0 --stage nominal_sanity --workers 1
LOGURU_LEVEL=WARNING scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v1_static_gated --stage smoke --workers 1
LOGURU_LEVEL=WARNING scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v1_static_gated --stage nominal_sanity --workers 1
```

Summary:

| Candidate | Stage | Episodes | Success | Collision | Near miss | Low-progress timeouts | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tentabot_value_scorer_v0` | nominal_sanity | 18 | 0.2222 | 0.0556 | 0.1667 | 11 | revise |
| `tentabot_value_scorer_v1_static_gated` | smoke | 1 | 1.0000 | 0.0000 | 0.0000 | 0 | pass |
| `tentabot_value_scorer_v1_static_gated` | nominal_sanity | 18 | 0.2222 | 0.1111 | 0.2222 | 9 | revise |

Classification: **revise / stop this static-gate lane**.

Interpretation: the static-gated scorer is executable and its diagnostics make the route/safety
tradeoff inspectable, but it is not an improvement over v0 on the same nominal-sanity slice. It
reduced low-progress timeouts from 11 to 9 while doubling static collisions from 1 to 2 and
increasing near-miss episodes from 3 to 4. Treat this as negative diagnostic evidence: future
Tentabot-style work should move to trace-level recovery policy design or a genuinely learned value
model rather than stronger hand-tuned static penalties.

## 2026-05-31 Issue #1877 Route-Arc Progress Probe

Issue #1877 next tested `tentabot_value_scorer_v2_route_arc`, a clean-room scorer variant that keeps
the v1 static-safety gate unchanged and adds a first-class `route_arc_progress` value term for every
accepted candidate when route geometry is available. This is a route-local progress mechanism, not a
speed, clearance, or static-gate retune.

Validation commands run:

```bash
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/planner/test_hybrid_rule_local_planner.py -q
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/validation/test_run_policy_search_candidate.py tests/validation/test_validate_policy_search_registry.py -q
LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v2_route_arc --stage smoke --workers 1 --output-dir output/policy_search/tentabot_value_scorer_v2_route_arc/smoke/issue1877_v2
LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v2_route_arc --stage nominal_sanity --workers 2 --output-dir output/policy_search/tentabot_value_scorer_v2_route_arc/nominal_sanity/issue1877_v2
```

Summary:

| Candidate | Stage | Episodes | Success | Collision | Near miss | Low-progress timeouts | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tentabot_value_scorer_v2_route_arc` | smoke | 1 | 1.0000 | 0.0000 | 0.0000 | 0 | pass |
| `tentabot_value_scorer_v2_route_arc` | nominal_sanity | 18 | 0.2222 | 0.1111 | 0.2778 | 8 | revise |

Classification: **revise / stop this hand-scored route-arc lane**.

Interpretation: route-arc scoring is executable and the unit tests prove it can rank route-local
progress ahead of misleading Euclidean goal progress. The nominal-sanity result still fails the
issue acceptance rule: low-progress timeouts drop from the #1832 retained baseline's 11/18 to 8/18,
but static collisions rise from 1/18 to 2/18 and near misses rise from 3/18 to 5/18. Treat this as
negative diagnostic evidence for hand-scored route-progress weighting. Further Tentabot-style work
should be trace-level recovery policy or learned value estimation, not larger route-progress weights.

## 2026-05-31 Issue #1908 Trace-Recovery Probe

Issue #1908 tested `tentabot_value_scorer_v3_trace_recovery`, a clean-room scorer variant that keeps
the v1 static-safety gate unchanged and adds an explicit trace-level recovery selector. The selector
only chooses already accepted `corridor_subgoal` or `route_guide` candidates when recent route
diagnostics show route regression or combined route/goal stall. It does not relax hard static or
dynamic rejection, speed, clearance, static-gate, or scalar route-progress weights.

Validation commands run:

```bash
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/planner/test_hybrid_rule_local_planner.py -q
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/validation/test_run_policy_search_candidate.py tests/validation/test_validate_policy_search_registry.py -q
LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v3_trace_recovery --stage smoke --workers 1 --output-dir output/policy_search/tentabot_value_scorer_v3_trace_recovery/smoke/issue1908_v3
LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v3_trace_recovery --stage nominal_sanity --workers 2 --output-dir output/policy_search/tentabot_value_scorer_v3_trace_recovery/nominal_sanity/issue1908_v3
```

Summary:

| Candidate | Stage | Episodes | Success | Collision | Near miss | Low-progress timeouts | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tentabot_value_scorer_v3_trace_recovery` | smoke | 1 | 0.0000 | 0.0000 | 0.0000 | 1 | pass |
| `tentabot_value_scorer_v3_trace_recovery` | nominal_sanity | 18 | 0.2222 | 0.1111 | 0.1667 | 9 | revise |

Classification: **revise / stop this trace-recovery lane**.

Interpretation: trace recovery is executable and diagnostics show when it activates, is held, or is
blocked. It does not satisfy the issue acceptance rule. The smoke stage times out from low progress,
and nominal sanity matches v1's 9/18 low-progress timeouts while keeping the same 2/18 static
collisions. Near misses return to the #1832 retained baseline level at 3/18, but the static-collision
regression remains. Treat this as negative diagnostic evidence for hand-authored Tentabot recovery
logic. The next meaningful lane is a learned value estimator or a different planner family, not
another hand-tuned Tentabot recovery variant.
