# Issue #1387 Tentabot-Style Value Scorer Spike

## Scope

This spike adds a Robot SF-native `tentabot_value_scorer_v0` candidate without importing upstream
Tentabot code, training data, ROS/Gazebo dependencies, OctoMap assets, or source checkpoints. It is
an experimental candidate behind `allow_testing_algorithms: true`, not a Tentabot parity claim and
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
