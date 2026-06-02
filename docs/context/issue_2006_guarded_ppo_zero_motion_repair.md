# Issue #2006 Guarded-PPO Zero-Motion Repair 2026-06-01

## Scope

Issue #2006 repairs the first post-training smoke failure for
`shielded_ppo_issue1474_collision20_v1`. The failure was not a new training failure: the runtime
handoff evaluated the #1474 checkpoint through the default guarded-PPO observation path instead of
the SocNav structured observation contract used during training.

## Root Cause

Two local guard assumptions made the failure look like an overconservative stop:

- `configs/policy_search/candidates/shielded_ppo_issue1474_collision20_v1.yaml` did not request
  `observation_mode: socnav_state`, so the PPO and guard received the wrong benchmark observation
  surface.
- `GuardedPPOAdapter` selected `goal.next` whenever it was nonzero and did not honor array-shaped
  pedestrian counts. With padded/default observations this could produce repeated
  `goal_reached` decisions or treat padded zero pedestrian rows as real near-field blockers.

## Local Repair Evidence

Local validation from branch `issue-2006-guarded-ppo-zero-motion`:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/planner/test_guarded_ppo.py \
  tests/validation/test_run_policy_search_candidate.py::test_shielded_ppo_issue1474_candidate_requests_training_observation_contract -q
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/validate_policy_search_registry.py \
  docs/context/policy_search/candidate_registry.yaml --max-age-days 30
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/validate_shielded_ppo_launch_packet.py \
  --config configs/training/shielded_ppo_issue_1396_launch_packet.yaml --json
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
  scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/run_policy_search_candidate.py \
  --candidate shielded_ppo_issue1474_collision20_v1 \
  --stage smoke --workers 1 --output-dir output/policy_search_issue2006_local
```

The repaired local smoke passed the launch-packet smoke gate:

| Metric | Value |
| --- | ---: |
| `success_rate` | `1.0` |
| `collision_rate` | `0.0` |
| `near_miss_rate` | `0.0` |
| `mean_avg_speed` | `1.9996905469041861` |
| `shield_decision_count` | `71` |
| `shield_intervention_rate` | `0.0` |
| `shield_override_rate` | `0.0` |

The guard diagnostics were `decision_counts.ppo_clear=71`, replacing the previous
`decision_counts.goal_reached=80` zero-motion timeout.

## SLURM Replay

Issue #2029 replayed the repaired handoff through SLURM on 2026-06-02:

- job: `12700_0` / `rsf-pol-smoke`, `COMPLETED`, exit `0:0`, elapsed `00:00:58`;
- commit: `2b86ef4cf5c10553ae2a0cf32c91a243164c44d2`;
- run id: `issue2029_shielded_ppo_smoke_replay`;
- summary:
  `output/policy_search/shielded_ppo_issue1474_collision20_v1/smoke/issue2029_shielded_ppo_smoke_replay/summary.json`;
- report:
  `docs/context/policy_search/reports/2026-06-02_shielded_ppo_issue1474_collision20_v1_smoke.md`;
- decision: `pass`;
- smoke metrics: `success_rate=1.0`, `collision_rate=0.0`, `near_miss_rate=0.0`,
  `mean_avg_speed=1.9997802215217393`;
- guard diagnostics: `shield_decision_count=71`, `shield_intervention_count=0`,
  `shield_override_count=0`, `decision_counts.ppo_clear=71`.

## Boundary

The zero-motion handoff blocker is repaired for both local and SLURM smoke. This remains
single-episode smoke evidence, not benchmark promotion evidence. Nominal-sanity is the next gate;
do not run stress or full-matrix escalation unless nominal-sanity passes with guard diagnostics and
artifact provenance recorded.
