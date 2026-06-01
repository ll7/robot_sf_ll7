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

## Boundary

This is local smoke evidence, not benchmark promotion evidence. The local machine context forbids
SLURM submission from this host, so SLURM smoke replay remains a follow-up gate before any
nominal-sanity, stress, or full-matrix escalation.
