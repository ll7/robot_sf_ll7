# Issue #1474 Shielded PPO Repair Closeout 2026-06-01

Date: 2026-06-01

## Scope

This note records the completed issue #1474 shielded-PPO repair training run and the post-training
evidence boundary. It supersedes the interim milestone comments/notes for job `12674` but does not
promote the checkpoint to benchmark evidence.

## Result

SLURM job `12674` completed successfully on `a30`:

- branch/worktree: `issue-1474-shielded-ppo-repair` at commit
  `cc3e8552b0fa1ae47ddb3f42cd74443576c6e9c0`;
- config:
  `configs/training/ppo/ablations/expert_ppo_issue_1474_shielded_repair_collision20_5m.yaml`;
- W&B run: `ll7/robot_sf/d8w8uykh`;
- W&B artifact:
  `ll7/robot_sf/ppo_expert_issue_1474_shielded_repair_collision20_5m-best-success:v5`;
- local synced output root:
  `output/slurm/issue1474-shielded-ppo-repair-job-12674`.

The best-success checkpoint was selected at 5M steps:

| Step | Success | Collision | SNQI | Path Efficiency | Eval Return |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500000 | 0.36 | 0.61 | -1.5825 | 0.7817 | -0.4372 |
| 1000000 | 0.66 | 0.33 | -0.6573 | 0.8169 | 16.1612 |
| 1500000 | 0.64 | 0.37 | -0.7081 | 0.8281 | 20.9957 |
| 2000000 | 0.73 | 0.27 | -0.4549 | 0.8240 | 23.4117 |
| 2500000 | 0.75 | 0.25 | -0.3275 | 0.8181 | 24.2932 |
| 3000000 | 0.80 | 0.19 | -0.1731 | 0.8258 | 24.7754 |
| 4000000 | 0.75 | 0.24 | -0.3784 | 0.8153 | 26.8277 |
| 5000000 | 0.83 | 0.16 | -0.1023 | 0.8145 | 27.9261 |

## Interpretation

The repair direction is promising as training evidence: final success is higher and collision is
lower than every earlier scheduled eval gate. It still missed the launch packet's training
convergence target (`success_rate >= 0.9`, `collision_rate <= 0.05`) and PPO frequently early-stopped
on max KL, so the result should be treated as a candidate checkpoint rather than a solved policy.

The aggregate generated expert-policy manifest remains `validation_state: draft`; its mean metrics
summarize the full training evaluation history and are weaker than the final checkpoint:
`success_rate.mean=0.69`, `collision_rate.mean=0.3025`.

## Artifact Boundary

Compact provenance is tracked in
`docs/context/evidence/issue_1474_shielded_ppo_repair_2026-06-01/artifact_manifest.md`.
The raw checkpoint is durable through W&B, not git. Local `output/slurm/...` files are caches and
should not be cited without the W&B artifact pointer.

## Post-Training Candidate

The runtime-guarded policy-search candidate is
`configs/policy_search/candidates/shielded_ppo_issue1474_collision20_v1.yaml`. It keeps the
`risk_guarded_ppo_v1` guard parameters active and swaps only the PPO model id to the issue #1474
repair checkpoint.

Required stop gates from the launch packet:

- smoke: success `1.0`, collision `0.0`, guard fallback rate at most `0.60`;
- nominal-sanity: success at least `0.2778`, collision at most `0.0556`, guard fallback rate at
  most `0.50`.

Stress or full-matrix escalation remains blocked until the guarded smoke and nominal-sanity evidence
exists and diagnostics separate raw PPO proposals from guarded/fallback actions.

## Guarded Smoke Result

After wiring the candidate, SLURM job `12685` ran the guarded smoke gate:

```bash
scripts/dev/sbatch_policy_search_sweep.sh --stage smoke --throttle 1 --workers 1 \
  --run-id issue1474_collision20_post_training_smoke \
  --candidates-file output/policy_search/issue1474_post_training_candidates.txt \
  --sbatch-arg --time=02:00:00
```

Result:

- SLURM job: `12685_0`, `COMPLETED`, exit `0:0`, elapsed `00:00:59`;
- output summary:
  `output/policy_search/shielded_ppo_issue1474_collision20_v1/smoke/issue1474_collision20_post_training_smoke/summary.json`;
- generated report:
  `docs/context/policy_search/reports/2026-06-01_shielded_ppo_issue1474_collision20_v1_smoke.md`;
- generic policy-search decision: `pass`;
- launch-packet smoke gate: `fail`, because `success_rate=0.0` while the launch packet requires
  `1.0`.

Smoke metrics:

| Episodes | Success | Collision | Near Miss | Mean Avg Speed | Failure Mode |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `overconservative_stop` |

Guard diagnostics in the smoke JSONL show `shield_decision_count=80`,
`shield_intervention_count=0`, `shield_override_count=0`, and all 80 guarded-PPO decisions were
classified as `goal_reached`, yielding zero motion and a timeout. This blocks nominal-sanity
escalation. The next repair should investigate the guarded-PPO goal/tolerance or observation/action
handoff for this new checkpoint before spending more validation allocation.

## Validation Commands

Metadata and launch-packet checks:

```bash
uv run python scripts/validation/validate_shielded_ppo_launch_packet.py \
  --config configs/training/shielded_ppo_issue_1396_launch_packet.yaml --json
uv run python scripts/validation/check_learned_policy_eligibility.py \
  docs/context/evidence/issue_1474_shielded_ppo_repair_2026-06-01/learned_policy_eligibility.yaml
```

Runtime gates should run through the policy-search SLURM wrapper from this worktree:

```bash
printf '%s\n' shielded_ppo_issue1474_collision20_v1 > output/policy_search/issue1474_post_training_candidates.txt
scripts/dev/sbatch_policy_search_sweep.sh --stage smoke --throttle 1 --workers 1 \
  --run-id issue1474_collision20_post_training_smoke \
  --candidates-file output/policy_search/issue1474_post_training_candidates.txt
scripts/dev/sbatch_policy_search_sweep.sh --stage nominal_sanity --throttle 1 --workers 2 \
  --run-id issue1474_collision20_post_training_nominal \
  --candidates-file output/policy_search/issue1474_post_training_candidates.txt
```

Do not run the nominal-sanity command above until the guarded smoke gate succeeds under the
launch-packet stop criteria.
