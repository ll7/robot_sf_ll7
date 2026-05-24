# Issue #1396 Shielded PPO Repair Launch Packet

Date: 2026-05-24

## Scope

This note records the pre-SLURM launch packet for the Shielded PPO Repair Campaign. It does not
train PPO, submit SLURM, run stress/full-matrix evaluation, or promote a guarded learned policy.

## Launch Packet

- Config: `configs/training/shielded_ppo_issue_1396_launch_packet.yaml`
- Validator: `scripts/validation/validate_shielded_ppo_launch_packet.py`
- SLURM handoff: `docs/context/policy_search/SLURM/002_shielded_ppo_repair_campaign.md`
- Evidence fixture:
  `docs/context/evidence/issue_1396_shielded_ppo_launch_packet_2026-05-24/`

## Repair Hypothesis

The packet permits exactly one repair delta:

- keep the BR-06 v3 PPO architecture and seed strategy unchanged,
- keep the `risk_guarded_ppo_v1` runtime guard active in all evaluations,
- increase only `env_factory_kwargs.reward_kwargs.weights.collision` to `-20.0` for future
  fine-tuning.

This is intentionally narrow. Architecture sweeps, guard disabling, broad curriculum work, and
multiple reward hypotheses remain out of scope for #1396.

## Frozen References

- PPO baseline: `ppo_issue791_best_v1`
- Guarded reference: `risk_guarded_ppo_v1`
- Scenario slice: `nominal_sanity`
- Seeds: `111`, `112`, `113`
- Guarded candidate config: `configs/policy_search/candidates/risk_guarded_ppo_v1.yaml`
- Base training config:
  `configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml`

The tracked comparison-freeze stub is not benchmark evidence. A follow-up run must replace pending
artifact aliases with concrete durable pointers before reporting outcomes.

## Stop Gates

- Smoke gate: success `1.0`, collision `0.0`, guard fallback rate at most `0.60`.
- Nominal-sanity gate: success at least `0.2778`, collision at most `0.0556`, guard fallback rate
  at most `0.50`.

Stress or full-matrix escalation is allowed only after the nominal-sanity gate passes with the
runtime guard active and diagnostics present.

## Guard Diagnostics

The validator requires the future report or preflight contract to expose:

- `guard_veto_count`
- `guard_fallback_count`
- `raw_ppo_action`
- `guarded_action`
- `fallback_action_source`

## Validation

```bash
uv run python scripts/validation/validate_shielded_ppo_launch_packet.py \
  --config configs/training/shielded_ppo_issue_1396_launch_packet.yaml --json
```

Expected result: `status=valid`, campaign `issue_1396_shielded_ppo_repair_v1`, one repair delta,
runtime guard active, and smoke/nominal gates present.

Targeted tests:

```bash
uv run pytest -q tests/training/test_shielded_ppo_launch_packet.py
```

## Follow-Up Boundary

The follow-up SLURM issue must record the exact branch/commit, local preflight result, base
checkpoint, baseline artifacts, submission command, stop-gate outcomes, guard diagnostics, and
continue/revise/reject criteria. It must not disable the runtime guard to gain success.
