# Issue #1428 ORCA-Residual Training Lineage

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1428>

## Decision

The first learned ORCA-residual policy uses behavior cloning. Its supervised target is the bounded
delta between a PPO leader action and the ORCA command, evaluated as an ORCA command plus clipped
learned residual under the existing hard guard.

The packet is intentionally pre-SLURM:

- `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml`
- `scripts/validation/validate_orca_residual_lineage_packet.py`
- `docs/context/policy_search/SLURM/005_orca_residual_bc_lineage.md`

## Boundary

This work stages the lineage, observation, artifact, and diagnostic contract. It does not train a
checkpoint, submit Slurm, or promote `orca_residual_guarded_ppo_v0` as learned-policy evidence.

## Evidence

Small tracked evidence lives in
`docs/context/evidence/issue_1428_orca_residual_lineage_2026-05-24/`.

The existing `orca_residual_guarded_ppo_v0` runtime surface was rerun as a smoke check on commit
`e14e2f8bc2058d9f0e071219629915dd5b5dd5a8` and passed one nominal smoke episode with success
`1.0`, collision `0.0`, and near-miss `0.0`. This is runtime-surface proof only, not learned
residual training evidence.

The packet requires:

- residual dataset manifest,
- candidate YAML,
- checkpoint pointer,
- diagnostic report path,
- residual contribution/clipping diagnostics,
- guard veto and fallback/degraded status.

Fallback/degraded rows must remain caveats and must not count as successful learned-residual rows.

## Validation

Local preflight:

```bash
uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --json
```

The bounded Slurm follow-up should replace the pending artifact aliases with concrete durable
pointers before collection/training begins, then run smoke before nominal and stop rather than
escalate if residual contribution or hard-guard status is ambiguous.
