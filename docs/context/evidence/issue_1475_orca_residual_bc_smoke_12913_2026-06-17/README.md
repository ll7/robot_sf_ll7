# Issue 1475 ORCA Residual BC Smoke Closeout

This bundle preserves the compact retrieved evidence for Slurm job `12913`,
submitted from `robot_sf_ll7@dcb14927f08277d123d9666ad91b8d6abbc4fe9d`.

The job reached the smoke/finalizer path and failed closed. It should not be
escalated to a nominal rerun until the smoke evidence contract and progress
behavior are fixed.

## Result

- Slurm job: `12913`
- Private ops ledger label: `issue1475-orca-residual-bc`
- Candidate: `orca_residual_guarded_ppo_progress_v1`
- Stage/profile: `smoke` / `experimental`
- Smoke episodes: `1`
- Success rate: `0.0`
- Collision rate: `0.0`
- Termination reason: `max_steps`
- Failure mode: `timeout_low_progress`
- Finalizer classification: `missing_required_smoke_evidence`
- Nominal escalation allowed: `false`

## Blocking Evidence Gap

The finalizer required these smoke evidence fields and found them missing:

- `residual_clipping_rate`
- `guard_veto_rate`
- `fallback_degraded_status`
- `artifact_pointer_status`

The smoke JSONL includes shield/intervention metrics, but the required residual
BC guardrail fields are not emitted at the evidence contract boundary. Treat this
as an instrumentation/candidate readiness blocker, not as a transient Slurm
failure.

## Files

- `summary.json`: compact closeout summary and next action.
- `policy_search/orca_residual_guarded_ppo_progress_v1/smoke/issue1475_smoke/summary.json`:
  smoke summary emitted by the policy-search runner.
- `policy_search/orca_residual_guarded_ppo_progress_v1/smoke/issue1475_smoke/issue1475_smoke_evidence_manifest.json`:
  failed-closed finalizer manifest.
- `policy_search/orca_residual_guarded_ppo_progress_v1/smoke/issue1475_smoke/smoke__orca_residual_guarded_ppo_progress_v1.jsonl`:
  one smoke episode record.
- `issue1475_materialized_candidate/`: materialized candidate metadata used by the run.
- `source_slurm_checksum_manifest.sha256`: retrieved Slurm checksum manifest, including
  large NPZ/ZIP artifacts intentionally not tracked in this bundle.
- `SHA256SUMS`: checksums for this tracked bundle.

## Validation

```bash
python3 -m json.tool docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/summary.json >/dev/null
python3 -m json.tool docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/policy_search/orca_residual_guarded_ppo_progress_v1/smoke/issue1475_smoke/summary.json >/dev/null
(cd docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17 && sha256sum -c SHA256SUMS)
```
