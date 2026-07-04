# Issue #3300 False-Positive Actor-Injection Replay

## Claim Boundary
CPU-local observation-quality replay smoke for false-positive actor injection. Diagnostic only; not a full benchmark campaign, hardware sensor model, or paper-facing claim.

## Classification
- Label: `blocked_unavailable`
- Reason: no perturbed episode recorded pedestrians_added > 0
- Replay mode: `executable`

## Pairing
- Paired rows: `1`
- Unmatched nominal rows: `0`
- Unmatched perturbed rows: `0`

## Injection Summary
- Pedestrians added: `0`
- Steps with noise: `0`
- Perturbation profiles: `issue_3300_false_positive_actor_injection_v1`
- Perturbation hashes: `f3149ea5da06`

## Episode Deltas

| planner | scenario | seed | pedestrians added | changed fields |
|---|---|---:|---:|---|
| goal | single_ped_crossing_orthogonal | 0 | 0 | none |

## Caveats
- CPU replay smoke only.
- False-positive effects are reported separately from other observation noise.
- No full benchmark campaign, Slurm/GPU submission, or paper-facing claim.
