# Issue #3952 Observation Robustness Smoke

## Claim Boundary
Same-scenario same-seed diagnostic robustness delta under non-calibrated observation perturbations. Not hardware sensor model and not paper-facing benchmark evidence.

## Inputs
- Nominal JSONL: `output/benchmarks/issue_3952/issue_3952_nominal_smoke/runs/goal__differential_drive/episodes.jsonl`
- Perturbed JSONL: `output/benchmarks/issue_3952/issue_3952_perturbed_smoke/runs/goal__differential_drive/episodes.jsonl`
- Paired rows: `1`
- Unmatched nominal rows: `0`
- Unmatched perturbed rows: `0`

## Planner Robustness Delta

| planner | algo | paired episodes | nominal success | perturbed success | success delta | nominal collision | perturbed collision | collision delta | perturbation profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| goal | goal | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | issue_3952_robustness_smoke_v1 |

## Caveats
- CPU smoke only.
- Same-scenario same-seed diagnostic comparison only.
- Observation perturbations are non-calibrated benchmark perturbations, not hardware sensor evidence.
- Nominal metric definitions unchanged.
