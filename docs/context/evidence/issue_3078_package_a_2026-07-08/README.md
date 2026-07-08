# Issue #3078 Package A Evidence

- Issue: #3078
- Parent: #3057
- Package: A (seed/planner-rank stability + held-out-family transfer)
- Generated: 2026-07-08
- Classification: `diagnostic`

## Scope

Package A seed-sufficiency analysis, planner-rank stability, and held-out-family transfer
pilot using the retained campaign bundles from issue_1484 and issue_1454. This is
diagnostic evidence; real held-out-family campaign execution is required before
benchmark-level classification.

## Claim Boundary

Diagnostic evidence only. Claim-card review and real campaign execution on held-out
families are required before promotion to benchmark classification. No planner
superiority or generalization claims are promoted from this evidence.

## Artifacts

| Artifact | Description |
|----------|-------------|
| `pipeline_output.json` | Top-level pipeline payload with classification |
| `seed_sufficiency_analysis.json` | Seed-sufficiency analysis from retained bundles |
| `baseline_table.csv` | Per-planner metric means with confidence intervals |
| `transfer_delta.csv` | Benchmark-set vs held-out-family transfer deltas |
| `fig_transfer_delta.png` | Transfer-delta visualization |
| `claim_card.json` | Preliminary claim card with classification |
| `reproduction.md` | Human-readable summary and reproduction notes |

## Classification Reasons

- `synthetic_fixture_heldout_used`: Held-out-family transfer used synthetic fixtures
  because retained campaign bundles do not contain held-out-family evaluation data
- `diagnostic: claim-card review required before promotion`: Classification requires
  review before any paper-facing use

## Validation Commands

```bash
# Validate Package A readiness
uv run --extra analytics python scripts/validation/check_package_a_readiness.py --json

# Run pipeline on retained bundles
uv run python scripts/tools/run_package_a_pipeline.py \
  --output-dir output/benchmarks/issue_3078_package_a \
  --campaign-root docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28 \
  --campaign-root docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23 \
  --partition-manifest configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml
```

## Remaining Work (out of scope for this evidence)

- Actual held-out-family campaign execution on real data
- Promotion of classification from `diagnostic` to `benchmark` or stronger
- Package B adversarial-falsification and Package C prediction/observation work
