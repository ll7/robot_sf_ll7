<!-- AI-GENERATED (robot_sf#7724) - NEEDS-REVIEW -->
# Issue #7724 collision-pressure release packet

This packet materializes a deterministic descriptive collision-pressure slice from the published
RobotSF release `0.0.3.post1`. It reads immutable typed event-ledger rows and runs no
simulation, campaign, GPU, SLURM job, or private-operations workflow.

## Claim boundary

- **Evidence status:** `analysis_only` / `diagnostic-only` candidate artifact.
- **Admission:** this packet does not admit evidence, change an evidence tier, or automatically
  restore the withdrawn dissertation Table 7.5 row in [ll7/diss#613](https://github.com/ll7/diss/issues/613).
- **Forbidden inference:** the counts are not a probability, ranking, causal mechanism, severity,
  physical-risk, deployment-safety, or real-world safety result.
- **Scope:** exact descriptive counts over `doorway`, `narrow_doorway`, and `robot_crowding`.

## Immutable input

- Release: `0.0.3.post1`
- Publication commit: `ded9027d2928512c14bc241397e0ab1d8f586654`
- Row-production commit: `a307ef276d701f8d14dead1aa0513f44ee97c0b0`
- Archive: `paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_publication_bundle.tar.gz`
- Archive SHA-256: `9bf6ea35a17ce812f0a9c841c3681bc072dcf7ba8c121cbcf05113b8514f4de1`
- Internal payload checksums: `85` entries verified before reading rows

## Reproduction

```bash
uv run python scripts/analysis/generate_collision_pressure_release_0_0_3_post1.py \
  --bundle <path-to-paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_publication_bundle.tar.gz> \
  --output-dir docs/context/evidence/diss_613_collision_pressure_release_0_0_3_post1
```

Verify the packet bytes from the repository root with:

```bash
shasum -a 256 -c docs/context/evidence/diss_613_collision_pressure_release_0_0_3_post1/SHA256SUMS
```

The adapter records the only two transformations: `scenario_family` is copied from
`scenario_params.metadata.archetype`, and `episode_key` is formatted as
`<release-arm>::<episode_id>`. No scientific category is inferred or normalized.

## Reconciled result

| Quantity | Count |
| --- | ---: |
| Release rows across 14 run files | 20,160 |
| Eligible arm-qualified episodes | 2,100 |
| Contact episodes | 1,545 |
| Typed collision events | 1,546 |
| Pedestrian-contact episodes | 520 |
| Obstacle-contact episodes | 1,026 |
| Pedestrian/obstacle overlap episodes | 1 |
| Unexplained exclusions | 0 |

The report's exact counts reconcile independently with the published
`payload/reports/scenario_family_breakdown.csv`. Missing `collision_partner_id` values remain
explicit: 1,026 static-geometry contacts;
relative-speed missingness is 0.

## Packet files

- `source_manifest.json`: immutable input identity, adapter locators, row totals, and reconciliation.
- `normalized_typed_ledger_slice.jsonl`: sorted, arm-qualified EpisodeEventLedger.v2 slice.
- `collision_pressure_report.json` / `.csv`: generic report outputs.
- `SHA256SUMS`: checksums for every packet file except the checksum manifest itself.

This packet is linked to [RobotSF issue #7724](https://github.com/ll7/robot_sf_ll7/issues/7724)
and requires a separate dissertation-side evidence-pin/card and author decision before any
manuscript use.
