<!-- AI-GENERATED (#10074) - NEEDS-REVIEW -->
# Pedestrian validation suite for the next release

Refs #10074; draft #10075 remains stacked on #10073. The source protocol suite measures the unchanged 0.0.8 substrate and successor selections using the same estimators. See source_definition_audit.md for exact definitions and equivalence limits. The previous five tables and raw_science.tar.gz are historical diagnostics; the source_* tables supersede them for these new measurements.

Full dev grid: seeds1001–1030, 750 case episodes plus 450 no-interferer baseline trials per source arm. Local execution is limited to two workers. Full campaigns use Slurm compute allocations; submit only through imech192 after squeue. Print resolved seeds first with --check-only. V9 is context only.

```bash
uv run python -m scripts.validation.pedestrian_validation_10074 --out output/source-baseline --workers 8 --check-only
uv run python -m scripts.validation.pedestrian_validation_10074 --out output/source-baseline --workers 8
uv run python -m scripts.validation.pedestrian_validation_10074 --out output/source-028 --mode radius --radius .28 --workers 8
uv run python -m scripts.validation.pedestrian_validation_10074 --out output/source-028-literature --mode radius --radius .28 --speed-tier literature --workers 8
```

A successor uses those commands with --wall-profile gradient_v3 (or calibrated_v2 for the rejected prototype diagnostic). Acquisition returns0 on successful execution; the model verdict is separate:

```bash
uv run python -m scripts.validation.pedestrian_validation_10074 --out output/source-baseline --gate-only
```

Gate-only verifies the exact config digest, complete unique declared case grid, dev-seed identity, all required raw files and their SHA-256 manifest. Exit2: censored/missing primary measurements. Exit3: body overlap or wall penetration. Exit4: invalid/incomplete/modified acquisition. Exit5: complete feasible observations but numeric tolerances/domain approval still unavailable. There is no automatic release acceptance from matching published standard deviations. Gate JSON retains both missing and physical violations even when one exit takes precedence. Current archived producer gate exits are rechecked with this strengthened gate; pass the archived producer config snapshot explicitly when gating older acquisitions. The original command receipts remain unedited.

--protocol legacy preserves the old grid and estimators for bit-compatibility proof only; it cannot admit a release. The new source protocol records lossless compressed XY/speed trajectories, V6 baseline trajectories, row/identity/config/source/dependency hashes, desired speeds, controlled interferer provenance, pair-step counts, eligibility/censor reasons and comparison tables. Spread is sample SD across seeds; attempted and observed n are distinct. Acquisition SHA256SUMS covers all files except itself; the outer archive manifest pins it too.

Production radius: opt-in SimulationSettings(pedestrian_radius_m=.28) or scenario simulation override. The selector drives resolved ped_radius for geometry/placement/metrics and SceneConfig.agent_radius for the force kernel. The gradient wall profile's body-edge origin follows it. Fitted legacy wall offsets remain law parameters. Without the selector, physical .40/force .35, serialization/config hashes and trajectories remain byte-identical. Explicit selector survives replace/deepcopy/serialization and rejects nonfinite/nonpositive values. Frozen planner assumptions and independent safety margins retain their versioned configurations.

Literature variant samples desired speeds from N(1.3,.2), clipped to[0,3], with an independent 3 m/s integration cap; these are explicitly recorded engineering choices. Source-controlled V6 speeds remain 1.15/1.42/1.78. The released default speed and wall law are unchanged. Tests protect early acceleration fitting, temporal aperture censoring, interpolated finite-N/stationary specific flow, obstacle-edge origin, filtered nonreactive onset, strict grouped pair counts, production radius consumers and artifact admission. Four-question test-value answers and fail-on-base witnesses are in test_value.md and the final lane report.
