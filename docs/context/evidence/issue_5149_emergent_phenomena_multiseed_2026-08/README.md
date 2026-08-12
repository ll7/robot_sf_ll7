<!-- AI-GENERATED (robot_sf_ll7#5149, 2026-08-12) - NEEDS-REVIEW -->

# Issue #5149: Multi-Seed Emergent-Phenomena Campaign (Measured Face-Validity)

Plain-language summary: this bundle measures whether THIS repository's pedestrian simulator (the bundled `fast-pysf` / PySocialForce Social Force model) reproduces the canonical crowd-dynamics emergent phenomena (lane formation in bidirectional flow, doorway oscillation, and an exit arching diagnostic) across 10 seeds per scenario and speed calibration, elevating the pinned single-seed exhibit (`issue_5149_emergent_phenomena_2026-07/`) to measured evidence with seed-level dispersion.

## Provenance
- Generated at (UTC): `2026-08-12T08:56:51Z`
- Git head: `a31686e51f624638255d22968b89cda0257c2e4c`
- Substrate: `pysocialforce==2.0.0`
- Generation command: `uv run python scripts/validation/build_issue_5149_emergent_phenomena_campaign.py`
- Harness modules: `robot_sf/research/emergent_phenomena.py`, `robot_sf/research/emergent_phenomena_campaign.py`
- Maintainer authorization: https://github.com/ll7/robot_sf_ll7/issues/5149#issuecomment-5264374182
- Seeds: `[5149, 5150, 5151, 5152, 5153, 5154, 5155, 5156, 5157, 5158]`
- Full machine-readable provenance: `manifest.json`.

## Claim boundary
This is **measured face-validity (smoke-tier) evidence**: per-seed order parameters with dispersion across a pinned seed list, at the released parameterization and a literature-typical speed calibration. It is NOT benchmark-matrix evidence and NOT paper-grade validation against real human trajectory datasets (tracked separately in issue #4975). Verdicts are conservative threshold labels on simple order parameters, suitable as a behavioral-validity exhibit and regression anchor for force-model changes (#4972 speed recalibration, #4973 anticipatory variant).

## Results (aggregated across seeds)

| Scenario | Calibration | n seeds | Primary order parameter (mean +/- std [min, max]) | Verdicts |
| --- | --- | --- | --- | --- |
| bidirectional_corridor | literature_typical | 10 | lane_segregation_index = 0.151 +/- 0.101 [0.041, 0.283] | absent_or_negligible: 6, weak_partial: 4 (majority: absent_or_negligible) |
| bidirectional_corridor | released_default | 10 | lane_segregation_index = 0.153 +/- 0.088 [0.052, 0.281] | absent_or_negligible: 5, weak_partial: 5 (majority: absent_or_negligible) |
| high_density_exit | literature_typical | 10 | exit_density_ratio = 7.441 +/- 0.237 [7.064, 7.910] | clearly_present: 10 (majority: clearly_present) |
| high_density_exit | released_default | 10 | exit_density_ratio = 6.953 +/- 0.188 [6.583, 7.228] | clearly_present: 10 (majority: clearly_present) |
| narrow_doorway | literature_typical | 10 | oscillation_flips = 1.800 +/- 1.135 [0.000, 3.000] | absent_or_negligible: 5, clearly_present: 5 (majority: absent_or_negligible) |
| narrow_doorway | released_default | 10 | oscillation_flips = 3.200 +/- 1.033 [1.000, 5.000] | absent_or_negligible: 1, clearly_present: 9 (majority: clearly_present) |

Secondary order parameters (lane purity, throughput, burst length, arch lateral spread) are in `summary.json`; per-seed rows are in `runs.jsonl`.

## Interpretation
Read the verdict counts literally: they are per-seed conservative threshold labels, and the majority verdict tie-breaks toward the weaker label so a split seed population never overclaims. The expected pattern from the 2026-07 single-seed exhibit is that doorway oscillation and exit arching emerge clearly at both calibrations while lane formation is weak at the slow released default and somewhat stronger at the literature-typical speed; this campaign measures how stable that pattern is across seeds rather than asserting it from one run.

## Thresholds (documented, conservative)
- Lane formation `clearly_present` if `lane_segregation_index >= 0.5`; `weak_partial` if `>= 0.15`.
- Doorway oscillation `clearly_present` if `oscillation_flips >= 2`.
- Exit arching `clearly_present` if `exit_density_ratio >= 2.0`.

## Reproducibility
Re-run with the generation command above from the repository root. Output is deterministic given the pinned seed list and the released force parameters; pass `--generated-at` with the timestamp above for a byte-stable re-run on the same platform/environment (`manifest.json` records the runtime; cross-platform floating-point drift is possible). File integrity is in `SHA256SUMS`.

## Files
- `README.md` — this file.
- `manifest.json` — full provenance manifest.
- `summary.json` — aggregate statistics + verdict distributions.
- `runs.jsonl` — one record per scenario x calibration x seed.
- `bidirectional_corridor__released_default__seed5149.png` — figure.
- `bidirectional_corridor__literature_typical__seed5149.png` — figure.
- `narrow_doorway__released_default__seed5149.png` — figure.
- `narrow_doorway__literature_typical__seed5149.png` — figure.
- `high_density_exit__released_default__seed5149.png` — figure.
- `high_density_exit__literature_typical__seed5149.png` — figure.
- `bidirectional_corridor__order_parameter_by_seed.png` — figure.
- `narrow_doorway__order_parameter_by_seed.png` — figure.
- `high_density_exit__order_parameter_by_seed.png` — figure.
- `SHA256SUMS` — integrity manifest for the bundle.
