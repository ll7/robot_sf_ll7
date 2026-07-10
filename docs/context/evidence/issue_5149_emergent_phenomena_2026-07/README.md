<!-- AI-GENERATED (robot_sf_ll7#5149, 2026-07-10) - NEEDS-REVIEW -->

# Issue #5149: Emergent-Phenomena Demonstration for the Released Pedestrian Substrate

Plain-language summary: this bundle demonstrates whether THIS repository's pedestrian simulator (the bundled `fast-pysf` / PySocialForce Social Force model) reproduces the three canonical crowd-dynamics emergent phenomena (lane formation in bidirectional flow, doorway oscillation, and exit arching/clogging), run at the released-default speed calibration (~0.65 m/s desired) and at a literature-typical calibration (~1.3 m/s).

## Provenance
- Generated at (UTC): `2026-07-10T18:27:48Z`
- Git head: `41e7d5e5638b37291cbbd972edf0d917364a1585`
- Substrate: `pysocialforce==2.0.0`
- Generation command: `uv run python scripts/validation/build_issue_5149_emergent_phenomena_demo.py`
- Harness module: `robot_sf/research/emergent_phenomena.py`

## Claim boundary
This is **diagnostic behavioral-validity (smoke-tier) evidence**, not paper-grade validation against real human trajectory datasets (tracked separately in issue #4975). It establishes whether the phenomena are reproducible in this implementation at the released parameterization, and pins a regression anchor for force-model changes (e.g. the anticipatory variant in #4973, speed recalibration in #4972).

## Speed calibrations
- `released_default`: desired speed ~0.65 m/s, reproducing the released default regime (`initial_speed=0.5`, `max_speed_multiplier=1.3`; see #4972).
- `literature_typical`: desired speed ~1.3 m/s (Moussaid et al. 2010, doi:10.1371/journal.pone.0010047).
- The desired speed is realized through the released substrate's own speed-derivation logic (`max_speeds = max_speed_multiplier * initial_speeds`); the harness sets spawn velocity magnitude to `desired / max_speed_multiplier` along the goal direction rather than patching the force stack.

## Results

| Scenario | Calibration | Order parameters | Phenomenon verdict |
| --- | --- | --- | --- |
| bidirectional_corridor | released_default | lane_segregation_index=0.231, lane_purity=0.368 | weak_partial |
| bidirectional_corridor | literature_typical | lane_segregation_index=0.283, lane_purity=0.415 | weak_partial |
| narrow_doorway | released_default | oscillation_flips=3.000, throughput_peds_per_sec=0.359, mean_burst_windows=2.000 | clearly_present |
| narrow_doorway | literature_typical | oscillation_flips=3.000, throughput_peds_per_sec=0.359, mean_burst_windows=1.750 | clearly_present |
| high_density_exit | released_default | exit_density_ratio=6.823, arch_lateral_spread=0.855 | clearly_present |
| high_density_exit | literature_typical | exit_density_ratio=7.664, arch_lateral_spread=0.730 | clearly_present |

## Interpretation
Read the verdict column literally: `clearly_present` means the order parameter crossed a conservative documented threshold; `weak_partial` means a detectable but non-robust signal; `absent_or_negligible` means the phenomenon did not emerge at these parameters in this run. Lane formation in particular is expected to be the weakest signal at the slow released-default regime (the issue itself flags this as genuinely open); the literature-typical speed is expected to strengthen it. Each scenario is a single seeded run (deterministic given the seed), so a verdict is a regression anchor, not a population statistic.

## Thresholds (documented, conservative)
- Lane formation `clearly_present` if `lane_segregation_index >= 0.5`; `weak_partial` if `>= 0.15`.
- Doorway oscillation `clearly_present` if `oscillation_flips >= 2`.
- Exit arching `clearly_present` if `exit_density_ratio >= 2.0`.

## Reproducibility
Re-run with the generation command above from the repository root. Output is deterministic given the pinned seed (`5149`) and the released force parameters. Trajectory plots are PNG; numeric results are in `summary.json`; file integrity is in `SHA256SUMS`.

## Files
- `README.md` — this file.
- `summary.json` — provenance + per-scenario order parameters.
- `bidirectional_corridor__released_default.png` — trajectory plot.
- `bidirectional_corridor__literature_typical.png` — trajectory plot.
- `narrow_doorway__released_default.png` — trajectory plot.
- `narrow_doorway__literature_typical.png` — trajectory plot.
- `high_density_exit__released_default.png` — trajectory plot.
- `high_density_exit__literature_typical.png` — trajectory plot.
- `SHA256SUMS` — integrity manifest for the bundle.
