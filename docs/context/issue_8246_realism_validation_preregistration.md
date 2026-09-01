# Issue #8246 interaction-conditioned realism validation preregistration

Date: 2026-09-01

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/8246>

Status: implementation and synthetic-fixture proof complete; external-data execution is
`blocked-external-input` by issue #6530. This note does not report a Stanford Drone Dataset (SDD)
result or a paper-facing realism claim.

## Purpose

Issue #8246 freezes the analysis contract needed before staging SDD pedestrian trajectories. The
contract makes interaction-specific counts, scene separation, baseline arms, and promotion rules
explicit before any real-data result can influence a research direction.

## Implemented surfaces

- Contract loader and validator: `robot_sf/benchmark/realism_validation_contract.py`.
- Versioned metadata contract: `configs/benchmark/realism_validation_contract.v1.yaml`.
- Interaction-window segmenter and typed context/result records:
  `robot_sf/benchmark/pedestrian_realism_validation.py`.
- Per-class rows, pooled counts, and event-floor status in the existing realism scorecard and
  Markdown renderer.
- Synthetic tests with planted free-walking, pedestrian-pedestrian, obstacle-avoidance,
  robot-approach, crossing-conflict, overtaking, and group fixtures where the relevant context is
  available. Missing robot or trusted obstacle context remains explicitly non-inferable.

The seven labels are `free_walking`, `ped_ped_interaction`, `obstacle_avoidance`, `robot_approach`,
`crossing_conflict`, `overtaking`, and `group`. A deterministic precedence order resolves windows
that satisfy more than one predicate. The segmenter is descriptive trajectory-window analysis; it
does not infer human intent or establish model quality.

## Frozen validation contract

The shipped YAML declares:

- calibration scenes `bookstore_calibration` and `coupa_calibration`;
- held-out scenes `death_circle_held_out` and `gates_held_out`, disjoint from calibration;
- constant velocity, `social_force_default`, and the registered HSF model variants as baseline arms;
- trajectory RMSE, fundamental-diagram, lane-formation, speed-distribution, and proxemic-
  distribution metric families;
- per-class minimum event counts, with sparse classes reported as `insufficient_events` rather than
  pooled into an apparent success;
- a held-out-only promotion rule compared with `social_force_default`, including a maximum allowed
  free-walking regression and required interaction classes.

ORCA is deliberately excluded from this hierarchy. The contract is metadata-only until the external
data gate is satisfied.

## Required SDD author packet before revival

When #6530 makes SDD staging available, the author should attach one immutable packet containing:

1. SDD scene IDs and the calibration/held-out split.
2. Raw annotation checksums and the staged-manifest checksum.
3. Metres-per-pixel conversion and source frame rate.
4. Coordinate convention, including the y-axis direction.
5. Accepted annotation labels and the policy for lost, occluded, or interpolated tracks.
6. Dataset license acknowledgement, citation, and provenance receipt.
7. Calibration and held-out scene IDs repeated in the executable contract.
8. Baseline artifact/config hashes for every evaluated model arm.
9. Metric and analysis-version identifiers plus the output receipt.
10. A statement that no real-data or paper-facing claim is made until all event floors, provenance,
    and held-out promotion predicates pass.

The revived execution must preserve the calibration/held-out separation, record unavailable classes
without silently pooling them, and distinguish native, adapter, fallback, or degraded model modes.
Fallback or degraded rows cannot be promoted as benchmark evidence.

## Evidence boundary and next action

Current evidence is implementation-integrity and synthetic-fixture evidence only. It demonstrates
that the contract validates, the segmenter labels known fixtures, the scorecard carries per-class
counts, and sparse event floors remain visible. It does not demonstrate SDD coverage, calibration,
realism, planner performance, or a model ranking.

The next smallest proof step is to revive the contract only after #6530 supplies the author packet,
stage the licensed data through the external-data workflow, and run the held-out validation with
the declared provenance and output receipt. Until then, keep this issue open and labeled
`state:blocked-external-input`.

## Reproducibility commands

```bash
uv run pytest tests/benchmark/test_realism_segmenter.py \\
    tests/benchmark/test_realism_validation_contract.py -q
uv run pytest tests/benchmark/test_pedestrian_realism_validation.py -q
uv run python -c "from pathlib import Path; from robot_sf.benchmark.realism_validation_contract import load_realism_validation_contract; print(load_realism_validation_contract(Path('configs/benchmark/realism_validation_contract.v1.yaml')).to_dict())"
```
