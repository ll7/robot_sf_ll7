# Realism validation preregistration (Issue #8246)

Plain-language summary: this note freezes the interaction-conditioned, held-out-by-scene
contract needed before a future staged-data run can be interpreted as pedestrian-model validation.

Date: 2026-09-01

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/8246>

Status: implementation and synthetic-fixture proof complete; terminal state is
`BLOCKED_EXTERNAL` (serialized as `blocked-external-input`). The revival condition is exactly
`#6530 staged`: the maintainer-provided SDD author packet below must be recorded and pass the
staging preflight before any real-data execution is considered. This note does not report a
Stanford Drone Dataset (SDD) result or a paper-facing realism claim.

## Purpose

Issue #8246 freezes the analysis contract needed before staging SDD pedestrian trajectories. The
contract makes interaction-specific counts, scene separation, baseline arms, and promotion rules
explicit before any real-data result can influence a research direction.

## Implemented surfaces

- Contract loader and validator: `robot_sf/benchmark/realism_validation_contract.py`.
- Versioned metadata contract: `configs/benchmark/realism_validation_contract.v1.yaml`.
- Interaction-window segmenter and typed context/result records:
  `robot_sf/benchmark/pedestrian_realism_validation.py`.
- Per-class rows, diagnostic window counts, independent event counts, and event-floor status in
  the existing realism scorecard and Markdown renderer. Overlapping or touching windows with the
  same label and participant tracks count as one event episode for floor evaluation.
- Synthetic tests with planted free-walking, pedestrian-pedestrian, obstacle-avoidance,
  robot-approach, crossing-conflict, overtaking, and group fixtures where the relevant context is
  available. Missing robot or trusted obstacle context remains explicitly non-inferable.

The seven labels are `free_walking`, `ped_ped_interaction`, `obstacle_avoidance`, `robot_approach`,
`crossing_conflict`, `overtaking`, and `group`. A deterministic precedence order resolves windows
that satisfy more than one predicate. Only tracks covering the complete window are considered;
windows without sufficient movement or caller-supplied robot/scene context are excluded instead of
being labeled `free_walking`. Crossing requires opposing tracks that are closing or at their
predicted closest approach. The segmenter is descriptive trajectory-window analysis; it does not
infer human intent or establish model quality.

## Frozen validation contract

The shipped YAML declares:

- calibration scenes `bookstore_calibration` and `coupa_calibration`;
- held-out scenes `death_circle_held_out` and `gates_held_out`, disjoint from calibration;
- constant velocity, `social_force_default`, and the registered HSF model variants as baseline
  arms;
- trajectory RMSE, fundamental-diagram, lane-formation, speed-distribution, and proxemic-
  distribution metric families;
- per-class minimum independent event counts, with sparse classes reported as
  `insufficient_events` rather than pooled into an apparent success;
- a held-out-only promotion rule compared with `social_force_default`, including a maximum allowed
  free-walking regression and required interaction classes.

ORCA is deliberately excluded from this hierarchy. The contract is metadata-only until the
external data gate is satisfied.

## Exact maintainer-provided SDD author packet before revival

The revival condition is exactly `#6530 staged`. When it is met, the author must attach one
immutable packet with every field below; placeholders, inferred values, or a substitute dataset do
not satisfy the condition:

1. Source/access provenance: accepted archive identity, source path or URI, access provenance, and
   the non-canonical-mirror designation.
2. Rights and license: rights basis, license acknowledgement, citation, and the privacy/publication
   boundary.
3. Checksums: per-file hashes, the aggregate raw-annotation checksum, and the staged-manifest
   checksum.
4. Scene identity: scene identifiers and the calibration/held-out split identity, repeated in the
   executable contract.
5. Coordinate and time units: metres-per-pixel conversion, source frame rate, coordinate units,
   time units, and y-axis direction.
6. Annotation schema: required columns, accepted annotation labels, and the policy for lost,
   occluded, or interpolated tracks.
7. Staging receipt: exact staging destination, provenance-manifest path, and the successful
   fail-closed validation receipt.

The revived run must additionally bind each evaluated arm to its baseline artifact/config hash,
metric and analysis-version identifiers, and output receipt. It must state that no real-data or
paper-facing claim is made until all event floors, provenance, and held-out promotion predicates
pass.

The revived execution must preserve the calibration/held-out separation, record unavailable classes
without silently pooling them, and distinguish native, adapter, fallback, or degraded model modes.
Fallback or degraded rows cannot be promoted as benchmark evidence.

## Evidence boundary and next action

Current evidence is implementation-integrity and synthetic-fixture evidence only. It demonstrates
that the contract validates, the segmenter labels known fixtures, the scorecard carries per-class
counts, and sparse event floors remain visible. It does not demonstrate SDD coverage, calibration,
realism, planner performance, or a model ranking.

The next smallest proof step is to revive the contract only after the exact condition `#6530
staged` is recorded with the author packet above, stage the licensed data through the external-data
workflow, and run the held-out validation with the declared provenance and output receipt. Until
then, the terminal state remains `BLOCKED_EXTERNAL` / `state:blocked-external-input`.

## Reproducibility commands

```bash
uv run pytest tests/benchmark/test_realism_segmenter.py \\
    tests/benchmark/test_realism_validation_contract.py -q
uv run pytest tests/benchmark/test_pedestrian_realism_validation.py -q
uv run python -c "from pathlib import Path; from robot_sf.benchmark.realism_validation_contract import load_realism_validation_contract; print(load_realism_validation_contract(Path('configs/benchmark/realism_validation_contract.v1.yaml')).to_dict())"
```
