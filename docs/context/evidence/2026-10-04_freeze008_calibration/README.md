# Frozen-source SNQI v2 development acquisition

AI-GENERATED NEEDS-REVIEW. Scientific review remains pending.

Job 21331 acquired 14 arms × 48 authored cells × dev seeds 1001/1002
at `3e73b04b43aa99b9fbe4a6ab34b89a5a9f1933b6`. The 1,344 unique rows and complete producer
hashes/sidecars passed the repository's strict freezer. All 14 arms completed;
no fallback, degraded, failed or unavailable execution was accepted. Navigation
collisions and timeouts remain outcomes, separate from planner execution failures.

[Acquired anchors](anchors.v2.0.acquired.json), [recomputed proof](acquisition-proof.json)
and [scalar rows](calibration-scalars.csv) retain the actual acquisition. Modes:
goal and PPO native (96 each), guarded PPO mixed (96), eleven adapter arms
(96 each). All rows declare recorded robot/pedestrian social force sampled
pre-integration: model acceleration, not measured human discomfort. Force sample
statistics are finite; 28 authored `classic_bottleneck_low` zero-pedestrian cells
are explicitly counted separately from 1,316 sampled cells. No missing force
value is zero-imputed. The compact rows retain recorded scalars and provenance;
raw force vectors are not claimed reconstructed from the scalar CSV.
[Scalar metadata](metadata.json) declares `distance_convention=surface_clearance`
for the underlying near-miss predicate; the close-clearance fraction is dimensionless.

## Anchors and rehearsal comparison

| Term | Acquired p95 | Change from d56092ed rehearsal |
| --- | ---: | ---: |
| F | 78.44270546210876 | 0.0 |
| J | 1.957837635866961 | 0.018161933593934476 |
| K | 0.5160654761904677 | 0.0 |

T=3 and N=0.25 remain normative. The selected F field remains
`robot_force_impulse_total`, because the recomputed absolute Spearman rho
0.60110908310645 is below the preregistered 0.9 boundary.
J rose by approximately 0.93634%. Its zero-based p95 index is 1275.85; both
bracketing acquired values are 1.957837635866961, from the seed-1001
`classic_doorway_high` rows of `hybrid_rule_v4_fast_progress_static_escape` and
`scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4`. The proof records those
actual order statistics. This is a changed empirical upper anchor, not rounding
of the rehearsal's 1.9396757022730264.

The acquisition configuration is byte-identical across rehearsal d56092ed and
freeze 3e73b04b; metric definitions, force kernel and dependency lock are unchanged.
The producer source and acquired trajectories are fresh. Runtime source differences
are recorded in the proof. These two acquisitions do not isolate a causal effect
of any single source change or runtime nondeterminism; no such attribution is made.
The independent reviewer must assess this empirical difference against raw custody.
The new anchor SHA also binds the new source, run ID, manifest and row/sidecar hashes.

## Preservation and authority boundary

W&B `ll7/robot_sf/campaign-issue9667_snqi_v2_calibration_dev1001_1002_3e73b04b43_20261003:v0` is COMMITTED and checksum-verified.
Manifest `sha256:691d21210c5f0e73da560fc81749fa0346c121d95d73f1bdb0b909aef54414c9` covers the full producer,
startup/source/config/checkpoint/runtime and scheduler/watcher receipts and anchors.
All 117 source members passed stored,
gzip-decoded and independent snapshot SHA-256 comparison using a fresh API and empty
cold cache. The operator retains those inputs and the receipt outside public Git.
Raw-custody and artifact-only attachment to the unchanged candidate passed without
any reset or step. The source pending-anchor file and acquisition YAML remain intact.
The placeholder pin test is inherited from #10125; stale comment work remains #10124.
All six runbook scientific-review boxes and private Git-blob trust pins remain pending.
No smoke, final mint, DOI change, sealed execution or publication was performed.

## Regeneration

Obtain the preserved complete producer and its independent snapshot, cold download
and original preservation receipt. Operator-supplied roots below contain no fixed
machine locator. `PRODUCER_SOURCE` must be clean at the named freeze with rebuilt
all-extras dependencies and staged checkpoints; `REVIEW_REPO` contains this script.
Run from the producer checkout because learned-checkpoint registry lookup is relative
to it. These commands analyze existing data only:

```bash
cd "$PRODUCER_SOURCE"
export PYTHONPATH="$PRODUCER_SOURCE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/tools/analyze_snqi_contract.py   --campaign-root "$PRODUCER_ROOT/benchmarks/issue9667_snqi_v2_calibration_dev1001_1002_3e73b04b43_20261003"   --freeze-v2-anchors "$ANCHORS_OUTPUT"
.venv/bin/python "$REVIEW_REPO/scripts/dev/build_snqi_v2_acquisition_evidence.py"   --producer-root "$PRODUCER_ROOT" --anchors "$ANCHORS_OUTPUT"   --snapshot-root "$SNAPSHOT_ROOT" --cold-root "$COLD_ROOT"   --preservation-receipt "$PRESERVATION_RECEIPT"   --rehearsal docs/context/evidence/2026-10-03_issue10112_mintorder/calibration-d56092ed-grid-proof.json   --output-dir "$REVIEW_REPO/docs/context/evidence/2026-10-04_freeze008_calibration"   --source-commit 3e73b04b43aa99b9fbe4a6ab34b89a5a9f1933b6   --runtime-commit c857a30b77fbd82b73fe4f72b71c045aae9c4d10   --launcher-sha256 da8e6480426ef0eb6651dd3f4c5dec3ec6436d6d38a93bcfa881330f473b0280
```
