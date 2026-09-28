# Corrected 0.0.8 evaluation before publication approval

## Goal and evidence boundary

Run and verify a corrected 0.0.8 scientific matrix before the author authorizes
a tag or Zenodo publication. A candidate is not a release. The frozen 0.0.7
source inputs, archive, tag and record remain unchanged; publication identity
must not rewrite any raw 0.0.8 episode row.

The author ruling on issue #9668 (2026-09-28) keeps the 14 planner keys, 48
scenario identities, seeds 111–140, horizon 600 and dt 0.1. It permits
versioned corrections to scenario, planner, spawn and pedestrian-model inputs
and accepts causally supported changes to outcomes and old metrics. The 0.0.7
archive remains the comparison baseline at SHA-256
`684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`.
The earlier 18-arm proposal in #9751 is superseded: arm keys remain paired
identities while corrected implementation/config versions are recorded separately.
Any additional v3/v4 arm comparison uses a diagnostic manifest outside this grid.
The four historical hybrid keys in the main grid bind explicit, hashed v4
configurations for nominal corrected behavior; v3 remains only in the frozen
predecessor or a separately labelled diagnostic comparison.
The historical 2 m doorway reference is a separate infeasibility probe, not a
nominal feasible doorway result. The 48-identity main grid needs a versioned
feasible successor for that scenario, with unchanged route-completion success
semantics; the preregistered three-width H400 slice has its own manifest.

## Source and custody

- `run_camera_ready_benchmark.py` and `camera_ready/` own raw execution;
  `release_protocol.py` owns source and scientific identity;
  `check_release_metric_equivalence.py` owns predecessor comparison;
  `finalize_benchmark_data_v008.py` owns derivative publication packaging.
- A pre-DOI mode admits only the exact, committed, checksummed corrected template
  at the exact clean source. It binds scenario/config/model-registry pins,
  checkpoint staging, exact-source runtime smoke, complete SNQI-v2 calibration
  anchors, and release preflight before evaluation. DOI and tag fields are
  absent from candidate artifacts.
- Candidate custody records source and template SHA-256, ordered planner keys
  and config hashes, unchanged checkpoint pins, scenario and seed hashes,
  all 20,160 expected cell keys, producer sidecars, and every raw
  `episodes.jsonl` digest. Corrected scientific input bytes are compared with,
  but need not equal, 0.0.7 input bytes. The scientific config hash excludes
  only publication identity fields.
- On a permitted compute host, preserve both local and independently recoverable
  copies of raw rows and receipts. This host's `local.machine.md` forbids Slurm
  submission. The earlier 1,152/1,344-row calibration is failed and supplies no
  eligible anchor.

## Acceptance and publication stop rule

1. The corrected manifest preflight and full 20,160-row acceptance pass, with
   fallback/degraded/partial rows excluded and all failures accounted for.
2. Join every 0.0.8 row to its 0.0.7 identity. Compare all common outcomes and
   metrics at absolute tolerance `1e-12`, retaining every changed field.
   A reviewed disposition attributes changed fields to named versioned
   corrections with causal evidence and rates/rankings impact. Missing,
   unexplained, or unsupported changes stop acceptance. Comparison findings
   do not invalidate custody of already completed raw rows.
3. Validate the robot-force reductions and SNQI-v2 terms against their stored
   inputs, then independently review the candidate, checksums and limitations.
4. Only after the author approves publication, bind real DOI/tag values in a
   derivative copy. Verify unchanged raw-row digests and scientific hash, rerun
   release acceptance, and export the publication bundle. The 18-row H400
   doorway slice remains under its own preregistered manifest and gate.

## Proof and recovery

Test placeholder DOI leakage, input-pin drift after freezing, missing or
changed rows and sidecars, incomplete matrix, degraded row, duplicate key,
stale receipt, unexplained changed field, and publication-copy row mutation.
Run focused tests, Ruff/format and exact-head PR readiness before any launch.
A failed gate produces a diagnostic or blocked receipt with the smallest repair
step; no failed or partial run becomes benchmark evidence. The earlier
identity-only candidate template and comparator assertions are superseded and
must not be used to launch this corrected campaign.
