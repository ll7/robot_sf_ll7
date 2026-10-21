# Confinement development trace results

AI-GENERATED NEEDS-REVIEW

**Diagnostic-only development evidence.** Traversal is post-treatment, duration-censored,
and changes cell composition across arms. No causal motion/wall-force, held-out,
release-equivalence, real-world, or dissertation-admission claim follows.

The small hybrid release lead is **not established as persistent** in traversal episodes.
Static-constriction hybrid-minus-ORCA success differences (percentage points; paired 95% intervals):

| Configuration | Difference | 95% interval |
|---|---:|---:|
| Hybrid v4 | +2.81 | [−2.42, 8.33] |
| Hybrid v4 continuous | −0.67 | [−6.62, 5.23] |
| Hybrid ORCA v2 yield | +0.20 | [−5.17, 6.00] |
| Hybrid ORCA v2 guard | +0.20 | [−5.17, 6.00] |

Portal hybrids each succeed in **4/4** traversal episodes, versus ORCA **8/12**:
+33.33 points, interval [0.00, 63.64]. Their [100%,100%] success intervals only resample
four observed successes, omit 142/10,000 empty draws, and do not establish perfect performance.
There are just 34/1,260 portal traversal episodes. Eight cells have no complete traversal;
double-bottleneck traversals occur at the east neck only, with none at the west or both.

Unconditional development success differs from release by −4.76 to +1.90 points across
12 group/configuration comparisons. The portal ordering reverses: ORCA 97.14%, hybrids
93.33–94.29%. This proximity check cannot isolate seed, tracing, or host/runtime effects.

## Contract and proof

- Preregistered plan: `ded06fc990b032cb4bf7fd2889a59aa234e8e53f`,
  2026-10-18T10:00:27+02:00, before any episode. See [PLAN.md](PLAN.md).
- Unchanged producer: `373dbfde4f39667cf9e8732dabe7df5118bdeab1`; all six effective configs
  match release, authored horizons 400–700, dt 0.1 s, development seeds 1091–1120 only.
- 2,700 complete unique traced records; four initial failed storage attempts preserved and
  repeated after a failing serialization regression. Four-to-six-worker resume adopted
  drained records without rerunning completed episodes.
- Per-cell episode and actor-slot shares/censoring: [cells.csv](cells.csv).
  Conditional success: [group_success.csv](group_success.csv).
  Contrasts: [hybrid_orca_contrasts.csv](hybrid_orca_contrasts.csv).
  Release comparison: [unconditional_release_comparison.csv](unconditional_release_comparison.csv).
- CSV rates/intervals are fractions, contrasts are fraction differences; `difference_from_release_pp`
  is percentage points. All counts retain denominators. Bootstrap: 10,000 shared complete-seed-block
  draws, Random(42), dissertation rounded 2.5/97.5-percentile indices; undefined draws flagged.
- All 2,700 pass stored-schema and canonical outcome/event-ledger checks; full reset obstacle
  geometry, map IDs and frozen SVG hashes agree. Native map-path identity is unavailable in
  the direct executor; geometry is the explicit compensating proof.
- Native-schema validation omits only the unused optional `min_separation_corrupted_m=null`
  in a validation projection (the native writer permits Infinity). Raw records are unchanged.
  First-frame zero velocities are documented sentinels; position differences provide speed.
  Later velocity components agree within 1.91e−11 m/s. Excluded transitions: 2,305.
  See [record_validation.json](record_validation.json) and [EXECUTION_NOTES.md](EXECUTION_NOTES.md).

## Preservation and delivery

Full traces, source archive, environment/input/tool receipts, checksums and reproducible analysis
are preserved in the campaign archive. **W&B upload is blocked: no credential path/helper was
available; zero upload attempts.** Artifact custody remains `durable-required` until upload and
remote SHA-256 readback. No frozen release, manuscript or claim ledger changed.

Focused analysis tests: 15 pass after demonstrated pre-implementation failures. The broad core
run has 11,978 passes and one inventory-only guard failure; the generated inventory is refreshed
with these final files. The joint focused analysis/registry suite passes 71 tests. Finding allowances are
unchanged. Independent exact-head review remains required; never merge from this lane.
