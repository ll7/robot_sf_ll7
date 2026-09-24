<!-- AI-GENERATED (robot_sf#6151, 2026-09-13) - NEEDS-REVIEW -->

# Issue #6151 simulator-dependence synthesis (2026-09-13)

Plain-language summary: the only defensible overall verdict is
`invalid_missing_evidence`. The recovered job-13512 bytes are now durable and checksum-covered,
but their execution lineage conflicts and the materialized metric is `success_rate`, not the
requested Social Navigation Quality Index (SNQI). The older bounded slice is independently
non-identifiable. No ranking, stability, benchmark, realism, or paper-facing claim is admitted by
this synthesis, and no new campaign is run.

## Decision

| Surface | Verdict |
| --- | --- |
| #6151 overall validity-boundary verdict | `invalid_missing_evidence` |
| Per-axis SNQI rank margin | Not materialized for any axis; no SNQI column exists in the promoted compact metric rows. |
| Per-axis collapse or flip | Not assessable from the accepted evidence. The source `success_rate` report is preserved as an unaccepted source output. |
| Accepted planner ranking or rank stability | None. The older slice is `non-identifiable`; the recovered bundle is `invalid_provenance_or_scope`. |
| Benchmark, planner-superiority, safety, realism, sim-to-real, paper, and dissertation use | Not eligible. |

This is a negative synthesis of existing evidence, not a new empirical result. The verdict follows
the parent #3207 decision and the #5890 maintainer ruling: preserve the bytes and contradictions,
but do not choose a metric or execution lineage by inference.

## Reproducible inputs and method

The synthesis compares only tracked, checksum-covered records and the older tracked bounded slice.
It does not read the remote raw episode stream, reconstruct missing SNQI values, recompute ranks, or
merge the two packets.

| Input | Role | Checksum / identity |
| --- | --- | --- |
| [`#5890 registration`](../issue_5890_job13512_promotion_2026-08-19/registration.json) | Machine-readable custody and eligibility ruling | `sha256: dad819ceffb0294d1e0731c31a78a20dce893815a45e2ef430c29c60dbc24ef3` |
| [`#5890 reconciliation`](../issue_5890_job13512_promotion_2026-08-19/reconciliation.md) | Contradiction-preserving comparison | `sha256: 071d08eac518b547a3741ee6b2adcfff0e9196dfd05f71c06e9ae225bd5b147d` |
| [`#5890 rank report`](../issue_5890_job13512_promotion_2026-08-19/fidelity_rank_stability_report.json) | Source `success_rate` rank output, not an accepted result | `sha256: ad191efdca328b1d0f1475b001aae9ac9c2fb75456fae6de4ff31b8b9dbe4dcd` |
| [`#3207 older slice`](../issue_3207_fidelity_sensitivity_actual_slice_2026-06-23/summary.json) | Independent bounded slice with zero-variance primary metric | `sha256: 0afc24852f92578fc0f456fc7dfda63d942af0fe3a4beab49e4ad5ce9653fd50` |
| [`#3207 no-claim packet`](../issue_3207_simulator_dependence_validity_boundary_packet_2026-06-29/decision_packet.json) | Existing fail-closed checker output | `sha256: 0602524f9ab9714ed2aba7e1118a3a887e3e69c654fc31302fad7942d897515f` |
| [`#3207 radius contract`](../../issue_6643_radius_rank_stability_gate3.md) | Separate footprint/radius sensitivity gate; no shared admitted rank table | `sha256: 61e1c2c4cf57e6e13b9dfc54788b66bba2e7827f7f1088aaf3bd256e3a923696` |

The #5890 bundle's complete tracked files are independently verified with:

```bash
cd docs/context/evidence/issue_5890_job13512_promotion_2026-08-19
sha256sum -c SHA256SUMS
```

The existing decision checker was also run against the promoted summary:

```bash
uv run python scripts/validation/check_simulator_dependence_validity_boundary.py \
  --summary docs/context/evidence/issue_5890_job13512_promotion_2026-08-19/summary.json
```

It returns `no_claim`. That checker result is consistent with, but does not replace, the stronger
parent-level `invalid_missing_evidence` synthesis verdict below: the promoted summary does not
provide a source-complete, provenance-verified SNQI study packet.

## Per-axis reconciliation

The requested four axes are retained as separate rows. The recovered report has two variants for
each axis, while the older bounded slice has one perturbation per axis. A `success_rate` Kendall
tau or source ordering is not substituted for an SNQI rank margin.

| Axis | Recovered job-13512 source facts | Older tracked slice | SNQI margin / collapse / flip | Per-axis verdict |
| --- | --- | --- | --- | --- |
| Integration timestep | `dt_0_05`, `dt_0_20`; compact rows materialize `success_rate` | `dt_0_05`; all `success_rate` values are `0.0` | SNQI unavailable; rank non-identifiable in older slice; no accepted flip result | `invalid_missing_evidence` |
| Social-force speed archetypes | `mixed_balanced`, `rush_hour`; compact rows materialize `success_rate` | `mixed_balanced`; primary metric has zero variance | SNQI unavailable; no accepted collapse/flip result | `invalid_missing_evidence` |
| Observation noise | `pose_heading_low`, `pedestrian_dropout_low`; compact rows materialize `success_rate` | `pose_heading_low`; primary metric has zero variance | SNQI unavailable; no accepted collapse/flip result | `invalid_missing_evidence` |
| Clearance / radius semantics | `radius_0_30`, `radius_0_50`; compact rows materialize `success_rate` | `radius_0_30`; primary metric has zero variance | SNQI unavailable; no accepted collapse/flip result | `invalid_missing_evidence` |

The recovered `fidelity_rank_stability_report.json` records `success_rate`, `kendall_tau: 1.0`,
and no source-reported rank flips for its eight variants. Those values remain preserved source
facts only: they cannot answer the SNQI question, and the #5890 registration sets ranking,
identifiability, and stability eligibility to `false`. The older slice explicitly reports
`primary_metric_zero_variance`, so its deterministic tie ordering is not a ranking result.

Footprint and threshold semantics are related methodological inputs, not an additional accepted
rank axis in this synthesis. The radius contract is separately gated and currently has no complete
production sweep; its documentation explicitly keeps physical-footprint, realism, sim-to-real,
and safety claims out of scope. No cross-packet threshold or footprint result is therefore merged
into the four-axis table.

## Contradictions that remain visible

The synthesis does not silently select one record over another:

1. `execution_context.txt` and `run_summary_source.txt` identify commit
   `ae0130d65cf232e0322cfd4800659a87d481490a`, config
   `configs/research/issue_3207_fidelity_sensitivity_full_fixed_scope.yaml`, primary metric
   `snqi`, and 5,184 expected episodes.
2. The promoted `summary.json` identifies commit
   `c153848d7be2851b5c5e89c11055bf96ea778a84`, config
   `configs/research/fidelity_sensitivity_v1.yaml`, and a bounded actual slice. Its compact rows
   and rank report materialize `success_rate`, not SNQI.
3. The promoted summary lists three materialized planners but retains stale bounded-two-planner
   wording. The older tracked packet is a separate 30-episode, two-planner slice with zero
   variance in `success_rate`.

These are provenance and scope conflicts, not evidence that either metric is numerically wrong.
They prevent verified execution reproducibility and any accepted ranking interpretation.

## What survives and what does not

The following narrow statements survive because they are directly supported by the tracked inputs:

- the job-13512 compact bytes and their checksums are preserved for custody;
- the contradictory source records, metric names, planner scope, and execution lineages remain
  visible rather than overwritten;
- the older bounded slice exercises the internal fidelity-sensitivity pipeline and records metric
  drift, but its primary rank is non-identifiable;
- radius/footprint and threshold work defines useful methodological contracts, but is not a shared
  source-complete ranking result.

The following statements do not survive the evidence gates:

- any SNQI value, per-axis SNQI margin, SNQI ranking, or SNQI rank-stability conclusion;
- any accepted `success_rate` planner ordering, planner superiority, or rank-flip conclusion;
- verified reproduction of job 13512;
- full #3207 acceptance, benchmark admission, simulator-realism, sim-to-real, physical-safety,
  paper-facing, or dissertation evidence.

## Unresolved validity threats and stop rule

- Resolve the exact execution commit/config conflict before treating the recovered run as
  reproducible.
- Supply a source-complete metric contract that actually materializes SNQI before any SNQI
  analysis; do not rename `success_rate`.
- Keep raw episode rows remote-only unless a separate custody decision changes the retention rule;
  compact files do not justify reconstructing them.
- Do not combine the older bounded slice with job 13512: their scope, metrics, provenance, and
  rank-identifiability states differ.
- Any new campaign or new radius/threshold axis requires a separately approved, source-bound
  issue and evidence contract.

The stop rule is therefore `invalid_missing_evidence`: close this synthesis only as a bounded
negative result, and reopen it only if independently reviewed evidence resolves the lineage and
metric-contract conflicts. This document contains no benchmark or scientific claim.

## Parent propagation

After this document is merged, propagate the exact verdict to [parent issue #3207](https://github.com/ll7/robot_sf_ll7/issues/3207), link this tracked synthesis and the #5890
registration/reconciliation, keep the older packet visible, and leave #3207 open for any separately
authorized evidence. The propagation must state `invalid_missing_evidence` and must not imply that
the preserved `success_rate` report is accepted ranking evidence.
