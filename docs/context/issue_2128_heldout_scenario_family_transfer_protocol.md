# Issue #2128 Held-Out Scenario-Family Transfer Protocol 2026-06-02

Status: draft protocol, proposal evidence only

Related issue: #2128

This note defines a conservative pilot protocol for separating benchmark-set performance from
held-out scenario-family transfer. It is a planning and validation contract, not benchmark evidence.
No transfer claim is established until the pilot is executed, artifacts are durable, and the leakage
audit below is completed.

## Claim Boundary

Allowed wording before execution:

- "planned held-out scenario-family transfer pilot"
- "draft transfer protocol"
- "benchmark-set and held-out-family claims are separated by design"

Allowed wording after a valid pilot run:

- "observed transfer behavior on the named held-out family partition"
- "pilot transfer evidence for the selected scenario families, seeds, planners, and metrics"
- "diagnostic transfer delta, subject to the pilot sample size and planner availability"

Disallowed wording:

- "OOD generalization" unless the protocol, data, tuning history, and artifact provenance support
  that stronger claim.
- "real-world generalization" from Robot SF scenario families alone.
- "transfer success" when rows ran in fallback, degraded, failed, or not-available modes.
- "architecture causality" from a planner/policy comparison without a separate ablation contract.

## Partition Contract

The pilot uses four named surfaces:

- `training_family_pool`: scenario families used to train learned policies or tune planner
  hyperparameters.
- `validation_family_pool`: scenario families used for early stopping, checkpoint selection, and
  threshold tuning.
- `benchmark_set_evaluation`: the in-distribution or already-known benchmark surface. Results here
  remain benchmark-set performance.
- `heldout_family_evaluation`: scenario families excluded from training, validation, reward tuning,
  normalization fitting, checkpoint selection, and report design decisions.

The initial partition manifest is
`configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml`.

## Pilot Execution Order

1. Freeze the partition manifest and pilot benchmark config before running evaluations.
2. Record all planner/policy inputs, checkpoint manifests, normalization state, reward settings, and
   scenario-family exposure history in the experiment card.
3. Run the benchmark-set evaluation and held-out-family evaluation with the same planner set, seed
   schedule, metrics, and availability policy.
4. Preserve fallback, degraded, failed, and not-available rows as visible exclusions or caveats.
5. Report benchmark-set metrics and held-out-family metrics in separate tables.
6. Compute transfer deltas only between comparable native or otherwise explicitly eligible rows.

## Leakage Audit

Complete this checklist before interpreting pilot outputs as held-out transfer evidence:

- [ ] Held-out scenario IDs and family labels were fixed before any pilot result was inspected.
- [ ] Held-out maps and scenario templates were not used for training, validation, reward tuning,
      normalization fitting, checkpoint selection, or planner hyperparameter tuning.
- [ ] Seed schedules are fixed and shared across comparable planner rows.
- [ ] Metric definitions match the camera-ready benchmark contract or name any deliberate pilot
      deviation.
- [ ] Planner inclusion rules identify native, adapter, fallback, degraded, failed, and
      not-available modes.
- [ ] Learned-policy checkpoint manifests name the training scenario surface and exclude held-out
      family exposure.
- [ ] Artifact outputs include source-data manifests and checksums for episode rows, summaries,
      tables, figures, and trace panels.
- [ ] Any missing planner or incompatible dependency is reported as an exclusion, not a successful
      transfer outcome.

## Output Contract

The pilot should produce these proposal-grade outputs before promotion:

- `summary_benchmark_set.csv`: benchmark-set aggregate table.
- `summary_heldout_family.csv`: held-out aggregate table.
- `scenario_family_breakdown.csv`: per-family metric table with availability and execution-mode
  columns.
- `transfer_delta.csv`: comparable-row delta table with uncertainty columns.
- `fig_transfer_delta.{png,pdf}`: figure showing held-out minus benchmark-set deltas.
- `artifact_catalog.yaml`: paths, checksums, durable references, and evidence roles.
- `leakage_audit.md`: completed checklist with reviewer/date/provenance.

Local files under `output/` are disposable staging artifacts until promoted to W&B, release storage,
or `docs/context/evidence/` with a tracked manifest.

## Pilot Configs

- Partition manifest:
  `configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml`
- Partition validator:
  `uv run python scripts/tools/validate_heldout_transfer_partitions.py configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml`
- Held-out pilot benchmark config:
  `configs/benchmarks/issue_2128_heldout_family_transfer_pilot.yaml`
- Training/tuning pool scenario set:
  `configs/scenarios/sets/issue_2128_heldout_family_transfer_training_pool.yaml`
- Held-out pilot scenario set:
  `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`
- Experiment card:
  `experiments/issue_2128_heldout_family_transfer_pilot.yaml`
