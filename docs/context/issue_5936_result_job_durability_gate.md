# Issue #5936 Result-Job Durability Gate

Date: 2026-07-17

Related issue: [ll7/robot_sf_ll7#5936](https://github.com/ll7/robot_sf_ll7/issues/5936)

Motivating failure instances: #5890 (job 13512 SNQI rows not durable), #5891
(job 13509 threshold-sweep rows not durable), #5912 (job 13516 latency rows
reachable only via a private pointer).

## Problem (one root cause)

Analysis issues were opened before their decisive inputs were durably reachable.
Each was filed from a worker's/job's report of an artifact on the compute host,
not from a durable repository artifact, then blocked in the same way and
consumed triage repeatedly.

## The gate

A completed result job MUST publish, before any `analysis:` successor issue may
be created or admitted:

1. a **checksummed** raw or sufficient-derived input (private-safe pointer if
   the rows must stay private);
2. a **versioned schema** for that input;
3. a **canonical rerun command** that regenerates the analysis from it; and
4. a **durable public-safe pointer** (tracked path or registry entry).

Acceptance: a clean-checkout reproduction — the analysis runs from the tracked
pointer alone, with no host-local state.

## Producer-side probe (this repository)

This issue's enforcement has two halves. The **admission discipline and fleet
invariant** live in the orchestrator (`ll7/codex-orchestrator`); this repository
provides the **producer-side counterpart** that a result job runs before it
declares success and that an analysis-issue author runs before filing.

- Schema: [`robot_sf/benchmark/schemas/result_job_durability.v1.json`](../../robot_sf/benchmark/schemas/result_job_durability.v1.json)
- Probe: [`robot_sf/benchmark/result_job_durability.py`](../../robot_sf/benchmark/result_job_durability.py)

Probe command (exit `0` = durable, `2` = not durable):

```bash
uv run python -m robot_sf.benchmark.result_job_durability \
  docs/context/evidence/<bundle>/result_job_durability.yaml
```

The probe verifies, on the current checkout, that:

- the `analysis_input.pointer` resolves and is **not** local-only
  (`output/`, `results/`, `.venv/`, worktrees, `/tmp`, `/home` are rejected);
- the declared `sha256` matches the resolved artifact (tracked-path inputs);
- the `input_schema` is versioned and, when `schema_path` is set, resolves on a
  clean checkout;
- the `rerun_command` is present and does not name a bare host-only artifact as
  its sole input; and
- the manifest carries `source_job`, `claim_boundary`, and a `gate_id`.

Registry/release pointers (private-safe, e.g. W&B artifacts) are durable by
reference: they require a `hydration_command` and are checksummed at hydration
time rather than from the repository tree. This matches the #5912 case where raw
rows must stay private behind a checksummed sufficient-derived input.

## Relationship to adjacent contracts

This gate specializes the existing
[Artifact Evidence Vocabulary](artifact_evidence_vocabulary.md) and the
[Issue #1053 Durable Artifact Reference Audit](issue_1053_durable_artifact_references.md):
those define what counts as durable; this gate is the enforced check that an
analysis successor's declared input actually is. It does not duplicate the
[reusable artifact catalog](issue_2008_artifact_catalog.md) (figures/tables) or
the campaign result store; it covers the analysis-input durability boundary
those surfaces do not enforce.

## Evidence boundary

- Classification: workflow/tooling, CPU-only.
- This PR adds the producer-side probe and schema; it does **not** wire the
  orchestrator admission/invariant (separate repository) and makes no
  benchmark, metric, or paper-facing claim.

## Validation

- `uv run pytest tests/benchmark/test_result_job_durability.py -q`
- `uv run ruff check robot_sf/benchmark/result_job_durability.py tests/benchmark/test_result_job_durability.py`
- `uv run ruff format --check robot_sf/benchmark/result_job_durability.py tests/benchmark/test_result_job_durability.py`
