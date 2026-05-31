# Issue #1894 SLURM Job Finalizer

Date: 2026-05-31

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1894>

## Scope

`scripts/tools/slurm_job_finalize.py` is a local, metadata-only closeout helper for completed or
observed SLURM jobs. It does not submit jobs, upload artifacts, copy raw `output/` trees, or turn a
local file into durable benchmark evidence.

The helper accepts an issue number, job id, observed job state, required artifact paths, and optional
artifact paths. It writes a compact JSON manifest and optional Markdown issue-update draft with:

- fail-closed completion classification;
- artifact presence, size, and checksum fields;
- issue-ready and ledger-ready update text;
- an explicit claim boundary that durable benchmark evidence still needs retrieval URIs and the
  relevant benchmark policy checks.

## Command

```bash
uv run python scripts/tools/slurm_job_finalize.py \
  --issue 1894 \
  --job-id 12345 \
  --job-state COMPLETED \
  --expected-artifact output/slurm/job-12345/summary.json \
  --output docs/context/evidence/issue_1894_slurm_job_12345.json \
  --markdown-output docs/context/evidence/issue_1894_slurm_job_12345.md
```

Non-success classifications intentionally return a non-zero CLI status so shell workflows do not
silently treat missing artifacts or failed jobs as completed evidence.

## Classifications

| Classification | Meaning |
| --- | --- |
| `success` | Job state is completed and every required artifact exists. Still local-only until promoted. |
| `missing_artifacts` | Job state is completed but one or more required artifacts are absent. |
| `failed` | Observed SLURM state is failed, cancelled, timed out, or equivalent. |
| `incomplete` | Observed SLURM state is pending, running, suspended, or requeued. |
| `not_available` | Job state or scheduler record is unavailable. |
| `manual_decision_required` | Inputs are insufficient for automatic classification. |

## Validation

```bash
uv run pytest tests/tools/test_slurm_job_finalize.py -q
uv run ruff check scripts/tools/slurm_job_finalize.py tests/tools/test_slurm_job_finalize.py
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
