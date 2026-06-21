# Durable artifact backend decision (#3075)

**Status:** current · **Decision date:** 2026-06-21 · **Decider:** maintainer (ll7)

## Decision

**Weights & Biases (W&B) Artifacts is the approved durable artifact backend** for sprint
research studies. Finalized SLURM/campaign runs record their durable pointer as a
`wandb://entity/project/artifact:version` URI; that URI is what lets a run's outputs be cited
after worktree cleanup.

Rationale: W&B Artifacts already integrates with the repository's run tracking
(`robot_sf/common/artifact_paths.py` declares a `wandb` artifact category; the reconciler
already recognizes `wandb://` / `wandb-artifact://` durable pointers), gives versioned,
checksum-addressable storage with retention, and avoids standing up new infrastructure. Other
durable schemes (`https://`, `s3://`, `gs://`, `dvc://`) remain accepted so a run may point at
an equivalent durable store; a bare local `output/` path is **never** durable.

## How the contract is enforced (already implemented)

The durable-evidence contract is split across two existing tools — this decision wires them to
the approved backend rather than adding a new finalizer:

- **`scripts/tools/slurm_job_finalize.py`** — idempotent, metadata-only finalizer. Validates the
  research-control-plane required-file set (`run_manifest.json`, `episodes.parquet`,
  `summary.json`, `checksums.sha256`, `stdout.log`, `stderr.log`, `environment.json`), computes
  SHA256 checksums, and classifies fail-closed (missing required files → `missing_artifacts`,
  non-zero exit; never partial success). It now also accepts `--durable-uri` and records:
  - `durable_uri` — the validated `wandb://...` (or other durable-scheme) pointer.
  - `durable_status` — `durable` only for a `success` run **with** a recorded durable URI;
    a successful run without one stays `pending_durable`; non-success runs are `not_applicable`.
- **`scripts/tools/reconcile_slurm_evidence.py`** — reconciles pending artifact aliases against
  durable pointers. It reads the finalizer's `durable_uri` (and `wandb_url` / `artifact_uri` /
  `s3_uri` / `gs_uri` / `dvc_uri`) so a `completed_pending_artifact_promotion` row is replaced
  by its durable W&B reference.

## Operator flow

1. The SLURM job writes the required-file set under its run root and uploads artifacts to W&B.
2. `slurm_job_finalize.py --control-plane-run-root <run> --durable-uri wandb://... ...` produces
   the compact finalization manifest (`durable_status: durable`).
3. `reconcile_slurm_evidence.py` replaces any pending aliases with the durable W&B pointer.
4. Only `durable`-status rows with a recorded W&B URI are cited as durable evidence; everything
   else stays caveated per `docs/context/artifact_evidence_vocabulary.md`.

## Acceptance criteria (#3075) mapping

- [x] Preferred W&B artifact path accepted — recorded here, with reason.
- [x] Finalizer requires the seven control-plane files unless a narrower run-class contract is
  documented — `CONTROL_PLANE_RUN_ARTIFACTS` in `slurm_job_finalize.py`.
- [x] Retries idempotent, no silent semantic change — the report builder is pure; only
  `generated_at` varies (covered by `test_report_semantics_are_idempotent`).
- [x] Missing required files → fail-closed, not partial success — `missing_artifacts` + exit 1.

## Caveats

- The finalizer remains metadata-only: it does **not** upload to W&B. Upload is performed by the
  job (or a dedicated promotion tool); the finalizer validates and records the resulting durable
  URI and fails closed when none is present.
- W&B network credentials/access are an operator prerequisite, not enforced by this tooling.
