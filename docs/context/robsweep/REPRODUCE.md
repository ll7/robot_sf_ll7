# Reproduce the examiner diagnostic

Use the pinned runtime source in `evidence/acquisition_source.json`, its
dependency lock, and the registry's immutable model assets. The archived driver
is the exact acquired implementation. Later tooling adds output guards, corrects
diagnostic manifest labels, and fixes rank arithmetic without changing planner
or simulator code. These data are diagnostic, not a frozen-release replay.

All Python commands use the repository environment wrapper. CPU inference and
single-thread numerical libraries are required. The run uses only dev seeds
1061–1090. No training or protected evaluation is part of this procedure.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/examiner_robustness_20261012.py prepare \
  --output-dir docs/context/robsweep_replay/evidence
GH_TOKEN="$(gh auth token)" scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/examiner_robustness_20261012.py canary \
  --output-dir docs/context/robsweep_replay/evidence
GH_TOKEN="$(gh auth token)" scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/examiner_robustness_20261012.py sweep --workers 8 \
  --output-dir docs/context/robsweep_replay/evidence
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/examiner_robustness_20261012.py sanity \
  --output-dir docs/context/robsweep_replay/evidence
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/examiner_crowd_probes_20261012.py \
  --output-dir docs/context/robsweep_replay/evidence
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/analyze_examiner_robustness_20261012.py \
  --evidence-dir docs/context/robsweep_replay/evidence \
  --config-dir configs/benchmarks/robsweep --manifest
```

Run model hydration with `GH_TOKEN` obtained from `gh auth token` in process;
never print it. The initial canary must show the intended radius, preferred
caps, checkpoint, and native or documented adapter execution before a sweep.
Fallback/degraded/invalid/missing rows cannot establish a rank comparison.
Existing phase outputs are refused rather than overwritten. Keep interrupted
rows and execution receipts; do not automatically repeat an unchanged failure.

`prepare` creates only the new `configs/benchmarks/robsweep` namespace; its
archived copies live under `inputs/`. The archive's relative map paths describe
the original execution layout, so execute regenerated files in the original
config namespace. The config PR corrects four inherited manifest names; numeric
settings are identical to the acquired copies.

Verify the delivered bundle from the repository root:

```bash
sha256sum -c docs/context/robsweep/evidence/SHA256SUMS
```

The reported success denominator is eligible episodes in each cell. The
requested denominator, missing identities and excluded rows remain explicit.
Collision is episode incidence, with separate typed counts in the CSV. Exact
integer counts are aggregated before normalization, retaining real ties.
Kendall tau-b is undefined for a constant ordering; it is withheld for an
incomplete matched roster. Bootstrap intervals are conditional on defined tau
replicates; `summary.json` records both defined and requested resample counts.

Sanity probes explicitly select the release's `wrapped_v2` pair kernel. They
enforce clear initial placement for 0.4 m physical bodies while retaining the
0.35 m force radius. Bidirectional doorway targets are aligned with the opening;
the unidirectional target lies beyond it. Trajectories are retained, including
wall violations. Earlier legacy-kernel or invalid-input probes remain separate
excluded diagnostics; they never replace the final release-kernel result.

Delivery custody uses `project_custody.py` after all 73 batches complete. It
rebinds producer sidecars to the compressed artifacts and archived inputs,
verifies archived input hashes, and records the transformations in
`evidence/portable_custody.json`. Scientific values and binary trajectories
remain unchanged. Exact original study metadata is in `inputs/study.source.txt`;
`inputs/study.json` relocates retrieval pointers for this independent evidence
checkout. Original acquisition config locations are retained as path components
in the input/source ledgers. Embedded producer digests identify structured input
descriptors; `hash_locations.json` locates JSON custody digests, while SHA256SUMS
verifies actual delivered file bytes. Superseded raw-JSONL byte checksums are
replaced by actual compressed-file checksums, not claimed as byte parity.

The acquired MPPI arms use an identity-aware continuation after the other 71
batches finish. Completed prefixes are retained under `evidence/resume/`; four
canonical episode workers per arm append only missing identities. The dated
`resume_examiner_mppi_20261012.py` requires the old controller to have exited,
verifies unchanged runtime and acquisition input bytes, refuses a repeated
continuation, and checks that final raw bytes retain their exact prefix. Its
result summary describes the continuation only; the outer execution receipt
separately records prefix counts and the complete 300-row total. All reported
rates are recomputed from the complete episode records.

Real prefix checks execute no new episodes and prove identity deduplication.
Five-step native serial/process canaries compare outcomes, steps, effective
physics and prediction metrics. Existing resume/batch tests also pass. This
is a dispatch/source binding proof, not full-trajectory determinism certification.
The dated helper binds this acquisition; fresh runs need their own source and
input ledger rather than reusing its completed identities.
