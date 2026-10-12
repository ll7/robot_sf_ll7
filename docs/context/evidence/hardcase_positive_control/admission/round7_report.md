<!-- AI-GENERATED (#9652) - NEEDS-REVIEW -->
# Round-7 stop: incomplete admission authority

**Admission FAIL. Recommend parking #10334 as a draft with `state:parked`.**
The complete reconstruction design was not reached in this round. #10331's
replay fix is independent and unchanged; neither PR is merged.

The gate refuses the entire continuation v1/v2 evidence class. It no longer
calculates a supposedly certified event from source radii, threshold metadata
or collision flags, and cannot return a successful window admission. Existing
custody, complete recorded-input comparisons and trajectory parity checks
still run before the dedicated `UnsupportedWindowAdmissionError`. Refusal
precedes the corpus lock, index and case writes. Historical records remain
available for forensic inspection through `read_corpus`; replay export refuses
nonempty historical corpora rather than trusting their old recorded passes.

This is a containment change, not a claim that the 16 forged declarations are
individually validated. The unchanged positive control also refuses. The
retained `corpus/`, `persistence/`, `admission_evidence.json`, `replay_slice.json`
and their checksums preserve historical results, not valid current admissions.
No test is deleted, marked xfail, or assigned a new baseline to hide this stop.

## Why this packet cannot establish the requested authority

The bound checkpoint in `packet/checkpoint.json` records
`configuration`, `run`, `seed`, `map` and `window`. Its configuration digest
owns the environment, planner and drive configuration, including radii. It
has no recorded metric threshold profile or source/replay Git commits. Its
`source_episode_id` is an identifier, not a commit binding. The code revision
is a digest of seven selected source files, not a full Git commit identity.
Different commits can contain exactly those same files.

Raw positions determine distances; they do not determine whether the intended
near-miss threshold was 0.5 or 5.0. Inferring current benchmark defaults or
accepting a metric profile from a native row would restore the prohibited
metadata authority. The old contact computation also trusted trace collision
flags rather than deriving contacts from geometry. Likewise, agreement between
manifest/row/receipt copies cannot recover a commit absent from the checkpoint.
The complete native redundancy and identity reconstruction design therefore
requires a new configuration-owning capture contract and fresh bound proof.
No such capture class is supported by this patch; none is accepted implicitly.

Reopening requires checkpoint-owned metric/contact definitions and complete
source/run identity, deterministic reconstruction of every admitted input and
quantity through existing producers, refusal of unsupported quantities, and
fresh seed-1001 source/two-replay evidence. All stored aliases must be compared
to the reconstruction, including planner, visibility, tracking, observations,
scenario/config/commit identities, metrics and events. This must replace the
partial-envelope design, not add another list of field checks. #9652/#5600
remain unresolved. The original positive control through replay parity remains
historical evidence in #10331.

## Regression proof and test value

The real `admit_window` API is exercised with the retained seed-1001 packet,
unchanged raw trajectories, valid rebound artifact hashes, and fresh temporary
corpus destinations. Seven native input contradictions, four planner dictionary
shape/type/precision changes, three manifest identity changes and two source
radius/threshold forgeries were all admitted before the fix. The manifest
config-hash forgery is preserved after rebinding; no fixture reidentification
silently overwrites it. The unchanged control was admitted too.

- Behaviour protected: incomplete checkpoint authority cannot produce a corpus
  admission or export historical admissions as currently verified.
- Credible regression: reintroducing success after trajectory parity, or letting
  export trust an old recorded pass, reopens the bypass.
- Coverage gap: prior input-envelope mutations missed the redundant native
  declarations; successful export tests used freshly admitted records.
- Test seam: none; both tests call the existing public APIs and check the actual
  filesystem for refusal before writes.

Exact commands (all from the child worktree unless stated):

```sh
export PYTHONDONTWRITEBYTECODE=1 HARDCASE_NO_TEST_SIMULATION=1 ROBOT_SF_PYTEST_SEED_GUARD=1
export PYTHONPATH=../test_guard:tests/support:.
scripts/dev/run_worktree_shared_venv.sh --venv ../typing-venv -- uv run pytest tests/adversarial/test_window_admission_authority.py -q
scripts/dev/run_worktree_shared_venv.sh --venv ../typing-venv -- uv run pytest tests/adversarial/test_window_admission_authority.py::test_historical_admission_is_not_exported_as_currently_verified -q
scripts/dev/run_worktree_shared_venv.sh --venv ../typing-venv -- uv run pytest tests/adversarial/test_counterexample_corpus.py tests/benchmark/test_source_bound_window_receipt.py tests/benchmark/test_scenario_persistence_gate.py tests/benchmark/test_pipeline_persistence_gate_wiring.py -q
scripts/dev/run_worktree_shared_venv.sh --venv ../typing-venv -- uv run pytest tests/adversarial/test_window_admission_input_contract.py -q
scripts/dev/run_worktree_shared_venv.sh --venv ../typing-venv -- uv run pytest tests/benchmark/test_benchmark_result_provenance.py -q
```

RED at reviewed b1a8339ec2b89085dd4b2d430ac697b78a5c1b66: **17 failed in
33.78s**, each with `DID NOT RAISE`, not fixture/import/locator errors. The
historical-export regression separately failed before its fix with
`DID NOT RAISE` (**1 failed in 11.64s**). GREEN: **18 passed in 17.52s**.
These passing tests prove class refusal only; they do not prove admission works.

## Corrected provenance-module claim

The claim that twelve focused failures were pre-existing main defects is
withdrawn. The old local comparison used a partial Python 3.13 dependency
profile and established only that environment-specific result. It was not
valid evidence of main defects. With the all-extras Python 3.12 environment,
the dedicated `tests/benchmark/test_benchmark_result_provenance.py` command
above passes **59/59 on head (8.49s)** and **59/59 on pinned main (8.48s)**,
matching the reviewer's result. The main command ran from the separate
`../main-baseline` worktree at 53721be3248600e08641a9816d094b605e3c93e4.
No unrelated provenance implementation or test was changed.

The old full `tests/benchmark` audits remain historical. They were not rerun
or relabeled as proof of this stopped head. The four current positive-admission
failures are introduced by intentional class refusal, not attributed to main.

## Per-stage disposition

| Stage | Verdict | Evidence / limitation |
| --- | --- | --- |
| Source, extraction, materialization, actual two replays | PASS, historical | Completed seed-1001 episodes were not restarted; their runtime files and packet bytes remain unchanged. |
| Stored trajectory parity | PASS | The unchanged control reaches class refusal after the two full channel comparisons. This does not certify event criteria or commit identity. |
| Complete admission authority / critical-event certification | FAIL | The packet lacks checkpoint-owned criteria and commit identities. No new event count is certified. |
| Corpus admission / re-admission / replay export | FAIL | The unmodified case refuses; historical exports refuse. Old positive tests are preserved. |
| Reviewer regressions and unchanged refusal control | PASS | 18 tests; all 16 forgeries fail first on the reviewed head. |
| Touched gate/API tests | FAIL | 29 passed, 4 failed (positive admission/receipt/re-admission expectations). No masking or ratchet change. |
| Existing effective-input regressions | PENDING | Exact command above; final result in lane report. |
| Provenance comparison | PASS | 59/59 on each checkout in all-extras environment. |
| Lint / ruff | PASS | `UV_PROJECT_ENVIRONMENT=../typing-venv scripts/dev/ci_driver.sh lint`; focused ruff also passes. |
| ty advisory ratchet | PASS, pending final rerun | No baseline increase; pinned ty 0.0.58. |
| Registry / docs integrity / PR contract | PENDING | Final exact-head results are recorded in the lane report; behaviour receipt remains orchestrator-owned. |

The four preserved failing tests are
`test_admit_real_seed1001_window_and_export_slice`,
`test_idempotent_readmission_keeps_original_evidence_and_recovers_index`,
`test_retained_seed1001_window_receipt_verifies_actual_replays`, and
`test_native_window_gate_accepts_parity_without_promoting_unknown_persistence`.
They are the evidence that the end-to-end success criterion remains unmet.

No frozen artifact, prohibited-seed simulation/reset, sealed replay, new
adversarial campaign, debt finding baseline, attribution line or merge is
introduced. Admission and export use no environment step/reset. #10334 should
stay parked until the full reconstruction design and positive chain pass;
#10331 can be reviewed and landed independently by its owners.

TERMINAL=DEFER_RECOMMENDED
