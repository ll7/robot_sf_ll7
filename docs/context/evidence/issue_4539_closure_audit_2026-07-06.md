# Issue #4539 Closure Audit

Issue #4539 asked to harden the release-claim-matrix publication gate
(`scripts/validation/check_release_claim_matrix_publication_gate.py`, the Issue #2910 gate) against
two *latent* fail-open paths surfaced during the PR #4536 gate review, while preserving the gate's
current fail-closed behavior on the committed matrix. This audit maps each acceptance criterion to
the merged PR(s) and tests that satisfy it and finds the issue closable at the
validation-hardening/tooling boundary.

## Closure Decision

Status: `complete`

Decision: `keep_diagnostic` (validation-hardening only)

Closure keyword for the handoff PR: `Closes #4539`

Claim boundary: CPU-only static-gate hardening over the existing tracked release claim matrix. This
audit and the referenced PRs do not run a benchmark campaign, submit Slurm/GPU work, create
certification records, promote any planner ranking or nominal benchmark result, or edit
paper/dissertation claims.

## Acceptance Criteria Mapping

The maintainer implementation plan (issue comment, 2026-07-05) restated the acceptance criteria as
three numbered hardening steps plus required tests. All are merged.

| Acceptance criterion | Implementing PR(s) | Evidence in current `origin/main` | Test | Status |
| --- | --- | --- | --- | --- |
| Unrecognized `classification` values fail closed (blocker), with a test. | PR #4544 (`fedf6cf8a`) | `build_gate_report` enumerates `BENCHMARK_EVIDENCE_CLASSIFICATIONS` and `NON_BENCHMARK_CLASSIFICATIONS`; any other value takes the `else` branch and appends a `check="classification"` blocker (`unrecognized classification …`). Unknown labels no longer fall to the lenient non-benchmark path. | `test_gate_fails_closed_for_unknown_classification` (asserts `nominal benchmark evidence` yields a `classification` blocker). | Met. |
| `scenario_certification` requires a positive accepted state (allowlist / accepted pattern), with a test, once the `scenario_cert.v1` accepted vocabulary is confirmed. | PR #4606 (`2a78618e4`, filed under #2910) | Denylist replaced by allowlist `ACCEPTED_SCENARIO_CERTIFICATION_VALUES = {"scenario_cert.v1:accepted", "scenario_cert.v1:accepted_reviewed"}` — the exact vocabulary the maintainer plan named as preferred. `_check_benchmark_evidence_row` blocks when `certification not in ACCEPTED_SCENARIO_CERTIFICATION_VALUES`, so `pending`/`draft`/`rejected`/unknown all block; only accepted vocabulary passes. | `test_gate_blocks_benchmark_row_with_pending_certification`; `test_committed_matrix_remains_blocked_until_certification_is_attached`. | Met. |
| (Optional) `artifact_uri`/`source_refs` reject absolute / non-repo-relative paths. | PR #4569 (`f592bfb11`) + PR #4606 (`2a78618e4`) | Helper renamed to `_repo_relative_file_exists`; rejects `None`/empty, absolute paths, `..` components, and `output/…`, and (per #4606) restricts to `DURABLE_PREFIXES = {docs, configs, scripts, robot_sf, tests}`, returning true only for existing regular files under `repo_root`. Applied to both `artifact_uri` and every `source_refs` entry. | `test_gate_rejects_absolute_artifact_uri_even_when_file_exists`; `test_gate_rejects_source_refs_with_parent_directory_components`; `test_gate_blocks_non_durable_prefix_path`; `test_gate_fails_closed_for_dot_artifact_uri_without_crashing`. | Met (the optional criterion is delivered). |

## Behavior Preservation Evidence

The core non-regression requirement — "preserve current fail-closed behavior on the committed
matrix" — holds. Running the gate on the tracked matrix:

```text
uv run python scripts/validation/check_release_claim_matrix_publication_gate.py --json
# status: blocked
# summary: row_count=21, benchmark_evidence_rows=3, diagnostic_or_non_claim_rows=18, blocker_count=3
# blocker checks: {scenario_certification}
```

The three `benchmark evidence` rows are blocked solely on `scenario_certification` (they carry no
accepted `scenario_cert.v1` value), and the 18 diagnostic/non-claim rows pass the non-promotion
guard. No false-pass is produced. The full targeted suite passes:

```text
uv run pytest tests/validation/test_check_release_claim_matrix_publication_gate.py -q
# 11 passed
```

## Residual Risks

| Risk | Why it remains |
| --- | --- |
| The gate still reports `status: blocked` on the committed matrix. | This is the intended fail-closed design, not an unmet criterion: no benchmark-evidence row has yet been positively certified. Turning any row `pass` requires a real `scenario_cert.v1:accepted` certification, which is a separate maintainer-owned certification action, not gate code. |
| Issue #4539 is still `OPEN` at audit time with no labels. | Closure and label state is GitHub state propagation, not repository evidence. This audit's handoff PR carries `Closes #4539`; the merge closes the issue. The implementation lane here is not authorized to close or comment on the issue directly. |
| Criterion 2's allowlist was implemented under issue #2910 (PR #4606), not a #4539-tagged PR. | The change is the exact vocabulary the #4539 maintainer plan requested; attribution is cross-linked here so the criterion→evidence trail is complete regardless of the tagging PR's issue header. |

No full benchmark campaign was run for this audit, no Slurm or GPU submission was made, no
certification record was created, and no paper/dissertation claim text was edited.
