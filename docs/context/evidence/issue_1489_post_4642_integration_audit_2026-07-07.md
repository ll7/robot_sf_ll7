# Issue #1489 Post-#4642 Integration Audit

Plain-language summary: issue #1489 should stay open. The merged work now provides
the shared schema, validator, prerequisite matrix, integration-status summary,
synthesis-report builder, command-line report path, and one durable blocked report
artifact. The remaining acceptance criterion is the actual comparative synthesis:
at least two component lanes must produce durable comparable outputs or explicit
fail-closed classifications. That work is campaign-backed and outside this
closure-audit slice.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/1489>
- Audit date: 2026-07-07
- Live issue thread reviewed through comment timestamp `2026-07-06T14:39:01Z`.
- Relevant latest maintainer guidance:
  - PR #4628 added `build_hybrid_synthesis_report()` but kept issue #1489 open.
  - PR #4642 added the reproducible `--synthesis-report` command and first durable
    blocked report artifact, but kept issue #1489 open.
- Open pull request dedupe: no open PR with `1489` in title/body at audit start.
- Fragmentation guard: two issue #1489 PRs merged on 2026-07-06, so this report
  consolidates the post-#4642 state instead of adding another micro-guard.

## Merged Evidence Reviewed

| PR | Merge time | Evidence contribution | Closure effect |
| --- | --- | --- | --- |
| [#4452](https://github.com/ll7/robot_sf_ll7/pull/4452) | 2026-07-04 13:52 UTC | Adds compact `integration_status` for the hybrid prerequisite matrix. | Keeps #1489 fail-closed until at least two component lanes are `complete`. |
| [#4563](https://github.com/ll7/robot_sf_ll7/pull/4563) | 2026-07-05 02:39 UTC | Adds the first criterion-to-evidence closure audit note. | Keeps #1489 open on missing downstream comparable component evidence. |
| [#4628](https://github.com/ll7/robot_sf_ll7/pull/4628) | 2026-07-06 11:46 UTC | Adds `build_hybrid_synthesis_report()` with fail-closed per-mechanism recommendations. | Delivers synthesis-report machinery only; no component lane becomes complete. |
| [#4642](https://github.com/ll7/robot_sf_ll7/pull/4642) | 2026-07-06 14:36 UTC | Adds `--synthesis-report` and `docs/context/evidence/issue_1489_synthesis_report_2026-07-06/synthesis_report.json`. | Produces a durable artifact with `status: blocked`, `complete_count: 0`, and no promoted verdicts. |

## Acceptance Criteria Map

| Criterion | Evidence | Status after #4642 |
| --- | --- | --- |
| Define shared synthesis table diagnostics contract across component campaigns. | Schema and validator work existed before this audit and is exercised by the hybrid evidence matrix tests; PR #4452 consumes it for prerequisite status. | Met for local contract/tooling. |
| Component outputs classified as paper-grade, stress/full-matrix, nominal-only, smoke-only, failed, degraded, unavailable, or insufficient. | Hybrid evidence matrix tiering and #4452 prerequisite matrix classify lane readiness; PR #4642 durable report records current lane states. | Met for classification machinery; current inputs remain non-synthesis-complete. |
| Hard-guard authority and learned-component contribution diagnostics visible in synthesis. | Matrix row schema and validator require guard authority and learned-component contribution fields; synthesis-report builder consumes validated rows. | Met for schema/tooling; not yet met as comparative evidence because durable complete component outputs are absent. |
| Synthesis recommends continue, revise, stop, or gather-more-evidence for each mechanism. | PR #4628 implements per-mechanism recommendations; PR #4642 runs the command on the committed component matrix. | Met as fail-closed diagnostic output: all mechanisms currently `gather_more_evidence`; no promoted synthesis verdict. |
| Reader can tell whether learned components improved downstream navigation metrics, not only intermediate diagnostics. | No merged PR provides at least two durable comparable complete component lanes. PR #4642 explicitly reports `complete_count: 0`. | Not met; blocked on component campaign outputs or explicit fail-closed classifications. |
| Follow-up issues opened only for concrete missing proof discovered during synthesis. | Existing prerequisite lanes remain #1470, #1472, #1474, #1475, and #1358. Latest comments say no new synthesis child is justified yet. | Met as routing discipline; no new issue needed from this audit. |

## Current Integration State

- Blockers remaining: two more complete lanes required; four lanes blocked; one lane
  ready but not synthesis-complete, per the durable synthesis report.
- New blockers found by this audit: none.
- Intentional blockers: full comparative synthesis waits for campaign-backed durable
  component evidence and must not count launch packets, fallback, degraded, or
  smoke-only rows as benchmark success.
- Next empirical action: finish or fail-close component campaign evidence in the
  prerequisite lanes, then rerun the #1489 synthesis report.

## Closure Decision

Do not close issue #1489. The merged PRs satisfy the local contract, validation,
status, and reporting machinery criteria, but the issue's scientific acceptance
criterion still requires downstream comparable component evidence. That remaining
work is excluded from this CPU-only closure audit because it depends on component
campaign outputs, not another local code or documentation slice.

## Validation Plan

This report is a documentation/evidence integration artifact. Minimum proof is:

```bash
uv run python scripts/dev/check_docs_evidence_integrity.py \
  --files docs/context/evidence/issue_1489_post_4642_integration_audit_2026-07-07.md \
  docs/context/evidence/README.md \
  docs/context/INDEX.md \
  docs/context/catalog.yaml
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/evidence/issue_1489_post_4642_integration_audit_2026-07-07.md \
  --path docs/context/evidence/README.md \
  --path docs/context/INDEX.md
git diff --check
```

The full explicit `docs/context/catalog.yaml` proof-consistency path currently
reports unrelated pre-existing catalog rows pointing at ignored `output/`
artifacts, so this audit uses the changed-file evidence-integrity check for the
new catalog registration.

No full benchmark campaign run, no Simple Linux Utility for Resource Management
(SLURM) or graphics processing unit (GPU) submission, and no paper or dissertation
claim edits are included.
