# Issue #1489 Closure Audit

Plain-language summary: issue #1489 is not ready to close. Merged PRs delivered
the shared hybrid-learning evidence schema, validator, prerequisite gate,
component-readiness matrices, and integration-status report, but the live issue
acceptance criteria still require at least two component campaigns with durable
comparable outputs before comparative synthesis can start.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/1489>
- Audit date: 2026-07-05
- Latest maintainer guidance reviewed: 2026-07-04 issue comment stating #1489
  remains blocked until component campaigns produce durable evidence.
- Merged PRs reviewed:
  - PR #1516, `eb06adf7bef425d0012597bb42cbdf5bec1b5d55`,
    <https://github.com/ll7/robot_sf_ll7/pull/1516>
  - PR #1535, `674fcc99a37b3e8cd725bf15f35f02a654691dc4`,
    <https://github.com/ll7/robot_sf_ll7/pull/1535>
  - PR #1547, `f9d796eb0e1dfbbdbb12581b8bd2644785da2ee7`,
    <https://github.com/ll7/robot_sf_ll7/pull/1547>
  - PR #2286, `66eb0a33b99105be32d9b5771b7763be0bb77745`,
    <https://github.com/ll7/robot_sf_ll7/pull/2286>
  - PR #2419, `8a277b48d9ca319fa6bb586c96a13afa7e7942e0`,
    <https://github.com/ll7/robot_sf_ll7/pull/2419>
  - PR #3736, `a3ea20a47de4ce3e46c7fb93c25f342a60b9ede6`,
    <https://github.com/ll7/robot_sf_ll7/pull/3736>
  - PR #4452, `ebe782f9cc0505e7ec3dad3c15632dfafda5b427`,
    <https://github.com/ll7/robot_sf_ll7/pull/4452>

## Claim Boundary

This is a closure-audit evidence note only. It does not run a benchmark campaign,
submit Slurm or GPU compute, synthesize component performance, upgrade
launch-packet, smoke-only, fallback, degraded, or local-only outputs, or edit any
paper or dissertation claim.

## Acceptance Mapping

| Acceptance criterion from #1489 | Delivered evidence | Audit status |
| --- | --- | --- |
| Define a shared synthesis table and diagnostics contract across component campaigns. | PR #1516 added `docs/context/issue_1499_hybrid_evidence_matrix_schema.md`. PR #1535 implemented row validation in `robot_sf/benchmark/hybrid_evidence_matrix.py` and `scripts/validation/validate_hybrid_evidence_matrix.py`. PR #1547 added opt-in git-history provenance checks. | Met for schema and validator readiness. |
| Link and consume completed evidence from #1470, #1472, #1474, #1475, and #1358 rather than rerunning inside #1489. | PR #2286 added `docs/context/issue_2274_hybrid_component_matrix.md` plus validator-readable rows under `docs/context/evidence/issue_2274_hybrid_component_matrix_2026-06-05/`. PR #2419 refreshed the component-readiness matrix under `docs/context/evidence/issue_2410_hybrid_component_readiness_refresh_2026-06-06/`. | Met for status-matrix consumption, but the consumed rows remain non-synthesis-eligible. |
| Separate paper-grade, stress/full-matrix, nominal-only, smoke-only, failed, degraded, unavailable, and insufficient evidence. | PR #1516 defined the evidence-tier vocabulary and consumer rules. PR #1535 and PR #1547 enforce row shape, fallback/degraded semantics, and provenance checks. PR #3736 added lifecycle states for `missing`, `blocked`, `ready`, and `complete`. | Met for classification machinery. |
| Require each synthesis row to cite exact campaign source, commit or artifact, evaluation slice, evidence tier, and fail-closed status. | PR #1535 validates required row fields and repository-local provenance tokens. PR #1547 adds strict SHA validation. PR #3736 builds the prerequisite matrix from validated rows. | Met for checker contract; no current component row satisfies the synthesis gate. |
| Hard-guard authority and learned-component contribution diagnostics visible in synthesis. | PR #1516 requires `guard_authority`, `learned_component_contribution`, intervention/fallback rates, and outcome fields. PR #1535 validates those fields for rows. | Met as a schema/checker contract; not yet met as synthesis evidence because component campaigns have not produced enough durable comparable rows. |
| Produce a conservative context note classifying each mechanism as continue, revise, stop, or gather-more-evidence. | PR #2286 and PR #2419 provide conservative component status/readiness notes and matrices. They conclude #1489 remains blocked and component rows are non-synthesis-eligible. | Partially met as readiness/status notes; not the final #1489 synthesis note because prerequisite campaign outputs are missing. |
| Synthesis recommends continue, revise, stop, or gather-more-evidence for each mechanism. | Existing matrices carry row-level verdicts, but the issue body requires the synthesis after component campaigns produce comparable evidence. The latest 2026-07-04 issue comment says no implementable slice exists until component campaigns complete. | Not met; blocked on missing durable comparable component campaign outputs. |
| Identify gaps and open follow-up issues only when component evidence is concrete. | Existing child/component issues #1470, #1472, #1474, #1475, and #1358 remain the named prerequisite lanes. Issue comments explicitly say do not open more synthesis children until at least two lanes have durable comparable outputs or fail-closed classifications. | Met as routing discipline; no new follow-up issue is justified by this audit. |
| Reader can tell whether learned components improved downstream navigation metrics rather than only intermediate diagnostics. | No merged PR provides at least two durable comparable component campaign outputs. PR #2286, PR #2419, PR #3736, and PR #4452 all preserve the blocked status instead of promoting readiness/preflight rows into results. | Not met; this is the principal blocker. |
| Synthesis does not count fallback, degraded, launch-readiness, smoke-only, or missing durable artifacts as benchmark success. | PR #1516, PR #1535, PR #1547, PR #3736, and PR #4452 all encode or report this fail-closed boundary. | Met as a guardrail; it is also why #1489 remains open. |

## Closure Decision

Do not close Issue #1489. The enabling contract and integration-reporting
criteria are delivered, but the core acceptance criteria remain blocked by
missing durable comparable component campaign outputs. The current blocker is
external to this closure-audit PR: at least two component lanes among Issue #1470,
Issue #1472, Issue #1474, Issue #1475, and Issue #1358 must produce
durable comparable outputs or
explicit fail-closed classifications suitable for conservative synthesis.

## Local Verification

Audit-time validation is docs-only:

```bash
uv run python scripts/dev/check_docs_evidence_integrity.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/evidence/issue_1489_closure_audit_2026-07-05.md --path docs/context/evidence/README.md --path docs/context/INDEX.md --path docs/context/catalog.yaml
git diff --check
```
