# Issue 1052 Claim-Language Audit

Date: 2026-05-07

Related issue: `ll7/robot_sf_ll7#1052`

## Goal

Audit paper-facing claim language against the narrow benchmark-set decision for issue 791. The
current manuscript-safe claim is benchmark-set performance on the evaluated scenario matrix, not
OOD generalization, transfer, unseen-environment behavior, or architecture-causality proof.

## Canonical Boundary

The active decision is
`memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md`. It allows reporting a strong
policy evaluated on a broad social-navigation benchmark with seed/bootstrap evidence. It does not
allow external wording that the issue-791 leader solved OOD generalization, transferred to novel
environments, or proved a particular architecture caused the lift.

## Audit Findings

The main camera-ready workflow already contains the necessary long-horizon caveat in
`docs/benchmark_camera_ready.md`: success gains under scenario-specific horizons are route-budget
sensitivity results unless collision, near-miss, SNQI, fallback, and runtime evidence are preserved.

`docs/plan/plan_big_picture_2026-04-30.md` already points at the narrow decision and explicitly
rejects OOD, transfer, and unseen-environment claims as current paper scope.

The risky language was concentrated in historical issue-791 notes:

* `docs/context/issue_791_benchmark_rerun_issue_body.md` was an old draft issue body that still
  described a held-out OOD requirement as active.
* `docs/context/issue_791_promotion_campaign_128k_256k.md` correctly records internal engineering
  attribution, but phrases such as OOD gap and distribution alignment could be misquoted as current
  paper framing.
* `docs/context/issue_848_issue_791_eval_aligned_benchmark_rerun.md` treated the old held-out OOD
  requirement as remaining open.

Those notes now carry explicit status or risk text that keeps them historical and points readers
back to the narrow benchmark-set decision.

## Reporting Rule

Use these phrases for paper-facing summaries:

* benchmark-set performance
* evaluated scenario matrix
* scenario-matrix coverage
* seed/bootstrap stability on the benchmark surface

Avoid these phrases unless a separate held-out study exists:

* OOD generalization
* transfer
* unseen or novel environments
* architecture-caused lift
* fixed generalization gap

## Validation

Audit command:

```bash
rtk rg -n "generaliz|transfer|unseen|OOD|DreamerV3|architecture.*lift" docs memory
```

The remaining matches after this audit are either canonical guardrails, historical engineering
records with explicit status text, or unrelated training/planner documentation. They should not be
used as paper-facing claim language without checking the current decision first.
