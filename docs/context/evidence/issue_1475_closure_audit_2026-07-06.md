# Issue #1475 Closure Audit

Plain-language summary: issue #1475 asked a Slurm-capable owner to run a bounded
ORCA-residual behavior-cloning (BC) smoke job and, if it passed, escalate to a
nominal job producing a durable learned-residual dataset and checkpoint. **This
audit recommends keeping #1475 open.** All CPU-implementable enabling criteria
(launch packet, runtime observation-contract adapter, residual telemetry
emission, smoke-evidence contract, and the smoke-to-nominal gate) are delivered
by merged PRs, and the earlier telemetry-instrumentation stop trigger from
Issue #2445 is now resolved. But two acceptance criteria — a durable learned
residual dataset/checkpoint and a nominal-run continue/revise/stop classification
— remain blocked on a Slurm smoke rerun plus nominal escalation. The most recent
issue status (2026-07-05, on PR #4561) states the same remaining action: "a
Slurm-capable owner still needs to run one bounded smoke rerun and pass the gate
before nominal escalation." That step cannot run on a no-Slurm host, so this
audit consolidates the fragmented history into one criterion→evidence map and
names the exact next empirical action rather than closing the issue.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/1475>
- Audit date: 2026-07-06
- Issue state at audit time: OPEN, 35 comments reviewed.
- Latest status reviewed: 2026-07-05 comment — "Gate update: PR #4561 merged …
  **Remaining:** a Slurm-capable owner still needs to run one bounded smoke rerun
  and pass the gate before nominal escalation." This is the authoritative current
  remaining-work statement and is consistent with the issue-body acceptance
  criteria.
- Interim lane decisions reviewed:
  - Issue #2311 `revise_residual_objective` (2026-06-05) — job 12749 produced a usable
    smoke row after adapter/runtime blockers were fixed, but the residual
    objective needed revision.
  - Issue #2445 `stop` (2026-06-25),
    `docs/context/issue_2445_orca_residual_progress_probe_decision.md` — stopped
    the **unchanged** command-imitation lane. Two triggers fired: (a)
    `success_rate=0.0` + `timeout_low_progress` reproducing the v0 failure, and
    (b) all four required smoke-evidence telemetry fields `null`. Trigger (b) has
    since been fixed (see PRs #2998/#3502); trigger (a) is the policy-objective
    reason a fresh rerun must use the revised residual objective, not a relabeled
    v0/v1 rerun. #2445 also deferred its GitHub update ("handoff only; do not edit
    GitHub from this lane").
- Merged PRs reviewed:
  - PR #1875, `727b397d7f6df5906dcae2151fc4685f053b85ba`,
    <https://github.com/ll7/robot_sf_ll7/pull/1875> — Slurm prep / launch packet.
  - PR #2457, `c28ae03308d77d1819ec9a930234e4f9bdd86839`,
    <https://github.com/ll7/robot_sf_ll7/pull/2457> — smoke checksums.
  - PR #2989, `5b94970458f72f4bfb9104931370a3ef7e0caf0f`,
    <https://github.com/ll7/robot_sf_ll7/pull/2989> — promoted Slurm closeout
    evidence for #1470 and #1475 (job-12913 smoke bundle).
  - PR #3502, `332ffe4e6cbe5cf4bd6ee06944831e84db4e2a45`,
    <https://github.com/ll7/robot_sf_ll7/pull/3502> — emit `residual_clipped`
    on no-residual GuardedPPO decisions (resolves the #2445 missing-telemetry
    trigger for pass-through/fallback decisions).
  - PR #3844, `0b57d9d54dfdce11ab1732d8884df1056564e7f0`,
    <https://github.com/ll7/robot_sf_ll7/pull/3844> — ORCA-residual BC smoke
    readiness packet.
  - PR #4561, `28807d1f1dcf5889fd04e2f9b9bd5f1f324517cb`,
    <https://github.com/ll7/robot_sf_ll7/pull/4561> — CPU-only smoke-to-nominal
    gate (`validate_smoke_nominal_gate`).

## Claim Boundary

This is a closure-audit / integration-report evidence note only. It does not run a
benchmark campaign, collect a BC dataset, submit Slurm or GPU compute, train or
rerun any smoke or nominal job, promote fallback/degraded/smoke-only/launch-readiness
output to benchmark success, or edit any paper or dissertation claim. It maps
already-merged evidence to the acceptance criteria and names the next empirical
action.

## Acceptance Mapping

| Acceptance criterion from #1475 | Delivered evidence | Audit status |
| --- | --- | --- |
| Residual dataset manifest and NPZ are recorded as durable artifacts. | No successful run produced a durable BC dataset. The BC dataset remained a `wandb-artifact://pending/...` alias; the #2445 decision notes "the BC dataset itself is a pending wandb artifact." | **Not met** — blocked on a Slurm run. |
| Learned residual checkpoint pointer is durable and included in the completion update. | No learned-residual checkpoint exists; every submitted job (12670/12672/12749/12900/12913/13034) failed or failed-closed before producing a durable checkpoint. | **Not met** — blocked on a Slurm run. |
| Diagnostics report includes ORCA command, raw/bounded residual, final guarded command, residual clipping rate, guard veto rate, and fallback/degraded status. | PR #3502 makes the planner emit `residual_clipped` even on no-residual GuardedPPO decisions; PR #3844 adds the smoke readiness packet defining the required fields; PR #4561 enforces them (`residual_clipping_rate`, `guard_veto_rate`, `fallback_degraded_status`, `artifact_pointer_status`, success/collision, `nominal_escalation_allowed`). The last recorded smoke (job 12913) still had all four telemetry fields `null`, which is exactly what PRs #2998/#3502 were merged to fix, so a fresh gated rerun is required to populate them. | **Contract met; data pending** — emission + gate exist; no post-fix passing run has populated them yet. |
| Fallback/degraded rows are not counted as learned-residual success evidence. | PR #4561's `validate_smoke_nominal_gate` fails closed when `fallback_degraded_status` is not clear or `artifact_pointer_status` is not durable/complete; the job-12913 `summary.json` sets `claim_boundary` to "Failed-closed smoke evidence only … must not be used to justify nominal or larger Slurm reruns." | **Met** as a guardrail. |
| Smoke result is recorded before nominal escalation. | Job-12913 smoke is durably recorded at `docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/summary.json` (`status: failed_closed`, `success_rate: 0.0`, `nominal_escalation_allowed: false`). | **Partially met** — a smoke result is recorded, but it is failed-closed and pre-dates the telemetry-emission fix; a passing smoke is still required before nominal escalation. |
| Nominal result is classified as ready for #1358 continuation, revise, or stop. | No nominal run occurred. Interim decisions exist over the smoke stage (#2311 `revise_residual_objective`, #2445 `stop` of the unchanged lane), but the post-#2445 telemetry-emission fixes reopened the operational path for a fresh gated rerun on the revised objective. | **Not met** — no nominal-run classification; blocked on a Slurm smoke rerun then nominal escalation. |

## Closure Decision

**Do not close Issue #1475.** The enabling, adapter, telemetry-emission, and
smoke-to-nominal gate criteria are delivered by the merged PRs listed in Source
Thread, and the Issue #2445 missing-telemetry stop trigger is resolved. But the
core execution criteria — a durable learned-residual dataset and checkpoint,
and a nominal continue/revise/stop classification — remain blocked on a Slurm
smoke rerun and nominal escalation that a no-Slurm host cannot perform. This
matches the current 2026-07-05 remaining-work statement on the issue.

Next empirical action (Slurm-capable owner, out of scope for this CPU-only audit):

1. Run one bounded ORCA-residual BC smoke rerun on the **revised** residual
   objective (per #2311 / #2445 — not a relabeled unchanged v0/v1 rerun), from a
   fresh `origin/main` worktree, so the now-emitted telemetry fields
   (`residual_clipping_rate`, `guard_veto_rate`, `fallback_degraded_status`,
   `artifact_pointer_status`) are populated.
2. Validate the produced summary with
   `uv run python scripts/validation/validate_orca_residual_lineage_packet.py --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --smoke-summary <summary.json> --json`.
   The gate requires clear fallback/degraded status, durable artifact pointers,
   and success ≥ 0.80 / collision ≤ 0.02 before `nominal_escalation_allowed`.
3. Only if the gate passes, escalate to the bounded nominal job, promote the
   durable dataset/checkpoint pointers, and record the nominal classification for
   Issue #1358 continuation.

Not in scope for this audit: it does not itself edit GitHub. Because the issue is
blocked (not resolved), the PR uses `Refs #1475`, not a closure keyword.

## Local Verification

Audit-time validation is docs-only:

```bash
uv run python scripts/dev/check_docs_evidence_integrity.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/evidence/issue_1475_closure_audit_2026-07-06.md --path docs/context/evidence/README.md --path docs/context/INDEX.md
git diff --check
```
