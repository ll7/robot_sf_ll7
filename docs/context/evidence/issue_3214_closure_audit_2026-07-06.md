# Issue #3214 Closure Audit (2026-07-06) — Closable as Negative Result

Issue: [#3214](https://github.com/ll7/robot_sf_ll7/issues/3214)

Status on 2026-07-06: **closable as a completed negative research result**.

This note supersedes the prior [`issue_3214_closure_audit_2026-07-05.md`](issue_3214_closure_audit_2026-07-05.md)
`blocked_not_closable` audit. It re-maps the acceptance criteria against current tracked evidence and
argues that the model-side hypothesis of #3214 has been tested and rejected, so the issue is complete
as a documented negative result rather than blocked. It is a closure-audit artifact only: no Slurm
scheduler or GPU job was submitted, no full benchmark campaign was run, no model checkpoint was
promoted, and no paper or dissertation claim was edited.

## Why This Supersedes The 2026-07-05 Audit

The prior audit held #3214 open for two reasons that do not survive scrutiny against the issue's own
scope and the repository fail-closed policy:

1. **"Checkpoint promoted durably (registry/W&B)" was read as an unmet hard requirement.** For a
   checkpoint that fails the closed-loop success gate (0.087 < 0.30), registry promotion is
   *prohibited* by the fail-closed benchmark policy
   (`docs/context/issue_691_benchmark_fallback_policy.md`): a degraded/gate-failing model must not be
   promoted as usable evidence. Durable provenance for the negative run already exists via W&B run id
   `3tu3tmee` plus the SHA256-checked evidence bundle. So this sub-clause is satisfied *as appropriate
   for a negative result* — durable record kept, false promotion correctly withheld — not left
   "blocked".
2. **"Public control-law-side change" was treated as a remaining #3214 criterion.** A control-law
   change is a *different bet* (the maneuver-authority / control-law lane, gestured at by #4205), not
   part of #3214. The issue's own **Scope / Out of scope** section reads: *"Out: new model
   architectures (tracked separately)."* #3214 is the model-side data-augmentation bet; retrying with a
   control-law change is a distinct follow-up, and keeping #3214 open to track it conflates two bets.

The one genuine sequencing constraint the maintainer recorded — *"sequenced after #3216"* (the
`/issue-audit` note, 2026-06-30) — is now satisfied: **#3216 is CLOSED**. The maintainer note also
anticipated completion ("completing this may make a #3216 rerun worthwhile"), i.e. it expected #3214
to reach a terminal result, which it now has.

## Hypothesis And Stop-Rule Outcome

- **Hypothesis (from the issue):** retraining the predictive model with crossing-conflict-weighted
  data raises hard-seed success over the current lineage.
- **Outcome:** tested via follow-up #3254 (Slurm job `13042`). Trajectory-quality gate passed
  (validation ADE `0.04837`, FDE `0.09735`); closed-loop success `0.08696` — below the `0.30` gate and
  below documented baselines (`predictive_proxy_selected_v2_full` 0.101, camera-ready
  `prediction_planner` 0.069). ADE improved while hard success did not.
- **Stop-rule branch fired:** the issue's stop rule says *"if ADE improves but hard success does not,
  that corroborates the #3204 finding that ADE ≠ benchmark success and is recorded as such."* That is
  exactly what happened; the result is a **legitimate recorded negative result** corroborating #3204,
  classified `negative_training_result_not_benchmark_promotion`.

## Criteria To Evidence

| Acceptance criterion | Status | Evidence |
| --- | --- | --- |
| Hard-case-weighted training config and explicit reproducible data-weighting spec. | **Met.** | PR [#3255](https://github.com/ll7/robot_sf_ll7/pull/3255): `configs/training/predictive/predictive_crossing_conflict_hardcase_mixing_issue_3214.yaml`, `mixing.weighting_spec` resolution + `repeat_hardcase_rows` oversampling in `scripts/training/build_predictive_mixed_dataset.py`, pipeline plumbing, and focused unit tests. |
| ≥1 retrained checkpoint evaluated vs baseline on `predictive_hard_seeds_v1`, nav gate separate from ADE/FDE gate. | **Met (as diagnostic negative evidence).** | Follow-up #3254 ran Slurm `13042`; PR [#3515](https://github.com/ll7/robot_sf_ll7/pull/3515) preserved the compact bundle `docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23/` (ADE 0.04837 / FDE 0.09735 trajectory gate vs closed-loop success 0.08696, reported separately; baselines documented). |
| Decision per stop rule with uncertainty; checkpoint promoted durably, not local-only. | **Met (negative-result reading).** | Decision recorded as negative/corroborates-#3204 in `docs/context/issue_3254_predictive_crossing_conflict_negative_result.md`. Durable provenance: W&B run `3tu3tmee` + SHA256-checked evidence bundle. Registry promotion correctly withheld per fail-closed policy (a gate-failing checkpoint must not be promoted). |
| Result classified on the evidence ladder; no fallback/degraded promotion. | **Met.** | Classified `negative_training_result_not_benchmark_promotion` (not paper-facing, not benchmark promotion). Tracked readiness packet `configs/training/predictive/predictive_retraining_readiness_issue_3214.yaml` records `prior_result.status: verified_negative` and keeps `launch_decision.state: blocked_until_control_law_change` — i.e. no further *launch* without a distinct control-law bet. |

## Reading The Readiness Packet Correctly

`predictive_retraining_readiness_issue_3214.yaml` sets `launch_decision.state:
blocked_until_control_law_change`. That flag governs **whether to launch another retraining run**, not
whether #3214 is complete. The model-side bet #3214 asked about has already been launched and answered
(negative). The `required_before_rerun` inputs (`control_law_change_config`, …) belong to a *future,
different* experiment, which should live as its own issue. The launch-block and issue-closure are
therefore consistent: do not re-launch this exact bet, and close #3214 as a resolved negative result.

## Closure Decision

**Recommend closing #3214** as a completed negative research result. All four acceptance criteria are
satisfied under an honest negative-result reading; the only "remaining" items are (a) promotion of a
gate-failing checkpoint, which the fail-closed policy prohibits, and (b) a control-law-side change,
which is out of #3214's stated model-side scope and belongs to a separate follow-up.

Confidence ~80%. The residual uncertainty is intent: if the maintainer prefers #3214 to remain an
umbrella that stays open until a *second* (control-law-assisted) retraining attempt is tried, the PR
gate can demote the `Closes` keyword and the issue stays open. In that case the correct next action is
to open a dedicated control-law-side retraining issue rather than reopening this model-side bet.

## Validation

```bash
# Bundle integrity + provenance consistency for the underlying negative result
cd docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23 && sha256sum -c SHA256SUMS
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_3254_predictive_crossing_conflict_negative_result.md \
  --path docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23/README.md \
  --path docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23/summary.json
```

## Claim Boundary

Closure-audit / research-triage artifact only. It makes no new benchmark, metric, model-provenance, or
paper-facing claim; it reclassifies existing tracked evidence and records a closure recommendation with
explicit uncertainty. The underlying #3254 result remains an analysis-only negative training result.
