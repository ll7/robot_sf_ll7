# Issue #2568 Adversarial Expansion Gate (2026-06-07)

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2568>

Status: current workflow gate for learned adversarial expansion. This note does not train RL,
diffusion, or learned-proposal models, and it does not claim adversarial coverage, planner
weakness, leaderboard movement, or paper-facing benchmark evidence.

## Result

Broad adversarial RL, diffusion, and learned-proposal work remains gated behind manifest smoke and
quality evidence. The gate is no longer just "create a manifest generator": the minimum prerequisite
is a generated batch that passes the Issue #2562 planner-smoke path and is summarized under the
Issue #2567 quality metrics without invalid, duplicate, degenerate, or low-yield behavior becoming
the claimed result.

Current prerequisite surfaces:

- [issue_2524_adversarial_manifests.md](issue_2524_adversarial_manifests.md) defines the
  validator-backed `adversarial_scenario_manifest.v1` generation path.
- [issue_2562_adversarial_manifest_smoke.md](issue_2562_adversarial_manifest_smoke.md) proves a
  route-materialized planner smoke path for one bounded seed/config.
- [issue_2567_adversarial_manifest_quality.md](issue_2567_adversarial_manifest_quality.md) adds
  `adversarial_manifest_quality_summary.v1` for validity, degeneracy, novelty, perturbation, and
  optional planner-yield signals.

## Gate Rule

Do not start broad adversarial RL training, diffusion training, or learned-proposal expansion until
the candidate batch being used for that expansion has a compact Issue #2567-style quality summary
showing:

- manifests parse and validate against the manifest contract;
- invalid and degenerate rates are low enough for the intended experiment and are reported instead
  of hidden;
- normalized-control-hash duplicates are measured and duplicate-heavy batches are rejected or
  explicitly labeled diagnostic-only;
- perturbation distance from a reference or seed batch is reported when the expansion claims
  candidate diversity;
- optional planner-smoke yields distinguish native, adapter, fallback, degraded, simulation-error,
  and not-available modes;
- any failure-yield signal is described as diagnostic generator quality unless a later certified
  benchmark issue supplies durable benchmark proof.

The current Issue #2567 smoke shows a useful tooling path and a small valid sample, but it is still
quality-metric smoke evidence. It is not enough by itself to claim adversarial coverage or planner
weakness for RL/diffusion.

## Open Issue Sweep

Searches on 2026-06-07 found:

- Issue #2470 and Issue #2471 are closed proposal/interface notes, not active training issues.
- Issue #2521 is an open empirical-followups epic whose freeze note mentioned broad diffusion/RL
  work. It was updated with a GitHub comment during this issue so downstream execution sees
  Issue #2568, not manifest generation alone, as the active learned-expansion gate.
- Issue #1488 is an older blocked bounded-search umbrella, not a learned/RL/diffusion expansion issue.

## Claim Boundary

This gate only controls when learned adversarial expansion is allowed to proceed. Passing the gate
does not promote generated scenarios into benchmark evidence. Benchmark-facing claims still need a
separate issue with certified scenarios, durable inputs, native/adapter/fallback status accounting,
seed discipline, and recorded validation.

## Validation

This was a docs/workflow update. Validation should check:

```bash
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
