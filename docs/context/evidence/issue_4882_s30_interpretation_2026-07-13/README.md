<!-- AI-GENERATED (robot_sf#4882, 2026-07-13) — NEEDS-REVIEW -->
# Issue #4882 — S30 Hybrid-vs-ORCA Interpretation Packet (independent read)

Independent, diagnostic-only interpretation of the arm-restricted h600 **S30** (30-seed,
seeds 111–140) hybrid-vs-ORCA benchmark wave pre-registered in
[docs/context/issue_4365_h600_hybrid_vs_orca_s30_preregistration.md](../../issue_4365_h600_hybrid_vs_orca_s30_preregistration.md)
(config `configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml`, scenario matrix
`configs/scenarios/classic_interactions_francis2023.yaml`).

This packet is an **independent re-derivation** built to give the maintainer the one independent
domain-aware verdict that draft PR #5514 flagged as its remaining readiness gate. It does not
replace or overturn #5514; it recomputes the six-arm result from the raw episodes under the
pre-registration's stated CI method and reports where the branch verdict is sensitive to
methodological choices the pre-registration left unspecified.

## Classification

- `evidence_tier`: `diagnostic-only`
- `result_classification`: `branch_b_boundary`
- `status`: `evidence`
- Claim boundary: diagnostic-only; feeds a maintainer branch decision; promotes **no** paper,
  dissertation, leaderboard, record-breaking, or universal-planner claim. No campaign, Slurm, or GPU
  job was launched for this analysis.

## Headline verdict — method-dependent

The pre-registration frames the intended analysis as an **F-C4(ii) separation-strengthening test**:
verbatim, *"whether the targeted hybrid-vs-ORCA success lead survives the predeclared 30-seed
schedule on the h600 surface."* It declares **neither** the bootstrap resampling unit **nor** a
CI-overlap-vs-paired-delta decision rule. The branch verdict flips on those two unspecified choices:

| Inference target (bootstrap unit) | Leading-hybrid vs ORCA success, per-arm 95% CIs | Paired Δ success 95% CI | Branch |
| --- | --- | --- | --- |
| **Scenario-clustered** (resample scenarios→seeds; generalize to new scenarios) | `[0.688, 0.849]` vs `[0.591, 0.767]` → **overlap** | `+0.091 [+0.024, +0.162]` → excludes 0 | **B** under the per-arm-CI rule / A under the paired rule |
| **Seed-block** (fixed 48-scenario suite; the scheme in the maintainer closeout and #5514) | `[0.747, 0.794]` vs `[0.657, 0.703]` → **non-overlap** | `+0.091 [+0.065, +0.117]` → excludes 0 | **A** |

**Headline = `branch_b_boundary`**, selected per the task guardrail *"apply the most conservative
reading"*: under the CI method the pre-registration/task specifies (scenario-clustered hierarchical
bootstrap) and the rule as literally worded ("the leading hybrid arm's CI separates from ORCA's"),
the leading hybrid's per-arm success CI still **overlaps** ORCA's at S30, so the lead does not
separate under the strictest reading.

**Under the fixed-suite seed-block bootstrap the result is `branch_a_separation`** — and my
seed-block re-derivation reproduces the maintainer closeout's exact intervals
(`+9.10 pp [+6.46, +11.67]` success, `−13.47 pp [−15.83, −10.97]` collision), confirming #5514's
computation is internally correct for the method it chose.

The decisive, pre-registration-unspecified fork is therefore the **inference target**: whether the
48 curated scenarios are treated as a *fixed benchmark suite* (seed-block → separation) or as a
*sample from a scenario population* (scenario-clustered → boundary under the per-arm-CI rule). This
is the single decision the maintainer must ratify. Both branches are internally valid; see
[branch_verdict.md](./branch_verdict.md).

## Key numbers (S30, n = 1440 per arm)

- Point estimates (identical across both schemes): leading hybrid **`hybrid_rule_v3_fast_progress_static_escape_continuous`**
  success **77.15%** / collision-event **14.86%**; ORCA **68.06%** / **28.33%**; PPO **71.74%** / **26.46%**.
- All six arms rank hybrids > PPO > ORCA on success; direction is uniform (every hybrid point estimate
  exceeds ORCA on success and undercuts it on collision).
- **All four hybrids separate from ORCA on both success and collision under the seed-block scheme**
  (each paired CI excludes 0). Under the conservative scenario-clustered scheme, **only the leading
  hybrid** retains clean paired separation on both metrics; the other three hybrids' success and/or
  collision paired CIs touch or include 0.
- PPO does **not** separate from ORCA on success or collision under the scenario-clustered scheme
  (success `+3.68 pp [−5.90, +13.61]`); under seed-block PPO beats ORCA on success but not collision.

## Contents

- [`s30_arm_statistics.csv`](./s30_arm_statistics.csv) / [`.json`](./s30_arm_statistics.json): per-arm
  success, collision-event, SNQI, near-misses, normalized-time, path-efficiency, mean wall-time, with
  **both** scenario-clustered and seed-block 95% CIs.
- [`pairwise_deltas.csv`](./pairwise_deltas.csv) / [`.json`](./pairwise_deltas.json): paired deltas
  (each hybrid vs ORCA, PPO vs ORCA, PPO vs leading hybrid) on success/collision/SNQI under both schemes.
- [`branch_verdict.json`](./branch_verdict.json) / [`branch_verdict.md`](./branch_verdict.md): the
  branch classification, the two readings, and the maintainer decision fork.
- [`s20_s30_rank_stability.json`](./s20_s30_rank_stability.json) / [`.md`](./s20_s30_rank_stability.md):
  S20-prefix (seeds 111–130) vs S30 rank stability via the S10⊂S20⊂S30 prefix-nesting property.
- [`input_audit.json`](./input_audit.json): identity-filter provenance — contaminated source,
  filter rule, rows before/after per arm, source SHA-256 digests, limitations.
- [`campaign_crosscheck.json`](./campaign_crosscheck.json): roster/row-count/grid cross-check.
- [`summary.json`](./summary.json): machine-readable classification fields.
- [`SHA256SUMS`](./SHA256SUMS): checksums for the generated files in this directory.

## Identity filter (provenance)

The five non-PPO arms come from the resume-append **contaminated** camera-ready root (8,640 rows/arm).
Each file is exactly **six complete append-blocks** keyed by `git_hash`, and each block holds all 1,440
canonical `(scenario_id, seed, horizon=600)` identities (48×30). I verified **zero cross-block metric
disagreements** on success / collision-event / SNQI across all 1,440 identities in every arm, so the
block selection is provably immaterial to the result; the canonical representative is the frozen public
commit `a596a33b…` (last-timestamped attempt / pre-registered frozen S30 SHA). The PPO arm is the clean
job-13388 rerun (1,440 rows, 1,440 unique identities, single commit `502564935…`, `resume:false`). Every
arm resolves to exactly 1,440 unique identities; the builder is fail-closed and STOPs otherwise. See
[`input_audit.json`](./input_audit.json).

## Limitations

- **PPO commit offset**: the PPO arm ran at public commit `502564935` (main + config-only) vs the five
  clean arms at `a596a33b`; the delta is configuration-only and the scenario-matrix hash / seed set /
  horizon match, so the arm is matrix-compatible but the offset is stated honestly.
- **SNQI** carries a documented contract-failure caveat (#5512); reported as a companion only.
- **Efficiency**: the leading hybrid's mean episode wall time is ~13.6× ORCA (18.25 s vs 1.34 s).
- Simulator-only evidence; incomplete AMV coverage; no independent clean-machine or real-world validation.
- **Bootstrap-unit sensitivity is the headline finding**, not a footnote: the S30 separation verdict is
  not robust to the inference target, and the pre-registration did not fix it.

## Reproduction

Bootstrap: 20,000 resamples, RNG seed 20260713, percentile 2.5/97.5. Scenario-clustered scheme resamples
the 48 scenarios with replacement then 30 seeds within each (nan-aware mean); seed-block scheme resamples
the 30 seed-blocks (each = all 48 scenarios at one seed) with replacement. Paired deltas reuse the same
resample indices for both arms on the shared 48×30 grid. Integrity:
`uv run python scripts/dev/check_docs_evidence_integrity.py --files <files in this directory>`.
