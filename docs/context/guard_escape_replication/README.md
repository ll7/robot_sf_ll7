# Guard/escape preregistered replication

**Diagnostic-only, fixed-roster within-controller evidence.** This packet uses
development seeds 1031–1060, fresh relative to the original guard/escape study
(seeds 1001–1030), and the original execution source. It does not
establish historical between-planner rankings, individual-component mediation,
release evidence or manuscript admission. Intervals are pointwise, conditional
on the 18-cell roster, and do not measure uncertainty over new scenarios.

This is a separate guard/escape experiment. The earlier
[settings-off experiment](https://github.com/ll7/robot_sf_ll7/issues/10006#issuecomment-6134330473)
already used the same seed numbers, 1031–1060, across 48 cells at source
`373dbfde4f39667cf9e8732dabe7df5118bdeab1`. Freshness here is relative to the
original guard/escape study, not prior use in other experiments. No settings-off
rows are pooled or compared with this replication. The permitted secondary
60-seed pool contains only the original guard/escape study (1001–1030) and this
replication (1031–1060); the settings-off experiment remains distinct.

Primary stop-rule outcome: **EFFECT**. Guard: **EFFECT**.
Escape: **NO EFFECT**. Overall agreement with the original:
**True**. The original review's
“guard and/or escape” OR interpretation and the exact copied rule were frozen
before the replication in [PREREGISTRATION.md](PREREGISTRATION.md).

## Registration proof

Registration commit: `ab9b418e69fb61a2ed85133a1036d5a87299cd51`.
Push completed UTC: `2026-10-16T10:57:58Z`. Remote branch head readback
matched that commit. Draft PR: <https://github.com/ll7/robot_sf_ll7/pull/10426>.
[Registration receipt](provenance/registration.json) pins every registered file.
[Acquisition receipt](provenance/acquisition.json) proves acquisition started
later. The registration file and all registered input/analyzer bytes remain
unchanged. Original packet: `acd878b4028341846e1fc1c8b0bf0c3ef9923dad`;
execution: `0035fba05c9b679e39f58ec286f5d57b96bb9a0a`.

## Results

All 2,160 episodes completed: 540 per arm, 18 cells, 30 paired seed blocks.
All declare `HybridRuleLocalPlannerAdapter`, `differential_drive` and
`unicycle_vw`. Fallback: 0; degraded: 0; missing/duplicate slots: 0;
exclusions: 0. Exact producer census and same-input comparisons are in
[raw-audit.json](provenance/raw-audit.json).

| Group | Factor | Original, pp [95% CI] | Replication, pp [95% CI] | Pooled 60 seeds, secondary |
| --- | --- | ---: | ---: | ---: |
| static_constriction | guard | +46.0417 [+42.5000, +50.0000] | +44.1667 [+38.9583, +49.1667] | +45.1042 [+41.8750, +48.2292] |
| static_constriction | escape | +3.9583 [+1.2500, +6.8750] | +0.4167 [-2.5000, +3.3333] | +2.1875 [+0.2083, +4.2708] |
| static_constriction | interaction | -6.2500 [-11.2500, -1.2500] | +2.5000 [-5.0000, +9.1667] | -1.8750 [-6.4583, +2.2969] |
| portal | guard | +34.0476 [+30.7143, +37.1429] | +34.7619 [+32.1429, +37.3810] | +34.4048 [+32.2619, +36.4286] |
| portal | escape | +0.2381 [-2.1429, +2.8571] | -0.9524 [-3.0952, +0.9524] | -0.3571 [-2.0238, +1.1905] |
| portal | interaction | +2.3810 [-1.4286, +6.1905] | -0.0000 [-3.3333, +3.3333] | +1.1905 [-1.4286, +3.8095] |
| crowding_control | guard | -1.1111 [-6.6667, +5.0000] | -10.0000 [-16.1111, -3.8889] | -5.5556 [-10.0000, -1.1111] |
| crowding_control | escape | -6.6667 [-15.0000, +1.1111] | -2.2222 [-9.4444, +4.4444] | -4.4444 [-9.7222, +0.8333] |
| crowding_control | interaction | +2.2222 [-5.5556, +8.8889] | +2.2222 [-8.8889, +12.2222] | +2.2222 [-4.4444, +8.8889] |

The guard's crowding-control effect is **−10.0000 pp [−16.1111, −3.8889]**,
an adverse effect whose interval excludes zero. The original control estimate
was −1.1111 pp [−6.6667, +5.0000]. The registered control rule still passes
because 10 pp is smaller in magnitude than both qualifying confinement effects;
this does not establish a null control or a uniformly beneficial guard. The
static-constriction interaction also does not reproduce the original negative
interval: replication +2.5000 pp [−5.0000, +9.1667] versus original −6.2500 pp
[−11.2500, −1.2500]. Agreement concerns the registered main-effect decision,
not every secondary pattern. Escape's upper bounds are 3.3333 pp and 0.9524 pp,
so both satisfy the pre-declared bounded-below-five-points reading.

The pooled estimate is **secondary, not preregistered as primary**. It samples
60 complete independent seed blocks with the same 10,000-draw method and RNG;
it does not enter the primary decision. The original and replication seed sets
are disjoint. All endpoints and interactions are retained in
[paired_effects.csv](evidence/paired_effects.csv) and
[pooled_effects.csv](evidence/pooled_effects.csv); exact rates/counts and intervals
are in [figure_data.csv](evidence/figure_data.csv).

## Mechanical hypothesis decisions

Guard: **EFFECT**; MIXED=False; control moves as much=False.

- static_constriction: +44.1667 pp [+38.9583, +49.1667]; qualifies=True; positive hypothesis supported=True.
- portal: +34.7619 pp [+32.1429, +37.3810]; qualifies=True; positive hypothesis supported=True.

Escape: **NO EFFECT**; MIXED=False; control moves as much=False.

- static_constriction: +0.4167 pp [-2.5000, +3.3333]; qualifies=False; positive hypothesis supported=False. Secondary 'bounded below five points'=True.
- portal: -0.9524 pp [-3.0952, +0.9524]; qualifies=False; positive hypothesis supported=False. Secondary 'bounded below five points'=True.

These decisions recommend the registered keep/fold disposition only. They do
not authorize thesis edits. The guard intervention removes the named collision,
clearance and braking rejection bundle while retaining speed limits, prediction
and scoring. A contribution to this controller's success is not an identified
explanation of its comparative historical lead. Escape-off disables the named
four-switch bundle; common route guidance remains, and inactive switched branches
cannot be attributed individual effects. Failure of the five-point rule alone
is not equivalence or proof that recovery has no role.

## Reproduce and verify

The original `analyze.py` is byte-identical. It hard-codes original seed labels;
the preregistered [reproduce.py](reproduce.py) supplies a temporary, reversible
label view while compact/raw evidence retains actual acquisition seeds. The
output adapter corrects identity metadata and removes the inherited original
review verdict. It does not alter any rate, contrast, interval, outcome or pairing.

With Python 3.13.14, NumPy 2.4.6 and PyYAML, from repository root:

```bash
uv run python docs/context/guard_escape_replication/reproduce.py \
  --output /tmp/guardesc-replication-recomputed --verify --crosscheck-original
uv run python docs/context/guard_escape_replication/compare.py \
  --output /tmp/guardesc-replication-comparison
sha256sum -c docs/context/guard_escape_replication/SHA256SUMS
```

The archived original acquisition analyzer crosschecks both CSVs byte for byte;
a separate seed-contrast calculation checks all 36 contrasts and their interval
bounds. Six invalid-packet controls are rejected. The eight pre-result rule
fixtures and the label/pooling regressions used no replication outcomes.
[verification.json](evidence/verification.json) records exact checks. Full runtime
suite and extra smoke/identity simulations were not run: runtime is unchanged,
and the registered task is this complete factorial acquisition.

Full acquisition replay uses a linked worktree at the execution SHA (or public
release plus the original acquisition-source patch), the exact environment in
[environment.json](provenance/environment.json), and all registered inputs copied
into that worktree under this packet path:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES= SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
uv run --frozen --no-sync python scripts/tools/run_camera_ready_benchmark.py \
  --config docs/context/guard_escape_replication/inputs/full.yaml \
  --output-root /tmp/guardesc-replication-full --label exp_guardesc_replication \
  --log-level INFO
```

## Deviations and custody

No deviation from the registered primary design, seeds, stopping rule, execution
source, arm settings, analysis, validity checks or decision rule. The label-only
adapter was declared before acquisition; it is not a post-result change. The
secondary pooled estimate is separated from the registered primary result.
Study fields describing the original smoke, identity and retrospective custody
were retained to satisfy the identical-study requirement; they did not trigger
new episodes or redefine this registration. Isolated Ruff 0.16.10 validates packet
scripts because the unchanged acquisition environment provides Ruff 0.16.7.

The compact projection retains endpoints, actual seeds, exact factor/source
identity, budgets, termination and SHA-256 of each producer line. Raw trajectories,
diagnostics and full logs stay outside Git under the original large-output rule.
The full archive includes all producer files, logs, input packet, source patch,
registration/acquisition receipts and a manifest; archive members are cold-read
and SHA-256 checked before upload. The final custody receipt is produced after
that archive, avoiding a self-referential manifest. Final Git SHA256SUMS covers
the final compact packet, including that receipt.

Full run uploaded **once** to W&B: [ll7/robot_sf/guard-escape-replication-development-1031-1060:v0](https://wandb.ai/ll7/robot_sf/runs/0y6h7fvo). A fresh W&B API download matched the archive and manifest SHA-256 values. See [wandb.json](provenance/wandb.json) and [archive.json](provenance/archive.json). Credentials were loaded only in process memory and never printed.
