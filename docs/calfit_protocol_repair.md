<!-- AI-GENERATED NEEDS-REVIEW -->
# CALFIT repaired protocol — author ruling 2026-10-02

The inherited V3 lattice placed 54 pairs in overlap before the model acted.
The author ruled this a suite input defect, authorized feasible holding layouts,
and retained zero physical violations and the empirical flow targets. The old
screen remains immutable diagnostic evidence, not model acceptance evidence.

Before every source-protocol simulation, admission checks the initial positions
against disc diameter plus .02 m and all wall segments. Invalid input raises
before Simulator construction. V3/V4 source holding layouts now use a deterministic
hex lattice with bounded dev-seed jitter; its pitch includes the worst possible
jitter reduction, so actual pair spacing remains >=2r+.02 m. Keep 60/350 people;
enlarge the holding area where needed rather than remove people.

[Seyfried section 2/Figure 2](https://arxiv.org/pdf/physics/0702004) describes
holding areas with equal initial density3.3/m², the first holding area's centre
3 m upstream, and runs with20/40/60 people. The suite retains the60-person runs,
4 m upstream width and2.8 m channel; holding length increases only when needed
by its nonoverlapping hex packing. At radii.25/.27 the area stays4×4.545455 m
and density3.3; at.28 it becomes4×5.206157 m (density2.881204); at.30 it is
4×5.566157 m (density2.694858). Old area18.181818 m² for all radii.

[Liao section 2](https://doi.org/10.1016/j.trpro.2014.09.005) supplies the
350-person, density3/m² holding semicircle, original radius8.618 m. The suite
chooses the first .01 m radius enlargement that fits350 sites on its density-based
hex lattice with the disc separation and entrance clearance. At radii.25/.27/.28,
new radius8.688 m, area118.565818 m², density2.951947; at.30, radius8.958 m,
area126.049751 m², density2.776681. Every row carries old/new area and density,
source citation and actual initial-state distances. This is an explicit protocol
equivalent, not a claim that the density changed in the source experiment.

The static audit checks actual constructed V2/V3/V4/V5/V6 inputs at dev1001–1003
for radii.25/.27/.28/.30, including all five V6 baseline starts. All pass the same
admission check. V2/V5/V6 starts require no geometry repair.

V2's retained raw traces show actual upstream stalls, not dropped measurements.
For diagnostic5c1f68bcf0f0, seed1001, final x is7.468097/7.546542/7.690361 m for
widths.55/.758/.966; terminal speeds are below4e-15 m/s. None crosses the8 m
passage plane. Legacy control50a1923042bc stops around6.40 m. Reapplying the
unchanged source estimator to raw traces reproduces missing passage; complete
synthetic passage traces remain measured correctly. The rerun records maximum
x, final x/speed and displacement over the final10 s for diagnosis.

V5/V6 use a reported band or mean±reported SD when available. Where none is
reported, the new author ruling supplies **±20% of the literature mean**, explicitly
labelled “no reported SD; author fallback” and “author-delegated engineering
tolerance, 2026-10-02”. No synthetic/published SD is invented. Feasible V2 and
all other acceptance targets remain as already ruled.

Only the original safety-first top10 settings plus one legacy wall-law control
are rerun, on dev1001/1002/1003, with no refinement or further calibration.
Original ranking: physical-failure rows; dense-wall penetration; most complete
V3 observations; fewer missing/numeric checks; ID. The legacy wall control uses
factor10, offset−.57, sigma0, radius.25, cap2 and the required desired N(1.29,.19),
so it isolates wall-law defaults without changing the calibration body/speed contract.
The recorded11-ID grid is bound to the preceding ranking before submission.

Regression evidence before repair: V3 dev1001 minimum centre distance .421414 m <.52 m;
V4 dev1001 radius.30 minimum spacing .563106 m <.62 m; overlapping inputs reached Simulator.
The same controls pass after repair, including direct refusal of the independently specified old60-person lattice (54 overlapping pairs) before Simulator construction. Distribution controls returned exit5 before
the fallback and PASS0 afterward. The new fallback exposed a preflight float
comparison (.61 < .6100000000000001); the independent isolated geometry test
failed before a1e-12 comparison slack and passes afterward. The CLI now admits
the complete ideal bank; that remains software proof, not a model calibration.

Test value: real constructed initial arrays are checked at the simulator boundary;
credible regressions are removing pre-step refusal, reverting the old lattice,
dropping people to fit, or presenting fallback bands as measured SD. Existing
suite tests measured post-step overlap, not input admissibility. These tests use
existing simulation interfaces, no new production seam. Static wall/boundary and
area-expansion controls protect the new geometry; they do not claim an old-model
bug. The preflight fingerprint also binds the new input-module bytes.

## One authorized rerun: result

**NO-PASS: 0/11 settings**, all gate exit3 (hard physics), source commit
`1ffad0099e44a178d3843bb3ff7ae98391640079`. Slurm16351 from imech192:
11 completed tasks ×2 CPUs, observed peak22 CPUs. Full `squeue` was checked
before submission and each worker checked priority; no queued sealed0.0.8
campaign was present. No additional acquisition or tuning followed this rerun.
The immutable grid was the original safety-first top10 plus the legacy wall control;
the table below preserves that order. Desired N(1.29,.19), positive draws with no
upper clipping, separate2/3 m/s integration cap; wall sigma0 throughout.

Independent workstation SHA256 verification covers11 manifests and594 lossless
primary NPZ acquisitions. Reopening their actual first positions finds zero
initial pair/wall overlaps and valid starting gaps in all594 cases, plus all495
V6 baseline starts. V3/V4 retain60/350 people. The row `holding_layout` and
`initial_admissibility` receipts describe actual inputs; inherited config prose
about overlapping starts is historical metadata, superseded by these receipts.

| ID | Law; factor; offset; radius; cap | Dense wall penetration m | Complete V3 /15 | V3 1.2 mean upper bound | Dense overlap pair-steps | Gate exit |
|---|---|---:|---:|---:|---:|---:|
| 5c1f68bcf0f0 | calibrated_v2; 0.003; 0.375; 0.25; 2 | 0.000000 | 1 | 0.316999 | 3363065 | 3 |
| cfdd6a222b7a | calibrated_v2; 0.0003; 0.375; 0.25; 2 | 0.000000 | 0 | 0.316925 | 3430539 | 3 |
| 12971af2fdcd | calibrated_v2; 0.003; 0.375; 0.25; 3 | 0.000000 | 0 | 0.316889 | 3402198 | 3 |
| 52db086a5eaf | calibrated_v2; 0.03; 0.25; 0.25; 3 | 0.007070 | 0 | 0.316943 | 3283960 | 3 |
| 6960f416697f | calibrated_v2; 0.03; 0.375; 0.25; 2 | 0.000000 | 0 | 0.321197 | 3306438 | 3 |
| f01a084ef9f0 | calibrated_v2; 0.03; 0.375; 0.25; 3 | 0.000000 | 0 | 0.324837 | 3315281 | 3 |
| daf1eb32e31d | calibrated_v2; 0.03; 0.25; 0.25; 2 | 0.000000 | 0 | 0.316913 | 3283835 | 3 |
| 160fddb5f1e2 | calibrated_v2; 0.0003; 0.375; 0.25; 3 | 0.009959 | 2 | 0.433167 | 3485638 | 3 |
| 2c0a9572c18f | calibrated_v2; 0.0006; 0.375; 0.27; 3 | 0.020198 | 1 | 0.366237 | 4987947 | 3 |
| 7dfb0af8f517 | calibrated_v2; 0.00015; 0.375; 0.27; 3 | 0.041824 | 4 | 0.748671 | 5028621 | 3 |
| 50a1923042bc | legacy_v1; 10; -0.57; 0.25; 2 | 0.249985 | 0 | 0.318843 | 2977772 | 3 |

The complete-measurement safety/flow **Pareto front is empty**: every candidate
has incomplete V3 width banks, so aggregate flow is null, never imputed zero.
At width1.1 alone, the partial-observation front is5c1f68bcf0f0
(wall0, flow.721034, n=1/3) and7dfb0af8f517 (wall.041824, flow.844643, n=1/3).
At width1.2 alone it is160fddb5f1e2 (wall.009959, flow.665499, n=1/3)
and7dfb0af8f517 (wall.041824, flow.748671, n=3/3). These conditional means are
not a whole-suite Pareto front or passing settings; all develop body overlap.

For every censored width1.2 trace, eventual finite-N flow, if all60 ever cross,
is <=60/((t_end−t_first)×1.2), because its last crossing must be later than the
observed end. Complete traces use the observed value. All33 per-seed values
are below the accepted1.576 minimum. The table reports the mean of those upper
bounds/complete values, **not an uncensored point estimate**. Thus none of these
11 repaired-input settings meets narrow flow, including all six with zero dense
wall penetration. This is a bounded failure of the rerun, not a theorem that
empirical safety and flow are intrinsically incompatible under every law:
other families in the earlier screen had inadmissible starts and were not rerun.
Every setting also develops V3/V4 disc overlaps from valid starts; the rigid-disc
pair kernel has no hard-contact resolution. The zero-overlap target is retained.

## V2 passage diagnosis

Of99 feasible-aperture acquisitions,85 reach stationary upstream stalls,
two remain nonstationary upstream without crossing, and12 cross and receive
finite speed-drop estimates. The latter are widest-width.966 trials of
cfdd6a222b7a/160fddb5f1e2/2c0a9572c18f/7dfb0af8f517. The first two settings
pass that single V2 width; no setting passes all three widths. Replay of the
unchanged estimator matches every saved estimate/missing result (cross-host
polynomial roundoff <1e-12; observed4.4e-16). No missing passage was dropped
by measurement. The best setting's nine V2 traces all stall upstream; the two
nonstationary cases belong to7dfb0af8f517 at seed1001, widths.59/.778.

## Best candidate: full gate table

Safety-first best remains5c1f68bcf0f0: calibrated_v2, factor.003, offset.375 m,
sigma0, radius.25 m, desired N(1.29,.19), separate cap2 m/s. Estimates are
population means over dev1001–1003; n records complete observations. MISSING
with a finite value is a conditional partial mean and cannot pass. Δ is
estimate−literature target; outside is distance beyond the accepted range.
V1/V2 units m/s; V3/V4 persons/(m s); V5/V6 m.

| Gate / variant | Estimate | Target | Δ to target | Accepted range | Outside range | n/3 | Status |
|---|---:|---:|---:|---|---:|---:|---|
| V1 / native | 1.291114 | 1.290000 | 0.001114 | [1.100000, 1.480000] | 0.000000 | 3/3 | PASS |
| V2 / 0.55 | — | 0.207826 | — | [0.096522, 0.333913] | — | 0/3 | MISSING |
| V2 / 0.758 | — | 0.140000 | — | [0.060000, 0.240000] | — | 0/3 | MISSING |
| V2 / 0.966 | — | 0.140000 | — | [0.060000, 0.240000] | — | 0/3 | MISSING |
| V3 / 0.8 | — | 1.610000 | — | [1.288000, 1.932000] | — | 0/3 | MISSING |
| V3 / 0.9 | — | 1.860000 | — | [1.488000, 2.232000] | — | 0/3 | MISSING |
| V3 / 1.0 | — | 1.900000 | — | [1.520000, 2.280000] | — | 0/3 | MISSING |
| V3 / 1.1 | 0.721034 | 1.930000 | -1.208966 | [1.544000, 2.316000] | 0.822966 | 1/3 | MISSING |
| V3 / 1.2 | — | 1.970000 | — | [1.576000, 2.364000] | — | 0/3 | MISSING |
| V5 / diagnostic | 0.865624 | 0.500000 | 0.365624 | [0.400000, 0.600000] | 0.265624 | 3/3 | FAIL |
| V6 / 1.15 | 2.938622 | 2.100000 | 0.838622 | [1.680000, 2.520000] | 0.418622 | 3/3 | FAIL |
| V6 / 1.42 | 2.928919 | 2.400000 | 0.528919 | [1.920000, 2.880000] | 0.048919 | 3/3 | FAIL |
| V6 / 1.78 | 2.930237 | 2.700000 | 0.230237 | [2.160000, 3.240000] | 0.000000 | 3/3 | PASS |
| V4 / all-data slope | 1.709378 | 2.300000 | -0.590622 | [1.840000, 2.760000] | 0.130622 | 3/3 | FAIL |
| V4 / steady slope | 2.573363 | 2.500000 | 0.073363 | [2.000000, 3.000000] | 0.000000 | 3/3 | PASS |

Physical gate: initial overlap/wall penetration0; dense-wall penetration0 PASS;
post-step pair overlap FAIL in all30 V3/V4 rows. V3 has2,918,554 overlapping
pair-steps (6.870419%, minimum centre spacing.000231 m); V4 has3,363,065
(0.229435%, minimum.001044 m), versus diameter.50 m. No hard-contact model
permits these. Overall exit3. V1 fitted tau.448142 s is informational, not an
acceptance target. The .9-shoulder-width case remains “not reproducible with
rigid discs (shoulder rotation)”, outside the gate.

All tolerance rows cite the source below and carry **“author-delegated engineering
tolerance, 2026-10-02”**. V1 mean±1 SD; V2 same positive sign and±50% source
magnitude (including documented anchor interpolation); V3/V4±20%; V5/V6
explicit **no reported SD; author fallback±20%**, not a measured SD.

- V1: Moussaid et al. 2009, section 4(a), Fig. 1b; https://doi.org/10.1098/rspb.2009.0405
- V2: Wilmut et al. 2015, Fig. 3; https://doi.org/10.1371/journal.pone.0124695
- V3: Seyfried et al. 2009, Table 2; https://arxiv.org/abs/physics/0702004
- V5: Gerin-Lajoie et al. 2008 pp.239-240, citing 2005; https://doi.org/10.1016/j.gaitpost.2007.03.015
- V6: Huber et al. 2014, Table 1, 180-degree condition; https://doi.org/10.1371/journal.pone.0089589
- V4: Liao et al. 2014, section 4; https://doi.org/10.1016/j.trpro.2014.09.005

## Verification and custody

Final runtime code7e8ee234c05187dd9bc192fd3039afcc85658da3 only adds return
annotations/test strengthening relative to the acquisition producer; dynamics,
geometry and policy are unchanged. Full default suite with explicit audited
safe file list:10,981 passed,18 existing skips,7 xfails;294 protected-seed
stepping files excluded, verified by collect-only, guard log absent. Focused
coverage:52 passed. Initial full run exposed three missing public return
annotations; those were fixed and the entire safe suite rerun successfully.
No skip/xfail was added. Old-lattice fail-on-base proof fails both source paths
with `AssertionError: simulator reached an inadmissible initial state`; fixed tests refuse
before construction. Protected defaults/releases/artifacts are untouched.

Custody under `/home/luttkule/lanes/calfit/evidence/protocol_repair/`:
`rerun_grid.json`, `campaign/rerun/<ID>/SHA256SUMS` and matching NPZ,
`custody_receipt.json`, `analysis.json` (all33 width1.2 bounds and99 V2 replay
results), `cpu_audit.json`, `static_admission_audit.json`, pre-fix proof logs,
`full.log`, `full.xml`, `full_receipts.json`, `seed_excluded_files.json`, and
changed-coverage verdict. Reproducible environment: lane `repo/.venv`, frozen
uv extras maps/analytics/recurrent/training/viz/benchmark/progress.
Strict curated Sphinx has the inherited `Argument list too long` failure,
previously reproduced on the stack base; no docs-build success is claimed.

**Stop after this rerun.** Keep draftPR10094 and issues10074/10061 open;
no profile promotion, default change, release claim, merge or ready transition.
The author's physical/empirical targets stay unchanged. Reopen calibration
only with a new author scope/rerun ruling or new material evidence; the earlier
inadmissible campaign does not support choosing a target to relax.
