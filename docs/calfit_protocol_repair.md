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
