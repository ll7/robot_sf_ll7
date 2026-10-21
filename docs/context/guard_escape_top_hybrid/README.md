# Top confinement hybrid: guard/escape development campaign

**Diagnostic-only. Registered result: EFFECT, guard only**, agreeing with
[the continuous-hybrid replication #10426](https://github.com/ll7/robot_sf_ll7/pull/10426).
All 2,160 episodes on development seeds 1061–1090 are valid, complete and paired.
This is within-controller evidence on a fixed 18-cell roster; it does not explain
historical rankings or establish a general hybrid-class advantage.

The 0.0.8 Table 7.2 / Section 7.6 sources select **hybrid v4**: first in static
constriction (234/240), tied first in portal (206/210; mean tied rank 2).
The continuous configuration ranks fourth in both (230/240 and 205/210).
See [selection provenance](provenance/controller-selection.json).
Escape-on retains this controller's native three active escape switches;
corridor-subgoal is off in every arm. The comparator had four active switches.
This limits cross-controller interpretation; no component is separately identified.

The [plan](PLAN.md) was committed and pushed before acquisition:
`55a98a1e6143a71e18278478ba4d3dff9bbf8bd4`, **2026-10-18 07:51:43 UTC**.
Remote readback completed before 2026-10-18T07:53:55.905240+00:00;
acquisition ran 2026-10-18T07:53:55.924744+00:00 to
2026-10-18T09:46:38.016110+00:00. [Registration receipt](provenance/registration.json).

## Arm success

G=hard guard, E=native escape bundle. Counts are collision-free goal completion.
Arm intervals and all four endpoints are in [figure_data.csv](evidence/figure_data.csv).

| Group | G0E0 | G0E1 | G1E0 | G1E1 |
| --- | ---: | ---: | ---: | ---: |
| Static constriction | 125/240 (52.08%) | 121/240 (50.42%) | 230/240 (95.83%) | 228/240 (95.00%) |
| Portal | 137/210 (65.24%) | 137/210 (65.24%) | 204/210 (97.14%) | 202/210 (96.19%) |
| Crowding control | 49/90 (54.44%) | 48/90 (53.33%) | 51/90 (56.67%) | 49/90 (54.44%) |

## Predeclared effects and comparison

On-minus-off main effects average across the other factor. Intervals are pointwise
95% percentile intervals from 10,000 whole-seed-block draws, NumPy default_rng(1001),
equal cell weighting. They condition on this roster, not unseen scenarios.
Five-point qualification uses unrounded values: absolute effect >=5 pp and an
interval excluding zero in at least one confinement group.

| Group | Factor | Top hybrid, pp [95% interval] | Five-point rule | #10426 continuous, pp [95% interval] |
| --- | --- | ---: | --- | ---: |
| Static constriction | guard | +44.17 [+39.79, +48.75] | yes | +44.17 [+38.96, +49.17] |
| Static constriction | escape | -1.25 [-2.71, -0.21] | no | +0.42 [-2.50, +3.33] |
| Portal | guard | +31.43 [+27.86, +35.00] | yes | +34.76 [+32.14, +37.38] |
| Portal | escape | -0.48 [-1.43, +0.48] | no | -0.95 [-3.10, +0.95] |
| Crowding control | guard | +1.67 [-4.44, +7.78] | control only | -10.00 [-16.11, -3.89] |
| Crowding control | escape | -1.67 [-5.56, +1.11] | control only | -2.22 [-9.44, +4.44] |

The guard qualifies in both confinement groups. Its crowding-control point effect
(+1.67 pp) is smaller in magnitude than either confinement effect, so the registered
control comparison passes. Its interval includes zero; this does not prove a null control.
Escape qualifies in neither group. Its static effect is small and negative with
an interval excluding zero, so the threshold result is not a claim of no detectable
change. Both confinement escape upper bounds are below +5 pp under the registered
secondary reading; no equivalence or individual-component claim is made.
The two controllers agree on the registered guard-only decision. Comparison is
descriptive across disjoint seed cohorts and different native escape bundles;
there is no pooled estimate or cross-controller causal contrast.
The keep/fold RQ consequence remains a recommendation, without manuscript admission.

| Group | Interaction, pp [95% interval] |
| --- | ---: |
| Static constriction | +0.83 [-0.83, +2.50] |
| Portal | -0.95 [-2.86, +0.95] |
| Crowding control | -1.11 [-3.33, +0.00] |

No confinement interaction interval excludes zero. All 36 contrasts, including
pedestrian collision, wall collision and timeout, are in
[paired_effects.csv](evidence/paired_effects.csv).

## Validity, source and preservation

- All 2,160 producer rows: declared source, actual seeds, registered effective arm
  config, exact selected SVG path/hash, budgets, consistent endpoints and termination.
- 540 complete seed/cell blocks; zero duplicates, missing slots, exclusions,
  fallback or degraded rows; all executions use the declared differential-drive adapter.
- Numerical analyzer byte-identical to #10426; actual seed labels are mapped only
  in a temporary analysis view. Both CSVs reproduce byte for byte through the archived
  acquisition analyzer and a fresh rerun. Independent contrast/interval error is
  at most 1.67e-16 in proportions. [Verification](evidence/verification.json).
- Original execution commit object is unavailable. Acquisition uses reconstructed
  commit `6a3ac329d25a97202b050edd69190142fd7553e7`: recorded release base plus the
  archived complete source/test patch, with matching result blob identities.
  This is disclosed in the predeclaration and [source receipt](provenance/source-reconstruction.json).
  It is not recovery of the original commit object. Full source snapshot and Git bundle are archived.
- Python 3.13.14, NumPy 2.4.6, PyYAML 6.0.3 and recorded lock identity; four CPU
  workers, one numerical thread each. Launch free disk was 144.59 GB.
- Guard fixtures fail on main and pass on reconstructed source (7 tests); the new
  seed-label adapter regression fails on main and passes after the change.
  Existing replication outputs remain byte-identical. Core and optional test lanes
  exited zero through the three-slot host helper, two workers each, at commit
  `22c164dde4486a4106c4b2eb2ca0f03fbd6410b0`; final changes add results/provenance only.
  Ruff, format and typing checks pass. Full PR readiness is not claimed: independent
  domain and final-head review remain pending. [Validation](provenance/validation.json).

Full run uploaded **once** to [W&B](https://wandb.ai/ll7/robot_sf/runs/8820f7ba),
artifact `ll7/robot_sf/guard-escape-top-hybrid-development-1061-1090:v0`. Fresh API download matches archive and manifest
SHA-256. [W&B receipt](provenance/wandb.json), [archive identity](provenance/archive.json).
Raw episode records and their compact projection stay outside Git in that archive.
The producer receipt corrects only the copied helper's inherited campaign-role label;
no producer bytes or numerical method changed. Public custody files omit private
host paths. Thesis wording, ledger, release artifacts and evidence admission are unchanged.

For standalone numerical reproduction, extract the verified artifact archive and run
from its root with the declared Python/NumPy/PyYAML environment:

```bash
python analysis-helpers/reproduce.py --packet analysis-packet \
  --output /tmp/top-hybrid-recomputed --acquisition-seed-start 1061 \
  --verify --crosscheck-original
sha256sum -c CAMPAIGN_SHA256SUMS
```
