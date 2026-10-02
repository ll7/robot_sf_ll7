# CALFIT preflight and bounded search contract

The preflight checks V1–V6 estimators with known synthetic trajectories, then
sends complete target-valued records to the actual suite gate. It runs no
simulation and changes no default or admission policy:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m scripts.validation.calfit_preflight_10074 --out output/calfit/preflight.json
```

Exit 2 means the search contract is blocked. The receipt separates numeric
estimator correctness, gate policy, and rigid-disc geometry. Synthetic records
are isolated from trajectory feasibility; they are not model measurements.

At stack base `8a40737de5cb861e29535c91c554221c9ae4cbc4`, complete feasible
measurements always receive gate exit 5. No numeric tolerance policy or PASS
path exists. Published SDs are descriptive statistics, not acceptance tolerances.

V2 fixes shoulder width at 0.46 m. Its 0.9 ratio gives a 0.414 m aperture.
Any continuous rigid-disc trajectory through that aperture has at least
`radius - 0.207` m wall penetration: 0.043/0.073/0.093 m for radii
0.25/0.28/0.30 m. These are geometric lower bounds, not simulated outcomes.
Changing a wall force cannot resolve this conflict. Sampled integration can
miss contact; skipping across a wall would not establish physical feasibility.

## Search grid recorded before execution

This grid was recorded before execution. The author ruling below authorizes its
execution; acquisition receipts, rather than this contract, establish which points ran.

- All candidates: radius 0.25/0.28/0.30 m, persistent positive desired speeds
  from underlying N(1.29, 0.19) m/s; negative draws rejected, no upper clipping;
  independent integration cap 2.0/3.0 m/s. V6 retains source-controlled speeds.
- `legacy_v1` reference: factor 10, offset −0.57 m, sigma 0.
- `calibrated_v2` law refits: factor 0.0003/0.003/0.03, offset 0.25/0.375 m,
  sigma 0; includes exact profile reference 0.003/+0.375.
- `gradient_v3` law: factor 0.0003/0.001/0.003, offset equal to body radius,
  sigma 0; includes exact profile reference 0.001/body radius.
- Diagnostic body-edge exponential: amplitude 5/15/40, decay length
  0.02/0.04/0.08 m. All selectors remain opt-in.
- Coarse grid: (1 + 6 + 3 + 9) × 3 radii × 2 caps = 114 candidates.
  Screen seeds 1001–1003 with all V1–V6 variants and dense V4 safety; retain raw
  traces, censoring and every physical violation. No missing flow becomes zero.
- One refinement only: for up to four nondominated coarse survivors, factor
  multipliers 0.5/1/2 (amplitude for exponentials), radii `r−0.01/r/r+0.01`
  clipped to [0.25,0.30], same offset rule/decay length, cap and speed law.
  Deduplicate points and screen on the same seeds. Validate finalists on the
  full dev set 1001–1030. Stop after this round if none pass.
- Pareto objectives: minimize worst dense-V4 wall penetration and maximize
  complete-supply V3 narrow flow. Report each width separately; any censored
  width makes a candidate ineligible for a numeric flow ranking. Retain pair
  contacts and every other gate; Pareto survival is not acceptance.
- Slurm submission from the designated login host only; check squeue before each submission,
  hold CALFIT while the sealed 0.0.8 campaign is pending, at most 32 allocated
  CPUs total. Use at most two simulation workers per task and cap active
  allocations accordingly. Preserve commit/config/environment identity, raw
  arrays, scheduler receipts and checksums outside disposable output.

## Author ruling, 2026-10-02

Zero wall penetration remains mandatory. CALFIT uses three aperture widths from
`2r + .05` through `.966` m. The original 0.9 shoulder-width case is recorded as
"not reproducible with rigid discs (shoulder rotation)", a model limitation,
not a failed gate. No rotating or compressible body is implemented.

Every numeric check records its estimate, target, signed residual, tolerance
range, distance outside that range, source and
"author-delegated engineering tolerance, 2026-10-02". V1 uses 1.29 ± .19 m/s;
V3/V4 use ±20% flow; V2 uses the positive sign and ±50% of the source drop.
For intermediate V2 widths below ratio 1.3, the face-validity reference linearly
interpolates the .9/.40 and 1.3/.12–.16 source anchors, explicitly labelled as
an engineering interpolation. Physical violations are hard failures.
V5 and V6 use reported mean ±1 SD or a reported band; missing source spread
remains unspecified, with no invented SD. Huber Table 1 supplies head-on means
only, and the supplied approximate V5 mean supplies no verified spread.

`robot_sf/research/pedestrian_acceptance.py` contains the uniform numeric policy.
The suite now has exit 0 for PASS, 1 for numeric failure, 2 for censoring, 3 for
physical violations, 4 for invalid/empty acquisition, 5 for unspecified tolerance.
Population estimates require every attempted measurement; censored episodes
are retained. Full-bank admission additionally checks the exact seed/variant grid.

`python -m scripts.validation.calfit_search_10074 grid --out GRID.json` records
all 114 settings. `run --grid GRID.json --index INDEX --out ROOT` requires Slurm,
explicit dev seeds and a maximum of two workers. `summarize --root ROOT --out
SUMMARY.json` verifies raw hashes before computing the measured Pareto front;
`refine` emits the single bounded round. Desired speed and execution cap are
separate, and all settings remain opt-in.

The new ruling regressions failed on the previous head: target-valued V1
returned 5 instead of 0; missing measurement masked a physical violation (2
instead of 3); V2 contained seven infeasible protocol widths rather than three
feasible widths. The additional synthetic full-bank control demonstrates that
each gate can pass when test-only spreads are supplied; those spreads are not
literature evidence. Search controls prevent censored flow entering the Pareto
front and ensure the frozen 114 settings are distinct. No production test seam.

## Historical decision packet

Owner: author/domain reviewer. Define V1–V6 numeric tolerances, aggregation
(per seed or population), and approved handling of documented equivalents.
For V2, keep zero penetration and either add a body representation that can
rotate/compress, or explicitly scope calibration to a feasible engineering
aperture range. Relabeling that range requires a reviewed protocol decision;
the human 0.9-shoulder-width target stays unvalidated.

This preflight does not establish incompatibility of dense-wall safety and
V3 narrow flow at smaller radii. That question still requires the declared
search. Reopen CALFIT once the two preflight blockers are resolved; then implement
the grid runner, submit it, and produce the measured Pareto packet.

## Test value

The new tests protect observable synthetic estimators, policy/physics separation,
radius-dependent geometric infeasibility, and a marked CLI receipt bound to source
bytes. Credible regressions are returning censored values for these complete
traces, inferring admission from finiteness, silently changing the shoulder
proxy, or dropping the blocked verdict during output. Existing estimator tests
do not join all six cases with actual admission and geometry; the old gate test
uses only V1. No production test seam is introduced. This is added diagnostic
functionality, not a claimed model fix or a claimed failing-base regression.
