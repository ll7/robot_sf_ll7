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

This is a proposed grid, **not an executed experiment**. Launch remains blocked
until a numeric policy and a feasible treatment of V2 are specified.

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

## Decision and revival

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
