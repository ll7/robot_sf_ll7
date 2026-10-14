# Wall-law typical clearance and passage flow

Evidence status: **diagnostic-only**, native simulator execution, no fallback or
degraded rows. This is the same eight-person bidirectional corrected fixture as
#10310, not a matched human experiment or a saturated capacity measurement.

## Recommendation (two sentences)

The present evidence does not establish that physical_margin x10 is closer to
measured pedestrians: it improves numerical proximity in passage flow and rarely
comes within 10 cm of a wall, but the population, width, directionality, and
arrival conditions differ materially from the cited experiment.
Keep legacy as the 0.1.0 default pending a matched empirical comparison, and
retain physical_margin x10 as a candidate rather than breaking comparability
with released 0.0.x results on this diagnostic alone.

## Observed clearance

200 paired dev seeds, **1001–1200 only**, eight pedestrians per trial.
No seeds from 111–140 or 50036–59019 were executed.

| Law | Pedestrian-steps | Median clearance (m) | P10 clearance (m) | Steps below 0.10 m | Share below 0.10 m |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 840,048 | 3.796870 | 1.250571 | 0 | 0% |
| physical_margin x10 | 634,736 | 3.527739 | 0.900251 | 540 | 0.085075% |

Clearance is center-to-nearest-solid-polygon distance minus the circular body
radius, 0.40 m. Quantiles use linear interpolation over the pooled samples, not
an average of per-seed quantiles. Every pedestrian is sampled after every
integration step until all eight are within the runner's 1 m goal tolerance;
tick zero is excluded, and pedestrians already near their goal remain included
until the common termination, exactly as in the existing runner. The threshold
is strictly less than 0.10 m. Wider approach/corridor areas dominate the median.

Supplementary samples restricted to centers inside either opening's longitudinal
slab (x=18–22 or x=38–42 m) have median/P10 of 1.551872/1.250880 m for legacy
and 1.056516/0.303881 m for physical_margin x10; neither has a sub-10-cm sample
in those slabs. The 540 close samples occur outside the slabs, including wall
approaches, rather than in the openings themselves.

Both laws have zero initial/new/any wall overlaps, zero stalls, and 1600/1600
goals. Their mean episode lengths are 525.030 and 396.710 steps respectively
(dt=0.1 s). All 400 complete baseline trial dictionaries exactly match the
[corrected prior matrix](https://github.com/ll7/robot_sf_ll7/issues/10304#issuecomment-6098746155).
Historical buffered-footprint minimum clearances reproduce as 0.515472957 and
0.046436427 m; exact circular clearances are 0.515077135 and 0.045979182 m.
The sub-millimetre difference is the polygonal circle approximation, not changed
simulation behavior.

## Specific flow and literature comparison

First directed pedestrian-center crossings are interpolated linearly between
adjacent 0.1 s samples at x=20 m and x=40 m. Each line sees all eight people once
(four in each direction); both directions count positively. The physical
wall-to-wall widths are 4 m (y=13–17) and 5 m (y=12–17), respectively; neither
body radius nor a wall avoidance distance is subtracted from the denominator.
Do not sum the sequential openings into a unique-person flow.

To reproduce Seyfried et al.'s **Table 2** convention, each seed uses
`Js = N / ((t_last - t_first) * width)`, with N=8. Pooling independent trials
uses `sum(N) / (sum(t_last - t_first) * width)`; the intervals remain within
their respective trials. This is passage flow over a first-to-last window,
not flow inferred from goal completion or a claim about capacity.

| Opening | Law | Pooled flow (persons/m/s) | Per-seed P10–P90 (persons/m/s) |
| --- | --- | ---: | ---: |
| 4 m, x=20 m | legacy | 0.082249 | 0.082056–0.082795 |
| 4 m, x=20 m | physical_margin x10 | 0.096433 | 0.096380–0.096558 |
| 5 m, x=40 m | legacy | 0.046368 | 0.044011–0.057431 |
| 5 m, x=40 m | physical_margin x10 | 0.081117 | 0.078737–0.082350 |

Physical_margin x10 increases these flows by 17.25% and 74.94%, respectively.
The instrumentation also records the **Eq. (3)** reciprocal-mean-headway
estimator `(N-1)/((t_last-t_first)*width)`: legacy 0.071967/0.040572,
physical_margin x10 0.084379/0.070977 persons/m/s. With only eight pedestrians,
the N versus N−1 convention matters (factor 8/7); both are disclosed rather
than silently compared. The verified summary records both conventions.

[Seyfried et al., *New Insights into Pedestrian Flow Through Bottlenecks*,
Transportation Science 43(3), 395–406 (2009), DOI 10.1287/trsc.1090.0263](https://doi.org/10.1287/trsc.1090.0263)
([open author manuscript, Table 2 and §2–3](https://arxiv.org/html/physics/0702004))
reports 1.61, 1.86, 1.90, 1.93, and 1.97 persons/m/s for N=60 and widths
0.8, 0.9, 1.0, 1.1, and 1.2 m. The experiment used one-way purposeful normal
walking, a 2.8 m bottleneck, and initial density 3.3 persons/m².

Our flow is numerically closer under physical_margin x10, but still about
17–20 times below the 1.61 lower endpoint (legacy about 20–35 times below).
**These are not comparable widths or arrival conditions.** Eight widely spaced
walkers in two directions through 4–5 m openings are supply-limited; a long gap
between opposite-direction arrivals depresses the first-to-last estimate.
A mismatch with saturated flow cannot by itself be attributed to either wall
law. This source also supplies no target for our pooled all-step wall clearance,
so the smaller clearance distribution does not independently establish realism.

## Reproduction and provenance

- Repository: ll7/robot_sf_ll7; fresh clone, new branch
  `analysis/wall-law-typical-flow`, no merge.
- Base origin/main: `c26838b140d4e4073f7414c33496d781f4eea4a0`.
- Clean execution source: `ed243122b062f4ea5b5b36edc54e1fa4ebaa1235`.
- Scenario: `classic_realworld_double_bottleneck_high` in
  `configs/scenarios/issue_10304_wall_law_bottleneck_successor_v1.yaml`.
- Map: `maps/successor_svg_maps/issue_9762_classic_realworld_bottleneck_goal_zone_entry_v2.svg`,
  SHA256 `4de3404105d3cb5748992e0eeff2bf8478100db51eb47ee6938f0e1249fae6e4`.
- Laws: `legacy_shifted_gradient_v1`, and
  `body_edge_exponential_v3_physical_margin` with obstacle factor multiplied by
  10 (effective factor 100); force radius 0.35 m plus margin 0.05 m aligns to
  the 0.40 m body. Single walkers have authored desired speed 0.9 m/s.
- Python/NumPy global placement RNG and all pedestrian seed fields use the same
  episode seed. Robot action is stationary. Integration and stopping behavior
  are inherited unchanged from the corrected runner.
- The artifact contains input/source/lock SHA256s, all 400 per-seed records,
  all 6,400 crossing times (two lines), and raw-trace digests. Compact JSON is
  tracked evidence; raw traces remain outside repository `output/` and are
  reconstructable from the pinned source/command. Raw traces are not published
  as Git blobs, and no public result depends on a local-only retrieval pointer.

```bash
git checkout ed243122b062f4ea5b5b36edc54e1fa4ebaa1235
scripts/dev/bootstrap_worktree.sh --extra benchmark --extra viz --no-symlink-machine
nice -n 15 scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/wall_law_typical_flow.py --seeds 1001-1200 --workers 4 \
  --output-dir ../evidence
```

Use `summary[*].flow[*].pooled_specific_flow_persons_per_m_s * 8/7`
to convert the runner's headway convention into the reported Table 2 convention.
The committed verified summary also records the total crossing intervals and
the order-statistic neighbors used for the pooled clearance quantiles.

Validation performed:

- Full matrix command above: exit 0, 400/400 native diagnostic trials.
- Same command with `--seeds 1001-1005 --output-dir ../replay`: exit 0;
  all ten complete row dictionaries (including floats and raw NPZ hashes) equal
  their corresponding full-run rows exactly.
- Independently decoded prior matrix: every one of the 400 complete baseline
  dictionaries exactly matches its prior row.
- Independent checks over all 400 traces: XML rectangle-derived Euclidean body
  clearance agrees within 2e-14 m; crossing interpolation agrees within 2e-14 s;
  pooled sample counts, quantiles, and sub-10-cm counts recompute exactly.
- Analytical two-direction interpolation and known five-value percentile/
  strict-threshold probes: passed. The first probe used an inappropriate exact
  binary-float equality; it passed after using 1e-15 tolerance. An initial
  independent geometry probe assumed a y flip, failed, and was corrected to
  the verified loaded-map coordinates before the all-trace check passed.
- `scripts/dev/run_worktree_shared_venv.sh -- uv run ruff check scripts/validation/wall_law_typical_flow.py`:
  exit 0; corresponding `ruff format --check`: exit 0.
- No default-changing PR was opened; broad PR readiness/hosted CI were not run.

## Next decision and reopening condition

Maintainer owns the default ruling. Reopen promotion when a preregistered,
development-seed-only matched bottleneck experiment controls width, body size,
desired speed, density, directionality, and counting window and compares both
laws against the same measured targets; use #10074's V3/V4 setup rather than
reinterpret the present sparse fixture as capacity evidence. A future default
change must keep legacy selectable, document the effective parameters, update
the changelog, and explicitly break comparability with released 0.0.x results.

| Mechanism | Source | Tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Radius-aligned wall repulsion | #10310 | diagnostic-only | #10304 successor | 1001–1200 | paired JSON and verified summary | clearance distribution, passage flow | faster, closer numeric flow; no realism promotion | sparse bidirectional 4–5 m fixture |
