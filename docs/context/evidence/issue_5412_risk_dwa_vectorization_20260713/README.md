# RiskDWA vectorization timing record

Status: diagnostic performance evidence for PR #5511 and issue #5412. This
record is not benchmark evidence and does not promote RiskDWA to a benchmark
planner.

## Measurement contract

- Code under test: synchronized PR head d021e612; the measured planner code
  is the phase-2 implementation from 3307d495.
- Comparator: the scalar reference _scalar_rollout_score in
  tests/planner/test_risk_dwa.py.
- Observation seed: NumPy default_rng(5412); six pedestrians with positions
  sampled from [-2.0, 2.0] meters and velocities sampled from [-0.5, 0.5]
  meters/second.
- Command grid: 77 commands; linear speed 0.0..1.2 meters/second in 0.2
  increments and angular speed -1.2..1.2 radians/second in 0.24 increments.
- Rollout horizon: 20 steps.
- Repetitions: 200 after one warm-up pass; each result is the median of three
  timed trials.
- Environment: Python 3.13.14 on Linux x86_64, no fallback or degraded planner
  path.

## Result

| Path | Median runtime |
| --- | ---: |
| Scalar reference | 5.494337 s |
| Vectorized RiskDWA rollout | 2.463305 s |
| Ratio | 2.230x |

The result supports a local diagnostic speedup on this fixed fixture only.
Hardware, NumPy version, observation composition, and command distribution can
change the result. It is not a campaign-level runtime claim.

## Reproduction

Use the fixed-seed fixture and scalar reference in
tests/planner/test_risk_dwa.py. Time 200 repetitions of the 77-command loop
for _scalar_rollout_score and then _rollout_score, after one warm-up pass,
and report the median of three trials. The parity and regression command is:

    uv run pytest tests/planner/test_risk_dwa.py tests/planner/test_risk_dwa_mppi_hybrid.py -q

Rollback/failure criterion: revert the vectorization if the fixed-fixture
parity tolerance exceeds 1e-9, the trajectory endpoint tolerance exceeds
1e-9, or two repeated measurements on the same host show the changed median
slower than the scalar median.
