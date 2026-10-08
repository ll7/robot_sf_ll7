# Pedestrian group allocation

`simulation_config.groups` is a target fraction of background pedestrians in
multi-member groups. `group_allocation_mode: legacy` is the default, including
when the key is omitted. It preserves the existing size law and seeded random
draw order. No scenario opts into the new law automatically.

Set `group_allocation_mode: exact_small_crowd_v1` and `groups: <fraction>` to
allocate a nearest-feasible count. For a spawning pool of size n, choose G
minimizing |G - n*f|; ties choose the larger G. With maximum M >= 3, feasible
counts are {0} union {2,...,n}. With M=2 only even counts are feasible. For n=0
or 1, G=0. M=1 permits f=0 only. Missing/nonfinite/out-of-range f is rejected.

| n | f=0 | f=0.5 | f=1 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 2 | 2 |
| 3 | 0 | 2 | 3 |
| 4 | 0 | 2 | 4 |

The table gives G for M=3. Split G greedily into groups of at most M. If the
next maximum-sized group would leave one pedestrian, reduce that group by one
and put the final two together. For M=2, evenness prevents this remainder.
Each step consumes at least two pedestrians, so the partition terminates,
contains no singleton among grouped pedestrians, and sums to G. The remaining
n-G pedestrians are singletons. Shuffle the sizes with the existing private
spawn RNG before spatial placement; replay at a fixed seed remains identical.

Every integer >=2 is representable using 2s and 3s (even: 2s; odd >=3: one 3
and 2s), proving the feasible set for M>=3. Nearest-feasible rounding gives
error <=0.5 between adjacent integers and <=1 around the gap from 0 to 2 or
between even counts. For n=1 the error can reach 1. G is deterministic, so
E[G]=G; the upward tie rule is deliberately biased at ties, not stochastic
unbiased rounding. Realised fractions are G/n for n>0; report n=0 as no crowd.
For n=2/3/4 at f=0.5 these are 1, 2/3, and 1/2.

Allocation is independent for route and crowded-zone spawning pools (including
synthetic background). Groups cannot span different behavior controllers.
Explicit authored single actors and the ego pedestrian are outside the target.
Rounding errors can therefore accumulate across pools; this is not a promise
of an exact whole-map fraction including authored actors.

Compatibility: the exact mode applies at every pool size; it does not switch
laws at a hidden threshold. At large n, G/n converges to f. It changes the
within-group size distribution to greedy partitions, so it is not the legacy
conditional size law. The legacy large-crowd law remains available unchanged.

Validation: `uv run pytest tests/sim/test_exact_group_allocation.py -q` proves
rounding against an independent reachable-sum oracle without episodes. Run
`uv run python scripts/validation/run_group_allocation_diagnostics.py --output
<diagnostics.json>` for dev seeds 1001-1030 through the production map-runner
reset path. These are development diagnostics, not benchmark performance claims.
