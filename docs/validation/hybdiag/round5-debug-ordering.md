<!-- AI-GENERATED/NEEDS-REVIEW -->
The review's wall/map-bound witness crashed `plan()` at fd11c6159 because one
continuous rejection omitted arc padding: otherwise equal constraint keys compared
`None` with `0.0`. The point/map-bound producer now emits `arc_padding_m=0.0`;
the other continuous producers already emit their actual padding. Debug sorting
explicitly maps absent threshold names and numbers to ordered sentinels.

The regression uses a 30x30 map, robot (0.30, 0.12), heading pi, body radius
0.25 m, and wall [[0.05, 0.0, 0.05, 0.42]]. It calls the real `plan()` with physical
exclusion and debug evaluation enabled. Its second case removes padding from
one actual zero-padding rejection to protect against future incomplete metadata.
Both cases fail at fd11c6159 with:

`TypeError: '<' not supported between instances of 'float' and 'NoneType'`

The existing exact debug-output expectation also now requires the producer's
explicit zero padding and fails on fd11c6159 because that field is missing.
Reproduce all three intended failures using the committed script:

```sh
python -m scripts.validation.prove_hybdiag_counterexamples --group round5 \
  --base-ref fd11c6159d5b929aa28b0a17beb71c1dc39c1000 --output /path/inside/lane/proof
```

Test-value answers (both parameters):

1. Protect non-crashing, truthful debug output when wall and map-bound rejections coexist.
2. A rejection producer drops optional padding, or tuple sorting returns without sentinels.
3. `test_debug_wall_exclusion_reports_applied_arc_padding` checks individual mappings;
   it never combines rejection keys through `plan()` at the map boundary.
4. No production seam. The second case wraps the real evaluator only in the test;
   physical evaluations and candidate generation remain real.

No shared fixture changed. Round 5 changes debug metadata/order and explanatory
wording, not candidate evaluation or action selection. Round 4 measurements remain
the behavior evidence; wall exclusion exposes the latent goal defect, and no arm
isolates the goal-validity pair alone.

Updated exact-output assertion: protects explicit zero padding; a producer
omitting the field fails it; the old expectation did not require padding; no
production seam is needed.
