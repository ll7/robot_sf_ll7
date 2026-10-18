<!-- AI-GENERATED/NEEDS-REVIEW -->
Reverse and applied-radius witnesses are synthetic geometry tests: no seeded
environment reset, test-only production seam, or oracle derived from the method
under test. Main was normally merged first. The pre-fix reference babd77658f7c7c85da34d249d108de89c9d3e618
has platform injection removed and wall stopping enabled by the static flag.

`test_physical_wall_stop_rejects_reverse_toward_wall_behind` (two cases):
1. Protects the rear swept disc through committed reverse motion and full braking,
   both already reversing and starting reverse from rest.
2. A forward-only speed floor or `speed <= 0` early exit would miss the rear wall.
3. The retained forward wall-stop test and arc test check forward commands;
   main's limited-reverse tests do not cover this newly opt-in wall braking tail.
4. No seam: bind real DifferentialDriveSettings and exact wall geometry, then
   call the production stopping predicate. Independently computed rear body gaps
   0.30/0.005 m are less than stopping sweeps 0.40/0.010 m at 1 m/s².

`test_debug_wall_exclusion_reports_applied_arc_padding`:
1. Protects the reported physical rejection radius and visible arc padding for
   rollout wall overlap and braking-tail wall rejection.
2. Reporting the body radius without the actual arc expansion loses the
   rejecting threshold and falsely describes grazing contacts.
3. Existing debug tests cover unexpanded radii; the grazing-arc test verifies
   geometry but did not verify debug serialization of its expansion.
4. No seam: the evaluator's existing debug mapper and serializer receive literal
   physical radius 0.25 plus arc bound 0.0013, independently expected as 0.2513 m.

The historical platform-only tests are removed with the removed experiment.
The forward stopping-distance test is retained under the physical flag, without
platform injection. Its geometry, assertion and horizon witness are unchanged.

Reproduce the three intended failures:
`python -m scripts.validation.prove_hybdiag_counterexamples --group round4 --base-ref babd77658f7c7c85da34d249d108de89c9d3e618 --output ../artifacts/round4-counterproof`
It requires two “reverse wall stopping sweep was skipped” assertion failures and
“reported exclusion radius omits arc padding”; an import/fixture failure cannot
satisfy the proof. `round4-counterproof.json` commits module and test byte hashes.
All evaluator witnesses pass on the fixed code. Historical Round 3 proofs use
`--group all --test-ref f16a53f5a924e0d50a749ea8f74e30fa4ac2db4a` so removal does
not make their previously accepted reproduction command depend on current tests.
