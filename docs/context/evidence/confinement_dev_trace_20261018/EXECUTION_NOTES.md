# Execution notes

AI-GENERATED NEEDS-REVIEW

The authoritative preregistration is PLAN.md at commit
`ded06fc990b032cb4bf7fd2889a59aa234e8e53f` (2026-10-18 10:00:27 +02:00),
committed before the first episode. Its exact bytes are retained in the campaign inputs.
The current plan differs only by spelling its dissertation-repository citation as an
external pinned GitHub link so the repository-local path checker can resolve its scope.
No estimand, threshold, passage coordinate, seed, planner or budget changed.

The producer initializes previous_trace_ped_pos=None, and _trace_pedestrians emits
zero velocity for the first frame in that case. Analysis recognizes only that first-frame
zero sentinel, reports its discrepancy separately, and uses reset-to-frame displacement
for speed. Subsequent frames must agree with displacement/dt within 1e-6 m/s.
This explains a producer representation convention under the frozen velocity validation
rule; the raw trace is preserved unchanged. Velocities are finite differences, not an
independent measurement of integration velocity.

Four initial first-batch simulations could not be serialized because unchanged producer
aggregate metrics contained inf. No complete records were accepted; all partial bytes and
the launch receipt were preserved. A failing regression preceded the null-encoding fix,
which uses strict-JSON null for undefined aggregate metrics. The native writer instead permits
Infinity; the wrapper is not byte/schema-equivalent there. The only observed native-schema
exception is the unused optional metrics.min_separation_corrupted_m. A stored-record schema
allows null for that field; native-schema validation omits only that null optional field
in a validation projection and fails on every other exception. Raw records stay unchanged. Those four identities
were restarted with the same inputs. Final acquisition accounts for 2,700 accepted episodes
plus four failed storage attempts, never outcome-based extra sampling.

After measuring host headroom, the four-worker controller was interrupted alone. Its active
workers drained and all complete records were verified; four complete unjournaled records
were adopted with checksums. Missing identities alone resumed at the predeclared six-worker
cap. Original launch and resume receipts retain source/tool/input/environment identity.

Actor denominators refer to stable simulator slots within each episode. Respawn transitions
break crossing and stationarity segments, and a both-neck traversal must occur without an
intervening respawn. Counts never bridge missing actor observations or large discontinuities.
The hallway and cross-trap operational passages are not literal doorway claims.

The current tracked policies.json additionally repeats each original `path` as
`resolved_path` for the evidence-registry linter. The configuration paths/hashes/contents
are unchanged; the exact preregistration JSON at the plan commit and the frozen runtime
copy remain preserved. This fixes six hash_without_artifact_path findings without changing
the registry baseline or scientific inputs.
