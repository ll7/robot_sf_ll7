# Environment-level metamorphic tests

This package checks whether the public `CrowdSimEnv` contract preserves the same
crowd dynamics under transformations that should not change the underlying
interaction. It is a deterministic test suite, not a benchmark and not evidence
for a planner, pedestrian model, metric, or scientific result.

The fixture uses three explicit pedestrians, a square synthetic map, fixed
`0.1 s` steps, a fixed seed, and no pedestrian-obstacle force. Every comparison
uses `rtol=0` and `atol=1e-5`: the absolute tolerance covers only float32
serialization and integration round-off, while zero relative tolerance keeps
small-magnitude frame or identity drift visible. A failure reports the first
divergent step and observation field together with the maximum absolute error.

The seven relations are:

| Test | Contract exercised |
| --- | --- |
| `test_scene_translation.py` | Translating scene coordinates translates positions and goals, while velocities and forces remain unchanged. |
| `test_scene_rotation.py` | A 90-degree rotation transforms positions, goals, velocities, and forces in the corresponding frame. |
| `test_row_permutation.py` | Declared pedestrian rows may be reordered without changing actor-associated state after identity matching. |
| `test_reset_isolation.py` | Both A→B and B→A reset orders reproduce fresh seeded episodes. |
| `test_render_independence.py` | Rendering at every observation boundary does not alter simulation state. |
| `test_record_replay_roundtrip.py` | The environment’s compact JSONL state recording replays without numeric drift. This is state-trace replay, not action re-simulation. |
| `test_oracle_isolation.py` | Opt-in privileged traces and randomized simulator identity labels do not change or enter actor-visible observations. |

The suite is collected by the repository’s normal `tests/` discovery and
`run_tests_parallel.sh`; it does not alter production code or benchmark metrics.
If a base implementation violates a relation, the failure should be filed as a
separate bug and linked to a strict `xfail` only after the defect is reproduced
and its expected scope is documented.

## Numeric tolerance versus exact representation identity

Two comparison boundaries coexist in this package, and they are not
interchangeable:

- **Numeric metamorphic tolerance** (`assert_trace_equal`, `rtol=0`,
  `atol=1e-5`): the default for the frame-transform, reset, render, replay, and
  oracle-visibility relations. The absolute tolerance covers only float32
  serialization and integration round-off; zero relative tolerance keeps
  small-magnitude drift visible. It proves the *values* of a relation hold.
- **Exact representation identity** (`assert_trace_byte_identical`): used by
  the randomized simulator-identity relation (`test_oracle_isolation.py`).
  Observation dtype, shape, and C-order byte sequence must match exactly, and
  the JSON-serializable info payload must serialize identically (sorted keys,
  compact separators, strict finite floats). Numeric closeness never
  substitutes for representation identity here, because `allclose` can hide a
  changed dtype, byte ordering, signed zero, or metadata representation while
  values remain within tolerance.

A relation may use both: the tolerant comparison reports the first numeric
divergence with the maximum absolute error, and the byte-identity comparison
reports the first representation divergence with dtype, shape, and the first
differing byte offset.
