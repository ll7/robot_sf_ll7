# Issue #7128 exact-repeat execution context

The exact-repeat host report now carries a canonical
`benchmark_execution_context.v1` block and its SHA-256 digest. The block is
dependency-free to collect and binds CPU model, platform, Python, NumPy, Numba,
CPU-only/single-worker mode, and the numerical thread variables
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS`.

The `cpu_only` and `workers` fields are recorded only by callers that enforce
or observe the execution mode. The exact-repeat path does, and its host-report
verification requires CPU-only single-worker execution. The generic benchmark
result-provenance path does not, so it omits both fields rather than restating
an unobserved mode; the real worker count stays in the campaign run metadata.

Host identity is separate from the scientific equivalence rule. Reports retain
the raw machine identifier for local verification, plus a digest and public-safe
label. The context digest excludes that identity, so distinct hosts can be
compared when their numerical contexts match.

Cross-host comparison has three admitted machine-readable states:

- `exact_context_match`: all canonical context fields and source/lock identities match;
- `approved_numpy_numba_near_miss`: only NumPy and/or Numba versions differ;
- `incompatible_context`: CPU, platform, Python, thread, worker-mode, source, or lock identity differs.

Missing, malformed, unsupported, or digest-drifted context is rejected during
host verification. No legacy report is upgraded implicitly. The repair changes
provenance and verdict validity only; it does not establish determinism, rerun
the #5498 matrix, or make a benchmark, dissertation, safety, or sim-to-real
claim.
