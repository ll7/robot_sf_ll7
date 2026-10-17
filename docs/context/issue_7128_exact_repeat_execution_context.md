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

Failed native repeats also retain diagnostic evidence (#10222). Executed targets
with an unrunnable or process-isolation disposition keep `repeats: []` and a
separate `repeat_diagnostics` entry for every attempted repeat. Each entry has a
zero-based `repeat_index`, complete `algorithm_metadata`, and `worker_events`.
Events record `kind` (`crash`, `timeout`, `exception`, or normal `closed`), `phase`,
`exit_code`, `stderr_tail` (the final 8192 bytes, decoded with replacement), an
optional error, and the applied timeout in seconds. A timeout records the exit
code after termination, separately from its timeout kind. A live worker's caught
exception can have a null exit code; cleanup subsequently records its actual
exit code. Child Python tracebacks and native writes to stderr are captured.

The target cache, host result and verifier retain these diagnostics. They are
excluded from trajectory fingerprints and cannot turn a fallback into native
evidence. The required offline-PPO assertion prints the retained result on
failure, and the hosted proof runs in 20 fresh pytest processes. Existing
initialization, warmup, step timeout and retry budgets are unchanged; no new
retry is justified without a captured failure demonstrating a transient cause.
Runtime diagnostics can contain environment-specific paths and should be
reviewed before public publication.
