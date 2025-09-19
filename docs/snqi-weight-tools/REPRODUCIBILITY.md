# Reproducibility & Determinism for SNQI

This note summarizes what is deterministic today, known sources of nondeterminism, and how to mitigate them for repeatable results.

## Deterministic By Design
- Seeding: All scripts accept `--seed` which is applied to NumPy RNG and any seeded stochastic components (e.g., differential evolution initialization, sampling for Pareto/grid).
- Episode sampling: `--sample N` selects a deterministic subset under the chosen seed.
- Provenance: Outputs embed `_metadata` with `seed`, `git_commit`, timestamps, and invocation echo for traceability.

## Potential Nondeterminism
- SciPy internals: Differential evolution (or other SciPy routines) may invoke BLAS/threaded math with non-deterministic reduction order, leading to minor numeric drift.
- Parallel evaluation order (future): If we parallelize objective evaluations, ordering/numeric accumulation may vary slightly.
- Floating-point environment: Different CPUs/BLAS libraries can yield tiny differences (rounding, fma usage). Generally negligible but visible in strict diffs.

## Mitigations
- Pin seed and avoid parallel threads where possible. On Linux/macOS, set `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for stricter reproducibility.
- Prefer the unified CLI paths and keep dataset/baseline files constant.
- Use the same Python/package versions (managed via `uv`). Re-run `uv sync` to restore a locked environment.
- For benchmarking in CI: tolerate small numeric epsilons in assertions rather than exact bitwise equality.

## Validation Checklist
- Check `_metadata.seed` and `_metadata.git_commit` are present and match your run expectations.
- Verify `original_episode_count` vs `used_episode_count` when `--sample` is used.
- Compare summaries allowing small tolerances (e.g., atol=1e-9 for floats).

## References
- Detailed design discussion: `docs/dev/issues/snqi-recomputation/DESIGN.md#10-reproducibility--seeding`
- User guide overview: `docs/snqi-weight-tools/README.md#reproducibility--determinism`
