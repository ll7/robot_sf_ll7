# Issue #1416 Converted-Map Cache Evidence

This bundle keeps a compact, reviewable summary of the profiler runs used to decide whether
generated converted-map disk cache support is worth adding after the capability catalog/resolver
stack.

The raw profiler outputs are disposable because the runs are cheap and reproducible from the
commands in `docs/context/issue_1416_converted_map_cache_evaluation.md`.

Tracked files:

- `summary.json`: cache profile counters, timing summaries, and the resulting decision.
