# Issue #2302 Benchmark Worker Scaling Evidence

This directory contains compact, sanitized evidence for the local worker-scaling continuation in
Issue #2302. The JSON summary records two repeats of the same h80 nominal-sanity candidate slice
at worker counts 1, 2, 4, and 6.

Raw per-run outputs were local diagnostic byproducts and are not durable evidence. The promoted
summary keeps the timing rows, medians, warnings, and diagnostic interpretation needed to decide
the next simulator-speed step.
