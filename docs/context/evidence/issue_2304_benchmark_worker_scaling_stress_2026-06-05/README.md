# Issue #2304 Stress-Slice Worker Scaling Evidence

This directory contains compact, sanitized evidence for the Issue #2304 worker-scaling follow-up.
The profile repeated the policy-search `stress_slice` stage twice for
`hybrid_rule_v3_fast_progress` at worker counts 1, 2, 4, and 6.

Raw per-run JSONL and generated policy-search reports were local diagnostic byproducts. The
promoted JSON summary keeps the repeat rows, medians, warnings, and interpretation needed for the
local worker-count recommendation.
