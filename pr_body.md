## Reviewer-critical summary

### What / Why
This PR implements the paired statistical report builder and decision-maker script for issue #3465. 
Specifically, it adds:
- `scripts/benchmark/build_issue3465_topology_gate_paired_decision.py` which aggregates campaign output metrics (from `campaign_summary.json`), calculates safety/efficiency deltas between enabled and disabled arms, validates fallback/degraded status exclusions, calls the near-parity promotion gate classifier, and writes evidence artifacts (`summary.json`, `paired_deltas.csv`, `README.md`) to the evidence packet directory.
- `tests/benchmark/test_build_issue3465_topology_gate_paired_decision.py` which verifies the decision-maker script under all logic branches (corrective incomplete, missing summaries, missing metrics, mock promotion/regression/fallback verdict paths, and CLI execution).

### Successor Statement
successor slice: Paired statistical report builder and decision-maker script; does not duplicate PR #4465.

### Validation Output
All 20 test cases in `tests/benchmark/` passed cleanly on CPU execution:
```
============================= 20 passed in 24.77s ==============================
```
Ruff styling, complexity, formatting, and exception checks are fully satisfied.

This PR was produced by the agy/Gemini-3.5-Flash cheap implementation lane; the review gate hardens and merges.
