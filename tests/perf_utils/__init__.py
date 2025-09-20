"""Performance test utilities package.

Feature: 124 (Accelerate Slow Benchmark Tests)

Modules expected:
* policy: PerformanceBudgetPolicy dataclass and classify helper
* guidance: Heuristics producing optimization suggestions text
* reporting: Aggregation + top-N slow test selection
* minimal_matrix: Shared minimal scenario matrix builder for benchmark tests

These helpers decouple performance instrumentation from individual tests,
keeping benchmark files focused on semantic assertions.
"""
