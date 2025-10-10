"""Full Classic Interaction Benchmark package.

Provides orchestration utilities, planning logic, aggregation, effect sizes,
precision evaluation, plotting, and video generation for the classic interaction
scenario matrix benchmark.

Modules (incrementally implemented):
- planning: load & expand scenarios
- orchestrator: episode execution, adaptive sampling loop
- aggregation: metrics aggregation + bootstrap + CI
- effects: effect size computation
- precision: statistical sufficiency evaluation
- plots: plot generation
- videos: video generation / annotation
- io_utils: persistence helpers (episodes append, manifest write)

Public entrypoint (eventually): run_full_benchmark(BenchmarkConfig) -> BenchmarkManifest
"""

from __future__ import annotations

# Re-export (added progressively)
__all__: list[str] = []
