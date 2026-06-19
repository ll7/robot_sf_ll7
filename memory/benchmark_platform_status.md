---
name: benchmark-platform-status
description: Social Navigation Benchmark Platform operational status, readiness, and known limitations
metadata:
  type: project
  created: 2026-06-19
  category: benchmarking
---

# Social Navigation Benchmark Platform Status

**Status**: Fully operational  
**Last Updated**: 2026-06-19  
**Evidence Grade**: Nominal benchmark evidence (108 tests passing)  
**Related**: [Architecture Overview](architecture_overview.md)

## Platform Overview

The Social Navigation Benchmark Platform is a complete, production-ready system for evaluating robot
navigation planners in pedestrian-filled environments.

| Component | Status | Notes |
| --- | --- | --- |
| Episode Runner | ✅ Complete | Parallel execution, resume, deterministic seeding |
| Metrics Suite | ✅ Complete | SNQI composite + component breakdown |
| Baseline Planners | ✅ Complete | SocialForce, PPO, Random with unified interface |
| Statistical Analysis | ✅ Complete | Bootstrap CI, robust aggregation |
| Figure Orchestrator | ✅ Complete | Distribution plots, Pareto frontiers, force fields |
| CLI Tools | ✅ Complete | 15 subcommands covering full workflow |
| Documentation | ✅ Complete | Quickstart, CLI reference, example workflows |

## Operational Capabilities

### Ready-to-Use Workflows

1. **Quick Assessment** (~15 min)
   - Compare 2-3 robot policies against baselines
   - Output: Metric summary table, success/failure breakdown

2. **Research Study** (~2-4 hours)
   - Multi-parameter sweep (planner family, map set, seed range)
   - Output: SNQI Pareto frontier, figure suite, statistical summary

3. **Weight Sensitivity** (~45 min)
   - Analyze SNQI component importance
   - Output: Weight contribution breakdown, robustness report

### Key Metrics

**SNQI (Social Navigation Quality Index)**
- Composite index (0-100) combining 4 components:
  - Success rate (did robot reach goal?)
  - Collision avoidance (keep distance from pedestrians)
  - Lateral displacement (not excessive deviation from optimal path)
  - Time efficiency (reach goal in reasonable time)
- Weighting: Configurable; default weights in `configs/benchmarks/`
- Evidence: Published; supports sensitivity analysis

### Baseline Planners

| Planner | Type | Training | Status |
| --- | --- | --- | --- |
| SocialForce | Physics-based | N/A (deterministic) | ✅ Baseline |
| PPO | Learned RL | Bundled in `model/` | ✅ Reference model |
| Random | Fallback | N/A | ✅ Sanity check |

## Known Limitations & Constraints

### Route Clearance Certification

Route clearance is certified via `configs/benchmarks/route_clearance_certifications_v1.yaml`.
- Not all maps pass; certified routes listed in config
- When route is not certified, episode execution falls back to fallback/degraded mode
- **Always report fallback/degraded status in results**
- Reference: `docs/context/issue_1105_route_clearance_certification.md`

### Benchmark Fallback Policy

When benchmark execution cannot complete under ideal conditions:
1. Execution may degrade (fallback to simplified metrics)
2. **Never relabel degraded results as nominal evidence**
3. Mark results `not benchmark evidence` or `diagnostic-only`
4. Document fallback reason and degradation mode
- Reference: `docs/context/issue_691_benchmark_fallback_policy.md`

### Pedestrian Dynamics

- Simulation uses `pysocialforce` (physics-based, not ML-based)
- Crowd size typically 5-50 pedestrians (configurable)
- Deterministic given seed; reproducible across runs

### Output Artifacts

- All benchmark outputs live under git-ignored `output/` (machine-local only)
- Small durable evidence promoted to `docs/context/evidence/` with commit hash link
- Figures: PNG/PDF export via figure orchestrator
- Raw data: JSONL format (episode logs, metrics)

## Development Readiness

### Testing Coverage

- 108 tests passing (CI: `scripts/dev/run_tests_parallel.sh`)
- Covers factory API, planner interface, metric calculation, CLI commands
- Test failure classification: See `.github/copilot-instructions.md` for evaluation protocol

### Performance Budget

- Full test run: ~5-10 min (16 parallel workers)
- Single episode: ~2-10 sec (sim only, no rendering)
- Benchmark sweep (100 episodes): ~2-5 min
- Training PPO model: 1-4 hours on RTX 3070

### CI/CD Status

- Automated: Pre-commit checks (ruff format, mypy)
- Automated: Test suite via GitHub Actions
- Manual: Benchmark validation (requires paper-grade evidence)

## Research Directions (Blocked/Exploratory)

### Candidate Planners

Status: **Blocked** pending architectural decision  
- Hybrid (rule-based + learning): Design phase
- Graph-based (visibility graph, RRT): Integration pending
- Learning-based variants (DQN, SAC): Reference implementation only

**Revival condition**: blocker removal through a current planner roadmap or ADR.

### Scenario Generation

Status: **Exploratory**  
- Adversarial scenario search: in progress across issue-specific context notes.
- Map synthesis: Diagnostic-only results

**How to apply**: Open as exploratory issue with clear status label; don't treat as benchmark evidence
until production validation complete.

## Artifact Provenance & Validation

For any benchmark claim (whether in issue, PR, or paper):
1. **Claim boundary**: What exactly is being claimed? (e.g., "PPO outperforms SocialForce by 5 SNQI")
2. **Evidence grade**: `diagnostic-only` / `smoke` / `nominal` / `paper-grade`
3. **Caveats**: Route clearance status, fallback exclusions, seed range
4. **Reproduction path**: Exact config file, seed range, command to run
5. **Uncertainty**: If confidence < 95%, include numeric bound or condition

**Example**:
> Nominal benchmark evidence (108 tests passing): PPO achieves 85±3 SNQI vs. 72±4 SocialForce
> on certified H500 maps (Route Clearance v1) with 5-run bootstrap CI. Config:
> `configs/benchmarks/h500_ppo_baseline.yaml`. Non-certified maps excluded per fallback policy
> (`docs/context/issue_691_*`). Reproduction: `uv run python -m robot_sf.benchmark run <config>`

## Reference & Documentation

- **Quickstart**: `specs/120-social-navigation-benchmark-plan/quickstart.md`
- **CLI Reference**: `docs/dev/issues/social-navigation-benchmark/README.md`
- **Zenodo artifact**: DOI 10.5281/zenodo.19563812 (public release)
- **Architecture notes**: `docs/README.md` → Benchmarking & Metrics
- **Validator**: `docs/code_review.md` (benchmark-facing review criteria)

## Next Steps

1. Review route clearance certification config: `configs/benchmarks/route_clearance_certifications_v1.yaml`
2. Run quickstart workflow: `specs/120-social-navigation-benchmark-plan/quickstart.md`
3. For new benchmark claims, follow artifact provenance checklist above
4. For research directions, file exploratory issues with clear status labels
