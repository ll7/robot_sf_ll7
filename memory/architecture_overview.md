---
name: architecture-overview
description: Core Robot SF architecture layers, module organization, and factory pattern for gymnasium environments
metadata:
  type: reference
  created: 2026-06-19
---

# Robot SF Architecture Overview

**Status**: Stable reference
**Last Updated**: 2026-06-19
**Related**: [Benchmark Platform Status](benchmark_platform_status.md)

## Core Layers

### 1. Environment Factory (Gymnasium Integration)
- Entry point: `robot_sf.spec_factory` (factory-based environments)
- Design pattern: Reusable Gymnasium entry points for robot and pedestrian simulations
- Benefit: Decouples environment creation from simulation logic; enables reproducible benchmarking
- Reference: `examples/quickstart/01_basic_robot.py`

### 2. Simulation Engine
- Pedestrian dynamics: `pysocialforce` integration (physics-based crowd simulation)
- Robot navigation: Planner interface (`PlannerProtocol`)
- Episode runner: Parallel execution with deterministic seeding and resume capability
- Reference: `robot_sf/sim/`, `robot_sf/gym_env/`

### 3. Planner Interface
- Protocol: `PlannerProtocol` defines reset/act/done contracts
- Implementations: SocialForce (baseline), PPO (learned), Random (fallback)
- Status: All baseline planners functional; extensible for new families
- Reference: `robot_sf/planner/`

### 4. Benchmark Suite (Social Navigation Benchmark Platform)
- Metrics: SNQI composite index with weighted components
- Runners: Episode execution with observation/action/metric collection
- Tools: CLI with 15 subcommands, figure orchestration, statistical analysis
- Status: **Fully operational** (108 tests passing)
- Reference: `robot_sf.benchmark`, `specs/120-social-navigation-benchmark-plan/`

### 5. Utilities & Tooling
- Map handling: SVG loading, collision detection, pathfinding
- Visualization: Pygame rendering, figure export (distribution plots, Pareto frontiers)
- Analysis: Bootstrap confidence intervals, metric aggregation
- Reference: `utilities/`, `scripts/dev/`

## Module Organization

```
robot_sf/
├── spec_factory.py           # Gymnasium entry points
├── gym_env/                  # Gymnasium environment bindings
├── sim/                      # Simulator and backend glue
├── planner/                  # Planner implementations and adapters
├── nav/                      # Map parsing and path planning
├── benchmark/                # Social Navigation Benchmark Platform
├── metrics/                  # SNQI and component calculations
└── render/                   # Playback and visualization support

tests/                         # Test suite (run with run_tests_parallel.sh)
examples/quickstart/           # 3-step onboarding (basic, trained, custom map)
specs/120-social-navigation-benchmark-plan/  # Benchmark specification
configs/benchmarks/            # Route clearance certifications, benchmark configs
```

## Key Design Decisions

### Factory Pattern
- **Why**: Reproducible environment creation, decoupled from CLI/training code
- **Trade-off**: Slightly more indirection vs. hardcoded args
- **Status**: Stable; migration complete for all public APIs

### Gymnasium Compliance
- **Why**: Integrate with broader RL ecosystem (stable-baselines3, Ray RLlib)
- **Trade-off**: Constrains env step API (observation, reward, terminated, truncated)
- **Impact**: Enables training without Robot SF-specific wrappers

### Planner Protocol
- **Why**: Pluggable planner family (don't hardcode SocialForce or PPO)
- **Interface**: `reset(obs) -> state`, `act(state) -> action`, `done() -> bool`
- **Benefit**: New planners (e.g., learned, hybrid) plug in without simulator changes

### SNQI Metric
- **Components**: Success rate, collision avoidance, lateral displacement, time efficiency
- **Weighting**: Configurable (default weights in `configs/benchmarks/`)
- **Status**: Published; supports weight-sensitivity analysis
- **Reference**: [Benchmark Platform Status](benchmark_platform_status.md)

## Artifact Provenance

- **Generated output**: All artifacts (JSONL, figures, videos) must live under git-ignored `output/`
- **Durable evidence**: Small, reviewable copies promoted to `docs/context/evidence/` with commit hash link
- **Models**: Trained weights under `model/`, versioned by seed/config
- **Configs**: All benchmark runs use predeclared configs under `configs/benchmarks/`

## Performance Characteristics

- **Episode runtime**: ~2-10 sec per 1000-step episode (simulation only, no rendering)
- **Parallel testing**: 16 workers on local machine (see `local.machine.md`)
- **Benchmark suite**: Full 108-test run ~5-10 min on dev hardware
- **Training**: PPO convergence typically 1-4 hours on RTX 3070

## Next Steps for Contributors

1. Read `docs/dev_guide.md` for contributor workflow
2. Run `examples/quickstart/01_basic_robot.py` to understand environment API
3. Check `robot_sf/schemas/` for data contracts (observe, action, metric)
4. Consult `docs/README.md` for architecture deeper dives (ADRs, design rationale)
