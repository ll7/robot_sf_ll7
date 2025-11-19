# Research Findings: Accelerate PPO Training with Expert Trajectories

## Decision 1: Sample-Efficiency Target for Pre-Trained PPO
- **Decision**: Define success as achieving benchmark performance in ≤70% of the environment interactions required by the current PPO-from-scratch baseline, with wall-clock completion under 18 hours on the reference workstation.
- **Rationale**: Literature on behavioural cloning warm starts (e.g., SB3 pretraining guides, recent imitation learning papers) reports 25–40% reduction in data needs; 30% savings aligns with the spec’s SC-003 and is operationally meaningful while remaining achievable with available compute.
- **Alternatives Considered**:
  - *50%+ reduction*: Unrealistic without extensive expert coverage and risk of over-promising.
  - *Wall-clock only target*: Ignores primary benefit (sample efficiency) and conflicts with Constitution Principle III metrics emphasis.

## Decision 2: Trajectory Archive Size & Retention Constraint
- **Decision**: Cap individual curated trajectory datasets at 25 GB (≈200 episodes with full observation/action history) stored under `output/benchmarks/expert_trajectories/`, with mandatory pruning/archive policy after three approved iterations.
- **Rationale**: Keeps artefacts manageable for CI sync and local storage, aligns with existing output governance, and still supports the required scenario coverage.
- **Alternatives Considered**:
  - *Unlimited storage*: Violates repository cleanliness and risks CI slowdown.
  - *Strict 5 GB cap*: Insufficient for multi-sensor observations and limits dataset fidelity.

## Decision 3: Metric & Reporting Commitments
- **Decision**: Every workflow run must emit JSONL episodes with success rate, collision rate, path efficiency, comfort exposure, and SNQI metrics, plus aggregated summaries (mean/median/p95) with 95% bootstrap confidence intervals.
- **Rationale**: Satisfies Constitution Principles III and VI, matches current benchmark schema, and provides stakeholders with comparable numbers against historical baselines.
- **Alternatives Considered**:
  - *Success/collision only*: Too narrow; weakens statistical insight.
  - *Full metric suite including experimental metrics*: Adds noise and complicates validation.

## Decision 4: Configuration Extension Approach
- **Decision**: Introduce additive fields to existing unified config objects (e.g., `RobotSimulationConfig`, training config dataclasses) for dataset paths, expert policy references, and trajectory capture parameters; avoid ad-hoc kwargs by surfacing these through config files under `configs/`.
- **Rationale**: Upholds Constitution Principle IV, keeps reproducibility central, and leverages existing config loading utilities.
- **Alternatives Considered**:
  - *Inline CLI flags only*: Risks drift between runs and breaks deterministic replay.
  - *Separate bespoke config schema*: Duplicates existing infrastructure and increases maintenance burden.
