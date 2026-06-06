# Issue #2459 SocNavBench / HuNavSim Mapping (2026-06-07)

Status: scoped interop/literature positioning, not simulator equivalence or benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2459
- SocNavBench arXiv: https://arxiv.org/abs/2103.00047 — "SocNavBench: A Grounded Simulation Testing
  Framework for Evaluating Social Navigation"
- HuNavSim arXiv: https://arxiv.org/abs/2305.01303 — "HuNavSim: A ROS 2 Human Navigation Simulator
  for Benchmarking Human-Aware Robot Navigation"
- SocNavBench vendored planner subset: `third_party/socnavbench/`
- SocNavBench metrics: `robot_sf/benchmark/metrics.py` (socnavbench_path_length,
  socnavbench_path_length_ratio, socnavbench_path_irregularity)
- SocNavBench planner adapters: `robot_sf/planner/socnav.py`
- SocNavBench observation mode: `robot_sf/sensor/socnav_observation.py`
- SocNavBench assets guide: `docs/socnav_assets_setup.md`
- SocNavBench re-entry gate: `docs/context/issue_562_socnav_bench_reentry.md`
- CARLA parity lane (kept separate): `docs/context/issue_2276_carla_parity_lane_decision.md`
- Structured mapping fixture: `docs/context/evidence/issue_2459/mapping_table_fixture.yaml`

## Result

This note maps Robot SF metric, scenario, planner, and observation concepts to their SocNavBench and
HuNavSim analogues. The purpose is **interop/literature positioning** — understanding what Robot SF
shares with these external social-navigation benchmark/simulation frameworks and where claims must
stop short of claiming simulator equivalence, metric parity, or scenario transfer.

### SocNavBench in the Repository

SocNavBench has substantial in-repo support:

- **Vendored planner subset**: `third_party/socnavbench/` provides 40+ modules (agents, planners,
  simulators, costs, objectives, metrics, trajectories, utils) from upstream commit
  `2724ea85ff22ca61a88ee95285dfc3fa056656c6` (MIT license).
- **Planner adapters**: `SocNavBenchSamplingAdapter` and `SocNavBenchComplexPolicy` in
  `robot_sf/planner/socnav.py` wrap vendored modules with Robot SF observation/command interfaces.
- **Three adopted metrics**: `socnavbench_path_length`, `socnavbench_path_length_ratio`,
  `socnavbench_path_irregularity` in `robot_sf/benchmark/metrics.py`.
- **Observation compatibility**: `socnav_struct` observation mode in
  `robot_sf/sensor/socnav_observation.py` follows SocNavBench structured observation convention.
- **Re-entry gate**: Full SocNavBench benchmark use is restricted by
  `docs/context/issue_562_socnav_bench_reentry.md` — the vendored planner is not treated as
  benchmark-ready.

### HuNavSim in the Repository

**HuNavSim has zero in-repo support.** No adapter, fixture, data file, config, metric, or
documentation reference exists anywhere in the codebase. The worktree branch name
`issue-2459-socnav-hunav-mapping` reflects the mapping topic only; no HuNavSim code has been
imported.

## Mapping Table

| Robot SF Concept | SocNavBench / HuNavSim Analogue | Missing Semantics | Claim Boundary |
|---|---|---|---|
| `socnavbench_path_length` | SocNavBench `path_length` (vendored) | Robot SF exposes a SocNavBench-derived metric name, but this note does not include fixture-level parity proof against upstream SocNavBench outputs. | Direct conceptual/API analogue in the vendored subset. Does not claim outcome generalisation to SocNavBench test scenarios or real-world settings. |
| `socnavbench_path_length_ratio` | SocNavBench `path_length_ratio` (vendored) | Same missing fixture-level parity proof caveat. | Same as `socnavbench_path_length`. |
| `socnavbench_path_irregularity` | SocNavBench `path_irregularity` (vendored) | Same missing fixture-level parity proof caveat. | Same as `socnavbench_path_length`. |
| `personal_space_cost` (vendored) | SocNavBench `personal_space_objective` (vendored) | Vendored objective is a source-level analogue; this note does not run objective parity fixtures. | Same as `socnavbench_path_length`. |
| `SocNavBenchSamplingAdapter` | SocNavBench sampling planner (vendored) | Re-entry gate restricts benchmark use. Adapter execution mode — not native SocNavBench runtime mode. | Adapter maps Robot SF observations. Does not claim planner behavior is identical to upstream SocNavBench runs. |
| `SocNavBenchComplexPolicy` | SocNavBench complex policy planner (vendored) | Re-entry gate restricts benchmark use. Adapter execution mode. | Same as `SocNavBenchSamplingAdapter`. |
| `socnav_struct` observation mode | SocNavBench structured observation (inspired by) | Robot SF adds RobotEnv-specific fields (ego_state, goal_relative, pedestrian_state, obstacle_clusters) not in upstream SocNavBench spec. | Observation shape/convention follow SocNavBench. Does not claim byte-identity with any upstream format. |
| SVG map + YAML manifest scenario definition | SocNavBench JSON config scenarios (analogous design) | SocNavBench references 3D mesh/point-cloud map datasets (ETH, S3DIS). Robot SF uses 2D SVG walkable-area maps. Pedestrian density, robot start/goal conventions differ. | Both use config-first scenario definition. Does not claim scenario semantics, difficulty calibration, or failure-mode coverage transfer. |
| Scenario certification (`scenario_cert.v1`) | No direct SocNavBench analogue | SocNavBench scenarios are assumed valid by construction from curated datasets. Robot SF certification is a geometric/kinodynamic feasibility gate with explicit failure classes. | Certification is Robot SF-specific. SocNavBench scenarios may fail Robot SF certification; this is not a defect in either framework. |
| SNQI composite quality metric | No direct SocNavBench analogue (individual cost terms exist but not weighted into a single score) | SNQI composite weighting, time-normalisation, and difficulty calibration are Robot SF-specific. | SNQI is Robot SF-specific. Cost-term deltas to SocNavBench metrics are not equivalent to a composite quality comparison. |
| Adversarial scenario search (`robot_sf/adversarial/`) | No direct SocNavBench analogue (scenarios are authored from real-world data) | Adversarial search over spawn/route/timing perturbation is Robot SF-specific. | Adversarial search is a Robot SF methodology extension. SocNavBench scenarios are not designed for adversarial perturbation. |
| Benchmark metrics (success, collision, near_miss) | HuNavSim benchmark metrics — source-level analogue from arXiv:2305.01303 | The HuNavSim paper describes a social-navigation metric suite, but Robot SF has no in-repo HuNavSim adapter, fixture, data, or config. Robot SF uses SnQI plus separate near-miss/collision metrics. | Both frameworks evaluate navigation outcomes in social scenarios. Does not claim metric equivalence, calibration parity, or that Robot SF metrics map to HuNavSim definitions. No HuNavSim benchmark rows exist. |
| Pedestrian behavior models (Social Force, ORCA) | HuNavSim human-navigation behavior models — source-level analogue from arXiv:2305.01303 | The HuNavSim paper reports a richer human-navigation behavior simulator, but Robot SF has no in-repo HuNavSim support or behavior-profile parity checks. | Both concern human/pedestrian behavior simulation. Does not claim model parity, behavioral diversity equivalence, or that Robot SF pedestrian dynamics reproduce HuNavSim benchmark outcomes. |

See `docs/context/evidence/issue_2459/mapping_table_fixture.yaml` for the structured YAML version.

## CARLA Parity Separation

CARLA replay parity (`docs/context/issue_2276_carla_parity_lane_decision.md`) is a separate concern
from SocNavBench/HuNavSim mapping. CARLA is an external **simulator backend** for Robot SF replay
parity (native/aligned/adapted replay modes, coordinate alignment, metric comparability). SocNavBench
and HuNavSim are **social navigation benchmark/simulation frameworks** — they define scenarios,
metrics, and pedestrian behavior models for evaluating robot social navigation.

Key separations:

- CARLA parity asks: "Does Robot SF replay produce comparable metrics in the CARLA simulator?"
- SocNavBench mapping asks: "Which Robot SF metric, planner, and scenario concepts share terminology
  or design with SocNavBench?"
- HuNavSim mapping asks: "Which Robot SF pedestrian modeling and benchmark concepts share design
  intentions with HuNavSim?"

These are fundamentally different questions with different evidence requirements. CARLA parity needs
live replay evidence and coordinate alignment validation (`docs/context/issue_2276_carla_parity_lane_decision.md`).
SocNavBench/HuNavSim mapping needs literature-positioning tables and concept-inventory evidence (this note).

## Claim Boundary

This is interop/literature positioning only. It does **not**:

- Claim simulator equivalence between Robot SF and SocNavBench or HuNavSim
- Claim that Robot SF metrics produce comparable values to either framework's metrics on shared
  scenarios
- Claim that Robot SF planner rankings generalise to SocNavBench scenario sets or HuNavSim benchmark
  rows
- Change any planner ranking, benchmark scenario set, or metric definition
- Import or adapt HuNavSim code, data, or configurations into Robot SF
- Remove the SocNavBench re-entry gate or relax its benchmark-use restrictions
- Establish any evidence tier stronger than `proposal` for interop claims
- Replace or overlap with CARLA replay parity work

A mapping table records a literature-positioning claim. Metric parity, scenario transfer, and
planner generalisation remain separate gates that require executable evidence before any stronger
claim is made.

## Validation

```bash
# Verify referenced paths exist
test -f third_party/socnavbench/__init__.py && echo "vendored SocNavBench OK"
test -f third_party/socnavbench/UPSTREAM.md && echo "upstream metadata OK"
test -f robot_sf/benchmark/metrics.py && echo "metrics OK"
test -f robot_sf/planner/socnav.py && echo "planner adapters OK"
test -f robot_sf/sensor/socnav_observation.py && echo "observation OK"
test -f docs/socnav_assets_setup.md && echo "assets guide OK"
test -f docs/context/issue_562_socnav_bench_reentry.md && echo "re-entry gate OK"
test -f docs/context/issue_2276_carla_parity_lane_decision.md && echo "CARLA parity OK"

# Validate YAML evidence fixture
uv run python -c "import yaml; yaml.safe_load(open('docs/context/evidence/issue_2459/mapping_table_fixture.yaml'))"

# Confirm zero HuNavSim references (expected)
test "$(grep -ri 'hunavsim\|HuNavSim\|hunav\|HuNav' --include='*.py' --include='*.md' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.cfg' --include='*.toml' --include='*.txt' docs/ robot_sf/ third_party/ configs/ scripts/ tests/ 2>/dev/null | grep -v -e 'docs/context/issue_2459' -e 'docs/context/evidence/issue_2459' -e 'docs/context/README.md' | grep -v '.git/' | wc -l)" -eq 0 && echo "No HuNavSim refs outside this note — confirmed"

# Consistency check
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Follow-Up Boundary

If SocNavBench re-entry or benchmark integration is later pursued, the SocNavBench mapping in this
note should be updated with executable evidence (planner adapter benchmark rows, observation-format
smoke, scenario-transfer diagnostics). If HuNavSim integration is later pursued, a new issue should
open that defines the adapter burden, data prerequisites, and claim boundary — this mapping note
would then provide the concept-inventory starting point. Both follow-ups are currently out of scope.
