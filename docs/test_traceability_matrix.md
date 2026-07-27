# Test Traceability Matrix – Critical Contracts

[← Back to Documentation Index](./README.md) | [← Developer Guide](./dev_guide.md)

This analysis-only inventory maps 14 high-risk shared contracts to the source files and existing
tests that were directly inspected. It is a navigation aid for contributors: a listed test is
evidence of the stated, bounded contract only, not proof of complete coverage or a benchmark result.

## How to read the evidence

Each row records a production owner, direct test or validation path, and a runnable local command.
The evidence classes describe the kind of check, not the strength of a research result:

- **Nominal**: ordinary unit, integration, schema, or contract protection.
- **Diagnostic**: a guard or probe that helps detect a problem but does not establish a benchmark
  result.
- **Smoke**: a short operational check of a rendering or system surface.

No row is classified as `paper-grade`. The matrix has no run configuration, artifact provenance,
sample size, or domain-aware approval, so its tests cannot establish a fully reproducible,
manuscript-facing claim. Evidence status is consequently limited to existing test coverage, not
benchmark evidence. A gap is recorded only when the inspected owner and named direct tests establish
it; this inventory does not infer gaps from filenames or create speculative follow-up work.

## Risk-based traceability matrix

| ID | Contract name | Scope category | Production owner path | Existing direct test / validation path(s) | Test level | Canonical command / CI lane | Evidence class | Evidence status | Known omission / gap | Proposed follow-up |
|---|---|---|---|---|---|---|---|---|---|---|
| **ENV-01** | Gymnasium environment construction and reset signatures | Environment construction | `robot_sf/gym_env/environment_factory.py` | `tests/test_gymnasium_env_contracts.py`, `tests/test_environment_factory_signatures.py` | Integration / contract | `uv run pytest tests/test_gymnasium_env_contracts.py tests/test_environment_factory_signatures.py` | Nominal | Existing test coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **ENV-02** | Step termination and truncation flags | Environment semantics | `robot_sf/gym_env/robot_env.py` | `tests/test_gymnasium_env_contracts.py`, `tests/test_crowd_sim_env_contract.py` | Integration / unit | `uv run pytest tests/test_gymnasium_env_contracts.py tests/test_crowd_sim_env_contract.py` | Nominal | Existing test coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **SIM-01** | Fast PySF force computation and wrapper queries | Simulation stepping | `robot_sf/sim/fast_pysf_wrapper.py`, `fast-pysf/pysocialforce/forces.py` | `fast-pysf/tests/test_forces.py`, `tests/test_fast_pysf_wrapper.py` | C++ integration | `uv run pytest fast-pysf/tests/test_forces.py tests/test_fast_pysf_wrapper.py` | Nominal | Existing test coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **SIM-02** | Kinematic collision clearance and distance checks | Collision semantics | `robot_sf/gym_env/robot_env.py` | `tests/test_collision_sanity.py`, `tests/test_robot_state_robot_collision.py` | Kinematic / unit | `uv run pytest tests/test_collision_sanity.py tests/test_robot_state_robot_collision.py` | Nominal | Existing test coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **NAV-01** | Baseline planner protocol compliance | Planner interfaces | `robot_sf/baselines/interface.py` | `tests/unit/test_planner_interface.py` | Unit / protocol contract | `uv run pytest tests/unit/test_planner_interface.py` | Nominal | Existing test coverage; not benchmark evidence | No high-risk gap established by this inventory. The classic grid-helper adapter is outside this protocol row. | None proposed. |
| **CFG-01** | Environment configuration schema validation | Schema validation | `robot_sf/gym_env/config_validation.py` | `tests/test_config_validation.py`, `tests/sim_config_test.py` | Unit / schema | `uv run pytest tests/test_config_validation.py tests/sim_config_test.py` | Nominal | Existing test coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **CFG-02** | Scenario schema and SVG map parsing | Serialization | `robot_sf/benchmark/scenario_schema.py`, `robot_sf/nav/svg_map_parser.py` | `tests/test_scenario_schema.py`, `tests/test_svg_classic_maps_format.py` | Unit / schema | `uv run pytest tests/test_scenario_schema.py tests/test_svg_classic_maps_format.py` | Nominal | Existing test coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **DOC-01** | Documentation and evidence-integrity guard | Provenance / tooling | `scripts/dev/check_docs_evidence_integrity.py` | `tests/dev/test_check_docs_evidence_integrity.py` | System guard | `uv run python scripts/dev/check_docs_evidence_integrity.py --files docs/test_traceability_matrix.md docs/dev_guide.md docs/README.md` | Diagnostic | Existing validation coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **MET-01** | Benchmark metric aggregation | Metric aggregation | `robot_sf/benchmark/metrics.py`, `robot_sf/benchmark/aggregate.py` | `tests/test_metrics.py`, `tests/test_aggregate.py` | Metric contract | `uv run pytest tests/test_metrics.py tests/test_aggregate.py` | Nominal | Existing test coverage; not paper-grade or benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **MET-02** | Fail-closed disposition for degraded execution | Benchmark disposition | `robot_sf/benchmark/fallback_policy.py` | `tests/benchmark/test_fallback_policy.py` | Policy contract | `uv run pytest tests/benchmark/test_fallback_policy.py` | Nominal | Existing test coverage; not paper-grade or benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **CLI-01** | CLI environment discovery and doctor commands | Process boundaries | `robot_sf/cli.py` | `tests/test_cli_envs.py`, `tests/test_cli_doctor.py` | System / CLI | `uv run pytest tests/test_cli_envs.py tests/test_cli_doctor.py` | Nominal | Existing test coverage; not benchmark evidence | The listed tests do not evidence evaluation-pipeline or multi-worker exit behavior; those behaviors are outside this row. | None proposed; this inventory did not verify an actionable high-risk gap. |
| **DET-01** | Python, NumPy, and optional PyTorch seed initialization | Reproducibility | `robot_sf/common/seed.py` | `tests/test_seed_utils.py` | Unit / determinism | `uv run pytest tests/test_seed_utils.py` | Diagnostic | Existing test coverage; not benchmark evidence | The owner does not seed a C++ Social Force generator. Near-miss repeatability is covered by a separate subsystem and is outside this row. | None proposed. |
| **SMK-01** | Headless visual rendering and Pygame surface smoke | Visual smoke surface | `robot_sf/render/sim_view.py` | `tests/test_pygame_headless.py`, `tests/test_render_error_message.py` | Visual smoke | `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/test_pygame_headless.py` | Smoke | Existing smoke coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |
| **SMK-02** | Promoted-planner smoke runner | System smoke surface | `scripts/validation/run_pr_promoted_planner_smoke.py` | `tests/validation/test_run_pr_promoted_planner_smoke.py` | System smoke | `uv run pytest tests/validation/test_run_pr_promoted_planner_smoke.py` | Smoke | Existing smoke coverage; not benchmark evidence | No high-risk gap established by this inventory. | None proposed. |

## Maintenance boundary

When a contract owner or direct test changes, update the affected row and rerun its command. Add a
follow-up only after verifying an actionable high-risk omission against the owner and direct test;
link the resulting issue in the row. If a proposed contract cannot be tied to a current owner and
executable validation path, keep it out of this matrix and record it separately as an unverified
candidate rather than treating it as covered.

Validate documentation changes with:

```bash
uv run python scripts/dev/check_docs_evidence_integrity.py \
  --files docs/test_traceability_matrix.md docs/dev_guide.md docs/README.md
git diff --check
```
