# Issue #2016 Webots and Gazebo AMV Prototype Parity Audit (2026-06-01)

Date: 2026-06-01

Related issue:

- <https://github.com/ll7/robot_sf_ll7/issues/2016>

Related Robot SF context:

- [issue_2013_backend_adapter_contract.md](issue_2013_backend_adapter_contract.md)
- [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md)
- [issue_1075_operating_envelope.md](issue_1075_operating_envelope.md)
- [issue_1542_manuscript_claim_evidence_map.md](issue_1542_manuscript_claim_evidence_map.md)
- [issue_2001_amv_actuation_proxy_source_analysis.md](issue_2001_amv_actuation_proxy_source_analysis.md)

## Decision

Recommendation: `no_integration_for_now`.

Webots and Gazebo are reasonable robotics-simulator stacks to monitor for a future AMV prototype,
ROS 2 controller, sensor, or hardware-in-the-loop workflow, but neither should be integrated into
Robot SF now. There is no current concrete ROS 2 or physical AMV target that would justify adding
world files, robot assets, controller bridges, simulator runtime dependencies, or parity adapters.

If a future target exists, the next step should be a narrow prototype spike, not a general backend
integration. That spike should replay one Robot SF command trace through one declared robot model
and emit diagnostic output tied to `simulation_trace_export.v1` or a named derived schema.

## Evidence Sources

Robot SF already defines the evidence boundary for alternate backends:

- `docs/context/issue_2013_backend_adapter_contract.md` requires declared scenario input, command
  input, trace output, supported metrics, unsupported semantics, status classification, and
  fail-closed behavior for alternate simulator adapters.
- `docs/context/issue_1689_simulation_trace_export_schema.md` defines
  `simulation_trace_export.v1` as analysis and visualization input only, not benchmark evidence.
- `docs/context/issue_1075_operating_envelope.md` states that Robot SF evidence does not currently
  support physical AMV deployment readiness, external-simulator parity, ROS parity, or hardware
  claims.
- `docs/context/issue_1542_manuscript_claim_evidence_map.md` and
  `docs/context/issue_2001_amv_actuation_proxy_source_analysis.md` keep calibrated or paper-facing
  AMV actuation claims blocked until direct provenance exists.

External documentation checked:

- Webots ROS 2 tutorial, "Setting up a robot simulation (Basic)":
  <https://docs.ros.org/en/ros2_documentation/rolling/Tutorials/Advanced/Simulators/Webots/Setting-Up-Simulation-Webots-Basic.html>
- Webots ROS 2 tutorial, "The Ros2Supervisor Node":
  <https://docs.ros.org/en/rolling/Tutorials/Advanced/Simulators/Webots/Simulation-Supervisor.html>
- Gazebo Harmonic, "ROS 2 integration overview":
  <https://gazebosim.org/docs/harmonic/ros2_overview/>
- Gazebo Harmonic, "Launch Gazebo from ROS 2":
  <https://gazebosim.org/docs/harmonic/ros2_launch_gazebo/>
- Gazebo latest, "SDF worlds":
  <https://gazebosim.org/docs/latest/sdf_worlds/>
- Gazebo Fortress, "Gazebo Classic Migration":
  <https://gazebosim.org/docs/fortress/gazebo_classic_migration/>

## Comparison

| Criterion | Webots | Gazebo / modern Gazebo | Robot SF implication |
| --- | --- | --- | --- |
| AMV-like robot model | Plausible through Webots `Robot`/PROTO/world modeling and a driver or external controller. | Plausible through SDF/URDF-style models, plugins, and ROS/Gazebo bridges. | Both can probably host a differential-drive or bicycle-style prototype, but Robot SF would still need a declared command-to-actuation mapping. |
| ROS 2 parity | Official ROS 2 tutorials show Webots launch, driver, and supervisor patterns. | Official Gazebo docs focus on ROS 2 launch and `ros_gz` integration. | Either stack becomes useful only when a ROS 2 controller, sensor, or hardware-facing target is in scope. |
| Trace replay | A Webots prototype could replay commands through a controller and export sampled states. | A Gazebo prototype could replay commands through a plugin/bridge and export sampled states. | Neither provides Robot SF parity by default; both must satisfy the backend adapter contract and fail closed on unsupported semantics. |
| Asset/model burden | Requires `.wbt` worlds, PROTO/model choices, controller code, and explicit asset provenance. | Requires SDF/world/model/plugin choices, Gazebo version selection, and bridge/plugin provenance. | Both add maintenance surface well beyond an analysis-only issue. |
| Runtime cost | Adds simulator runtime and ROS 2/Webots package expectations outside current routine setup. | Adds Gazebo/ROS 2 packages, version compatibility decisions, and likely heavier runtime setup. | Do not add either to routine CI or local setup without a separate dependency decision. |
| Benchmark claim risk | High if a Webots demo is described as AMV validity or simulator parity. | High if a Gazebo demo is described as AMV validity, ROS parity, or hardware readiness. | Outputs remain diagnostic unless a later benchmark/parity proof surface is explicitly scoped and executed. |

## Conditions For A Future Spike

Open a separate spike only if at least one of these prerequisites exists:

- a named physical AMV or controller stack that needs ROS 2 parity;
- a specific sensor/controller integration question that Robot SF cannot answer locally;
- a command-trace diagnostic need where Webots or Gazebo supplies dynamics unavailable from Robot
  SF's native simulation and existing synthetic AMV actuation diagnostics;
- a durable source for AMV geometry, actuation, latency, and controller semantics that can be used
  without turning a demo into a paper-facing calibration claim.

The smallest acceptable spike would:

1. use one tracked `simulation_trace_export.v1` fixture or a declared command-trace fixture;
2. define one robot model and one command interface;
3. emit a compact diagnostic artifact with units and unsupported semantics;
4. report `execution_mode`, `readiness_status`, and `availability_status`;
5. state that the output is diagnostic-only and does not establish benchmark or physical AMV parity.

## Recommendation

Do not integrate Webots or Gazebo now.

Keep Webots as the lower-burden candidate to monitor for an education/demo-style AMV prototype or
ROS 2 controller walkthrough, because the ROS 2 tutorial path is compact and directly shows a robot
driver and supervisor workflow.

Keep Gazebo as the candidate to monitor only when the target is explicitly ROS 2 ecosystem parity,
Gazebo-native assets, sensor plugins, or compatibility with an upstream ROS/Gazebo planner source.
Existing Robot SF policy-search notes already treat ROS/Gazebo-heavy sources as source-side
reproduction targets, not direct benchmark adapters.

## Validation

This is an analysis-only note. Validation is by inspection and path/link checks:

```bash
test -f docs/context/issue_2013_backend_adapter_contract.md
test -f docs/context/issue_1689_simulation_trace_export_schema.md
test -f docs/context/issue_1075_operating_envelope.md
grep -q 'issue_2016_webots_gazebo_amv_parity_audit.md' docs/context/README.md
grep -q 'issue_2016_webots_gazebo_amv_parity_audit.md' docs/context/INDEX.md
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```
