# Issue #2014 Simulator Backend Decision Matrix (2026-06-01)

Date: 2026-06-01

Related issue:

- <https://github.com/ll7/robot_sf_ll7/issues/2014>

Related Robot SF context:

- [issue_2013_backend_adapter_contract.md](issue_2013_backend_adapter_contract.md)
- [issue_2016_webots_gazebo_amv_parity_audit.md](issue_2016_webots_gazebo_amv_parity_audit.md)
- [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md)
- [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)
- [issue_1485_carla_transfer_boundary_follow_up.md](issue_1485_carla_transfer_boundary_follow_up.md)
- [issue_1508_carla_native_aligned_eligibility.md](issue_1508_carla_native_aligned_eligibility.md)
- [issue_1509_carla_native_fixture_certification.md](issue_1509_carla_native_fixture_certification.md)
- [issue_2001_amv_actuation_proxy_source_analysis.md](issue_2001_amv_actuation_proxy_source_analysis.md)
- [issue_1556_amv_actuation_stress_slice.md](issue_1556_amv_actuation_stress_slice.md)
- [docs/debug_visualization.md](../debug_visualization.md)

Related GitHub issues without a local canonical note for every sub-scope:

- [#1646](https://github.com/ll7/robot_sf_ll7/issues/1646) trace/report/frontend direction
- [#1491](https://github.com/ll7/robot_sf_ll7/issues/1491) CARLA parity parent
- [#1510](https://github.com/ll7/robot_sf_ll7/issues/1510) and
  [#1511](https://github.com/ll7/robot_sf_ll7/issues/1511) CARLA parity siblings
- [#1585](https://github.com/ll7/robot_sf_ll7/issues/1585) AMV calibration gate
- [#1559](https://github.com/ll7/robot_sf_ll7/issues/1559) AMV evidence/claim boundary

## Recommendation

Continue the trace/report/frontend path before any full alternate-simulator integration. The
current Robot SF evidence base is strongest when alternate backends are treated as diagnostic
producers that must satisfy the backend adapter contract, emit trace-compatible artifacts, and fail
closed when semantics are missing or degraded.

Near-term implementation work should be limited to one possible spike: a MuJoCo AMV actuation
diagnostic that replays a small Robot SF command trace through one declared model and emits
`simulation_trace_export.v1`-compatible output or a named diagnostic variant. That spike should be a
separate issue and should remain diagnostic unless it also satisfies the AMV calibration gates in
[#1585](https://github.com/ll7/robot_sf_ll7/issues/1585) and
[#1559](https://github.com/ll7/robot_sf_ll7/issues/1559).

Monitor Webots, Gazebo, Isaac Lab, Habitat/Habitat-Sim, and CARLA, but do not integrate them now.
Webots and Gazebo are mainly monitor-only for a future ROS 2/controller/AMV prototype. CARLA remains
deferred to the native/aligned parity chain. Isaac Lab is worth monitoring for future large-scale
robot-learning experiments, not current Robot SF benchmark validation. Habitat/Habitat-Sim is useful
to watch for embodied-AI sensor and navigation semantics, but it is not a near-term AMV dynamics
backend.

Confidence: 0.78. This recommendation should change if a named AMV hardware/controller target
appears, a calibrated actuation source passes the AMV provenance gates, a certified native/aligned
CARLA fixture becomes available, or Robot SF policy research becomes blocked on RL scale or sensor
modalities that the native simulator and trace frontend cannot supply.

## Backend Matrix

| Backend | AMV actuation | Social-navigation semantics | Pedestrians | Sensors | ROS 2/controller parity | RL scale | Licensing | Install burden | Reproducibility | Artifact provenance | Evidence/claim boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuJoCo | Best near-term actuation-diagnostic candidate because it is a fast articulated-body physics engine, but Robot SF still needs an explicit command-to-actuation mapping. | Does not provide Robot SF social-force semantics by default. | Must be authored or replayed from Robot SF traces; not a native crowd/social-navigation answer. | Basic rendering and model sensors are possible, but sensor realism is not the reason to use it here. | No repo-local ROS 2 parity target; controller bridge would be new scope. | Strong fit for local RL/control iteration and parallel sampling. | Official project is open source under Apache-2.0. | Moderate: smaller than full robotics/AV simulators, but new model assets and bindings still add setup/provenance work. | Good candidate for deterministic micro-diagnostics if model, version, seed, timestep, and trace input are pinned. | Needs tracked model, command trace, environment version, and output schema variant. | Only diagnostic unless a separate benchmark/parity issue proves contract satisfaction and AMV calibration relevance. |
| Webots | Plausible AMV prototype host through robot/world/controller modeling. | No Robot SF social-force semantics by default. | Possible via authored agents/controllers, but not a native replacement for Robot SF pedestrian dynamics. | Good robotics-simulator sensor coverage. | Official ROS 2 packages/tutorials make it a reasonable future ROS 2 monitor target. | Usable for robotics demos; not the obvious high-throughput RL path for this repo. | Official repositories report Apache-2.0. | Moderate to high: `.wbt` worlds, PROTO/model choices, controllers, ROS 2 packages, and runtime expectations. | Reproducible only after versioned world/model/controller assets and adapter status fields are pinned. | Needs world/model/controller provenance plus trace export mapping. | Monitor only; `no_integration_for_now` per issue #2016. |
| Gazebo / modern Gazebo | Plausible AMV prototype host through SDF/URDF-style models, plugins, and bridges. | No Robot SF social-force semantics by default. | Possible through plugins or scripted actors, but not a current Robot SF crowd model. | Strong robotics sensor/plugin ecosystem. | Strongest ROS 2/controller parity candidate when a concrete ROS 2 target exists. | Useful for robotics integration; not the lowest-burden RL scale option. | Gazebo Sim libraries are documented under Apache-2.0. | High: Gazebo version, SDF assets, bridge/plugin provenance, ROS 2 compatibility, and CI/runtime burden. | Reproducible only with pinned Gazebo/ROS versions, world/model files, plugins, and status classification. | Needs SDF/world/plugin/controller provenance and durable trace outputs. | Monitor only; spike only for a named ROS 2/controller/AMV need. |
| Isaac Lab | Can model mobile robots/AMRs, but would require a declared Robot SF-to-Isaac actuation contract and assets. | Does not import Robot SF social-navigation semantics by default. | Pedestrian/social behavior would need task/asset authoring or trace replay. | Strong simulated sensor/rendering stack through Isaac Sim. | Possible ecosystem path, but no repo-local controller target exists. | Strongest candidate for future GPU-scale robot-learning workloads. | Official docs state BSD-3-Clause for Isaac Lab; Isaac Sim/Omniverse dependencies bring their own terms and hardware assumptions. | High: NVIDIA GPU stack, Isaac Sim dependency, USD assets, environment setup, and version sensitivity. | Reproducibility depends on pinned container/runtime/GPU/asset versions and task configs. | Needs USD/assets, task config, runtime version, checkpoint lineage, and trace export mapping. | Monitor only for future RL-scale or sensor-rich research; not a current benchmark backend. |
| Habitat / Habitat-Sim | Poor near-term AMV actuation fit; its strength is embodied-AI navigation/sensing rather than vehicle/robot dynamics parity. | Strong embodied-AI task framing, but not Robot SF social-force semantics. | Dataset/task-driven embodied agents, not a drop-in pedestrian dynamics model. | Strong visual/navigation sensor focus. | Not a ROS 2/controller parity route for this repo. | Good for embodied-AI experiments, less direct for Robot SF AMV control. | Official repository reports MIT license; verify dataset and dependency terms for any adopted asset/data pipeline. | Moderate to high: dataset/assets, simulator build/runtime, and task integration. | Depends heavily on dataset and asset provenance plus task config pinning. | Needs dataset license, scene asset provenance, task config, and trace/report mapping. | Do not integrate now as a dynamics backend; monitor only for perception/navigation diagnostics. |
| CARLA | Vehicle dynamics and sensors are rich, but AMV/social-navigation parity is blocked without native/aligned scenario fixtures. | Autonomous-driving semantics differ from Robot SF; only native/aligned parity can support stronger comparison. | CARLA traffic actors are not Robot SF pedestrian semantics by default. | Strong AV sensor and environment controls. | ROS bridge exists, but repo-local CARLA work remains host-dependent and parity-gated. | Useful for AV research; high runtime burden for routine Robot SF work. | Official CARLA code is MIT and assets are CC-BY; Unreal Engine follows its own terms. | Very high: Unreal/CARLA versions, assets, GPU/runtime setup, maps, host constraints. | Reproducibility blocked until fixture certification and native/aligned parity inputs exist. | Needs map/asset/version, scenario fixture, replay mode, alignment proof, and status classification. | Deferred to CARLA parity issues; adapted/degraded runs remain diagnostic or unavailable, not benchmark success. |

## Routing

Near-term:

- Continue trace export, report, and frontend work under
  [#1646](https://github.com/ll7/robot_sf_ll7/issues/1646), because it improves inspection and
  debugging without adding simulator runtime dependencies.
- Consider one MuJoCo AMV actuation diagnostic issue only if it names the command trace, model,
  schema output, and fail-closed adapter status fields up front.

Monitor only:

- Webots and Gazebo for a future named AMV/ROS 2/controller/sensor prototype.
- Isaac Lab for future GPU-scale robot-learning workloads that Robot SF cannot cover locally.
- Habitat/Habitat-Sim for sensor/navigation diagnostics where embodied-AI datasets are the real
  research object.
- CARLA for native/aligned parity after the #1491/#1510/#1511 chain has certified fixtures.

Do not integrate now:

- Any broad alternate backend as a routine Robot SF benchmark backend.
- Habitat/Habitat-Sim as an AMV dynamics backend.
- CARLA as evidence for Robot SF benchmark improvement before native/aligned parity is proven.
- Webots/Gazebo/Isaac Lab demos as AMV calibration, ROS parity, or paper-facing evidence.

## External Sources Checked

- MuJoCo overview: <https://mujoco.readthedocs.io/en/stable/overview.html>
- MuJoCo project page: <https://mujoco.org/>
- Webots repository: <https://github.com/cyberbotics/webots>
- Webots ROS 2 packages: <https://github.com/cyberbotics/webots_ros2>
- Gazebo ROS 2 integration: <https://gazebosim.org/docs/harmonic/ros2_integration/>
- Gazebo Sim library page: <https://gazebosim.org/libs/sim/>
- Isaac Lab documentation: <https://isaac-sim.github.io/IsaacLab/v2.0.1/index.html>
- NVIDIA Isaac Lab overview: <https://developer.nvidia.com/isaac/lab>
- Habitat-Sim documentation: <https://aihabitat.org/docs/habitat-sim/habitat_sim.html>
- Habitat-Sim repository: <https://github.com/facebookresearch/habitat-sim>
- CARLA introduction: <https://carla.readthedocs.io/en/latest/start_introduction/>
- CARLA repository and license notes: <https://github.com/carla-simulator/carla>

## Validation

This note is a docs-only decision matrix. It does not add a backend, schema, benchmark row, or
paper-facing claim. Validation should inspect the diff, verify that referenced local paths exist,
check discoverability through `docs/context/INDEX.md`, `docs/context/README.md`, and
`docs/context/catalog.yaml`, and run the docs proof-consistency checker against both
`docs/context/catalog.yaml` and this note.
