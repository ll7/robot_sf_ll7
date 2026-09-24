# Robot-attributable force implementation plan

Goal: record the simulator's robot force without modifying dynamics or legacy metrics.
Scope: #9666 only; Social Navigation Quality Index changes remain downstream.
Sources: issue #9666, PedRobotForce, PySocialForce social_force_ped_ped, simulator step ordering.

Steps: capture component and pre-integration inputs; add optional episode data and reductions;
compute reference from the configured pedestrian kernel; add explicit post-hoc and experimental
counterfactual paths; integrate schema/registry; verify focused contracts and full readiness.

Evidence: compare existing metric serialization with optional fields absent; analytical law and
reduction tests; simulator component sum and position-aligned recomputation tests. Campaign
acceptance needs 384 episodes, correlation tables and 20 disagreement examples. No claim of
campaign acceptance until recorded native runs exist. Keep campaign artifacts in durable storage,
with config/source hashes. Stop on mismatch or degraded execution.

Observed issue drift: actual SocialForce includes a lateral component at theta=0; reference must
use full vector magnitude. Simulator overrides robot force radius with each actual robot radius.
Snapshot positions occur after integration, so preserve distinct force-input positions.
Recovery: isolated branch codex/issue-9666-robot-force-008; retain incomplete proof as blocked.
