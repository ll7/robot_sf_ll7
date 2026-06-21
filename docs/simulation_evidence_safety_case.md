# Simulation-Evidence Safety Case Template

[Back to Documentation Index](./README.md)

**Status**: Public-safe template for organizing Robot SF simulation evidence.

**Related issue**: [#3291](https://github.com/ll7/robot_sf_ll7/issues/3291)

This template helps reviewers map Robot SF benchmark and falsification evidence into a structured
safety argument. It is not a safety certificate, legal compliance claim, deployment approval, or
proof that simulation alone establishes safety.

Use it when a report, release packet, or issue needs to state what simulation evidence supports,
what it does not support, and what evidence must come from outside the simulator.

## Claim Boundary

A filled template may support a bounded statement such as:

> Under the named ODD, scenario, seed, metric, and artifact contracts, this benchmark campaign gives
> diagnostic or benchmark evidence for the listed hazards and requirements.

It must not support these statements by itself:

- the robot is safe for real-world deployment,
- the planner satisfies a regulatory or operational safety case,
- unmodeled pedestrians, vehicle classes, weather, sensors, or actuation dynamics are covered,
- fallback, degraded, failed, not-available, or metadata-only rows are success evidence.

Use the repository evidence ladder from
[Maintainer Values And Hard Contracts](./maintainer_values.md): `diagnostic-only`,
`smoke evidence`, `nominal benchmark evidence`, and `paper-grade`. When confidence is below about
95 percent, put the caveat or uncertainty next to the claim.

## Required Links

Every filled safety-case instance should link these surfaces when they exist. If one is missing,
record `missing` plus the unblock condition instead of leaving the field blank.

| Field | Required reference |
| --- | --- |
| ODD scope | [ODD contract](./odd_contracts.md) or explicit `missing` status. |
| Hazard mapping | [Hazard traceability](./hazard_traceability.md) row or explicit gap. |
| Scenario intent | [Scenario contract](./scenario_contracts.md) or explicit gap. |
| Scenario feasibility | [Scenario certification](./scenario_certification.md) or exclusion reason. |
| Artifact provenance | Campaign root or manifest, command, commit, checksum, and artifact category from the [Artifact Evidence Vocabulary](./context/artifact_evidence_vocabulary.md). |
| Credibility assessment | Filled credibility checklist, planned assessment, or blocker. |
| External evidence | Required real-world, operational, hardware, human-factors, or regulatory evidence outside Robot SF. |

## Template

### 1. Header

| Field | Value |
| --- | --- |
| Safety-case id | `<stable id>` |
| Owner / reviewer | `<person or team>` |
| Date and source commit | `<date>`, `<commit>` |
| Evidence status | `diagnostic-only` / `smoke evidence` / `nominal benchmark evidence` / `paper-grade` |
| Intended use | `<analysis, benchmark release, paper appendix, internal gate, other>` |
| Non-claims | `<deployment readiness, legal compliance, real-world safety certification, etc.>` |

### 2. System And ODD Scope

| Field | Value |
| --- | --- |
| Robot or AMV class | `<platform class and modeled kinematics>` |
| Planner or policy | `<planner id, policy id, adapter mode>` |
| Operating context | `<public-space context, speed envelope, density envelope>` |
| ODD contract | `<path and contract id>` |
| ODD exclusions | `<weather, public-road autonomy, sensor limits, legal claims, etc.>` |
| Scenario set | `<config or manifest path>` |
| Seed policy | `<seed count, deterministic policy, excluded seeds>` |

### 3. Hazards And Requirements

| Hazard id | Hazard statement | Requirement or question | Simulation evidence expected | Outside-simulation evidence required |
| --- | --- | --- | --- | --- |
| `<hazard_id>` | `<harm or undesired state>` | `<requirement, threshold, or decision>` | `<metrics, scenarios, artifacts>` | `<field data, HIL, hardware, human study, regulator review, etc.>` |

Keep requirements audit-friendly. If a requirement is not measurable in Robot SF, say so and route
it to an external evidence row.

### 4. Evidence Map

| Evidence id | Artifact category | Source and provenance | Status | Caveats | Supports | Does not support |
| --- | --- | --- | --- | --- | --- | --- |
| `<id>` | `<durable evidence copy, benchmark claim, release artifact, etc.>` | `<manifest, command, commit, checksum>` | `<covered, partial, missing, excluded, unavailable>` | `<fallback/degraded rows, schema gaps, missing real-world data>` | `<bounded claim>` | `<claim gap>` |

Rules:

- Link to the durable artifact or manifest, not only a local `output/` path.
- Name fallback, degraded, failed, and not-available rows as caveats or exclusions.
- Keep metric/schema versions visible when they affect the claim.
- Treat metadata-only surfaces as claim boundaries, not execution evidence.

### 5. Credibility And Validation Limits

| Question | Current answer | Evidence or blocker |
| --- | --- | --- |
| Numerical implementation checked? | `<yes/no/partial>` | `<tests, fixtures, validators>` |
| Scenario validity checked? | `<yes/no/partial>` | `<scenario_cert.v1, contract, exclusions>` |
| Model assumptions reviewed? | `<yes/no/partial>` | `<pedestrian model, kinematics, sensor, actuation assumptions>` |
| Uncertainty quantified? | `<yes/no/partial>` | `<confidence intervals, seed count, sensitivity report>` |
| Real-world validation available? | `<yes/no/partial>` | `<external source or missing evidence>` |
| Known invalid domains named? | `<yes/no/partial>` | `<ODD exclusions, unsupported settings>` |

This section separates verification of repository behavior from validation against real-world
behavior. A campaign can be reproducible and still not validated for deployment use.

### 6. Residual Risk And Decision

| Residual risk or gap | Owner | Required next evidence | Blocks current claim? |
| --- | --- | --- | --- |
| `<gap>` | `<owner>` | `<smallest proof step>` | `yes/no` |

Decision:

- `continue`: evidence is enough for the bounded internal/research claim.
- `revise`: claim wording, scenario set, metric, or artifact provenance must be narrowed.
- `block`: required proof for the stated claim is missing.
- `external`: the next evidence must come from outside Robot SF.

## Toy Example Mapping

This example is intentionally limited. It shows how to fill the template without implying safety or
deployment readiness.

### Example Header

| Field | Value |
| --- | --- |
| Safety-case id | `toy_low_speed_public_space_hazard_coverage` |
| Evidence status | `diagnostic-only` |
| Intended use | Demonstrate how hazard/ODD coverage evidence maps into a safety-case scaffold. |
| Non-claims | No deployment readiness, legal compliance, real-world safety certification, or planner safety claim. |

### Example Scope

| Field | Value |
| --- | --- |
| Robot or AMV class | Low-speed public-space AMV/robot class represented by existing benchmark metadata. |
| Planner or policy | Mixed rows from the source campaign; this example does not compare planner safety. |
| ODD contract | [`configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml`](../configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml). |
| Hazard mapping | [`configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml`](../configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml). |
| Scenario contract | [`configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml`](../configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml). |
| Source campaign root | [`docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28`](./context/evidence/issue_1484_broader_cross_kinematics_2026-05-28/README.md). |
| Evidence artifact | [`docs/context/evidence/issue_2156_research_v1_hazard_odd_2026-06-03`](./context/evidence/issue_2156_research_v1_hazard_odd_2026-06-03/README.md). |
| Credibility assessment | `missing`: this toy example records the gap; issue #3290 tracks a reusable credibility checklist. |

### Example Hazard And Requirement Rows

| Hazard id | Requirement or question | Simulation evidence expected | Current mapped status | Outside-simulation evidence required |
| --- | --- | --- | --- | --- |
| `robot_pedestrian_collision` | Can the campaign support a collision-hazard coverage statement? | Non-caveated executed rows linked to collision-rate evidence for mapped scenarios. | `missing`: metadata exists, but no executed row matched the mapped scenarios in the Issue #2156 rollup. | Real-world incident/hazard analysis, hardware emergency-stop behavior, sensor-performance evidence, and operational safety review. |
| `near_miss` | Can the campaign support a near-miss coverage statement? | Non-caveated executed rows with `min_ttc` or `pet` evidence for mapped scenarios. | `missing`: no executed row matched the mapped near-miss scenarios. | Real pedestrian interaction data, validated TTC/PET thresholds, and human-factors review. |

### Example Evidence Map

| Evidence id | Artifact category | Source and provenance | Status | Supports | Does not support |
| --- | --- | --- | --- | --- | --- |
| `issue_2156_hazard_odd_rollup` | Durable evidence copy | Source campaign root `docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28`; rollup files `hazard_odd_coverage_summary.json`, `hazard_coverage_table.csv`, `odd_boundary_table.csv`, and `checksums.sha256` under the Issue #2156 evidence directory. Generated by `scripts/tools/hazard_odd_coverage_rollup.py` at commit `0b5a2efcdd738f5019e75a063218a99c9772d589`. | `diagnostic-only`; hazard statuses are `missing=5`, ODD statuses are `partial=2`, `excluded=8`, scenario-contract statuses are `missing=1`. | A traceability gap: current metadata and executed rows are not yet joined for the listed hazards. | Any claim that the campaign proves collision or near-miss mitigation, benchmark coverage completeness, or deployment safety. |

### Example Decision

Decision: `revise`.

The evidence is useful because it identifies missing joins between hazard metadata and executed
rows. The next smallest proof step is to add or select scenario contracts and hazard mappings that
name the executed AMV scenarios, rerun the same rollup, and only then decide whether the result
supports `smoke evidence` or `nominal benchmark evidence`. Real-world validation remains outside
the simulator and must be handled by a separate evidence source.

## Review Checklist

- The claim boundary appears before result interpretation.
- Evidence status is one of the repository ladder values.
- ODD, hazard, scenario, certification, provenance, and credibility links are present or explicitly
  marked `missing`.
- `output/` is not cited as the durable evidence location.
- Fallback, degraded, failed, and not-available rows are caveats, not success evidence.
- The template names at least one real-world or operational evidence gap that simulation cannot
  fill.
- The decision is `continue`, `revise`, `block`, or `external`, with the next proof step named.
