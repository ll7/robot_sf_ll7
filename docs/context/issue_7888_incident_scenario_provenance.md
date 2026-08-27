# Incident-to-Scenario Provenance Contract (`incident_scenario_provenance.v1`)

**Status:** diagnostic / contract-only — schema + fixture proof, not evidence of legal
attribution, real-world causality, scenario representativeness, or benchmark performance.
**Issue:** [#7888](https://github.com/ll7/robot_sf_ll7/issues/7888) (parent research
[#7881](https://github.com/ll7/robot_sf_ll7/issues/7881)).
**Related causal boundary:** [#7315](https://github.com/ll7/robot_sf_ll7/issues/7315) and
[#5440](https://github.com/ll7/robot_sf_ll7/issues/5440).
**Owner module:** `robot_sf/benchmark/collision/incident_scenario_provenance.py` +
`robot_sf/benchmark/schemas/incident_scenario_provenance.v1.json`.
**Method card source:** Bai et al., "LiOScen: Liability-oriented scenario generation from accident
reports for the validation of autonomous driving systems" (Journal of Systems and Software, 2026).

Plain-language summary: this contract turns one incident description into one replayable Robot SF
scenario record while keeping source facts, extracted hypotheses, simulator assumptions, parameter
mappings, execution identity, and observed outcomes in separate, auditable fields. It never assigns
legal or moral fault — `normative_fault` is always `not_assessed` — and rejected, ambiguous, or
unsupported records stay outside any admitted denominator.

## 1. Why a provenance contract first

The parent research question asks whether an accident or incident description can be transformed
into a reproducible Robot SF scenario while preserving actor roles, hazard initiation, source
evidence, and the distinction between model-scoped attribution and legal fault. The honest first
increment is a versioned, fail-closed schema plus a synthetic fixture that makes every
transformation step explicit. A scenario generator or an incident-source adapter would be premature
before the contract exists, because the repository needs a place to record *which* source fact,
hypothesis, or assumption produced *which* scenario parameter.

## 2. Field separation

| Field group | Purpose | Fail-closed rule |
| --- | --- | --- |
| `source` | Immutable source identity, SHA-256 digest, observed facts | digest must be a full 64-hex SHA-256 |
| `extraction` | Status (`verified` / `human_corrected` / `unverified` / `rejected`), actor roles, hypotheses, simulator assumptions | `verified` / `human_corrected` require an explicit `verification_record` (human review) |
| `scenario_parameters` | Parameter mappings with source field, transformation, unit, status, confidence | confidence is required; `unsupported` mappings must use `unavailable` confidence and must not be admitted |
| `execution` | Claimed execution identity: config digest, seed, software commit, replay identity, observed outcome | `claimed=true` requires all four identity fields and a full 40-hex software commit |
| `normative_fault` | Always `not_assessed` | schema `const` plus a semantic second net |

## 3. Actor-role boundary

Roles use the neutral model vocabulary `ego`, `pedestrian_initiator`, `affected_pedestrian`,
`infrastructure`, and `unknown`. Role labels describe the scenario model only. They never encode
legal liability, moral blame, or real-world causal certainty.

## 4. LiOScen boundary (method card)

The source method is car-centric and assumes an ADS-at-fault framing. This contract does **not**
transfer those assumptions: there is no fault vocabulary, no liability-oriented outcome, and no
claim that a scenario generated from an accident report reproduces the accident's legal or causal
conclusion. Completion of this issue proves contract validity only — not legal attribution,
real-world causality, scenario representativeness, or benchmark evidence.
