# Phase 0 Research - Architectural decoupling and consistency overhaul

## Decisions

- Simulator Facade as the stable contract: create/reset/step/get_state minimal surface selected via unified config key.
- Sensor and Fusion minimal interfaces: sensors provide typed outputs; fusion composes without hardcoding sensor types.
- Error handling policy: required resources → fatal with remediation; optional paths → warning and soft-degrade.
- Unified configuration with schema validation and conflict checks; resolved-config dump for reproducibility.

## Rationale

- Reduces tight coupling and allows backend swaps without touching env code (aligns with Constitution II, VII).
- Simplifies adding sensors by constraining to a small interface, avoiding fusion rewrites.
- Actionable errors reduce support burden and speed onboarding.
- Config consolidation removes ambiguity and enforces deterministic runs (Constitution IV).

## Alternatives Considered

- Direct env↔backend binding via subclassing: rejected; increases coupling and regression risk.
- Implicit sensor discovery via reflection: rejected; fragile and opaque error modes.
- Tolerant config (ignore unknown keys): rejected per Constitution (unknown fields must be rejected in strict modes).

## Open Questions (resolved - no blockers)

- External API exposure (REST/GraphQL) not applicable; internal component contracts documented instead.
- Performance: baseline maintained by avoiding per-step logging; interfaces are thin wrappers only.

## References

- Repository Constitution v1.3.1 (sections II, IV, VII, IX, XII).
- Existing unified test suite and performance targets in docs/dev_guide.md.
