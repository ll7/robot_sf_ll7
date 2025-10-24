# Phase 0 Research: Environment Factory Ergonomics

**Feature**: Improve Environment Factory Ergonomics  
**Branch**: 130-improve-environment-factory  
**Status**: In Progress  
**Purpose**: Resolve unknowns, document current state, justify design decisions before Phase 1 artifacts.

## 1. Current State Inventory (Baseline)
TODO: Enumerate existing factory functions, signatures, and internal kwargs usage patterns.

## 2. Legacy Kwarg Usage Frequency
TODO: Grep repository for prior factory invocations collecting argument names into a frequency table.

## 3. Performance Baseline (Env Creation)
Method: Time 30 sequential creations of each factory variant (robot, image, pedestrian).  
Metrics: mean, p95, std; compare post-change (< +5% budget).  
TODO: Capture measurements.

## 4. Unknowns & Decisions
| Unknown | Decision | Rationale | Alternatives | Status |
|---------|----------|-----------|-------------|--------|
| Unknown legacy kw policy | TBD |  | Strict raise / permissive | OPEN |
| Validation severity | TBD |  | Warn vs raise | OPEN |
| Multi-robot options split | TBD |  | Separate dataclass now | OPEN |
| Legacy permissive toggle name | TBD |  | Env var vs function param | OPEN |
| Python baseline features | Assume >=3.11 | Slots + dataclass features | Confirm runtime version | OPEN |

## 5. Alternatives Considered
### A. Builder Pattern
Pros: Chainable clarity; Cons: Overkill, increases ceremony, diverges from current simple functional API.  
Status: Rejected (added complexity; no multi-step state required).

### B. Single Monolithic Options Object
Pros: One param; Cons: Hides discoverability in IDE signature; discourages frequent simple usage.  
Status: Rejected in favor of two focused option dataclasses.

### C. Dynamic Registry of Factories
Pros: Extensibility; Cons: Adds indirection, not required for current scope.  
Status: Rejected (YAGNI).

## 6. Preliminary Decisions (Proposed)
1. Strict-by-default unknown kw error; override with env var `ROBOT_SF_FACTORY_LEGACY=1` enabling permissive mapping + warnings.
2. Incompatible combos produce warning and auto-normalization where safe (e.g., enabling minimal rendering path for recording); escalate to error only if impossible to normalize.
3. Multi-robot specialization deferred; keep pedestrian-specific flags inside existing config and generic render options.
4. Option dataclasses use `@dataclass(slots=True, frozen=False)` for performance and mutability clarity.
5. Deprecation window: 2 minor releases (document timeline; warnings now, potential removal later).

## 7. Open Research Tasks
- [ ] Extract list of all historical kwargs used in `make_*_env` calls.
- [ ] Time baseline creation costs (capture raw numbers).
- [ ] Validate Python version in CI matrix for dataclass feature compatibility.
- [ ] Confirm no existing user docs advise unsupported kwargs to avoid mismatch.

## 8. Decision Log
Will be appended as each OPEN item transitions to CLOSED with rationale.

## 9. Risk Analysis
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Silent behavior divergence due to normalization | Incorrect expectations | Medium | Explicit logging + tests |
| Performance regression from added object construction | Slower creation | Low | Micro-benchmark + slots |
| User confusion during transition | Support burden | Medium | Migration guide + warnings |
| Over-broad permissive mode enabling old typos | Hidden bugs | Low | Default strict, explicit opt-in |

## 10. Next Steps
Proceed to gather empirical data (Sections 1–3) → update Sections 2,3,4 tables → convert OPEN to CLOSED. Once all unknowns resolved, advance to Phase 1.
