# Issue #3064 Behavior Variant Preflight

- Generated at: `2026-06-18T22:55:00Z`
- Git head: `736de7781d1ccdf214866a3ea9c48612a7994bff`
- Social-Navigation-PyEnvs root: `<local-artifact-root>/repos/Social-Navigation-PyEnvs`
- Claim boundary: Preflight/inventory evidence only. Rows marked diagnostic_only, not_available, fallback, or degraded must not be counted as benchmark-success evidence.

## Classification

| Variant | Mode | Availability | Row status | Benchmark validity | Reason |
| --- | --- | --- | --- | --- | --- |
| `social_force` | `adapter` | `available` | `benchmark_valid_candidate` | `benchmark_valid_candidate` | required repository paths and runtime prerequisites are present |
| `ammv_social_force` | `adapter` | `available` | `diagnostic_only` | `diagnostic_only` | existing same-seed adapter-mode evidence found no default-vs-AMMV frame or episode metric delta; use as diagnostic mechanism evidence until a behavioral-difference execution path is proven |
| `social_navigation_pyenvs_orca` | `adapter` | `not_available` | `unavailable/excluded` | `not_available` | missing or incompatible runtime prerequisites |
| `social_navigation_pyenvs_socialforce` | `adapter` | `not_available` | `unavailable/excluded` | `not_available` | missing or incompatible runtime prerequisites |
| `social_navigation_pyenvs_sfm_helbing` | `adapter` | `not_available` | `unavailable/excluded` | `not_available` | missing or incompatible runtime prerequisites |
| `social_navigation_pyenvs_hsfm_new_guo` | `adapter` | `not_available` | `unavailable/excluded` | `not_available` | missing or incompatible runtime prerequisites |

## Runtime Limitations

- `social_navigation_pyenvs_orca`:
  - Social-Navigation-PyEnvs checkout: `missing`
- `social_navigation_pyenvs_socialforce`:
  - Social-Navigation-PyEnvs checkout: `missing`
  - python package: `missing`
- `social_navigation_pyenvs_sfm_helbing`:
  - Social-Navigation-PyEnvs checkout: `missing`
- `social_navigation_pyenvs_hsfm_new_guo`:
  - Social-Navigation-PyEnvs checkout: `missing`

## Interpretation

- `social_force` is the current benchmark-valid native/adapter baseline candidate.
- `ammv_social_force` remains diagnostic-only under the existing same-seed evidence.
- Social-Navigation-PyEnvs rows are excluded unless their checkout and runtime dependencies are present in the local environment.
