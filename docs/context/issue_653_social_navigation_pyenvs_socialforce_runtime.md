# Issue 653 Social-Navigation-PyEnvs SocialForce Runtime Reproduction Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#646` Prototype benchmark-facing Social-Navigation-PyEnvs SocialForce/SFM integration
- `robot_sf_ll7#653` Prototype compatible Social-Navigation-PyEnvs SocialForce runtime reproduction

## Goal

Determine whether the upstream `crowd_nav.policy_no_train.socialforce.SocialForce` policy can be
reproduced against a compatible external `socialforce` runtime without modifying the upstream
policy logic.

## Result

Verdict: `compatible runtime reproduced`

The upstream policy expects the external simulator constructor:

- `socialforce.Simulator(state, delta_t=self.time_step, initial_speed=self.initial_speed, v0=self.v0, sigma=self.sigma)`

The current published package API does not expose that signature directly. The resolved package is:

- `socialforce==0.2.3`
- `Simulator.__init__(..., ped_space=None, ped_ped=None, field_of_view=None, delta_t=0.4, tau=0.5, oversampling=10, dtype=None, integrator=None)`

A narrow compatibility runtime is still possible:

1. keep the upstream Social-Navigation-PyEnvs policy code unchanged,
2. wrap the external package with a CrowdNav-style shim that accepts
   `initial_speed`, `v0`, and `sigma`,
3. implement `step()` by forwarding the state through the external tensor-based simulator,
4. expose the resulting simulator state back through `.state` for the upstream policy.

This reproduces a minimal upstream `SocialForce.predict()` call successfully.

## Canonical validation commands

Inspect the external package contract:

```bash
uv run --with socialforce==0.2.3 python - <<'PY'
import inspect
import socialforce
from socialforce import Simulator
print(getattr(socialforce, '__version__', 'unknown'))
print(inspect.signature(Simulator.__init__))
PY
```

Run the checked-in compatibility probe:

```bash
uv run python scripts/tools/probe_social_navigation_pyenvs_socialforce_runtime.py \
  --repo-root output/repos/Social-Navigation-PyEnvs \
  --backend-spec socialforce==0.2.3 \
  --output-json output/benchmarks/external/social_navigation_pyenvs_socialforce_runtime_probe/report.json \
  --output-md output/benchmarks/external/social_navigation_pyenvs_socialforce_runtime_probe/report.md
```

## Why this matters

This closes the uncertainty from `#646`.

What is now justified:

1. the blocker was a concrete API mismatch in the external package, not a Robot SF observation-map error,
2. a source-faithful runtime reproduction exists without patching the upstream policy logic,
3. a benchmark-facing retry is now justified as a separate issue.

What is still not justified:

- claiming benchmark quality improvement,
- merging hidden changes into the upstream Social-Navigation-PyEnvs policy,
- skipping a dedicated benchmark-facing retry and performance comparison.

## Final verdict

- `social_navigation_pyenvs_socialforce` runtime: `compatible with explicit shim`
- next step: benchmark-facing retry with the compatibility runtime, keeping the shim behavior explicit
