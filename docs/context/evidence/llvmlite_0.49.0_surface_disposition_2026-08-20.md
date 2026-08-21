<!-- AI-GENERATED (robot_sf#7670, 2026-08-21) - NEEDS-REVIEW -->

# llvmlite 0.49.0 surface-specific disposition evidence

This note is the durable provenance pointer for the bounded maintainer ruling in
Issue #7653. It records package, lock, artifact, profile, and upstream notice
references; it is not a license opinion, redistribution authorization, or
technical approval.

## Exact package identity

- Package: `llvmlite` `0.49.0`
- Python requirement: `>=3.10`
- Source/index: `https://pypi.org/simple`
- Upstream repository: `https://github.com/numba/llvmlite`
- Upstream tag: `v0.49.0`
- Reported expression: `BSD-2-Clause AND Apache-2.0 WITH LLVM-exception`
- Root lock: `uv.lock`
- Standalone lock: `fast-pysf/uv.lock`

The target Linux x86_64 / CPython 3.13 wheel is
`llvmlite-0.49.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
with SHA-256
`ddc7aecd4f56397ed6e8f120ec5dcd5a1a8f0e6032ca4af413462792d4dca2e3`.
The lockfiles also record `llvmlite-0.49.0.tar.gz` with SHA-256
`00f16db782f4a13c78c5804aedc434e46794a77e89999a168f9401106270e50a`.

## Upstream notice references

The two upstream notice texts retained by the ruling are:

- [`LICENSE`](https://github.com/numba/llvmlite/blob/v0.49.0/LICENSE)
- [`LICENSE.thirdparty`](https://github.com/numba/llvmlite/blob/v0.49.0/LICENSE.thirdparty)

These URLs are provenance references. The repository does not vendor or copy
llvmlite source, wheels, sdists, or notice files into a Robot SF artifact.

## Surface and profile boundary

The allowed modes are `user_installed` and `not_distributed`. `bundled_source`
and `built_companion` remain blocked. Mirrored, vendored, container-bundled,
unknown, unavailable, and conflicting surfaces remain blocked pending a separate ruling.

The disposition is bound to the current frozen profile definitions: root
`core`, every current root extra and closure (`viz`, `maps`, `benchmark`, `gpu`,
`training`, `recurrent`, `rllib`, `progress`, `analytics`, `browser`,
`sacadrl`, `orca`, `socnav`, `criticality`, `all`), and standalone `fast-pysf`.
Any version, source, artifact, profile, expression, or packaging-surface change
reopens the decision.

## Claim boundary

This is dependency and release-surface governance evidence only. It is not
legal advice, a technical approval, redistribution authorization, or merge
authorization.
