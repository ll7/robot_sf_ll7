# Dependency Cache Profiles

Profiles declare one explicit environment (roots, extras, and target platform) so the exact
dependency artifact set for an offline reconstruction is derived from the canonical lock rather than
from a machine-local cache inspection.

| Profile | Roots | Environment |
| --- | --- | --- |
| [all_extras_linux_x86_64.json](all_extras_linux_x86_64.json) | `robot-sf` (all extras) | CPython 3.11, linux, x86_64 |

## Check a cache

```bash
uv run python scripts/tools/dependency_cache_manifest.py --check \
  --profile configs/dependency_profiles/all_extras_linux_x86_64.json \
  --cache-root "$POPULATED_CACHE_ROOT" --format markdown
```

The cache root must contain the original artifact files (`.whl`, `.tar.gz`, `.tgz`, `.zip`), for
example a staged wheelhouse; an unpacked package cache does not preserve the artifact identity the
lock records. The tool never downloads and never copies: it derives the closure from the lock,
matches each expected artifact by filename, verifies the lock's SHA-256, and classifies every
requirement as `available_verified`, `checksum_drift`, `duplicate_artifact`, `missing`,
`wrong_platform`, `source_only`, or `build_required`.

Rights are independent of cache presence: `rights.default` is `redistribution-unknown` unless a
profile explicitly permits a package or marks it as a private companion. Private preservation never
implies public redistribution rights, and the sanitized status
(`dependency_cache_reconstruction_status.v1`) carries no private paths, indexes, or credentials.

Write the private manifest and checksum list with `--manifest-out <json>` and `--sums-out
<SHA256SUMS>`; the checksum list includes the expected hash for missing artifacts so a transfer is
verifiable file by file. `--verify-offline-install` additionally runs `uv pip install --no-index`
against the cache in a temporary environment, but only when every requirement is
`available_verified` and the profile sets `offline_install.permitted`.
