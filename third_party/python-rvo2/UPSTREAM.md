upstream_repository: https://github.com/mit-acl/Python-RVO2
upstream_revision: 56b245132ea104ee8a621ddf65b8a3dd85028ed2
source_archive_url: https://github.com/mit-acl/Python-RVO2/archive/56b245132ea104ee8a621ddf65b8a3dd85028ed2.tar.gz
source_archive_sha256: 9789ad4807a1708a41a259ad1b4948af6385b0cb3c74acdcfb2c032aa15fb473
retrieved_at: 2026-08-09
license_spdx: Apache-2.0
upstream_notice_present: false
local_patch: LOCAL_CHANGES.patch

Local changes:
- Updated CMake minimum version to 3.5 to satisfy newer CMake policy handling.
- Added the CMake build wrapper used by the package build backend.
- Added the Robot SF collaboration-coefficient extension to the RVO2 C++ and
  Cython bindings; modified files carry local-change notices where the upstream
  Apache-2.0 source is retained.
- Added the Python build metadata and the platform-specific build adjustments.

The upstream commit is publicly resolvable. Its archive contains `LICENSE` but
does not contain an upstream `NOTICE` file. The local patch is generated against
the upstream archive while excluding generated build outputs, this provenance
record, and the patch itself. Recreate the archive from `source_archive_url`,
then run the following from the repository root before changing the pinned source:

```bash
diff -ruN --exclude='__pycache__' --exclude='*.egg-info' \
  --exclude='rvo2.cpp' --exclude='UPSTREAM.md' --exclude='LOCAL_CHANGES.patch' \
  upstream-python-rvo2/ third_party/python-rvo2/ > LOCAL_CHANGES.patch
```
