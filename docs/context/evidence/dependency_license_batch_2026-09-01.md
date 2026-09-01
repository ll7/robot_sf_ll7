<!-- AI-GENERATED (robot_sf#8163, 2026-09-01) - NEEDS-REVIEW -->

# v0.0.6 supported-dependency batch: archive and candidate evidence

This note preserves the bounded evidence for the first 36 exact dependency rows in Issue #8163. SPDX (Software Package Data Exchange) expressions and SBOM (Software Bill of Materials) identities below are observed metadata/provenance facts; they are not legal conclusions, redistribution authorization, release approval, or merge approval.

## Review state and boundary

- **Status:** `blocked_diagnostic_only`; all 36 policy rows are intentionally `pending_review`.
- **Independent reviewer:** `null` (not yet supplied); `reviewed_at: null`. No row is marked `reviewed` before a named independent maintainer records the exact row, two artifacts, archive notices, and candidate receipt.
- **Release surface:** profile `all` only; target Linux `x86_64` / CPython `3.13`; source index `https://pypi.org/simple`.
- **Scope:** exactly the 36 rows in the policy file; no root package, llvmlite control row, rllib/orca/pyrvo2 surface, class-b/c/d/e row, or release-state change is included.
- **Policy:** [`scripts/validation/dependency_license_policy.v1.json`](../../../scripts/validation/dependency_license_policy.v1.json).
- **Machine-readable receipt:** [`dependency_license_batch_2026-09-01.receipt.json`](dependency_license_batch_2026-09-01.receipt.json).

## Reproducibility inputs

- Base ref: `origin/main` at `e2ac1dcf145415552bd083315a1700c87c592eaf`; candidate source commit: `39c1840c16fb53d2b8a881bde8786922f54bc170` (merged current base plus this evidence change).
- `uv.lock` SHA-256: `de2b22de6327164a9abda93f9d0a1fdb38316471f17ab6428ebc52ed7da603c6`.
- Exact archive audit: operator-local `/tmp/issue8163-archive-audit/archive-audit.json`, SHA-256 `3bb2e3cba161be5f27427f42d2a72b12b4b581beed09aab0db3ffeae8c4f0447`, schema `robot-sf.issue-8163-archive-audit.v1`, 36 packages / 72 artifacts / zero audit failures.
- Exact upstream tag checks: operator-local `/tmp/issue8163-archive-audit/upstream-tags.json`, SHA-256 `8de0debdebc5664995a846fbfe4166a84c8a981370688ab3c88cfa4736a489c8`.
- The tag probe recorded one non-fatal GitHub tag-page pagination `404` while enumerating attrs; the exact `25.4.0` tag and notice blobs were separately verified and retained. This warning is not treated as license or release approval.

## Candidate binding receipt

- Bound status: `bound`; repository `ll7/robot_sf_ll7`; source SHA `39c1840c16fb53d2b8a881bde8786922f54bc170`; workflow run `1` attempt `1`.
- Candidate package: `robot_sf` `0.0.6`; selected profile IDs `['all']`; selected SBOM component count `145`.
- Candidate manifest SHA-256: `a0020b8954925c4413a08aa6db3496f6f8e6271d6a9945161ef73347365dc4dc`; SBOM `robot_sf-0.0.6.cyclonedx.json` SHA-256 `8602d598f32f4264e2a8a32088d0a41d9dd07bc19ffcd55780dd1e26531b5ce8`; component-set SHA-256 `aa6c94073eb332e09a4fab6433f308f1c41b46e87b08704cadb1dfef6a9ad186`.
- Candidate materialization: commit `5c7f7165e0ff844095625b5b91804fcc230d1cc1`, tree `6b2fa342920a7c4d7b733906eb26acc11cd16632`, policy `scripts/validation/software_candidate_policy.v1.json` SHA-256 `8e02808a9e342e7c50aff6aa1a038de4b00f9e60a875b0f0b2ddd5d036888d6d`.
- Candidate members (all hashes are retained in the receipt): `robot_sf-0.0.6-py3-none-any.whl`, `robot_sf-0.0.6.tar.gz`, CycloneDX SBOM, and provenance record.
- The SBOM was exported with the supported extras and normalized to the selected profile-all lock closure before assembly. The raw export contained ten target-inactive/unselected universal-lock identities; those were excluded, with no selected component omitted. This is diagnostic candidate evidence, not a release pass; the receipt records the exact transformation and report digest.
- Strict report: operator-local `/tmp/issue8163-final-candidate.IVUGJR/inventory-final-strict-all.json`, exit code `2`, `candidate_bound=true`, `surface.profile_ids=["all"]`, unresolved count `181`; the final report digest is recorded in the PR handoff because embedding it here would make the freshness input self-referential.

## Exact metadata and frozen artifacts

| Package | Version | Observed SPDX | `Requires-Python` | Exact PyPI JSON | Upstream tag |
| --- | --- | --- | --- | --- | --- |
| `absl-py` | `2.4.0` | `Apache-2.0` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/absl-py/2.4.0/json>) | [`v2.4.0`](<https://github.com/abseil/abseil-py/tree/v2.4.0>) |
| `alembic` | `1.18.4` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/alembic/1.18.4/json>) | [`rel_1_18_4`](<https://github.com/sqlalchemy/alembic/tree/rel_1_18_4>) |
| `attrs` | `25.4.0` | `MIT` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/attrs/25.4.0/json>) | [`25.4.0`](<https://github.com/python-attrs/attrs/tree/25.4.0>) |
| `click` | `8.3.1` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/click/8.3.1/json>) | [`8.3.1`](<https://github.com/pallets/click/tree/8.3.1>) |
| `cma` | `4.4.4` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | (not specified in exact metadata) | [`JSON`](<https://pypi.org/pypi/cma/4.4.4/json>) | [`r4.4.4`](<https://github.com/CMA-ES/pycma/tree/r4.4.4>) |
| `cyclopts` | `4.18.0` | `Apache-2.0` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/cyclopts/4.18.0/json>) | [`v4.18.0`](<https://github.com/BrianPugh/cyclopts/tree/v4.18.0>) |
| `fsspec` | `2026.2.0` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/fsspec/2026.2.0/json>) | [`2026.2.0`](<https://github.com/fsspec/filesystem_spec/tree/2026.2.0>) |
| `geopandas` | `1.1.4` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/geopandas/1.1.4/json>) | [`v1.1.4`](<https://github.com/geopandas/geopandas/tree/v1.1.4>) |
| `idna` | `3.11` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.8` | [`JSON`](<https://pypi.org/pypi/idna/3.11/json>) | [`v3.11`](<https://github.com/kjd/idna/tree/v3.11>) |
| `imageio` | `2.37.2` | `BSD-2-Clause` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/imageio/2.37.2/json>) | [`v2.37.2`](<https://github.com/imageio/imageio/tree/v2.37.2>) |
| `joblib` | `1.5.3` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/joblib/1.5.3/json>) | [`1.5.3`](<https://github.com/joblib/joblib/tree/1.5.3>) |
| `jsonschema` | `4.26.0` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/jsonschema/4.26.0/json>) | [`v4.26.0`](<https://github.com/python-jsonschema/jsonschema/tree/v4.26.0>) |
| `jsonschema-specifications` | `2025.9.1` | `MIT` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/jsonschema-specifications/2025.9.1/json>) | [`v2025.9.1`](<https://github.com/python-jsonschema/jsonschema-specifications/tree/v2025.9.1>) |
| `lazy-loader` | `0.5` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/lazy-loader/0.5/json>) | [`v0.5`](<https://github.com/scientific-python/lazy-loader/tree/v0.5>) |
| `markdown` | `3.10.2` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/markdown/3.10.2/json>) | [`3.10.2`](<https://github.com/Python-Markdown/markdown/tree/3.10.2>) |
| `narwhals` | `2.22.1` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/narwhals/2.22.1/json>) | [`v2.22.1`](<https://github.com/narwhals-dev/narwhals/tree/v2.22.1>) |
| `networkx` | `3.6.1` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `!=3.14.1,>=3.11` | [`JSON`](<https://pypi.org/pypi/networkx/3.6.1/json>) | [`networkx-3.6.1`](<https://github.com/networkx/networkx/tree/networkx-3.6.1>) |
| `opentelemetry-api` | `1.44.0` | `Apache-2.0` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/opentelemetry-api/1.44.0/json>) | [`v1.44.0`](<https://github.com/open-telemetry/opentelemetry-python/tree/v1.44.0>) |
| `opt-einsum` | `3.4.0` | `MIT` (sdist + wheel `License-Expression`) | `>=3.8` | [`JSON`](<https://pypi.org/pypi/opt-einsum/3.4.0/json>) | [`v3.4.0`](<https://github.com/dgasmith/opt_einsum/tree/v3.4.0>) |
| `osmnx` | `2.1.1` | `MIT` (sdist + wheel `License-Expression`) | `>=3.11` | [`JSON`](<https://pypi.org/pypi/osmnx/2.1.1/json>) | [`v2.1.1`](<https://github.com/gboeing/osmnx/tree/v2.1.1>) |
| `platformdirs` | `4.5.1` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/platformdirs/4.5.1/json>) | [`4.5.1`](<https://github.com/tox-dev/platformdirs/tree/4.5.1>) |
| `pooch` | `1.9.0` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/pooch/1.9.0/json>) | [`v1.9.0`](<https://github.com/fatiando/pooch/tree/v1.9.0>) |
| `proglog` | `0.1.12` | `MIT` (sdist + wheel `License-Expression`) | (not specified in exact metadata) | [`JSON`](<https://pypi.org/pypi/proglog/0.1.12/json>) | [`v0.1.12`](<https://github.com/Edinburgh-Genome-Foundry/proglog/tree/v0.1.12>) |
| `pydantic` | `2.12.5` | `MIT` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/pydantic/2.12.5/json>) | [`v2.12.5`](<https://github.com/pydantic/pydantic/tree/v2.12.5>) |
| `pyparsing` | `3.3.2` | `MIT` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/pyparsing/3.3.2/json>) | [`3.3.2`](<https://github.com/pyparsing/pyparsing/tree/3.3.2>) |
| `python-dotenv` | `1.2.1` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/python-dotenv/1.2.1/json>) | [`v1.2.1`](<https://github.com/theskumar/python-dotenv/tree/v1.2.1>) |
| `pyvista` | `0.48.4` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/pyvista/0.48.4/json>) | [`v0.48.4`](<https://github.com/pyvista/pyvista/tree/v0.48.4>) |
| `referencing` | `0.37.0` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/referencing/0.37.0/json>) | [`v0.37.0`](<https://github.com/python-jsonschema/referencing/tree/v0.37.0>) |
| `rich-rst` | `2.0.1` | `MIT` (sdist + wheel `License-Expression`) | (not specified in exact metadata) | [`JSON`](<https://pypi.org/pypi/rich-rst/2.0.1/json>) | [`v2.0.1`](<https://github.com/wasi-master/rich-rst/tree/v2.0.1>) |
| `scooby` | `0.11.2` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/scooby/0.11.2/json>) | [`v0.11.2`](<https://github.com/banesullivan/scooby/tree/v0.11.2>) |
| `setuptools` | `83.0.0` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/setuptools/83.0.0/json>) | [`v83.0.0`](<https://github.com/pypa/setuptools/tree/v83.0.0>) |
| `termcolor` | `3.3.0` | `MIT` (sdist + wheel `License-Expression`) | `>=3.10` | [`JSON`](<https://pypi.org/pypi/termcolor/3.3.0/json>) | [`3.3.0`](<https://github.com/termcolor/termcolor/tree/3.3.0>) |
| `typing-inspection` | `0.4.2` | `MIT` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/typing-inspection/0.4.2/json>) | [`v0.4.2`](<https://github.com/pydantic/typing-inspection/tree/v0.4.2>) |
| `urllib3` | `2.6.3` | `MIT` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/urllib3/2.6.3/json>) | [`2.6.3`](<https://github.com/urllib3/urllib3/tree/2.6.3>) |
| `werkzeug` | `3.1.5` | `BSD-3-Clause` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/werkzeug/3.1.5/json>) | [`3.1.5`](<https://github.com/pallets/werkzeug/tree/3.1.5>) |
| `wheel` | `0.46.3` | `MIT` (sdist + wheel `License-Expression`) | `>=3.9` | [`JSON`](<https://pypi.org/pypi/wheel/0.46.3/json>) | [`0.46.3`](<https://github.com/pypa/wheel/tree/0.46.3>) |

The policy records `python_requires: null` only where the exact PyPI response does not specify `Requires-Python`; no wildcard or inferred range is substituted.

| Package | Frozen sdist (URL / SHA-256 / bytes) | Frozen wheel (URL / tags / SHA-256 / bytes) |
| --- | --- | --- |
| `absl-py` | [`absl_py-2.4.0.tar.gz`](<https://files.pythonhosted.org/packages/64/c7/8de93764ad66968d19329a7e0c147a2bb3c7054c554d4a119111b8f9440f/absl_py-2.4.0.tar.gz>)<br>`8c6af82722b35cf71e0f4d1d47dcaebfff286e27110a99fc359349b247dfb5d4`<br>116543 bytes | [`absl_py-2.4.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/18/a6/907a406bb7d359e6a63f99c313846d9eec4f7e6f7437809e03aa00fa3074/absl_py-2.4.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`88476fd881ca8aab94ffa78b7b6c632a782ab3ba1cd19c9bd423abc4fb4cd28d`<br>135750 bytes |
| `alembic` | [`alembic-1.18.4.tar.gz`](<https://files.pythonhosted.org/packages/94/13/8b084e0f2efb0275a1d534838844926f798bd766566b1375174e2448cd31/alembic-1.18.4.tar.gz>)<br>`cb6e1fd84b6174ab8dbb2329f86d631ba9559dd78df550b57804d607672cedbc`<br>2056725 bytes | [`alembic-1.18.4-py3-none-any.whl`](<https://files.pythonhosted.org/packages/d2/29/6533c317b74f707ea28f8d633734dbda2119bbadfc61b2f3640ba835d0f7/alembic-1.18.4-py3-none-any.whl>)<br>`py3 / none / any`<br>`a5ed4adcf6d8a4cb575f3d759f071b03cd6e5c7618eb796cb52497be25bfe19a`<br>263893 bytes |
| `attrs` | [`attrs-25.4.0.tar.gz`](<https://files.pythonhosted.org/packages/6b/5c/685e6633917e101e5dcb62b9dd76946cbb57c26e133bae9e0cd36033c0a9/attrs-25.4.0.tar.gz>)<br>`16d5969b87f0859ef33a48b35d55ac1be6e42ae49d5e853b597db70c35c57e11`<br>934251 bytes | [`attrs-25.4.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/3a/2a/7cc015f5b9f5db42b7d48157e23356022889fc354a2813c15934b7cb5c0e/attrs-25.4.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`adcf7e2a1fb3b36ac48d97835bb6d8ade15b8dcce26aba8bf1d14847b57a3373`<br>67615 bytes |
| `click` | [`click-8.3.1.tar.gz`](<https://files.pythonhosted.org/packages/3d/fa/656b739db8587d7b5dfa22e22ed02566950fbfbcdc20311993483657a5c0/click-8.3.1.tar.gz>)<br>`12ff4785d337a1bb490bb7e9c2b1ee5da3112e94a8622f26a6c77f5d2fc6842a`<br>295065 bytes | [`click-8.3.1-py3-none-any.whl`](<https://files.pythonhosted.org/packages/98/78/01c019cdb5d6498122777c1a43056ebb3ebfeef2076d9d026bfe15583b2b/click-8.3.1-py3-none-any.whl>)<br>`py3 / none / any`<br>`981153a64e25f12d547d3426c367a4857371575ee7ad18df2a6183ab0545b2a6`<br>108274 bytes |
| `cma` | [`cma-4.4.4.tar.gz`](<https://files.pythonhosted.org/packages/67/ac/8c27720838e293898671f01b5c452236a0c74f4799a3f2d5fcccbbf50d71/cma-4.4.4.tar.gz>)<br>`632bd654b5dce04c0eaa3166679d3e4773ce7a79eab7934e7f363c341b9a8170`<br>316645 bytes | [`cma-4.4.4-py3-none-any.whl`](<https://files.pythonhosted.org/packages/d4/d4/ec46cedab6a6145e21768baa8110db3e2e836a320d8499e4ef18bc894e61/cma-4.4.4-py3-none-any.whl>)<br>`py3 / none / any`<br>`edb6d02eb2aac2d54650f16a8f0c70711ff17445957de7c9de92ff7fd4b7ef38`<br>328311 bytes |
| `cyclopts` | [`cyclopts-4.18.0.tar.gz`](<https://files.pythonhosted.org/packages/9a/19/5c438b428b3dca208eb920804dc16aeb3ca1e85d6163d17e8fb0785ead19/cyclopts-4.18.0.tar.gz>)<br>`fb7b730f21932e0784f7e54462df0447aaa1fbf034d65b605bd8a25dce58b188`<br>182821 bytes | [`cyclopts-4.18.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/5d/9f/b67f14c6b686ca90d317c0358f1a52ae171f43f83c808683fae3ba0b1f90/cyclopts-4.18.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`18ba2912e48e890a97ecc8a05c9beddf30a407b43f4e14cccfd40efddc41f029`<br>221216 bytes |
| `fsspec` | [`fsspec-2026.2.0.tar.gz`](<https://files.pythonhosted.org/packages/51/7c/f60c259dcbf4f0c47cc4ddb8f7720d2dcdc8888c8e5ad84c73ea4531cc5b/fsspec-2026.2.0.tar.gz>)<br>`6544e34b16869f5aacd5b90bdf1a71acb37792ea3ddf6125ee69a22a53fb8bff`<br>313441 bytes | [`fsspec-2026.2.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/e6/ab/fb21f4c939bb440104cc2b396d3be1d9b7a9fd3c6c2a53d98c45b3d7c954/fsspec-2026.2.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`98de475b5cb3bd66bedd5c4679e87b4fdfe1a3bf4d707b151b3c07e58c9a2437`<br>202505 bytes |
| `geopandas` | [`geopandas-1.1.4.tar.gz`](<https://files.pythonhosted.org/packages/39/a9/0478b74a33aea66827c10baaf6025cb2a17e44b1c117533eecd3e977bb75/geopandas-1.1.4.tar.gz>)<br>`06f2890a07e1a239047daa14b486a7c6ae5ce82dcf7405e13c46bf31f5d0dd66`<br>337445 bytes | [`geopandas-1.1.4-py3-none-any.whl`](<https://files.pythonhosted.org/packages/59/a8/bd530cc264e62ddbc1d1bb7225823992e6f2432c664693e9281bb6b9c359/geopandas-1.1.4-py3-none-any.whl>)<br>`py3 / none / any`<br>`1a0c459cbdb1537cd154dafe6174be20d1760844b7f1c967dc8520b180f2e773`<br>343292 bytes |
| `idna` | [`idna-3.11.tar.gz`](<https://files.pythonhosted.org/packages/6f/6d/0703ccc57f3a7233505399edb88de3cbd678da106337b9fcde432b65ed60/idna-3.11.tar.gz>)<br>`795dafcc9c04ed0c1fb032c2aa73654d8e8c5023a7df64a53f39190ada629902`<br>194582 bytes | [`idna-3.11-py3-none-any.whl`](<https://files.pythonhosted.org/packages/0e/61/66938bbb5fc52dbdf84594873d5b51fb1f7c7794e9c0f5bd885f30bc507b/idna-3.11-py3-none-any.whl>)<br>`py3 / none / any`<br>`771a87f49d9defaf64091e6e6fe9c18d4833f140bd19464795bc32d966ca37ea`<br>71008 bytes |
| `imageio` | [`imageio-2.37.2.tar.gz`](<https://files.pythonhosted.org/packages/a3/6f/606be632e37bf8d05b253e8626c2291d74c691ddc7bcdf7d6aaf33b32f6a/imageio-2.37.2.tar.gz>)<br>`0212ef2727ac9caa5ca4b2c75ae89454312f440a756fcfc8ef1993e718f50f8a`<br>389600 bytes | [`imageio-2.37.2-py3-none-any.whl`](<https://files.pythonhosted.org/packages/fb/fe/301e0936b79bcab4cacc7548bf2853fc28dced0a578bab1f7ef53c9aa75b/imageio-2.37.2-py3-none-any.whl>)<br>`py3 / none / any`<br>`ad9adfb20335d718c03de457358ed69f141021a333c40a53e57273d8a5bd0b9b`<br>317646 bytes |
| `joblib` | [`joblib-1.5.3.tar.gz`](<https://files.pythonhosted.org/packages/41/f2/d34e8b3a08a9cc79a50b2208a93dce981fe615b64d5a4d4abee421d898df/joblib-1.5.3.tar.gz>)<br>`8561a3269e6801106863fd0d6d84bb737be9e7631e33aaed3fb9ce5953688da3`<br>331603 bytes | [`joblib-1.5.3-py3-none-any.whl`](<https://files.pythonhosted.org/packages/7b/91/984aca2ec129e2757d1e4e3c81c3fcda9d0f85b74670a094cc443d9ee949/joblib-1.5.3-py3-none-any.whl>)<br>`py3 / none / any`<br>`5fc3c5039fc5ca8c0276333a188bbd59d6b7ab37fe6632daa76bc7f9ec18e713`<br>309071 bytes |
| `jsonschema` | [`jsonschema-4.26.0.tar.gz`](<https://files.pythonhosted.org/packages/b3/fc/e067678238fa451312d4c62bf6e6cf5ec56375422aee02f9cb5f909b3047/jsonschema-4.26.0.tar.gz>)<br>`0c26707e2efad8aa1bfc5b7ce170f3fccc2e4918ff85989ba9ffa9facb2be326`<br>366583 bytes | [`jsonschema-4.26.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/69/90/f63fb5873511e014207a475e2bb4e8b2e570d655b00ac19a9a0ca0a385ee/jsonschema-4.26.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`d489f15263b8d200f8387e64b4c3a75f06629559fb73deb8fdfb525f2dab50ce`<br>90630 bytes |
| `jsonschema-specifications` | [`jsonschema_specifications-2025.9.1.tar.gz`](<https://files.pythonhosted.org/packages/19/74/a633ee74eb36c44aa6d1095e7cc5569bebf04342ee146178e2d36600708b/jsonschema_specifications-2025.9.1.tar.gz>)<br>`b540987f239e745613c7a9176f3edb72b832a4ac465cf02712288397832b5e8d`<br>32855 bytes | [`jsonschema_specifications-2025.9.1-py3-none-any.whl`](<https://files.pythonhosted.org/packages/41/45/1a4ed80516f02155c51f51e8cedb3c1902296743db0bbc66608a0db2814f/jsonschema_specifications-2025.9.1-py3-none-any.whl>)<br>`py3 / none / any`<br>`98802fee3a11ee76ecaca44429fda8a41bff98b00a0f2838151b113f210cc6fe`<br>18437 bytes |
| `lazy-loader` | [`lazy_loader-0.5.tar.gz`](<https://files.pythonhosted.org/packages/49/ac/21a1f8aa3777f5658576777ea76bfb124b702c520bbe90edf4ae9915eafa/lazy_loader-0.5.tar.gz>)<br>`717f9179a0dbed357012ddad50a5ad3d5e4d9a0b8712680d4e687f5e6e6ed9b3`<br>15294 bytes | [`lazy_loader-0.5-py3-none-any.whl`](<https://files.pythonhosted.org/packages/8a/a1/8d812e53a5da1687abb10445275d41a8b13adb781bbf7196ddbcf8d88505/lazy_loader-0.5-py3-none-any.whl>)<br>`py3 / none / any`<br>`ab0ea149e9c554d4ffeeb21105ac60bed7f3b4fd69b1d2360a4add51b170b005`<br>8044 bytes |
| `markdown` | [`markdown-3.10.2.tar.gz`](<https://files.pythonhosted.org/packages/2b/f4/69fa6ed85ae003c2378ffa8f6d2e3234662abd02c10d216c0ba96081a238/markdown-3.10.2.tar.gz>)<br>`994d51325d25ad8aa7ce4ebaec003febcce822c3f8c911e3b17c52f7f589f950`<br>368805 bytes | [`markdown-3.10.2-py3-none-any.whl`](<https://files.pythonhosted.org/packages/de/1f/77fa3081e4f66ca3576c896ae5d31c3002ac6607f9747d2e3aa49227e464/markdown-3.10.2-py3-none-any.whl>)<br>`py3 / none / any`<br>`e91464b71ae3ee7afd3017d9f358ef0baf158fd9a298db92f1d4761133824c36`<br>108180 bytes |
| `narwhals` | [`narwhals-2.22.1.tar.gz`](<https://files.pythonhosted.org/packages/62/3c/c4ef2164a71c1a63d7f1ae411c4082c5fa872405106db60a4b7114989ad7/narwhals-2.22.1.tar.gz>)<br>`d62920805a0a43b7ff8b54b0c0d3142d796f8a9301836ada37e573d6a33cbcd9`<br>647493 bytes | [`narwhals-2.22.1-py3-none-any.whl`](<https://files.pythonhosted.org/packages/48/ca/36339329c4604adbcc99c899b7eb1ce1a555c499b6a6860757dc9bfed36d/narwhals-2.22.1-py3-none-any.whl>)<br>`py3 / none / any`<br>`60567d774edf77db53906f89d9fbd164e66e56d66d388e1e6990f17ac33cfb53`<br>454815 bytes |
| `networkx` | [`networkx-3.6.1.tar.gz`](<https://files.pythonhosted.org/packages/6a/51/63fe664f3908c97be9d2e4f1158eb633317598cfa6e1fc14af5383f17512/networkx-3.6.1.tar.gz>)<br>`26b7c357accc0c8cde558ad486283728b65b6a95d85ee1cd66bafab4c8168509`<br>2517025 bytes | [`networkx-3.6.1-py3-none-any.whl`](<https://files.pythonhosted.org/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl>)<br>`py3 / none / any`<br>`d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762`<br>2068504 bytes |
| `opentelemetry-api` | [`opentelemetry_api-1.44.0.tar.gz`](<https://files.pythonhosted.org/packages/ee/8b/aa9e2d8b8dfa7c946f7dec5d1f8f6ba8eca062f43509a06bdb5ce93d26c0/opentelemetry_api-1.44.0.tar.gz>)<br>`67647e5e9566edcf421166fdf022b3537f818635daa852b289e34604dc6fb33a`<br>72406 bytes | [`opentelemetry_api-1.44.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/ca/6f/a04e900f465ff3221ccc395522503e2d10e79fa21f2723c8e177aae1e0d1/opentelemetry_api-1.44.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`94b98c893a91b88657eaac1e3ba89618cdb85be6918196705354f34728b2cdef`<br>60018 bytes |
| `opt-einsum` | [`opt_einsum-3.4.0.tar.gz`](<https://files.pythonhosted.org/packages/8c/b9/2ac072041e899a52f20cf9510850ff58295003aa75525e58343591b0cbfb/opt_einsum-3.4.0.tar.gz>)<br>`96ca72f1b886d148241348783498194c577fa30a8faac108586b14f1ba4473ac`<br>63004 bytes | [`opt_einsum-3.4.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/23/cd/066e86230ae37ed0be70aae89aabf03ca8d9f39c8aea0dec8029455b5540/opt_einsum-3.4.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`69bb92469f86a1565195ece4ac0323943e83477171b91d24c35afe028a90d7cd`<br>71932 bytes |
| `osmnx` | [`osmnx-2.1.1.tar.gz`](<https://files.pythonhosted.org/packages/0c/6d/42279fb31b09f0532ec4de90604ee8078adad66b89865887cdf014f651f1/osmnx-2.1.1.tar.gz>)<br>`0785166487d4d8da5a551dac3d1562fe114fb39f3bf66f261c8b4d60baee9295`<br>90368 bytes | [`osmnx-2.1.1-py3-none-any.whl`](<https://files.pythonhosted.org/packages/ed/08/450c5c707493cf3a1da91050cf6b9b92e5279676f28e85d85b81d9e46f85/osmnx-2.1.1-py3-none-any.whl>)<br>`py3 / none / any`<br>`6680c98312a8c327dea01da8ec59328bf1d7b0e0cab42db6ce500d0cb83b87e5`<br>104711 bytes |
| `platformdirs` | [`platformdirs-4.5.1.tar.gz`](<https://files.pythonhosted.org/packages/cf/86/0248f086a84f01b37aaec0fa567b397df1a119f73c16f6c7a9aac73ea309/platformdirs-4.5.1.tar.gz>)<br>`61d5cdcc6065745cdd94f0f878977f8de9437be93de97c1c12f853c9c0cdcbda`<br>21715 bytes | [`platformdirs-4.5.1-py3-none-any.whl`](<https://files.pythonhosted.org/packages/cb/28/3bfe2fa5a7b9c46fe7e13c97bda14c895fb10fa2ebf1d0abb90e0cea7ee1/platformdirs-4.5.1-py3-none-any.whl>)<br>`py3 / none / any`<br>`d03afa3963c806a9bed9d5125c8f4cb2fdaf74a55ab60e5d59b3fde758104d31`<br>18731 bytes |
| `pooch` | [`pooch-1.9.0.tar.gz`](<https://files.pythonhosted.org/packages/83/43/85ef45e8b36c6a48546af7b266592dc32d7f67837a6514d111bced6d7d75/pooch-1.9.0.tar.gz>)<br>`de46729579b9857ffd3e741987a2f6d5e0e03219892c167c6578c0091fb511ed`<br>61788 bytes | [`pooch-1.9.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/2a/2d/d4bf65e47cea8ff2c794a600c4fd1273a7902f268757c531e0ee9f18aa58/pooch-1.9.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`f265597baa9f760d25ceb29d0beb8186c243d6607b0f60b83ecf14078dbc703b`<br>67175 bytes |
| `proglog` | [`proglog-0.1.12.tar.gz`](<https://files.pythonhosted.org/packages/c2/af/c108866c452eda1132f3d6b3cb6be2ae8430c97e9309f38ca9dbd430af37/proglog-0.1.12.tar.gz>)<br>`361ee074721c277b89b75c061336cb8c5f287c92b043efa562ccf7866cda931c`<br>8794 bytes | [`proglog-0.1.12-py3-none-any.whl`](<https://files.pythonhosted.org/packages/c1/1b/f7ea6cde25621cd9236541c66ff018f4268012a534ec31032bcb187dc5e7/proglog-0.1.12-py3-none-any.whl>)<br>`py3 / none / any`<br>`ccaafce51e80a81c65dc907a460c07ccb8ec1f78dc660cfd8f9ec3a22f01b84c`<br>6337 bytes |
| `pydantic` | [`pydantic-2.12.5.tar.gz`](<https://files.pythonhosted.org/packages/69/44/36f1a6e523abc58ae5f928898e4aca2e0ea509b5aa6f6f392a5d882be928/pydantic-2.12.5.tar.gz>)<br>`4d351024c75c0f085a9febbb665ce8c0c6ec5d30e903bdb6394b7ede26aebb49`<br>821591 bytes | [`pydantic-2.12.5-py3-none-any.whl`](<https://files.pythonhosted.org/packages/5a/87/b70ad306ebb6f9b585f114d0ac2137d792b48be34d732d60e597c2f8465a/pydantic-2.12.5-py3-none-any.whl>)<br>`py3 / none / any`<br>`e561593fccf61e8a20fc46dfc2dfe075b8be7d0188df33f221ad1f0139180f9d`<br>463580 bytes |
| `pyparsing` | [`pyparsing-3.3.2.tar.gz`](<https://files.pythonhosted.org/packages/f3/91/9c6ee907786a473bf81c5f53cf703ba0957b23ab84c264080fb5a450416f/pyparsing-3.3.2.tar.gz>)<br>`c777f4d763f140633dcb6d8a3eda953bf7a214dc4eff598413c070bcdc117cbc`<br>6851574 bytes | [`pyparsing-3.3.2-py3-none-any.whl`](<https://files.pythonhosted.org/packages/10/bd/c038d7cc38edc1aa5bf91ab8068b63d4308c66c4c8bb3cbba7dfbc049f9c/pyparsing-3.3.2-py3-none-any.whl>)<br>`py3 / none / any`<br>`850ba148bd908d7e2411587e247a1e4f0327839c40e2e5e6d05a007ecc69911d`<br>122781 bytes |
| `python-dotenv` | [`python_dotenv-1.2.1.tar.gz`](<https://files.pythonhosted.org/packages/f0/26/19cadc79a718c5edbec86fd4919a6b6d3f681039a2f6d66d14be94e75fb9/python_dotenv-1.2.1.tar.gz>)<br>`42667e897e16ab0d66954af0e60a9caa94f0fd4ecf3aaf6d2d260eec1aa36ad6`<br>44221 bytes | [`python_dotenv-1.2.1-py3-none-any.whl`](<https://files.pythonhosted.org/packages/14/1b/a298b06749107c305e1fe0f814c6c74aea7b2f1e10989cb30f544a1b3253/python_dotenv-1.2.1-py3-none-any.whl>)<br>`py3 / none / any`<br>`b81ee9561e9ca4004139c6cbba3a238c32b03e4894671e181b671e8cb8425d61`<br>21230 bytes |
| `pyvista` | [`pyvista-0.48.4.tar.gz`](<https://files.pythonhosted.org/packages/a6/11/554ae45f79d45039c733d93acb36a433b71e8c63a79bbf2f414b3685de18/pyvista-0.48.4.tar.gz>)<br>`c639dad1bddff5e366d77371f66f783f6e6a0581446810a66439902222d8db07`<br>2581423 bytes | [`pyvista-0.48.4-py3-none-any.whl`](<https://files.pythonhosted.org/packages/4d/91/696d869e4df2e25a5b201a69ce69a2204d37fad3e90c2b731f7b3f1d7c68/pyvista-0.48.4-py3-none-any.whl>)<br>`py3 / none / any`<br>`a46eda178e10e279afda550c341676a82dcee607c86db74565fa455ac0bd23e2`<br>2629373 bytes |
| `referencing` | [`referencing-0.37.0.tar.gz`](<https://files.pythonhosted.org/packages/22/f5/df4e9027acead3ecc63e50fe1e36aca1523e1719559c499951bb4b53188f/referencing-0.37.0.tar.gz>)<br>`44aefc3142c5b842538163acb373e24cce6632bd54bdb01b21ad5863489f50d8`<br>78036 bytes | [`referencing-0.37.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/2c/58/ca301544e1fa93ed4f80d724bf5b194f6e4b945841c5bfd555878eea9fcb/referencing-0.37.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`381329a9f99628c9069361716891d34ad94af76e461dcb0335825aecc7692231`<br>26766 bytes |
| `rich-rst` | [`rich_rst-2.0.1.tar.gz`](<https://files.pythonhosted.org/packages/57/56/3191bae66b08ccc637ea8120426068bcb361cc323c96404c310886937067/rich_rst-2.0.1.tar.gz>)<br>`cbe236ed0901d1ec8427cc6a50bf0a34353ba28ad014dc24def68bfe7f3b9e68`<br>300570 bytes | [`rich_rst-2.0.1-py3-none-any.whl`](<https://files.pythonhosted.org/packages/a0/3d/55c17d3ebdf3cd81356002afe5bef9bb8af631db2819785b6eac845b925b/rich_rst-2.0.1-py3-none-any.whl>)<br>`py3 / none / any`<br>`7ee15f345ce25fa02b582c272a6cdbaf0c21243e38061cea273cff659bf3ef61`<br>272922 bytes |
| `scooby` | [`scooby-0.11.2.tar.gz`](<https://files.pythonhosted.org/packages/5b/06/9a8600207fd72a29ee965e9a4c61b750cc3fa106768f14a7b3ee3e36cb61/scooby-0.11.2.tar.gz>)<br>`0575c73636ec4c2587bea1f8a038798ddcb249e02067fae897dac3bf4f4e444d`<br>242928 bytes | [`scooby-0.11.2-py3-none-any.whl`](<https://files.pythonhosted.org/packages/99/bc/1173f502f1870e3bae81c148326c5cbcc19ec77df79a9aaf17a59911355c/scooby-0.11.2-py3-none-any.whl>)<br>`py3 / none / any`<br>`f34c36bbee749b2c55816a080521f216d88304e635017e911c12249607d38c49`<br>20142 bytes |
| `setuptools` | [`setuptools-83.0.0.tar.gz`](<https://files.pythonhosted.org/packages/34/26/f5d29e25ffdb535afef2d35cdb55b325298f96debd670da4c325e08d70f4/setuptools-83.0.0.tar.gz>)<br>`025bccbbf0fa05b6192bc64ae1e7b16e001fd6d6d4d5de03c97b1c1ade523bef`<br>1154254 bytes | [`setuptools-83.0.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/5d/40/e1e72872c6354b306daef1703549e8e83b4d43cfea356311bf722a043752/setuptools-83.0.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3`<br>1008090 bytes |
| `termcolor` | [`termcolor-3.3.0.tar.gz`](<https://files.pythonhosted.org/packages/46/79/cf31d7a93a8fdc6aa0fbb665be84426a8c5a557d9240b6239e9e11e35fc5/termcolor-3.3.0.tar.gz>)<br>`348871ca648ec6a9a983a13ab626c0acce02f515b9e1983332b17af7979521c5`<br>14434 bytes | [`termcolor-3.3.0-py3-none-any.whl`](<https://files.pythonhosted.org/packages/33/d1/8bb87d21e9aeb323cc03034f5eaf2c8f69841e40e4853c2627edf8111ed3/termcolor-3.3.0-py3-none-any.whl>)<br>`py3 / none / any`<br>`cf642efadaf0a8ebbbf4bc7a31cec2f9b5f21a9f726f4ccbb08192c9c26f43a5`<br>7734 bytes |
| `typing-inspection` | [`typing_inspection-0.4.2.tar.gz`](<https://files.pythonhosted.org/packages/55/e3/70399cb7dd41c10ac53367ae42139cf4b1ca5f36bb3dc6c9d33acdb43655/typing_inspection-0.4.2.tar.gz>)<br>`ba561c48a67c5958007083d386c3295464928b01faa735ab8547c5692e87f464`<br>75949 bytes | [`typing_inspection-0.4.2-py3-none-any.whl`](<https://files.pythonhosted.org/packages/dc/9b/47798a6c91d8bdb567fe2698fe81e0c6b7cb7ef4d13da4114b41d239f65d/typing_inspection-0.4.2-py3-none-any.whl>)<br>`py3 / none / any`<br>`4ed1cacbdc298c220f1bd249ed5287caa16f34d44ef4e9c3d0cbad5b521545e7`<br>14611 bytes |
| `urllib3` | [`urllib3-2.6.3.tar.gz`](<https://files.pythonhosted.org/packages/c7/24/5f1b3bdffd70275f6661c76461e25f024d5a38a46f04aaca912426a2b1d3/urllib3-2.6.3.tar.gz>)<br>`1b62b6884944a57dbe321509ab94fd4d3b307075e0c2eae991ac71ee15ad38ed`<br>435556 bytes | [`urllib3-2.6.3-py3-none-any.whl`](<https://files.pythonhosted.org/packages/39/08/aaaad47bc4e9dc8c725e68f9d04865dbcb2052843ff09c97b08904852d84/urllib3-2.6.3-py3-none-any.whl>)<br>`py3 / none / any`<br>`bf272323e553dfb2e87d9bfd225ca7b0f467b919d7bbd355436d3fd37cb0acd4`<br>131584 bytes |
| `werkzeug` | [`werkzeug-3.1.5.tar.gz`](<https://files.pythonhosted.org/packages/5a/70/1469ef1d3542ae7c2c7b72bd5e3a4e6ee69d7978fa8a3af05a38eca5becf/werkzeug-3.1.5.tar.gz>)<br>`6a548b0e88955dd07ccb25539d7d0cc97417ee9e179677d22c7041c8f078ce67`<br>864754 bytes | [`werkzeug-3.1.5-py3-none-any.whl`](<https://files.pythonhosted.org/packages/ad/e4/8d97cca767bcc1be76d16fb76951608305561c6e056811587f36cb1316a8/werkzeug-3.1.5-py3-none-any.whl>)<br>`py3 / none / any`<br>`5111e36e91086ece91f93268bb39b4a35c1e6f1feac762c9c822ded0a4e322dc`<br>225025 bytes |
| `wheel` | [`wheel-0.46.3.tar.gz`](<https://files.pythonhosted.org/packages/89/24/a2eb353a6edac9a0303977c4cb048134959dd2a51b48a269dfc9dde00c8a/wheel-0.46.3.tar.gz>)<br>`e3e79874b07d776c40bd6033f8ddf76a7dad46a7b8aa1b2787a83083519a1803`<br>60605 bytes | [`wheel-0.46.3-py3-none-any.whl`](<https://files.pythonhosted.org/packages/87/22/b76d483683216dde3d67cba61fb2444be8d5be289bf628c13fc0fd90e5f9/wheel-0.46.3-py3-none-any.whl>)<br>`py3 / none / any`<br>`4b399d56c9d9338230118d705d9737a2a468ccca63d5e813e2a4fc7815d8bc4d`<br>30557 bytes |

Every row has exactly one sdist and one wheel selected from the current `uv.lock`; the policy stores the full lock SHA-256, byte size, and wheel tags. The 72 values above were rechecked against the downloaded exact PyPI archives and the lock entries.

## Archive LICENSE / NOTICE inventory

Both exact archives were inspected. The paths below are archive-member paths, including bundled/vendor notice paths. `archive_notice_absences: []` is recorded for every row because each exact archive contained at least one relevant path; no absence was inferred from registry metadata.

### `absl-py` `2.4.0`
- `sdist`: `absl_py-2.4.0/LICENSE`.
- `wheel`: `absl_py-2.4.0.dist-info/licenses/AUTHORS`; `absl_py-2.4.0.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/abseil/abseil-py/blob/v2.4.0/LICENSE>), [`AUTHORS`](<https://github.com/abseil/abseil-py/blob/v2.4.0/AUTHORS>).

### `alembic` `1.18.4`
- `sdist`: `alembic-1.18.4/LICENSE`; `alembic-1.18.4/docs/_static/vendor/fontawesome/6.5.2/LICENSE.txt`; `alembic-1.18.4/docs/_static/vendor/fontawesome/6.5.2/css/all.min.css`; `alembic-1.18.4/docs/_static/vendor/fontawesome/6.5.2/js/all.min.js`; `alembic-1.18.4/docs/_static/vendor/fontawesome/6.5.2/js/all.min.js.LICENSE.txt`.
- `wheel`: `alembic-1.18.4.dist-info/licenses/LICENSE`.
- The sdist Font Awesome 6.5.2 files are generated/injected documentation assets and are not present in the Alembic tag tree; their exact upstream references are [Font Awesome license](<https://github.com/FortAwesome/Font-Awesome/blob/6.5.2/LICENSE.txt>), [CSS](<https://github.com/FortAwesome/Font-Awesome/blob/6.5.2/css/all.min.css>), [JavaScript](<https://github.com/FortAwesome/Font-Awesome/blob/6.5.2/js/all.min.js>), [embedded JS license](<https://github.com/FortAwesome/Font-Awesome/blob/6.5.2/js/all.min.js.LICENSE.txt>), and [Font Awesome free-license page](<https://fontawesome.com/license/free>).
- Present upstream review references: [`LICENSE`](<https://github.com/sqlalchemy/alembic/blob/rel_1_18_4/LICENSE>).

### `attrs` `25.4.0`
- `sdist`: `attrs-25.4.0/LICENSE`; `attrs-25.4.0/docs/license.md`.
- `wheel`: `attrs-25.4.0.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/python-attrs/attrs/blob/25.4.0/LICENSE>), [`license.md`](<https://github.com/python-attrs/attrs/blob/25.4.0/docs/license.md>).

### `click` `8.3.1`
- `sdist`: `click-8.3.1/LICENSE.txt`; `click-8.3.1/docs/license.md`.
- `wheel`: `click-8.3.1.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/pallets/click/blob/8.3.1/LICENSE.txt>), [`license.md`](<https://github.com/pallets/click/blob/8.3.1/docs/license.md>).

### `cma` `4.4.4`
- `sdist`: `cma-4.4.4/LICENSE`.
- `wheel`: `cma-4.4.4.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/CMA-ES/pycma/blob/r4.4.4/LICENSE>).

### `cyclopts` `4.18.0`
- `sdist`: `cyclopts-4.18.0/LICENSE`.
- `wheel`: `cyclopts-4.18.0.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/BrianPugh/cyclopts/blob/v4.18.0/LICENSE>).

### `fsspec` `2026.2.0`
- `sdist`: `fsspec-2026.2.0/LICENSE`; `fsspec-2026.2.0/docs/source/copying.rst`.
- `wheel`: `fsspec-2026.2.0.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/fsspec/filesystem_spec/blob/2026.2.0/LICENSE>), [`copying.rst`](<https://github.com/fsspec/filesystem_spec/blob/2026.2.0/docs/source/copying.rst>).

### `geopandas` `1.1.4`
- `sdist`: `geopandas-1.1.4/LICENSE.txt`.
- `wheel`: `geopandas-1.1.4.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/geopandas/geopandas/blob/v1.1.4/LICENSE.txt>).

### `idna` `3.11`
- `sdist`: `idna-3.11/LICENSE.md`.
- `wheel`: `idna-3.11.dist-info/licenses/LICENSE.md`.
- Present upstream review references: [`LICENSE.md`](<https://github.com/kjd/idna/blob/v3.11/LICENSE.md>).

### `imageio` `2.37.2`
- `sdist`: `imageio-2.37.2/LICENSE`.
- `wheel`: `imageio-2.37.2.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/imageio/imageio/blob/v2.37.2/LICENSE>).

### `joblib` `1.5.3`
- `sdist`: `joblib-1.5.3/LICENSE.txt`.
- `wheel`: `joblib-1.5.3.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/joblib/joblib/blob/1.5.3/LICENSE.txt>).

### `jsonschema` `4.26.0`
- `sdist`: `jsonschema-4.26.0/COPYING`; `jsonschema-4.26.0/json/LICENSE`.
- `wheel`: `jsonschema-4.26.0.dist-info/licenses/COPYING`.
- Present upstream review references: [`COPYING`](<https://github.com/python-jsonschema/jsonschema/blob/v4.26.0/COPYING>), [`LICENSE`](<https://github.com/python-jsonschema/jsonschema/blob/v4.26.0/json/LICENSE>).

### `jsonschema-specifications` `2025.9.1`
- `sdist`: `jsonschema_specifications-2025.9.1/COPYING`.
- `wheel`: `jsonschema_specifications-2025.9.1.dist-info/licenses/COPYING`.
- Present upstream review references: [`COPYING`](<https://github.com/python-jsonschema/jsonschema-specifications/blob/v2025.9.1/COPYING>).

### `lazy-loader` `0.5`
- `sdist`: `lazy_loader-0.5/LICENSE.md`.
- `wheel`: `lazy_loader-0.5.dist-info/licenses/LICENSE.md`.
- Present upstream review references: [`LICENSE.md`](<https://github.com/scientific-python/lazy-loader/blob/v0.5/LICENSE.md>).

### `markdown` `3.10.2`
- `sdist`: `markdown-3.10.2/LICENSE.md`.
- `wheel`: `markdown-3.10.2.dist-info/licenses/LICENSE.md`.
- Present upstream review references: [`LICENSE.md`](<https://github.com/Python-Markdown/markdown/blob/3.10.2/LICENSE.md>).

### `narwhals` `2.22.1`
- `sdist`: `narwhals-2.22.1/LICENSE.md`.
- `wheel`: `narwhals-2.22.1.dist-info/licenses/LICENSE.md`.
- Present upstream review references: [`LICENSE.md`](<https://github.com/narwhals-dev/narwhals/blob/v2.22.1/LICENSE.md>).

### `networkx` `3.6.1`
- `sdist`: `networkx-3.6.1/LICENSE.txt`.
- `wheel`: `networkx-3.6.1.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/networkx/networkx/blob/networkx-3.6.1/LICENSE.txt>).

### `opentelemetry-api` `1.44.0`
- `sdist`: `opentelemetry_api-1.44.0/LICENSE`.
- `wheel`: `opentelemetry_api-1.44.0.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/open-telemetry/opentelemetry-python/blob/v1.44.0/LICENSE>).

### `opt-einsum` `3.4.0`
- `sdist`: `opt_einsum-3.4.0/LICENSE`.
- `wheel`: `opt_einsum-3.4.0.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/dgasmith/opt_einsum/blob/v3.4.0/LICENSE>).

### `osmnx` `2.1.1`
- `sdist`: `osmnx-2.1.1/LICENSE.txt`.
- `wheel`: `osmnx-2.1.1.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/gboeing/osmnx/blob/v2.1.1/LICENSE.txt>).

### `platformdirs` `4.5.1`
- `sdist`: `platformdirs-4.5.1/LICENSE`.
- `wheel`: `platformdirs-4.5.1.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/tox-dev/platformdirs/blob/4.5.1/LICENSE>).

### `pooch` `1.9.0`
- `sdist`: `pooch-1.9.0/LICENSE.txt`.
- `wheel`: `pooch-1.9.0.dist-info/licenses/AUTHORS.md`; `pooch-1.9.0.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/fatiando/pooch/blob/v1.9.0/LICENSE.txt>), [`AUTHORS.md`](<https://github.com/fatiando/pooch/blob/v1.9.0/AUTHORS.md>).

### `proglog` `0.1.12`
- `sdist`: `proglog-0.1.12/LICENSE`.
- `wheel`: `proglog-0.1.12.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/Edinburgh-Genome-Foundry/proglog/blob/v0.1.12/LICENSE>).

### `pydantic` `2.12.5`
- `sdist`: `pydantic-2.12.5/LICENSE`.
- `wheel`: `pydantic-2.12.5.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/pydantic/pydantic/blob/v2.12.5/LICENSE>).

### `pyparsing` `3.3.2`
- `sdist`: `pyparsing-3.3.2/LICENSE`.
- `wheel`: `pyparsing-3.3.2.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/pyparsing/pyparsing/blob/3.3.2/LICENSE>).

### `python-dotenv` `1.2.1`
- `sdist`: `python_dotenv-1.2.1/LICENSE`; `python_dotenv-1.2.1/docs/license.md`.
- `wheel`: `python_dotenv-1.2.1.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/theskumar/python-dotenv/blob/v1.2.1/LICENSE>), [`license.md`](<https://github.com/theskumar/python-dotenv/blob/v1.2.1/docs/license.md>).

### `pyvista` `0.48.4`
- `sdist`: `pyvista-0.48.4/LICENSE`.
- `wheel`: `pyvista-0.48.4.dist-info/licenses/AUTHORS.rst`; `pyvista-0.48.4.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/pyvista/pyvista/blob/v0.48.4/LICENSE>), [`AUTHORS.rst`](<https://github.com/pyvista/pyvista/blob/v0.48.4/AUTHORS.rst>).

### `referencing` `0.37.0`
- `sdist`: `referencing-0.37.0/COPYING`; `referencing-0.37.0/suite/LICENSE`.
- `wheel`: `referencing-0.37.0.dist-info/licenses/COPYING`.
- `suite/LICENSE` is present only in the sdist archive submodule snapshot. The archive `.gitmodules` declares `https://github.com/python-jsonschema/referencing-suite`; no submodule commit is asserted by this receipt. The package tag [tree](<https://github.com/python-jsonschema/referencing/tree/v0.37.0>) and archive member are retained so the unpinned submodule remains review-blocked.
- Present upstream review references: [`COPYING`](<https://github.com/python-jsonschema/referencing/blob/v0.37.0/COPYING>).

### `rich-rst` `2.0.1`
- `sdist`: `rich_rst-2.0.1/LICENSE`; `rich_rst-2.0.1/rich_rst/_vendor/LICENSES.txt`.
- `wheel`: `rich_rst-2.0.1.dist-info/licenses/LICENSE`; `rich_rst-2.0.1.dist-info/licenses/rich_rst/_vendor/LICENSES.txt`; `rich_rst/_vendor/LICENSES.txt`.
- `rich_rst/_vendor/LICENSES.txt` documents a vendored Docutils 0.22.4 subset (public-domain and BSD-2-Clause files); exact references are [Docutils](<https://docutils.sourceforge.io/>) and [COPYING](<https://docutils.sourceforge.io/COPYING.html>). The `vendored` surface remains blocked.
- Present upstream review references: [`LICENSE`](<https://github.com/wasi-master/rich-rst/blob/v2.0.1/LICENSE>), [`LICENSES.txt`](<https://github.com/wasi-master/rich-rst/blob/v2.0.1/rich_rst/_vendor/LICENSES.txt>).

### `scooby` `0.11.2`
- `sdist`: `scooby-0.11.2/LICENSE`.
- `wheel`: `scooby-0.11.2.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/banesullivan/scooby/blob/v0.11.2/LICENSE>).

### `setuptools` `83.0.0`
- `sdist`: `setuptools-83.0.0/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/autocommand-2.2.2.dist-info/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/licenses/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/jaraco.text-4.0.0.dist-info/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/jaraco_context-6.1.0.dist-info/licenses/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/licenses/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/more_itertools-10.8.0.dist-info/licenses/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.APACHE`; `setuptools-83.0.0/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.BSD`; `setuptools-83.0.0/setuptools/_vendor/packaging/licenses/__init__.py`; `setuptools-83.0.0/setuptools/_vendor/packaging/licenses/_spdx.py`; `setuptools-83.0.0/setuptools/_vendor/platformdirs-4.4.0.dist-info/licenses/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/tomli-2.4.0.dist-info/licenses/LICENSE`; `setuptools-83.0.0/setuptools/_vendor/wheel-0.46.3.dist-info/licenses/LICENSE.txt`; `setuptools-83.0.0/setuptools/_vendor/zipp-3.23.0.dist-info/licenses/LICENSE`; `setuptools-83.0.0/setuptools/config/NOTICE`; `setuptools-83.0.0/setuptools/config/_validate_pyproject/NOTICE`.
- `wheel`: `setuptools-83.0.0.dist-info/licenses/LICENSE`; `setuptools/_vendor/autocommand-2.2.2.dist-info/LICENSE`; `setuptools/_vendor/backports.tarfile-1.2.0.dist-info/LICENSE`; `setuptools/_vendor/importlib_metadata-8.7.1.dist-info/licenses/LICENSE`; `setuptools/_vendor/jaraco.text-4.0.0.dist-info/LICENSE`; `setuptools/_vendor/jaraco_context-6.1.0.dist-info/licenses/LICENSE`; `setuptools/_vendor/jaraco_functools-4.4.0.dist-info/licenses/LICENSE`; `setuptools/_vendor/more_itertools-10.8.0.dist-info/licenses/LICENSE`; `setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE`; `setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.APACHE`; `setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.BSD`; `setuptools/_vendor/packaging/licenses/__init__.py`; `setuptools/_vendor/packaging/licenses/_spdx.py`; `setuptools/_vendor/platformdirs-4.4.0.dist-info/licenses/LICENSE`; `setuptools/_vendor/tomli-2.4.0.dist-info/licenses/LICENSE`; `setuptools/_vendor/wheel-0.46.3.dist-info/licenses/LICENSE.txt`; `setuptools/_vendor/zipp-3.23.0.dist-info/licenses/LICENSE`; `setuptools/config/NOTICE`; `setuptools/config/_validate_pyproject/NOTICE`.
- The listed `_vendor`, `packaging/licenses`, and `config/NOTICE` paths are retained as bundled/vendor evidence. Each corresponding source-tag blob URL is present in `upstream.notice_paths`; `bundled_source` and `vendored` remain blocked.
- Present upstream review references: [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/autocommand-2.2.2.dist-info/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/licenses/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/jaraco.text-4.0.0.dist-info/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/jaraco_context-6.1.0.dist-info/licenses/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/licenses/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/more_itertools-10.8.0.dist-info/licenses/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE>), [`LICENSE.APACHE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.APACHE>), [`LICENSE.BSD`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.BSD>), [`__init__.py`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/packaging/licenses/__init__.py>), [`_spdx.py`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/packaging/licenses/_spdx.py>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/platformdirs-4.4.0.dist-info/licenses/LICENSE>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/tomli-2.4.0.dist-info/licenses/LICENSE>), [`LICENSE.txt`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/wheel-0.46.3.dist-info/licenses/LICENSE.txt>), [`LICENSE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/_vendor/zipp-3.23.0.dist-info/licenses/LICENSE>), [`NOTICE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/config/NOTICE>), [`NOTICE`](<https://github.com/pypa/setuptools/blob/v83.0.0/setuptools/config/_validate_pyproject/NOTICE>).

### `termcolor` `3.3.0`
- `sdist`: `termcolor-3.3.0/COPYING.txt`.
- `wheel`: `termcolor-3.3.0.dist-info/licenses/COPYING.txt`.
- Present upstream review references: [`COPYING.txt`](<https://github.com/termcolor/termcolor/blob/3.3.0/COPYING.txt>).

### `typing-inspection` `0.4.2`
- `sdist`: `typing_inspection-0.4.2/LICENSE`.
- `wheel`: `typing_inspection-0.4.2.dist-info/licenses/LICENSE`.
- Present upstream review references: [`LICENSE`](<https://github.com/pydantic/typing-inspection/blob/v0.4.2/LICENSE>).

### `urllib3` `2.6.3`
- `sdist`: `urllib3-2.6.3/LICENSE.txt`.
- `wheel`: `urllib3-2.6.3.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/urllib3/urllib3/blob/2.6.3/LICENSE.txt>).

### `werkzeug` `3.1.5`
- `sdist`: `werkzeug-3.1.5/LICENSE.txt`; `werkzeug-3.1.5/docs/license.rst`.
- `wheel`: `werkzeug-3.1.5.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/pallets/werkzeug/blob/3.1.5/LICENSE.txt>), [`license.rst`](<https://github.com/pallets/werkzeug/blob/3.1.5/docs/license.rst>).

### `wheel` `0.46.3`
- `sdist`: `wheel-0.46.3/LICENSE.txt`.
- `wheel`: `wheel-0.46.3.dist-info/licenses/LICENSE.txt`.
- Present upstream review references: [`LICENSE.txt`](<https://github.com/pypa/wheel/blob/0.46.3/LICENSE.txt>).

## Disposition boundary

The rows are restricted to `user_installed` and `not_distributed`; `bundled_source` and `built_companion` remain blocked. `mirrored`, `vendored`, `container_bundled`, `unknown`, `unavailable`, and `conflicting` conditions remain blocked. Any change to package identity, source, artifact, target, profile, archive contents, candidate identity, or reviewer evidence reopens the row.

The strict inventory remains nonzero because the rest of the selected closure is unresolved and unrepresented surfaces remain fail-closed. This batch is evidence preparation only and does not close Issue #8163, parent gate #8021, or release acceptance.
