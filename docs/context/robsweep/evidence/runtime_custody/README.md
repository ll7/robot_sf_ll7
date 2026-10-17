The episode commit `22172f6cacc8b3e63825793db604c820ff21080a` was recovered from the local
acquisition clone. Custody uses an archive, rather than a preserved branch. The original commit
object and all thirteen tree objects absent from public base
`33a4735900153e742828227cc2171436ea09cd90` are preserved byte for byte. SHA256 checksums and
artifact paths are in [identity.json](identity.json) and the study's [manifest](../SHA256SUMS).
Git's object hashes independently bind the commit to its complete tracked-tree identity.

[changes.raw.txt](changes.raw.txt) is the complete Git metadata diff against that public base.
All 157 changes are additions under the new diagnostic input/evidence namespaces and the acquisition
driver. Every existing public-base path, including simulator, planner, maps, dependencies, models
and release inputs, has the same Git blob identifier and mode. [runtime.diff.patch](runtime.diff.patch)
is empty. All 42 archived acquisition-source files also match the recovered commit's blob identifiers.
This includes the exact dirty-at-launch driver, which became tracked in the recovered commit.

Partial diagnostic output blobs from that intermediate commit are not republished. Their Git
identifiers are retained in the archived trees; the completed portable outcomes have separate custody
in the study bundle. This is runtime/input continuity evidence, rather than a complete checkout of
the intermediate output state. Later diagnostic tooling edits do not change the acquisition archive.

From the delivered repository root, run:

```bash
python docs/context/robsweep/evidence/runtime_custody/verify.py
sha256sum -c docs/context/robsweep/evidence/SHA256SUMS
```

To verify without any access to the local acquisition commit, fetch the public base into a fresh
repository and pass that repository to `verify.py --public-repository <public-clone>`. A shallow
fetch with `--filter=blob:none --depth=1` suffices. The verifier reconstructs Git metadata in a
temporary object directory, uses unchanged trees from the public base, checks the complete diff,
and binds archived source bytes to the recovered commit. It does not modify either repository or
need unavailable intermediate output blobs.
