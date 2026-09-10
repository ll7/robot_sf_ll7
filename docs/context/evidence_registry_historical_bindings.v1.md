# Historical evidence registry bindings

This manifest records two narrowly reviewed historical references that remain
valid for the current evidence records after the referenced source files moved
in the repository history. The linter verifies each record against immutable
Git commit, tree, blob, parent, and ancestry identities before it can suppress
an artifact checksum mismatch.

The consumer digest and blob identity are pinned separately for each record.
The forecast packet intentionally records its current consumer digest
(`c0a737...`) separately from the producer packet digest (`7e73...`); the
producer identity therefore does not claim that the whole current packet was
created at that commit. A binding applies only to the exact consumer path,
JSON pointer, reference path, and checksum occurrence. It provides provenance
for the historical source transition and does not make a rights, release, or
publication decision.
