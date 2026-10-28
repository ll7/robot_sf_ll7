# Acquisition and custody

The main acquisition and frozen contract are both at commit
`4e16d46908789c37651a4b851a23f10bf1540e7b`, reachable on the retained
`xperc-perception-degradation` branch. The starting release commit is
`373dbfde4f39667cf9e8732dabe7df5118bdeab1`. The packet-only branch is based
on later main and does not itself install the experimental runtime hook.

[`runtime.patch`](runtime.patch) is the zero-context release-to-acquisition diff for all
five changed runtime files and the two associated diagnostic tests. It
contains both the guard-switch ablation and the pedestrian-perception hook.
It can be applied to the release commit with `git apply --unidiff-zero`; its reverse applies
cleanly at the acquisition commit. The original preregistration is preserved
unchanged as [`../contract.yaml`](../contract.yaml).

The raw main, pilot and smoke stages have separately verified SHA-256 manifests
in the lab host archive. The pilot compact qualification is tracked in this
packet. The main-stage W&B URI, digest and file-manifest readback are recorded
in [`preservation.json`](preservation.json).
